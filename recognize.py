"""
Real-time multi-face recognition + emotion detection with IoU tracker.

Industrial-style pipeline that runs on a CPU laptop:
- Detect faces every frame (MediaPipe, ~30ms)
- Track them across frames by bounding-box IoU
- Identify (who is this) only every IDENTITY_INTERVAL_SEC per track
- Emotion only every EMOTION_INTERVAL_SEC per track
- Between updates: keep showing the last known name/emotion from the track

This is the same trick used by Class Insight / DeepSORT-based engagement systems.

Run once:
    python download_emotion_model.py

Then:
    python recognize.py
"""

import os
import time
import threading
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from pathlib import Path
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
load_dotenv()

# ===== Config =====
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream"
)
STUDENTS_DIR = Path("students")
MODEL_PATH = Path("models") / "emotion-ferplus-8.onnx"
TOLERANCE = 0.55
CONNECT_TIMEOUT_SECONDS = 10

MP_MODEL_SELECTION = 1
MP_MIN_CONFIDENCE = 0.5

RECOGNITION_SCALE = 0.5
DISPLAY_SCALE = 0.7

# Tracker thresholds
IOU_MATCH_THRESHOLD = 0.3    # below this -> new track
TRACK_TTL_SEC = 1.5          # drop track if not seen for this long

# How often per track we run the slow models
IDENTITY_INTERVAL_SEC = 2.0
EMOTION_INTERVAL_SEC = 1.0

EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


def load_known_faces(folder):
    known_encodings = []
    known_names = []
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print("[!] Created empty students/. Add photos and re-run.")
        return known_encodings, known_names

    photos = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
    if not photos:
        print("[!] No photos in students/.")
        return known_encodings, known_names

    for photo_path in photos:
        name = photo_path.stem
        print("[+] Loading " + name + "...")
        image = face_recognition.load_image_file(str(photo_path))
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print("    [!] No face in " + photo_path.name + ", skipping.")
            continue
        known_encodings.append(encodings[0])
        known_names.append(name)

    print("")
    print("[+] " + str(len(known_names)) + " students loaded: " + ", ".join(known_names))
    print("")
    return known_encodings, known_names


def open_rtsp(url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp"
        "|stimeout;" + str(CONNECT_TIMEOUT_SECONDS * 1000000) +
        "|max_delay;500000"
        "|buffer_size;1024000"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def load_emotion_session():
    if not MODEL_PATH.exists():
        print("[!] Emotion model not found. Run: python download_emotion_model.py")
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        print("[+] Emotion model loaded.")
        return sess
    except Exception as e:
        print("[!] Could not load emotion model: " + str(e))
        return None


def predict_emotion(sess, gray_face_64):
    if sess is None:
        return ""
    try:
        x = gray_face_64.astype(np.float32).reshape(1, 1, 64, 64)
        scores = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        idx = int(np.argmax(scores[0]))
        if 0 <= idx < len(EMOTION_LABELS):
            return EMOTION_LABELS[idx]
        return ""
    except Exception:
        return ""


def iou(box_a, box_b):
    """box format: (top, right, bottom, left)"""
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b
    inter_left = max(a_left, b_left)
    inter_top = max(a_top, b_top)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    a_area = (a_right - a_left) * (a_bottom - a_top)
    b_area = (b_right - b_left) * (b_bottom - b_top)
    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class Track:
    _next_id = 1

    def __init__(self, box, now):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box                  # (top, right, bottom, left), full-res coords
        self.name = ""
        self.emotion = ""
        self.last_seen = now
        self.last_identity_time = 0.0
        self.last_emotion_time = 0.0

    def needs_identity(self, now):
        return (not self.name) or (now - self.last_identity_time >= IDENTITY_INTERVAL_SEC)

    def needs_emotion(self, now):
        return (now - self.last_emotion_time >= EMOTION_INTERVAL_SEC)


class CameraReader(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = open_rtsp(url)
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.fail_count = 0

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.3)
                self.cap = open_rtsp(self.url)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.fail_count += 1
                if self.fail_count > 8:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    print("[!] Decoder unhealthy, reopening RTSP...")
                    self.cap = open_rtsp(self.url)
                    self.fail_count = 0
                time.sleep(0.02)
                continue
            self.fail_count = 0
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class TrackedRecognizer(threading.Thread):
    """
    Detect every iteration, but identify + emotion at throttled per-track intervals.
    """
    def __init__(self, camera, known_encodings, known_names, emotion_sess):
        super().__init__(daemon=True)
        self.camera = camera
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.emotion_sess = emotion_sess
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=MP_MODEL_SELECTION,
            min_detection_confidence=MP_MIN_CONFIDENCE,
        )
        self.lock = threading.Lock()
        self.tracks = []     # list of Track
        self.scale_back = 1.0 / RECOGNITION_SCALE
        self.running = True

    def _detect_boxes(self, frame):
        small = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
        h_s, w_s = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        det = self.detector.process(rgb)

        boxes_full = []   # in full-res coords
        boxes_small = []  # for face_encodings (small frame)
        if det.detections:
            for d in det.detections:
                b = d.location_data.relative_bounding_box
                x1 = max(0, int(b.xmin * w_s))
                y1 = max(0, int(b.ymin * h_s))
                bw = int(b.width * w_s)
                bh = int(b.height * h_s)
                x2 = min(w_s, x1 + bw)
                y2 = min(h_s, y1 + bh)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                boxes_small.append((y1, x2, y2, x1))
                top = int(y1 * self.scale_back)
                right = int(x2 * self.scale_back)
                bottom = int(y2 * self.scale_back)
                left = int(x1 * self.scale_back)
                boxes_full.append((top, right, bottom, left))
        return rgb, boxes_small, boxes_full

    def _match_tracks(self, boxes_full, now):
        """Match detection boxes to existing tracks by IoU. Returns list of (track, box_full, idx_small)."""
        assignments = []
        used_track_ids = set()
        used_det_idx = set()

        # Greedy: for each detection, find best track
        for det_idx, det_box in enumerate(boxes_full):
            best_track = None
            best_iou = IOU_MATCH_THRESHOLD
            for t in self.tracks:
                if t.id in used_track_ids:
                    continue
                v = iou(det_box, t.box)
                if v > best_iou:
                    best_iou = v
                    best_track = t
            if best_track is not None:
                best_track.box = det_box
                best_track.last_seen = now
                assignments.append((best_track, det_idx))
                used_track_ids.add(best_track.id)
                used_det_idx.add(det_idx)

        # Unmatched detections -> create new tracks
        for det_idx, det_box in enumerate(boxes_full):
            if det_idx in used_det_idx:
                continue
            new_t = Track(det_box, now)
            self.tracks.append(new_t)
            assignments.append((new_t, det_idx))

        # Drop stale tracks
        self.tracks = [t for t in self.tracks if now - t.last_seen < TRACK_TTL_SEC]

        return assignments

    def run(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                rgb_small, boxes_small, boxes_full = self._detect_boxes(frame)
                now = time.time()

                with self.lock:
                    assignments = self._match_tracks(boxes_full, now)

                # Identify: collect only tracks that need it
                idents_needed = []
                idents_boxes = []
                emos_needed = []

                with self.lock:
                    for track, det_idx in assignments:
                        if track.needs_identity(now):
                            idents_needed.append(track)
                            idents_boxes.append(boxes_small[det_idx])
                        if track.needs_emotion(now) and self.emotion_sess is not None:
                            emos_needed.append((track, det_idx))

                # Batch identity encodings only for tracks that need it
                if idents_needed:
                    encs = face_recognition.face_encodings(rgb_small, idents_boxes)
                    for track, enc in zip(idents_needed, encs):
                        distances = face_recognition.face_distance(self.known_encodings, enc)
                        if len(distances) > 0:
                            best = int(np.argmin(distances))
                            name = self.known_names[best] if distances[best] < TOLERANCE else "Unknown"
                        else:
                            name = "Unknown"
                        with self.lock:
                            track.name = name
                            track.last_identity_time = now

                # Emotions for tracks that need it
                for track, det_idx in emos_needed:
                    top, right, bottom, left = track.box
                    face_bgr = frame[top:bottom, left:right]
                    if face_bgr.size > 0:
                        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                        gray64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                        emo = predict_emotion(self.emotion_sess, gray64)
                        with self.lock:
                            track.emotion = emo
                            track.last_emotion_time = now

            except Exception as e:
                print("[!] Recognizer error: " + str(e))
                time.sleep(0.1)

    def get_snapshot(self):
        with self.lock:
            out = []
            for t in self.tracks:
                top, right, bottom, left = t.box
                name = t.name if t.name else "..."
                emotion = t.emotion
                label = name + " (" + emotion + ")" if emotion else name
                color = (0, 255, 0) if name not in ("", "Unknown", "...") else (
                    (0, 165, 255) if name == "..." else (0, 0, 255)
                )
                out.append((top, right, bottom, left, label, color))
            return out

    def stop(self):
        self.running = False
        try:
            self.detector.close()
        except Exception:
            pass


def main():
    print("=" * 60)
    print("Multi-Face Recognition + Emotion (tracked)")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)
    if len(known_encodings) == 0:
        print("[!] Add student photos to students/ and re-run.")
        return

    emotion_sess = load_emotion_session()

    print("[+] Connecting to camera (TCP): " + RTSP_URL)
    camera = CameraReader(RTSP_URL)
    camera.start()

    t0 = time.time()
    while camera.read() is None:
        if time.time() - t0 > CONNECT_TIMEOUT_SECONDS:
            print("[X] No frames from camera within timeout. Exiting.")
            camera.stop()
            return
        time.sleep(0.1)

    print("[+] Camera streaming. Starting tracker + recognizer...")
    recognizer = TrackedRecognizer(camera, known_encodings, known_names, emotion_sess)
    recognizer.start()

    print("[+] Press 'q' to quit.")
    print("")

    fps_t0 = time.time()
    fps_n = 0
    fps_value = 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            results = recognizer.get_snapshot()
            for (top, right, bottom, left, label, color) in results:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(frame, (left, bottom), (left + tw + 12, bottom + th + 12), color, cv2.FILLED)
                cv2.putText(
                    frame, label,
                    (left + 6, bottom + th + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (255, 255, 255), 1
                )

            fps_n += 1
            if fps_n >= 15:
                now = time.time()
                fps_value = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0

            cv2.putText(
                frame,
                "FPS: " + str(round(fps_value, 1)) + "  Tracks: " + str(len(results)),
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1
            )

            if DISPLAY_SCALE != 1.0:
                disp = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            else:
                disp = frame

            cv2.imshow("Recognition + Emotion (q to quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("")
        print("[+] Stopped.")


if __name__ == "__main__":
    main()
