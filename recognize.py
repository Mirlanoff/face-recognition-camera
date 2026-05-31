"""
Real-time multi-face recognition + emotion detection from IP camera (RTSP).
Threaded pipeline: display thread is never blocked by detection/recognition.

Run once:
python download_emotion_model.py

Then:
python recognize.py
"""

import os
import time
import threading
import queue
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

# MediaPipe: 0 = short-range (<2m), 1 = full-range (up to 5m)
MP_MODEL_SELECTION = 1
MP_MIN_CONFIDENCE = 0.5

# Downscale frame before recognition to speed it up (recognition is the slow part).
# 0.5 = process half-resolution. 0.4 even faster, 0.6 more accurate.
RECOGNITION_SCALE = 0.5

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
        "rtsp_transport;tcp|stimeout;" + str(CONNECT_TIMEOUT_SECONDS * 1000000)
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


class CameraReader(threading.Thread):
    """Reads frames as fast as the camera produces them, keeps only the latest."""
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
                time.sleep(0.5)
                self.cap = open_rtsp(self.url)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.fail_count += 1
                if self.fail_count > 5:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = open_rtsp(self.url)
                    self.fail_count = 0
                time.sleep(0.05)
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


class Recognizer(threading.Thread):
    """Background thread: pulls latest frame, detects + identifies + emotion. Updates shared results."""
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
        self.results = []  # list of (top, right, bottom, left, label, color)
        self.scale_back = 1.0 / RECOGNITION_SCALE
        self.running = True

    def run(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                # Downscale frame for ALL detection/recognition work
                small = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
                h_s, w_s = small.shape[:2]
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                det = self.detector.process(rgb)
                new_results = []
                boxes_small = []

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

                if boxes_small:
                    encodings = face_recognition.face_encodings(rgb, boxes_small)
                else:
                    encodings = []

                for (top_s, right_s, bottom_s, left_s), enc in zip(boxes_small, encodings):
                    # Identify
                    distances = face_recognition.face_distance(self.known_encodings, enc)
                    if len(distances) > 0:
                        best = int(np.argmin(distances))
                        name = self.known_names[best] if distances[best] < TOLERANCE else "Unknown"
                    else:
                        name = "Unknown"

                    # Scale box back to full-resolution coords for the display thread
                    top = int(top_s * self.scale_back)
                    right = int(right_s * self.scale_back)
                    bottom = int(bottom_s * self.scale_back)
                    left = int(left_s * self.scale_back)

                    # Emotion from full-res crop
                    emotion = ""
                    if self.emotion_sess is not None:
                        face_bgr = frame[top:bottom, left:right]
                        if face_bgr.size > 0:
                            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                            gray64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                            emotion = predict_emotion(self.emotion_sess, gray64)

                    label = name + " (" + emotion + ")" if emotion else name
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    new_results.append((top, right, bottom, left, label, color))

                with self.lock:
                    self.results = new_results
            except Exception as e:
                # Never let the worker die silently
                print("[!] Recognizer error: " + str(e))
                time.sleep(0.1)

    def get_results(self):
        with self.lock:
            return list(self.results)

    def stop(self):
        self.running = False
        try:
            self.detector.close()
        except Exception:
            pass


def main():
    print("=" * 60)
    print("Multi-Face Recognition + Emotion (threaded)")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)
    if len(known_encodings) == 0:
        print("[!] Add student photos to students/ and re-run.")
        return

    emotion_sess = load_emotion_session()

    print("[+] Connecting to camera (TCP): " + RTSP_URL)
    camera = CameraReader(RTSP_URL)
    camera.start()

    # Wait briefly for first frame
    t0 = time.time()
    while camera.read() is None:
        if time.time() - t0 > CONNECT_TIMEOUT_SECONDS:
            print("[X] No frames from camera within timeout. Exiting.")
            camera.stop()
            return
        time.sleep(0.1)

    print("[+] Camera streaming. Starting recognizer...")
    recognizer = Recognizer(camera, known_encodings, known_names, emotion_sess)
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

            results = recognizer.get_results()
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
                "FPS: " + str(round(fps_value, 1)) + "  Faces: " + str(len(results)),
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1
            )

            cv2.imshow("Recognition + Emotion (q to quit)", frame)
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
