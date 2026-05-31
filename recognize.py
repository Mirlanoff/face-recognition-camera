"""
Real-time multi-face recognition + emotion detection.

Stack:
- ffmpeg subprocess with HW decode (d3d11va/dxva2/sw fallback)
- InsightFace (SCRFD detector + ArcFace embeddings, ONNX runtime, CPU)
- ONNX FER+ emotion model
- IoU tracker: identity every 2s, emotion every 1s per track

InsightFace is the SOTA face recognition system used in production.
On CPU via ONNX Runtime it is 5-10x faster than dlib (face_recognition lib).
"""

import os
import sys
import time
import threading
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
load_dotenv()

# ===== Config =====
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream"
)
STUDENTS_DIR = Path("students")
EMOTION_MODEL_PATH = Path("models") / "emotion-ferplus-8.onnx"

# Cosine similarity threshold for ArcFace embeddings.
# 0.40 = strict, 0.35 = balanced, 0.30 = permissive.
SIMILARITY_THRESHOLD = 0.35

CONNECT_TIMEOUT_SECONDS = 15

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# InsightFace processes resized frame internally; this is detector input size.
DET_SIZE = 640

DISPLAY_SCALE = 1.0

IOU_MATCH_THRESHOLD = 0.3
TRACK_TTL_SEC = 1.5
IDENTITY_INTERVAL_SEC = 2.0
EMOTION_INTERVAL_SEC = 1.0

HWACCEL_ATTEMPTS = ["d3d11va", "dxva2", None]

EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


# ===== InsightFace =====
def load_insightface():
    """
    Loads InsightFace model 'buffalo_l' (SCRFD-10G detector + ArcFace r50).
    First run downloads ~250MB to ~/.insightface/models/buffalo_l/.
    """
    from insightface.app import FaceAnalysis
    print("[+] Loading InsightFace (buffalo_l, CPU)...")
    print("    First run downloads ~250MB; subsequent runs are instant.")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
    print("[+] InsightFace ready.")
    return app


def load_known_faces(app, folder):
    """Build embedding DB from students/*.jpg|jpeg|png."""
    known_embs = []
    known_names = []
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print("[!] Created empty students/. Add photos and re-run.")
        return known_embs, known_names

    photos = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
    if not photos:
        print("[!] No photos in students/.")
        return known_embs, known_names

    for photo_path in photos:
        name = photo_path.stem
        print("[+] Loading " + name + "...")
        img = cv2.imread(str(photo_path))
        if img is None:
            print("    [!] Could not read " + photo_path.name + ", skipping.")
            continue
        faces = app.get(img)
        if len(faces) == 0:
            print("    [!] No face in " + photo_path.name + ", skipping.")
            continue
        # Pick largest face if multiple
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding  # L2-normalized, shape (512,)
        known_embs.append(emb)
        known_names.append(name)

    print("")
    print("[+] " + str(len(known_names)) + " students loaded: " + ", ".join(known_names))
    print("")
    return known_embs, known_names


def match_face(emb, known_embs, known_names, threshold):
    """Cosine similarity (embeddings are already L2-normalized, so it's just dot product)."""
    if len(known_embs) == 0:
        return "Unknown", 0.0
    sims = np.dot(np.stack(known_embs), emb)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return known_names[best_idx], best_sim
    return "Unknown", best_sim


# ===== Emotion =====
def load_emotion_session():
    if not EMOTION_MODEL_PATH.exists():
        print("[!] Emotion model not found. Run: python download_emotion_model.py")
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(EMOTION_MODEL_PATH), providers=["CPUExecutionProvider"])
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


# ===== Tracking =====
def iou(box_a, box_b):
    # box format: (top, right, bottom, left)
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
        self.box = box  # (top, right, bottom, left)
        self.name = ""
        self.emotion = ""
        self.score = 0.0
        self.last_seen = now
        self.last_identity_time = 0.0
        self.last_emotion_time = 0.0

    def needs_identity(self, now):
        return (not self.name) or (now - self.last_identity_time >= IDENTITY_INTERVAL_SEC)

    def needs_emotion(self, now):
        return (now - self.last_emotion_time >= EMOTION_INTERVAL_SEC)


# ===== FFmpeg HW decoder =====
def build_ffmpeg_cmd(ffmpeg_path, url, width, height, hwaccel):
    cmd = [ffmpeg_path, "-loglevel", "warning"]
    if hwaccel is not None:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-rtsp_transport", "tcp",
        "-timeout", str(CONNECT_TIMEOUT_SECONDS * 1000000),
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", url,
        "-an",
        "-vf", "scale=" + str(width) + ":" + str(height),
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-"
    ]
    return cmd


def _stderr_pump(stream, label):
    try:
        for line in iter(stream.readline, b""):
            try:
                text = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                text = str(line)
            if text:
                print("[ffmpeg/" + label + "] " + text)
    except Exception:
        pass


def probe_ffmpeg_pipeline(ffmpeg_path, url, width, height, hwaccel, frame_size):
    label = hwaccel if hwaccel else "sw"
    print("[?] Probing ffmpeg with hwaccel=" + label + " ...")
    cmd = build_ffmpeg_cmd(ffmpeg_path, url, width, height, hwaccel)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * 1024 * 1024
        )
    except Exception as e:
        print("    [X] failed to spawn: " + str(e))
        return None

    t = threading.Thread(target=_stderr_pump, args=(proc.stderr, label), daemon=True)
    t.start()

    deadline = time.time() + CONNECT_TIMEOUT_SECONDS
    buf = b""
    while time.time() < deadline:
        if proc.poll() is not None:
            print("    [X] ffmpeg exited early (code " + str(proc.returncode) + ")")
            return None
        try:
            chunk = proc.stdout.read(frame_size - len(buf))
        except Exception:
            chunk = b""
        if not chunk:
            time.sleep(0.05)
            continue
        buf += chunk
        if len(buf) >= frame_size:
            print("    [+] First frame received with hwaccel=" + label)
            return proc

    print("    [X] timeout — no frames with hwaccel=" + label)
    try:
        proc.terminate()
    except Exception:
        pass
    return None


class FFmpegCameraReader(threading.Thread):
    def __init__(self, url, width, height):
        super().__init__(daemon=True)
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.hwaccel = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.ffmpeg_path = shutil.which("ffmpeg")

    def start_pipeline(self):
        if self.ffmpeg_path is None:
            print("[X] ffmpeg not on PATH")
            return False
        for hw in HWACCEL_ATTEMPTS:
            proc = probe_ffmpeg_pipeline(
                self.ffmpeg_path, self.url, self.width, self.height, hw, self.frame_size
            )
            if proc is not None:
                self.process = proc
                self.hwaccel = hw
                print("[+] Using hwaccel: " + (hw if hw else "software"))
                return True
        return False

    def run(self):
        while self.running:
            if self.process is None or self.process.poll() is not None:
                if self.process is not None:
                    print("[!] ffmpeg exited, restarting...")
                time.sleep(0.5)
                if not self.start_pipeline():
                    print("[!] All hwaccel attempts failed. Retrying in 3s...")
                    time.sleep(3.0)
                    continue
            try:
                raw = self.process.stdout.read(self.frame_size)
            except Exception:
                raw = b""
            if len(raw) != self.frame_size:
                try:
                    self.process.terminate()
                except Exception:
                    pass
                self.process = None
                time.sleep(0.2)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.latest = frame.copy()

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        try:
            if self.process is not None:
                self.process.terminate()
        except Exception:
            pass


# ===== Recognizer thread =====
class InsightFaceRecognizer(threading.Thread):
    def __init__(self, camera, app, known_embs, known_names, emotion_sess):
        super().__init__(daemon=True)
        self.camera = camera
        self.app = app
        self.known_embs = known_embs
        self.known_names = known_names
        self.emotion_sess = emotion_sess
        self.lock = threading.Lock()
        self.tracks = []
        self.running = True

    def _detect_and_recognize(self, frame):
        """
        Returns list of (box_tuple, embedding) for each detected face.
        box_tuple is (top, right, bottom, left) in original frame coords.
        """
        faces = self.app.get(frame)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            box = (y1, x2, y2, x1)
            out.append((box, f.normed_embedding))
        return out

    def _match_tracks(self, detections, now):
        """detections: list of (box, embedding). Returns list of (track, det_idx)."""
        assignments = []
        used_track_ids = set()
        used_det_idx = set()

        for det_idx, (det_box, _) in enumerate(detections):
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

        for det_idx, (det_box, _) in enumerate(detections):
            if det_idx in used_det_idx:
                continue
            new_t = Track(det_box, now)
            self.tracks.append(new_t)
            assignments.append((new_t, det_idx))

        self.tracks = [t for t in self.tracks if now - t.last_seen < TRACK_TTL_SEC]
        return assignments

    def run(self):
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                detections = self._detect_and_recognize(frame)
                if not self.running:
                    break
                now = time.time()

                with self.lock:
                    assignments = self._match_tracks(detections, now)

                # Update identity (rate-limited per track)
                for track, det_idx in assignments:
                    if track.needs_identity(now):
                        emb = detections[det_idx][1]
                        name, sim = match_face(
                            emb, self.known_embs, self.known_names, SIMILARITY_THRESHOLD
                        )
                        with self.lock:
                            track.name = name
                            track.score = sim
                            track.last_identity_time = now

                # Update emotion (rate-limited per track)
                if self.emotion_sess is not None:
                    for track, det_idx in assignments:
                        if track.needs_emotion(now):
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
                if self.running:
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
                if name == "...":
                    color = (0, 165, 255)
                elif name == "Unknown" or name == "":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                out.append((top, right, bottom, left, label, color))
            return out

    def stop(self):
        self.running = False


# ===== Main =====
def main():
    print("=" * 60)
    print("Multi-Face Recognition + Emotion (InsightFace + HW decode)")
    print("=" * 60)

    if shutil.which("ffmpeg") is None:
        print("[X] ffmpeg not found on PATH.")
        sys.exit(1)

    # Load InsightFace
    try:
        app = load_insightface()
    except Exception as e:
        print("[X] Failed to load InsightFace: " + str(e))
        print("    Install: pip install insightface onnxruntime")
        return

    known_embs, known_names = load_known_faces(app, STUDENTS_DIR)
    if len(known_embs) == 0:
        print("[!] Add student photos to students/ and re-run.")
        return

    emotion_sess = load_emotion_session()

    print("[+] Starting ffmpeg pipeline: " + RTSP_URL)
    print("    Output: " + str(FRAME_WIDTH) + "x" + str(FRAME_HEIGHT) + " BGR24")
    camera = FFmpegCameraReader(RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT)

    if not camera.start_pipeline():
        print("[X] Could not start ffmpeg.")
        return

    camera.start()

    t0 = time.time()
    while camera.read() is None:
        if time.time() - t0 > CONNECT_TIMEOUT_SECONDS:
            print("[X] Pipeline started but no decoded frames. Exiting.")
            camera.stop()
            return
        time.sleep(0.1)

    print("[+] Stream alive. Starting recognizer thread...")
    recognizer = InsightFaceRecognizer(camera, app, known_embs, known_names, emotion_sess)
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
                "FPS: " + str(round(fps_value, 1)) + "  Tracks: " + str(len(results)) +
                "  HW: " + (camera.hwaccel if camera.hwaccel else "sw") + "  Engine: InsightFace",
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
        time.sleep(0.2)
        camera.stop()
        cv2.destroyAllWindows()
        print("")
        print("[+] Stopped.")


if __name__ == "__main__":
    main()
