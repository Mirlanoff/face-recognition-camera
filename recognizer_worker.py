"""
Background worker that processes a camera stream during an active lesson.
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from models import (
    SessionLocal, Student, Lesson, Attendance, EmotionLog, Alert,
)

# ---------- Config ----------
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:Akniet12%40@192.168.205.14:554/h264/ch1/main/av_stream",
)
RECOG_EVERY_N_FRAMES = 8
EMOTION_EVERY_N_FRAMES = 15
MATCH_THRESHOLD = 0.45
NEG_EMOTION_ALERT_SECONDS = 120
NEG_EMOTIONS = {"sad", "angry", "fear", "disgust"}
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]
RECONNECT_DELAY_S = 5.0


# ---------- Lazy model loaders ----------
_insightface = None
_emotion_session = None


def get_insightface():
    global _insightface
    if _insightface is None:
        from insightface.app import FaceAnalysis
        print("[worker] Loading InsightFace buffalo_l ...")
        _insightface = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _insightface.prepare(ctx_id=0, det_size=(640, 640))
    return _insightface


def get_emotion_session():
    global _emotion_session
    if _emotion_session is None:
        import onnxruntime as ort
        model_path = Path("models/emotion-ferplus-8.onnx")
        if not model_path.exists():
            print("[worker] WARNING: emotion model not found at " + str(model_path))
            return None
        print("[worker] Loading emotion model ferplus-8 ...")
        _emotion_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    return _emotion_session


def classify_emotion(face_bgr):
    sess = get_emotion_session()
    if sess is None:
        return "neutral"
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        blob = gray.astype(np.float32).reshape(1, 1, 64, 64)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        idx = int(np.argmax(out[0]))
        label = EMOTION_LABELS[idx] if 0 <= idx < len(EMOTION_LABELS) else "neutral"
        short = {"happiness": "happy", "sadness": "sad", "anger": "angry"}
        return short.get(label, label)
    except Exception:
        return "neutral"


def open_camera():
    """Open the RTSP stream. Returns VideoCapture or None."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|hwaccel;d3d11va|stimeout;5000000"
    )
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


# ---------- LessonWorker ----------
class LessonWorker(threading.Thread):
    def __init__(self, lesson_id, event_queue):
        super().__init__(daemon=True)
        self.lesson_id = lesson_id
        self.event_queue = event_queue
        self.loop = None
        self.stop_flag = threading.Event()
        self.last_seen = {}
        self.neg_emotion_start = {}
        self.last_emotion = {}
        self.alerts_fired = set()
        self.currently_visible = set()
        self.VISIBLE_TIMEOUT_S = 3.0

    def stop(self):
        self.stop_flag.set()

    def push_event(self, payload):
        if self.loop is None or self.event_queue is None:
            return
        try:
            self.loop.call_soon_threadsafe(self.event_queue.put_nowait, payload)
        except Exception:
            pass

    def load_class_students(self):
        s = SessionLocal()
        try:
            lesson = s.query(Lesson).filter(Lesson.id == self.lesson_id).first()
            if lesson is None:
                return [], None
            students = list(lesson.school_class.students)
            return [
                {
                    "id": st.id,
                    "name": st.full_name,
                    "embedding": np.array(st.get_embedding(), dtype=np.float32),
                    "photo_path": st.photo_path,
                }
                for st in students
            ], lesson.class_id
        finally:
            s.close()

    def mark_attendance(self, student_id):
        s = SessionLocal()
        try:
            existing = s.query(Attendance).filter(
                Attendance.lesson_id == self.lesson_id,
                Attendance.student_id == student_id,
            ).first()
            if existing is None:
                att = Attendance(
                    lesson_id=self.lesson_id,
                    student_id=student_id,
                    entered_at=datetime.utcnow(),
                )
                s.add(att)
                s.commit()
                return "entered"
            return "already"
        finally:
            s.close()

    def log_emotion(self, student_id, emotion):
        s = SessionLocal()
        try:
            row = EmotionLog(
                lesson_id=self.lesson_id,
                student_id=student_id,
                emotion=emotion,
                timestamp=datetime.utcnow(),
            )
            s.add(row)
            s.commit()
        finally:
            s.close()

    def fire_alert(self, student_id, alert_type, message):
        key = (student_id, alert_type)
        if key in self.alerts_fired:
            return
        self.alerts_fired.add(key)
        s = SessionLocal()
        try:
            a = Alert(
                lesson_id=self.lesson_id,
                student_id=student_id,
                alert_type=alert_type,
                message=message,
            )
            s.add(a)
            s.commit()
            self.push_event({
                "type": "alert",
                "lesson_id": self.lesson_id,
                "student_id": student_id,
                "alert_type": alert_type,
                "message": message,
            })
        finally:
            s.close()

    def match_student(self, embedding, roster):
        if len(roster) == 0:
            return None, 0.0
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_id = None
        best_sim = -1.0
        for st in roster:
            ref = st["embedding"]
            ref = ref / (np.linalg.norm(ref) + 1e-8)
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_id = st["id"]
        if best_sim >= MATCH_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    def update_emotion_state(self, student_id, emotion, student_name):
        now = datetime.utcnow()
        if emotion in NEG_EMOTIONS:
            start = self.neg_emotion_start.get(student_id)
            if start is None:
                self.neg_emotion_start[student_id] = now
            else:
                elapsed = (now - start).total_seconds()
                if elapsed >= NEG_EMOTION_ALERT_SECONDS:
                    self.fire_alert(
                        student_id,
                        "negative_emotion",
                        student_name + " 2 минуты подряд испытывает негативные эмоции (" + emotion + ")",
                    )
        else:
            self.neg_emotion_start.pop(student_id, None)
        self.last_emotion[student_id] = emotion

    def run(self):
        print("[worker " + str(self.lesson_id) + "] starting")
        roster, class_id = self.load_class_students()
        print("[worker " + str(self.lesson_id) + "] roster size = " + str(len(roster)))

        cap = open_camera()
        if cap is None:
            print("[worker " + str(self.lesson_id) + "] CAMERA OFFLINE - wait-and-retry")
            self.push_event({
                "type": "worker_started",
                "lesson_id": self.lesson_id,
                "roster_size": len(roster),
                "mode": "waiting",
            })
            self.push_event({
                "type": "worker_error",
                "lesson_id": self.lesson_id,
                "message": "Камера офлайн. Текшерип жатам...",
            })
            while not self.stop_flag.is_set():
                for _ in range(int(RECONNECT_DELAY_S * 10)):
                    if self.stop_flag.is_set():
                        break
                    time.sleep(0.1)
                if self.stop_flag.is_set():
                    break
                cap = open_camera()
                if cap is not None:
                    print("[worker " + str(self.lesson_id) + "] camera online, switching to live")
                    self.push_event({
                        "type": "worker_started",
                        "lesson_id": self.lesson_id,
                        "roster_size": len(roster),
                        "mode": "live",
                    })
                    self.run_live(cap, roster)
                    break
        else:
            self.push_event({
                "type": "worker_started",
                "lesson_id": self.lesson_id,
                "roster_size": len(roster),
                "mode": "live",
            })
            self.run_live(cap, roster)

        print("[worker " + str(self.lesson_id) + "] stopped")
        self.push_event({"type": "worker_stopped", "lesson_id": self.lesson_id})

    def run_live(self, cap, roster):
        frame_idx = 0
        consecutive_failures = 0
        app_if = get_insightface()

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 100:
                    print("[worker] too many read failures - reconnecting")
                    cap.release()
                    cap = open_camera()
                    if cap is None:
                        self.push_event({
                            "type": "worker_error",
                            "lesson_id": self.lesson_id,
                            "message": "Камера менен байланыш үзүлдү",
                        })
                        return
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            frame_idx += 1
            if frame_idx % RECOG_EVERY_N_FRAMES != 0:
                continue

            try:
                small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
                faces = app_if.get(small)
            except Exception as e:
                print("[worker] insightface error: " + str(e))
                continue

            do_emotion = (frame_idx % EMOTION_EVERY_N_FRAMES == 0)

            for f in faces:
                emb = f.normed_embedding
                sid, sim = self.match_student(emb, roster)
                if sid is None:
                    continue
                student_name = next((st["name"] for st in roster if st["id"] == sid), "?")
                state = self.mark_attendance(sid)
                self.last_seen[sid] = datetime.utcnow()

                if state == "entered":
                    self.push_event({
                        "type": "student_entered",
                        "lesson_id": self.lesson_id,
                        "student_id": sid,
                        "student_name": student_name,
                        "similarity": round(sim, 3),
                    })
                    self.currently_visible.add(sid)

                if do_emotion:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in f.bbox]
                        x1 = int(x1 / 0.7); y1 = int(y1 / 0.7)
                        x2 = int(x2 / 0.7); y2 = int(y2 / 0.7)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            emo = classify_emotion(crop)
                            prev = self.last_emotion.get(sid)
                            if emo != prev:
                                self.log_emotion(sid, emo)
                                self.update_emotion_state(sid, emo, student_name)
                                self.push_event({
                                    "type": "emotion",
                                    "lesson_id": self.lesson_id,
                                    "student_id": sid,
                                    "student_name": student_name,
                                    "emotion": emo,
                                })
                    except Exception as e:
                        print("[worker] emotion error: " + str(e))

            now = datetime.utcnow()
            left = []
            for sid in list(self.currently_visible):
                last = self.last_seen.get(sid)
                if last is None or (now - last).total_seconds() > self.VISIBLE_TIMEOUT_S:
                    left.append(sid)
            for sid in left:
                self.currently_visible.discard(sid)

        cap.release()


# ---------- Registry ----------
_workers = {}


def start_worker(lesson_id, event_queue, loop):
    if lesson_id in _workers and _workers[lesson_id].is_alive():
        return False
    w = LessonWorker(lesson_id, event_queue)
    w.loop = loop
    w.start()
    _workers[lesson_id] = w
    return True


def stop_worker(lesson_id):
    w = _workers.get(lesson_id)
    if w is None:
        return False
    w.stop()
    return True


def is_worker_running(lesson_id):
    w = _workers.get(lesson_id)
    return w is not None and w.is_alive()
