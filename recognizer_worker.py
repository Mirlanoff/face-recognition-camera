"""
Background worker that processes a camera stream during an active lesson.

Optionally shows a native OpenCV window with live video + bbox + name + emotion overlay.
Set SHOW_WINDOW = False to run headless on a server.

Hotkeys (when window is focused):
Q / Esc  - close window (worker keeps running headless)
F        - toggle fullscreen
S        - save screenshot to screenshots/<lesson_id>_<ts>.jpg
Space    - pause/resume rendering (worker keeps running)
R        - force reconnect to camera
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
"rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream",
)
RECOG_EVERY_N_FRAMES = 8
EMOTION_EVERY_N_FRAMES = 15
MATCH_THRESHOLD = 0.45
NEG_EMOTION_ALERT_SECONDS = 15            # was 120 — faster trigger
NEG_EMOTION_ALERT_COOLDOWN_S = 60         # new — min seconds between repeated alerts for same student
NEG_EMOTIONS = {"sad", "angry", "fear", "disgust"}
EMOTION_LABELS = [
"neutral", "happiness", "surprise", "sadness",
"anger", "disgust", "fear", "contempt",
]
RECONNECT_DELAY_S = 5.0

# ---------- Live window config ----------
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "1") not in ("0", "false", "False")
WINDOW_DISPLAY_W = 1280     # downscale for display (camera is 1080p)
EMOTION_EMOJI = {
"happy": ":)", "neutral": "-", "surprise": "!",
"sad": ":(", "angry": ">:(", "disgust": "X(", "fear": "D:", "contempt": ":/",
}
SCREENSHOT_DIR = Path("screenshots")


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


# ---------- Drawing helpers ----------
def _color_for(state):
# state: "known_positive" | "known_neutral" | "known_negative" | "unknown"
return {
    "known_positive": (0, 200, 0),       # green
    "known_neutral":  (0, 200, 200),     # yellow
    "known_negative": (0, 0, 220),       # red
    "unknown":        (40, 40, 220),     # red-ish
}.get(state, (200, 200, 200))


def _draw_label(img, x1, y1, x2, y2, name, emotion, color):
cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
text = name
if emotion:
    text = name + "  " + emotion + " " + EMOTION_EMOJI.get(emotion, "")
(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
y_text = max(y1 - 8, th + 4)
cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
cv2.putText(img, text, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(img, lesson_id, fps, recognized_count, visible_count, alerts_count, paused):
h, w = img.shape[:2]
# Top-left status bar
cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
status = "PAUSED" if paused else "LIVE"
line = "[" + status + "]  Lesson #" + str(lesson_id) + "   FPS: " + str(int(fps)) + \
       "   Visible: " + str(visible_count) + "/" + str(recognized_count) + \
       "   Alerts: " + str(alerts_count)
cv2.putText(img, line, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
# Bottom-right hint
hint = "Q-quit  F-fullscreen  S-screenshot  Space-pause  R-reconnect"
(tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
cv2.putText(img, hint, (w - tw - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


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
    self.alerts_fired = set()                # kept for HUD counter (total alerts this session)
    self.last_alert_at = {}                  # new — (student_id, alert_type) -> datetime of last fire (for cooldown)
    self.alerts_count = 0                    # new — total alert fires this session (for HUD)
    self.currently_visible = set()
    self.VISIBLE_TIMEOUT_S = 3.0

    # Window state
    self.window_name = "Camera - Lesson #" + str(lesson_id)
    self.window_open = False
    self.fullscreen = False
    self.paused = False
    self.last_overlay = {}   # student_id -> (bbox, name, emotion, ts) for drawing between recog frames
    self.OVERLAY_TTL_S = 1.0
    self.fps_ema = 0.0
    self.last_frame_ts = None

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
    # Cooldown: same (student, alert_type) can re-fire after NEG_EMOTION_ALERT_COOLDOWN_S
    key = (student_id, alert_type)
    now = datetime.utcnow()
    last = self.last_alert_at.get(key)
    if last is not None and (now - last).total_seconds() < NEG_EMOTION_ALERT_COOLDOWN_S:
        return
    self.last_alert_at[key] = now
    self.alerts_fired.add(key)
    self.alerts_count += 1
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
                    student_name + " " + str(NEG_EMOTION_ALERT_SECONDS) + " секунд подряд испытывает негативные эмоции (" + emotion + ")",
                )
                # Reset timer so next 15-sec window can trigger another alert (subject to 60s cooldown in fire_alert)
                self.neg_emotion_start[student_id] = now
    else:
        self.neg_emotion_start.pop(student_id, None)
    self.last_emotion[student_id] = emotion

# ---------- Window helpers ----------
def _ensure_window(self):
    if not SHOW_WINDOW or self.window_open:
        return
    try:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, WINDOW_DISPLAY_W, int(WINDOW_DISPLAY_W * 9 / 16))
        self.window_open = True
    except Exception as e:
        print("[worker] cannot open window: " + str(e))

def _close_window(self):
    if self.window_open:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self.window_open = False

def _toggle_fullscreen(self):
    if not self.window_open:
        return
    self.fullscreen = not self.fullscreen
    try:
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )
    except Exception:
        pass

def _save_screenshot(self, frame):
    try:
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / ("lesson" + str(self.lesson_id) + "_" + ts + ".jpg")
        cv2.imwrite(str(path), frame)
        print("[worker] screenshot saved: " + str(path))
    except Exception as e:
        print("[worker] screenshot error: " + str(e))

def _classify_state(self, sid, emotion):
    if sid is None:
        return "unknown"
    if emotion in NEG_EMOTIONS:
        return "known_negative"
    if emotion in ("happy", "surprise"):
        return "known_positive"
    return "known_neutral"

def _render_and_show(self, frame, roster):
    if not SHOW_WINDOW:
        return True   # keep running

    self._ensure_window()
    if not self.window_open:
        return True

    # FPS
    now = time.time()
    if self.last_frame_ts is not None:
        dt = now - self.last_frame_ts
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst_fps if self.fps_ema > 0 else inst_fps
    self.last_frame_ts = now

    # Draw overlays from recent recognitions
    display = frame.copy()
    cutoff = time.time() - self.OVERLAY_TTL_S
    stale = [k for k, v in self.last_overlay.items() if v[3] < cutoff]
    for k in stale:
        self.last_overlay.pop(k, None)

    for key, (bbox, name, emotion, _ts) in self.last_overlay.items():
        x1, y1, x2, y2 = bbox
        state = self._classify_state(key if isinstance(key, int) else None, emotion)
        color = _color_for(state)
        _draw_label(display, x1, y1, x2, y2, name, emotion, color)

    recognized_count = len(roster)
    _draw_hud(
        display, self.lesson_id, self.fps_ema,
        recognized_count, len(self.currently_visible),
        self.alerts_count, self.paused,
    )

    # Downscale for display if huge
    h, w = display.shape[:2]
    if w > WINDOW_DISPLAY_W:
        scale = WINDOW_DISPLAY_W / float(w)
        display = cv2.resize(display, (WINDOW_DISPLAY_W, int(h * scale)))

    try:
        cv2.imshow(self.window_name, display)
    except Exception:
        self.window_open = False
        return True

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):    # Q or Esc
        print("[worker] window closed by user (Q/Esc) - going headless")
        self._close_window()
    elif key in (ord('f'), ord('F')):
        self._toggle_fullscreen()
    elif key in (ord('s'), ord('S')):
        self._save_screenshot(display)
    elif key == 32:    # Space
        self.paused = not self.paused
    elif key in (ord('r'), ord('R')):
        return False   # signal reconnect

    return True

def run(self):
    print("[worker " + str(self.lesson_id) + "] starting (SHOW_WINDOW=" + str(SHOW_WINDOW) + ")")
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

    self._close_window()
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

        do_recog = (frame_idx % RECOG_EVERY_N_FRAMES == 0) and (not self.paused)
        do_emotion = (frame_idx % EMOTION_EVERY_N_FRAMES == 0) and (not self.paused)

        if do_recog:
            try:
                small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
                faces = app_if.get(small)
            except Exception as e:
                print("[worker] insightface error: " + str(e))
                faces = []

            seen_this_pass = []

            for f in faces:
                emb = f.normed_embedding
                sid, sim = self.match_student(emb, roster)

                # rescale bbox to full frame
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                x1 = int(x1 / 0.7); y1 = int(y1 / 0.7)
                x2 = int(x2 / 0.7); y2 = int(y2 / 0.7)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                if sid is None:
                    # unknown - still draw overlay
                    self.last_overlay["unk_" + str(id(f))] = (
                        (x1, y1, x2, y2), "?", "", time.time(),
                    )
                    continue

                student_name = next((st["name"] for st in roster if st["id"] == sid), "?")
                state = self.mark_attendance(sid)
                self.last_seen[sid] = datetime.utcnow()
                seen_this_pass.append(sid)

                if state == "entered":
                    self.push_event({
                        "type": "student_entered",
                        "lesson_id": self.lesson_id,
                        "student_id": sid,
                        "student_name": student_name,
                        "similarity": round(sim, 3),
                    })
                    self.currently_visible.add(sid)

                emo_for_overlay = self.last_emotion.get(sid, "")

                if do_emotion and x2 > x1 and y2 > y1:
                    try:
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
                        emo_for_overlay = emo
                    except Exception as e:
                        print("[worker] emotion error: " + str(e))

                # Update overlay for this student
                self.last_overlay[sid] = (
                    (x1, y1, x2, y2), student_name, emo_for_overlay, time.time(),
                )

            # cleanup currently_visible
            now = datetime.utcnow()
            left = []
            for sid in list(self.currently_visible):
                last = self.last_seen.get(sid)
                if last is None or (now - last).total_seconds() > self.VISIBLE_TIMEOUT_S:
                    left.append(sid)
            for sid in left:
                self.currently_visible.discard(sid)

        # Render window every frame (uses cached overlays)
        keep_going = self._render_and_show(frame, roster)
        if not keep_going:
            # Reconnect requested via R
            print("[worker] reconnect requested via hotkey")
            cap.release()
            cap = open_camera()
            if cap is None:
                self.push_event({
                    "type": "worker_error",
                    "lesson_id": self.lesson_id,
                    "message": "Кайра туташуу ишке ашкан жок",
                })
                return

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
