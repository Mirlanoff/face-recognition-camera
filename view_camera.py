"""
Standalone live camera viewer.

Runs independently from the FastAPI server. Connects to the RTSP camera,
performs face recognition + emotion classification (same as recognizer_worker),
and shows a native OpenCV window with bbox + name + emotion overlay.

Does NOT touch the database. Does NOT mark attendance. It is purely a viewer.

Usage:
    python view_camera.py
    python view_camera.py --no-emotion        # disable emotion model (faster)
    python view_camera.py --rtsp rtsp://...   # custom RTSP url

Hotkeys (window must be focused):
    Q / Esc   close
    F         toggle fullscreen
    S         save screenshot to screenshots/view_<ts>.jpg
    Space     pause / resume
    R         reconnect to camera
    + / -     increase / decrease recognition frequency
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from models import SessionLocal, Student

# ---------- Defaults ----------
DEFAULT_RTSP = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream",
)
MATCH_THRESHOLD = 0.45
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]
NEG_EMOTIONS = {"sad", "angry", "fear", "disgust"}
EMOTION_EMOJI = {
    "happy": ":)", "neutral": "-", "surprise": "!",
    "sad": ":(", "angry": ">:(", "disgust": "X(", "fear": "D:", "contempt": ":/",
}
WINDOW_NAME = "Camera Viewer"
WINDOW_DISPLAY_W = 1280
SCREENSHOT_DIR = Path("screenshots")


# ---------- Model loaders ----------
def load_insightface():
    from insightface.app import FaceAnalysis
    print("[viewer] Loading InsightFace buffalo_l ...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def load_emotion():
    import onnxruntime as ort
    model_path = Path("models/emotion-ferplus-8.onnx")
    if not model_path.exists():
        print("[viewer] emotion model not found at " + str(model_path) + " - disabled")
        return None
    print("[viewer] Loading emotion model ferplus-8 ...")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def classify_emotion(sess, face_bgr):
    if sess is None:
        return ""
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
        return ""


# ---------- Roster ----------
def load_all_students():
    s = SessionLocal()
    try:
        out = []
        for st in s.query(Student).all():
            emb = st.get_embedding()
            if emb is None or len(emb) == 0:
                continue
            out.append({
                "id": st.id,
                "name": st.full_name,
                "embedding": np.array(emb, dtype=np.float32),
            })
        return out
    finally:
        s.close()


def match_student(embedding, roster):
    if len(roster) == 0:
        return None, 0.0
    emb = embedding / (np.linalg.norm(embedding) + 1e-8)
    best_id = None
    best_sim = -1.0
    best_name = None
    for st in roster:
        ref = st["embedding"]
        ref = ref / (np.linalg.norm(ref) + 1e-8)
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_sim = sim
            best_id = st["id"]
            best_name = st["name"]
    if best_sim >= MATCH_THRESHOLD:
        return best_name, best_sim
    return None, best_sim


# ---------- Camera ----------
def open_camera(rtsp_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|hwaccel;d3d11va|stimeout;5000000"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


# ---------- Drawing ----------
def color_for(name, emotion):
    if name is None:
        return (40, 40, 220)            # red - unknown
    if emotion in NEG_EMOTIONS:
        return (0, 0, 220)              # red
    if emotion in ("happy", "surprise"):
        return (0, 200, 0)              # green
    return (0, 200, 200)                # yellow


def draw_label(img, x1, y1, x2, y2, name, emotion, sim, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = name if name else "?"
    if emotion:
        label = label + "  " + emotion + " " + EMOTION_EMOJI.get(emotion, "")
    if sim > 0:
        label = label + "  " + str(round(sim, 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_text = max(y1 - 8, th + 4)
    cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
    cv2.putText(img, label, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img, fps, recog_n, faces_now, roster_size, paused):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
    status = "PAUSED" if paused else "LIVE"
    line = "[" + status + "]  FPS: " + str(int(fps)) + \
           "   Faces: " + str(faces_now) + \
           "   Known: " + str(roster_size) + \
           "   RecogEvery: " + str(recog_n) + " frames"
    cv2.putText(img, line, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    hint = "Q-quit  F-fullscreen  S-screenshot  Space-pause  R-reconnect  +/- speed"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(img, hint, (w - tw - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", default=DEFAULT_RTSP)
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument("--recog-every", type=int, default=8,
                        help="run face detection every N frames (default 8)")
    args = parser.parse_args()

    print("[viewer] RTSP: " + args.rtsp)
    roster = load_all_students()
    print("[viewer] roster size = " + str(len(roster)))

    app_if = load_insightface()
    emo_sess = None if args.no_emotion else load_emotion()

    cap = open_camera(args.rtsp)
    if cap is None:
        print("[viewer] CAMERA OFFLINE - exiting")
        sys.exit(1)
    print("[viewer] camera online")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_DISPLAY_W, int(WINDOW_DISPLAY_W * 9 / 16))

    recog_every = max(1, args.recog_every)
    emotion_every = 15

    frame_idx = 0
    fps_ema = 0.0
    last_ts = None
    paused = False
    fullscreen = False
    last_overlay = {}      # key -> (bbox, name, emotion, sim, ts)
    overlay_ttl = 1.0
    consecutive_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures > 100:
                print("[viewer] too many read failures - reconnecting")
                cap.release()
                cap = open_camera(args.rtsp)
                if cap is None:
                    print("[viewer] reconnect failed - exiting")
                    break
                consecutive_failures = 0
            time.sleep(0.05)
            continue
        consecutive_failures = 0

        frame_idx += 1

        # FPS
        now = time.time()
        if last_ts is not None:
            dt = now - last_ts
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema > 0 else inst
        last_ts = now

        # Recognition pass
        if (not paused) and (frame_idx % recog_every == 0):
            try:
                small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
                faces = app_if.get(small)
            except Exception as e:
                print("[viewer] insightface error: " + str(e))
                faces = []

            do_emotion = (frame_idx % emotion_every == 0)
            last_overlay.clear()

            for i, f in enumerate(faces):
                emb = f.normed_embedding
                name, sim = match_student(emb, roster)

                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                x1 = int(x1 / 0.7); y1 = int(y1 / 0.7)
                x2 = int(x2 / 0.7); y2 = int(y2 / 0.7)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                emotion = ""
                if do_emotion and emo_sess is not None and x2 > x1 and y2 > y1:
                    try:
                        emotion = classify_emotion(emo_sess, frame[y1:y2, x1:x2])
                    except Exception:
                        emotion = ""

                last_overlay[i] = ((x1, y1, x2, y2), name, emotion, sim if name else 0, time.time())

        # Render
        display = frame.copy()
        cutoff = time.time() - overlay_ttl
        for key in list(last_overlay.keys()):
            if last_overlay[key][4] < cutoff:
                last_overlay.pop(key, None)

        for _, (bbox, name, emotion, sim, _ts) in last_overlay.items():
            x1, y1, x2, y2 = bbox
            color = color_for(name, emotion)
            draw_label(display, x1, y1, x2, y2, name, emotion, sim, color)

        draw_hud(display, fps_ema, recog_every, len(last_overlay), len(roster), paused)

        # Downscale for display
        h, w = display.shape[:2]
        if w > WINDOW_DISPLAY_W:
            scale = WINDOW_DISPLAY_W / float(w)
            display = cv2.resize(display, (WINDOW_DISPLAY_W, int(h * scale)))

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[viewer] quit")
            break
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key in (ord('s'), ord('S')):
            try:
                SCREENSHOT_DIR.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = SCREENSHOT_DIR / ("view_" + ts + ".jpg")
                cv2.imwrite(str(path), display)
                print("[viewer] screenshot saved: " + str(path))
            except Exception as e:
                print("[viewer] screenshot error: " + str(e))
        elif key == 32:
            paused = not paused
            print("[viewer] " + ("paused" if paused else "resumed"))
        elif key in (ord('r'), ord('R')):
            print("[viewer] reconnecting...")
            cap.release()
            cap = open_camera(args.rtsp)
            if cap is None:
                print("[viewer] reconnect failed - exiting")
                break
        elif key in (ord('+'), ord('=')):
            recog_every = max(1, recog_every - 1)
            print("[viewer] recog every = " + str(recog_every))
        elif key in (ord('-'), ord('_')):
            recog_every = min(60, recog_every + 1)
            print("[viewer] recog every = " + str(recog_every))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
