"""
Real-time multi-face recognition + emotion detection from IP camera (RTSP).

Stack:
- MediaPipe FaceDetection (very fast, finds 10+ faces per frame on CPU)
- face_recognition (identifies who is who using known student photos)
- FER+ ONNX model via onnxruntime (emotion: happy/sad/angry/surprise/fear/disgust/neutral/contempt)

Run once before first launch:
python download_emotion_model.py

Then:
python recognize.py
"""

import os
import time
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
PROCESS_EVERY_N_FRAMES = 3
CONNECT_TIMEOUT_SECONDS = 10
# MediaPipe model: 0 = short-range (within 2m), 1 = full-range (up to 5m). Pick 1 for classroom.
MP_MODEL_SELECTION = 1
MP_MIN_CONFIDENCE = 0.5

EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


def load_known_faces(folder):
    known_encodings = []
    known_names = []
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print("[!] Created empty students folder. Add photos and re-run.")
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
        print("[!] Emotion model not found at " + str(MODEL_PATH))
        print("    Run: python download_emotion_model.py")
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
    """gray_face_64: 64x64 uint8 grayscale crop. Returns emotion label."""
    if sess is None:
        return ""
    try:
        x = gray_face_64.astype(np.float32)
        x = x.reshape(1, 1, 64, 64)
        input_name = sess.get_inputs()[0].name
        scores = sess.run(None, {input_name: x})[0]
        idx = int(np.argmax(scores[0]))
        if idx < 0 or idx >= len(EMOTION_LABELS):
            return ""
        return EMOTION_LABELS[idx]
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("Multi-Face Recognition + Emotion (MediaPipe + ONNX)")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)
    if len(known_encodings) == 0:
        print("[!] Add student photos to students/ and re-run.")
        return

    emotion_sess = load_emotion_session()

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        model_selection=MP_MODEL_SELECTION,
        min_detection_confidence=MP_MIN_CONFIDENCE,
    )

    print("[+] Connecting to camera (TCP): " + RTSP_URL)
    cap = open_rtsp(RTSP_URL)
    if not cap.isOpened():
        print("[X] Failed to open RTSP.")
        return

    ret, _ = cap.read()
    if not ret:
        print("[X] Connected but cannot read frames.")
        cap.release()
        return

    print("[+] Connected. Press 'q' to quit.")
    print("")

    frame_counter = 0
    consecutive_failures = 0
    cached_results = []  # list of (top, right, bottom, left, label, color)
    fps_t0 = time.time()
    fps_n = 0
    fps_value = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            print("[!] Failed to grab frame (" + str(consecutive_failures) + "). Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = open_rtsp(RTSP_URL)
            if consecutive_failures > 5:
                print("[X] Too many failures. Exiting.")
                break
            continue
        consecutive_failures = 0

        h, w = frame.shape[:2]

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.process(rgb)

            new_results = []

            if detections.detections:
                # Collect all face boxes first
                boxes = []
                for det in detections.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x2 = min(w, x1 + bw)
                    y2 = min(h, y1 + bh)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    boxes.append((y1, x2, y2, x1))  # top, right, bottom, left

                # Batch identity encodings for all boxes at once
                if boxes:
                    encodings = face_recognition.face_encodings(rgb, boxes)
                else:
                    encodings = []

                for (top, right, bottom, left), enc in zip(boxes, encodings):
                    # Identify
                    distances = face_recognition.face_distance(known_encodings, enc)
                    if len(distances) > 0:
                        best = int(np.argmin(distances))
                        name = known_names[best] if distances[best] < TOLERANCE else "Unknown"
                    else:
                        name = "Unknown"

                    # Emotion: crop, grayscale, resize to 64x64
                    emotion = ""
                    if emotion_sess is not None:
                        face_bgr = frame[top:bottom, left:right]
                        if face_bgr.size > 0:
                            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                            gray64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                            emotion = predict_emotion(emotion_sess, gray64)

                    label = name + " (" + emotion + ")" if emotion else name
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    new_results.append((top, right, bottom, left, label, color))

            cached_results = new_results

        # Draw cached results on every frame for smooth display
        for (top, right, bottom, left, label, color) in cached_results:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(frame, (left, bottom), (left + tw + 12, bottom + th + 12), color, cv2.FILLED)
            cv2.putText(
                frame, label,
                (left + 6, bottom + th + 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 255, 255), 1
            )

        # FPS counter
        fps_n += 1
        if fps_n >= 10:
            now = time.time()
            fps_value = fps_n / (now - fps_t0)
            fps_t0 = now
            fps_n = 0

        cv2.putText(
            frame,
            "FPS: " + str(round(fps_value, 1)) + "  Faces: " + str(len(cached_results)),
            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1
        )

        cv2.imshow("Recognition + Emotion (q to quit)", frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("")
    print("[+] Stopped.")


if __name__ == "__main__":
    main()
