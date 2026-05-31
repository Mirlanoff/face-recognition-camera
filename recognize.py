"""
Real-time face recognition + emotion detection from IP camera (RTSP).
Detects faces, identifies known students, and shows their current emotion.
"""

import os
import time
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()

# ===== Config =====
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream"
)
STUDENTS_DIR = Path("students")
TOLERANCE = 0.55
# 0.5 = good balance between speed and far-distance detection (~3m)
# 0.25 = faster but only sees close faces
RESIZE_FACTOR = 0.5
PROCESS_EVERY_N_FRAMES = 5
CONNECT_TIMEOUT_SECONDS = 10
# Run emotion detection every N face-processing cycles (it is slower than recognition)
EMOTION_EVERY_N_CYCLES = 2

# ===== Emotion detector (optional) =====
EMOTION_ENABLED = True
emotion_detector = None
try:
    from fer import FER
    emotion_detector = FER(mtcnn=False)
    print("[+] Emotion detection enabled.")
except Exception as e:
    EMOTION_ENABLED = False
    print("[!] Emotion detection disabled (fer not installed): " + str(e))


def load_known_faces(folder):
    known_encodings = []
    known_names = []

    if not folder.exists():
        print("[!] Folder '" + str(folder) + "' does not exist. Creating it.")
        folder.mkdir(parents=True, exist_ok=True)
        print("[!] Put student photos here, then re-run.")
        return known_encodings, known_names

    photos = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    if not photos:
        print("[!] No photos found in '" + str(folder) + "'.")
        return known_encodings, known_names

    for photo_path in photos:
        name = photo_path.stem
        print("[+] Loading " + name + " from " + photo_path.name + "...")

        image = face_recognition.load_image_file(str(photo_path))
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print("    [!] No face found in " + photo_path.name + ", skipping.")
            continue

        known_encodings.append(encodings[0])
        known_names.append(name)
        print("    [OK] Registered: " + name)

    print("")
    print("[+] Loaded " + str(len(known_names)) + " students: " + ", ".join(known_names))
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


def detect_emotion(frame_bgr, top, right, bottom, left):
    """Run emotion detection on a face crop. Returns label like 'happy' or ''."""
    if not EMOTION_ENABLED or emotion_detector is None:
        return ""
    # Add some padding
    h, w = frame_bgr.shape[:2]
    pad = 20
    y1 = max(0, top - pad)
    y2 = min(h, bottom + pad)
    x1 = max(0, left - pad)
    x2 = min(w, right + pad)
    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return ""
    try:
        result = emotion_detector.detect_emotions(face_crop)
        if not result:
            return ""
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        return top_emotion
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("Face Recognition + Emotion - IP Camera")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)

    if len(known_encodings) == 0:
        print("[!] No registered students. Add photos to 'students/' folder.")
        return

    print("[+] Connecting to camera (TCP): " + RTSP_URL)
    cap = open_rtsp(RTSP_URL)

    if not cap.isOpened():
        print("[X] Failed to open RTSP stream.")
        return

    ret, _ = cap.read()
    if not ret:
        print("[X] Connected but cannot read frames.")
        cap.release()
        return

    print("[+] Connected. Press 'q' to quit.")
    print("")

    frame_counter = 0
    detection_cycle = 0
    face_locations_full = []
    face_labels = []
    consecutive_failures = 0

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

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations_small = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

            scale = int(1 / RESIZE_FACTOR)
            face_locations_full = []
            new_labels = []

            run_emotion = EMOTION_ENABLED and (detection_cycle % EMOTION_EVERY_N_CYCLES == 0)

            for (top_s, right_s, bottom_s, left_s), face_encoding in zip(face_locations_small, face_encodings):
                # Scale up to full-frame coords
                top = top_s * scale
                right = right_s * scale
                bottom = bottom_s * scale
                left = left_s * scale
                face_locations_full.append((top, right, bottom, left))

                # Identify
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    best = np.argmin(distances)
                    name = known_names[best] if distances[best] < TOLERANCE else "Unknown"
                else:
                    name = "Unknown"

                # Emotion
                emotion = ""
                if run_emotion:
                    emotion = detect_emotion(frame, top, right, bottom, left)

                label = name + " (" + emotion + ")" if emotion else name
                new_labels.append(label)

            face_labels = new_labels
            detection_cycle += 1

        # Draw using last known detection
        for (top, right, bottom, left), label in zip(face_locations_full, face_labels):
            is_known = not label.startswith("Unknown")
            color = (0, 255, 0) if is_known else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, label,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (255, 255, 255), 1
            )

        cv2.imshow("Face Recognition + Emotion (press q to quit)", frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("")
    print("[+] Stopped.")


if __name__ == "__main__":
    main()
