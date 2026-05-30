"""
Real-time face recognition from IP camera (RTSP).
Detects faces in the video stream and labels each known student by name
inside a green bounding box. Unknown faces get a red box with "Unknown".

Usage:
1. Put student photos in ./students/ (filename = student name, e.g. Marlis.jpg)
2. Configure RTSP_URL in .env (or below)
3. python recognize.py
4. Press 'q' to quit

Requirements: see requirements.txt
"""

import os
import time
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===== Config =====
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream"
)
STUDENTS_DIR = Path("students")
TOLERANCE = 0.55
RESIZE_FACTOR = 0.25
PROCESS_EVERY_N_FRAMES = 3
CONNECT_TIMEOUT_SECONDS = 10


def load_known_faces(folder):
    known_encodings = []
    known_names = []

    if not folder.exists():
        print("[!] Folder '" + str(folder) + "' does not exist. Creating it.")
        folder.mkdir(parents=True, exist_ok=True)
        print("[!] Put student photos here, then re-run. Filename = student name.")
        return known_encodings, known_names

    photos = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    if not photos:
        print("[!] No photos found in '" + str(folder) + "'. Add some .jpg/.png files.")
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


def main():
    print("=" * 60)
    print("Face Recognition - IP Camera")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)

    if len(known_encodings) == 0:
        print("[!] No registered students. Add photos to 'students/' folder and re-run.")
        return

    print("[+] Connecting to camera (TCP): " + RTSP_URL)
    cap = open_rtsp(RTSP_URL)

    if not cap.isOpened():
        print("[X] Failed to open RTSP stream. Check URL, network, credentials.")
        print("    Tip: open the URL in VLC first to confirm the camera is reachable.")
        return

    ret, _ = cap.read()
    if not ret:
        print("[X] Connected but cannot read frames. Camera may be busy or stream invalid.")
        cap.release()
        return

    print("[+] Connected. Press 'q' to quit.")
    print("")

    frame_counter = 0
    face_locations = []
    face_names = []
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

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < TOLERANCE:
                        name = known_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            scale = int(1 / RESIZE_FACTOR)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (255, 255, 255), 1
            )

        cv2.imshow("Face Recognition - IP Camera (press q to quit)", frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("")
    print("[+] Stopped.")


if __name__ == "__main__":
    main()
