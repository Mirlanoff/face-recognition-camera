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
# Lower = stricter match. 0.6 is the default sweet spot.
TOLERANCE = 0.55
# Resize factor for faster processing (smaller = faster, less accurate)
RESIZE_FACTOR = 0.25
# Process every Nth frame to save CPU
PROCESS_EVERY_N_FRAMES = 3


def load_known_faces(folder: Path):
    """Load all student photos from folder. Filename (without ext) = student name."""
    known_encodings = []
    known_names = []

    if not folder.exists():
        print(f"[!] Folder '{folder}' does not exist. Creating it.")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[!] Put student photos here, then re-run. Filename = student name.")
        return known_encodings, known_names

    photos = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    if not photos:
        print(f"[!] No photos found in '{folder}'. Add some .jpg/.png files.")
        return known_encodings, known_names

    for photo_path in photos:
        name = photo_path.stem  # filename without extension
        print(f"[+] Loading {name} from {photo_path.name}...")

        image = face_recognition.load_image_file(str(photo_path))
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"    [!] No face found in {photo_path.name}, skipping.")
            continue

        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"    [OK] Registered: {name}")

    print(f"\n[+] Loaded {len(known_names)} students: {', '.join(known_names)}\n")
    return known_encodings, known_names


def main():
    print("=" * 60)
    print("Face Recognition — IP Camera")
    print("=" * 60)

    known_encodings, known_names = load_known_faces(STUDENTS_DIR)

    if len(known_encodings) == 0:
        print("[!] No registered students. Add photos to 'students/' folder and re-run.")
        return

    print(f"[+] Connecting to camera: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("[X] Failed to open RTSP stream. Check URL, network, credentials.")
        return

    print("[+] Connected. Press 'q' to quit.\n")

    frame_counter = 0
    face_locations = []
    face_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        # Process every Nth frame for performance
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            # Resize for speed
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare with all known faces
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

        # Draw boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up
            scale = int(1 / RESIZE_FACTOR)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            # Color: green for known, red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Box around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Label text
            cv2.putText(
                frame, name,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (255, 255, 255), 1
            )

        cv2.imshow("Face Recognition — IP Camera (press 'q' to quit)", frame)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[+] Stopped.")


if __name__ == "__main__":
    main()
