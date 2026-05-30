"""
Register students for face recognition.

Run this BEFORE recognize.py. It will:
1. Open the IP camera (or webcam if RTSP fails)
2. Ask you for the student's name
3. Show live video — press SPACE to capture, R to retake, ESC to cancel
4. Save the photo to ./students/<name>.jpg
5. Repeat for the next student

Run: python register.py
"""

import os
import cv2
import face_recognition
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv(
  "RTSP_URL",
  "rtsp://admin:Akniet12@192.168.205.14:554/h264/ch1/main/av_stream"
)
STUDENTS_DIR = Path("students")
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)


def open_camera():
  """Try RTSP first, fall back to webcam (index 0)."""
  print(f"[+] Trying IP camera: {RTSP_URL}")
  cap = cv2.VideoCapture(RTSP_URL)
  if cap.isOpened():
      ret, _ = cap.read()
      if ret:
          print("[+] IP camera connected.")
          return cap
      cap.release()

  print("[!] IP camera not available. Falling back to webcam (index 0)...")
  cap = cv2.VideoCapture(0)
  if cap.isOpened():
      print("[+] Webcam connected.")
      return cap

  print("[X] No camera available.")
  return None


def capture_student(cap, name: str) -> bool:
  """Show live feed, capture on SPACE. Returns True if saved."""
  print(f"\n[+] Capturing: {name}")
  print("    SPACE = capture | R = retake | ESC = cancel")

  while True:
      ret, frame = cap.read()
      if not ret:
          print("[!] Failed to read frame. Retrying...")
          continue

      # Show detected faces with a rectangle so user sees framing
      small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
      rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
      locations = face_recognition.face_locations(rgb, model="hog")

      display = frame.copy()
      for (top, right, bottom, left) in locations:
          top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
          cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

      # Overlay instructions
      cv2.putText(display, f"Registering: {name}", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
      cv2.putText(display, "SPACE = capture | R = retake | ESC = cancel",
                  (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      cv2.putText(display, f"Faces detected: {len(locations)}",
                  (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  (0, 255, 0) if len(locations) == 1 else (0, 0, 255), 2)

      cv2.imshow("Register Student", display)
      key = cv2.waitKey(1) & 0xFF

      if key == 27:  # ESC
          print("[!] Cancelled.")
          return False

      if key == ord(" "):  # SPACE — capture
          if len(locations) != 1:
              print(f"[!] Need exactly 1 face in frame, found {len(locations)}. Try again.")
              continue

          # Verify a face encoding can actually be extracted
          full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          encodings = face_recognition.face_encodings(full_rgb)
          if len(encodings) == 0:
              print("[!] Could not extract face encoding. Try again with better lighting/angle.")
              continue

          # Preview
          cv2.putText(display, "Saved! Press any key to continue, R to retake",
                      (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
          cv2.imshow("Register Student", display)
          k2 = cv2.waitKey(0) & 0xFF
          if k2 == ord("r"):
              print("[~] Retaking...")
              continue

          # Save
          path = STUDENTS_DIR / f"{name}.jpg"
          cv2.imwrite(str(path), frame)
          print(f"[OK] Saved: {path}")
          return True


def main():
  print("=" * 60)
  print("Student Registration")
  print("=" * 60)
  print(f"Photos will be saved to: {STUDENTS_DIR.resolve()}\n")

  cap = open_camera()
  if cap is None:
      return

  count = 0
  try:
      while True:
          print("\n" + "-" * 60)
          name = input("Student name (or press ENTER to finish): ").strip()
          if not name:
              break

          # Sanitize filename (keep letters, digits, basic chars)
          safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
          if not safe_name:
              print("[!] Invalid name. Try again.")
              continue

          # Warn if already exists
          existing = STUDENTS_DIR / f"{safe_name}.jpg"
          if existing.exists():
              ans = input(f"[!] {safe_name}.jpg already exists. Overwrite? (y/N): ").strip().lower()
              if ans != "y":
                  continue

          if capture_student(cap, safe_name):
              count += 1
              print(f"[+] Total registered this session: {count}")

  except KeyboardInterrupt:
      print("\n[!] Interrupted.")
  finally:
      cap.release()
      cv2.destroyAllWindows()

  # Final summary
  all_photos = list(STUDENTS_DIR.glob("*.jpg")) + list(STUDENTS_DIR.glob("*.png"))
  print("\n" + "=" * 60)
  print(f"Done. Total students in folder: {len(all_photos)}")
  if all_photos:
      print("Registered:")
      for p in sorted(all_photos):
          print(f"  - {p.stem}")
  print("=" * 60)
  print("\nNext step: run `python recognize.py` to start recognition.")


if __name__ == "__main__":
  main()
