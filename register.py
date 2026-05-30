"""
Register students for face recognition.

Run this BEFORE recognize.py. It will:
1. Open the IP camera (or webcam if RTSP fails)
2. Ask you for the student's name
3. Show live video — press SPACE to capture, ESC to cancel
4. Save the photo and immediately close the window
5. Ask for the next student's name

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

# Detect faces only every N frames (preview keeps running smoothly)
DETECT_EVERY_N_FRAMES = 5
# Smaller = faster detection during preview
PREVIEW_SCALE = 0.35


def open_camera():
  """Try RTSP first, fall back to webcam (index 0). Low-latency settings."""
  print(f"[+] Trying IP camera: {RTSP_URL}")
  # Force FFMPEG backend + tiny buffer = no lag buildup
  os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
  cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
  try:
      cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
  except Exception:
      pass

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


def flush_buffer(cap, frames: int = 5):
  """Drain old frames so we always work with the latest one."""
  for _ in range(frames):
      cap.grab()


def capture_student(cap, name: str) -> bool:
  """Show live feed, capture on SPACE, auto-close window. Returns True if saved."""
  print(f"\n[+] Capturing: {name}")
  print("    SPACE = capture | ESC = cancel")

  window_name = "Register Student"
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

  frame_counter = 0
  face_count = 0  # last known number of detected faces

  try:
      while True:
          # Drop stale frames for low latency
          flush_buffer(cap, 2)
          ret, frame = cap.read()
          if not ret:
              continue

          # Detect faces only every N frames (smooth preview)
          if frame_counter % DETECT_EVERY_N_FRAMES == 0:
              small = cv2.resize(frame, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
              rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
              locations = face_recognition.face_locations(rgb, model="hog")
              face_count = len(locations)

              # Draw boxes
              scale = int(1 / PREVIEW_SCALE)
              display = frame.copy()
              for (top, right, bottom, left) in locations:
                  cv2.rectangle(
                      display,
                      (left * scale, top * scale),
                      (right * scale, bottom * scale),
                      (0, 255, 0), 2
                  )
              cached_display = display
          else:
              # Reuse last detection overlay, but on a fresh frame
              display = frame
              cached_display = display

          # Overlay text
          cv2.putText(cached_display, f"Registering: {name}", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
          cv2.putText(cached_display, "SPACE = capture | ESC = cancel",
                      (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
          color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
          cv2.putText(cached_display, f"Faces detected: {face_count}",
                      (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

          cv2.imshow(window_name, cached_display)
          key = cv2.waitKey(1) & 0xFF
          frame_counter += 1

          if key == 27:  # ESC
              cv2.destroyWindow(window_name)
              cv2.waitKey(1)
              print("[!] Cancelled.")
              return False

          if key == ord(" "):  # SPACE — capture
              # Verify a face encoding can be extracted (full resolution)
              full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              encodings = face_recognition.face_encodings(full_rgb)
              if len(encodings) == 0:
                  print("[!] No face found. Better lighting / move closer. Try again.")
                  continue
              if len(encodings) > 1:
                  print(f"[!] Found {len(encodings)} faces, need exactly 1. Try again.")
                  continue

              # Save and close window immediately
              path = STUDENTS_DIR / f"{name}.jpg"
              cv2.imwrite(str(path), frame)
              print(f"[OK] Saved: {path}")
              cv2.destroyWindow(window_name)
              cv2.waitKey(1)
              return True
  finally:
      # Safety net: make sure no window stays open
      try:
          cv2.destroyWindow(window_name)
          cv2.waitKey(1)
      except Exception:
          pass


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

          # Sanitize filename
          safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
          if not safe_name:
              print("[!] Invalid name. Try again.")
              continue

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
      cv2.waitKey(1)

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
