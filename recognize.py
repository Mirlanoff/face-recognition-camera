"""
Real-time multi-face recognition + emotion detection with:
- Hardware-accelerated h264 decoding via ffmpeg subprocess (DXVA2 on Windows).
This bypasses OpenCV's software decoder which was producing 'bytestream' errors
and pixelation artifacts ("клетки") when CPU got busy.
- IoU tracker: identity every 2s, emotion every 1s per track (industry-style).

Requirements:
- ffmpeg.exe on PATH (https://www.gyan.dev/ffmpeg/builds/)

Run:
  python download_emotion_model.py
  python recognize.py
"""

import os
import sys
import time
import threading
import subprocess
import shutil
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
CONNECT_TIMEOUT_SECONDS = 10

# Frame size we ask ffmpeg to output. Lower = lighter on CPU.
# Detector + recognition will operate on this size.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MP_MODEL_SELECTION = 1
MP_MIN_CONFIDENCE = 0.5

# Inside the already-downscaled stream, downscale once more for recognition.
RECOGNITION_SCALE = 0.5
DISPLAY_SCALE = 1.0  # ffmpeg already gives us 1280x720, no further shrink needed

IOU_MATCH_THRESHOLD = 0.3
TRACK_TTL_SEC = 1.5
IDENTITY_INTERVAL_SEC = 2.0
EMOTION_INTERVAL_SEC = 1.0

EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


# ===== Face DB =====
def load_known_faces(folder):
  known_encodings = []
  known_names = []
  if not folder.exists():
      folder.mkdir(parents=True, exist_ok=True)
      print("[!] Created empty students/. Add photos and re-run.")
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


# ===== Emotion =====
def load_emotion_session():
  if not MODEL_PATH.exists():
      print("[!] Emotion model not found. Run: python download_emotion_model.py")
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
      self.box = box
      self.name = ""
      self.emotion = ""
      self.last_seen = now
      self.last_identity_time = 0.0
      self.last_emotion_time = 0.0

  def needs_identity(self, now):
      return (not self.name) or (now - self.last_identity_time >= IDENTITY_INTERVAL_SEC)

  def needs_emotion(self, now):
      return (now - self.last_emotion_time >= EMOTION_INTERVAL_SEC)


# ===== FFmpeg hardware-accelerated reader =====
class FFmpegCameraReader(threading.Thread):
  """
  Spawns ffmpeg as a subprocess that:
    - Reads RTSP over TCP
    - Decodes h264 on GPU via DXVA2 (Windows hardware decoder)
    - Outputs raw BGR24 frames at FRAME_WIDTH x FRAME_HEIGHT into stdout
  We read raw bytes from stdout into numpy arrays.
  This pipeline is what professional VMS systems use.
  """
  def __init__(self, url, width, height):
      super().__init__(daemon=True)
      self.url = url
      self.width = width
      self.height = height
      self.frame_size = width * height * 3
      self.process = None
      self.lock = threading.Lock()
      self.latest = None
      self.running = True

  def _spawn(self):
      ffmpeg_path = shutil.which("ffmpeg")
      if ffmpeg_path is None:
          print("[X] ffmpeg not found on PATH.")
          print("    Install from https://www.gyan.dev/ffmpeg/builds/ and add bin\\ to PATH.")
          return None

      cmd = [
          ffmpeg_path,
          "-loglevel", "error",
          "-hwaccel", "dxva2",                 # Windows GPU h264 decode
          "-rtsp_transport", "tcp",
          "-stimeout", str(CONNECT_TIMEOUT_SECONDS * 1000000),
          "-fflags", "nobuffer",
          "-flags", "low_delay",
          "-i", self.url,
          "-an",                               # no audio
          "-vf", "scale=" + str(self.width) + ":" + str(self.height),
          "-pix_fmt", "bgr24",
          "-f", "rawvideo",
          "-"
      ]
      try:
          return subprocess.Popen(
              cmd,
              stdout=subprocess.PIPE,
              stderr=subprocess.DEVNULL,
              bufsize=10 * 1024 * 1024
          )
      except Exception as e:
          print("[X] Failed to start ffmpeg: " + str(e))
          return None

  def run(self):
      while self.running:
          if self.process is None or self.process.poll() is not None:
              if self.process is not None:
                  print("[!] ffmpeg exited, restarting...")
              self.process = self._spawn()
              if self.process is None:
                  time.sleep(1.0)
                  continue

          try:
              raw = self.process.stdout.read(self.frame_size)
          except Exception:
              raw = b""

          if len(raw) != self.frame_size:
              # broken pipe / stream hiccup -> restart
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


# ===== Recognizer =====
class TrackedRecognizer(threading.Thread):
  def __init__(self, camera, known_encodings, known_names, emotion_sess):
      super().__init__(daemon=True)
      self.camera = camera
      self.known_encodings = known_encodings
      self.known_names = known_names
      self.emotion_sess = emotion_sess
      self.detector = mp.solutions.face_detection.FaceDetection(
          model_selection=MP_MODEL_SELECTION,
          min_detection_confidence=MP_MIN_CONFIDENCE,
      )
      self.lock = threading.Lock()
      self.tracks = []
      self.scale_back = 1.0 / RECOGNITION_SCALE
      self.running = True

  def _detect_boxes(self, frame):
      small = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
      h_s, w_s = small.shape[:2]
      rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

      if not self.running:
          return rgb, [], []

      try:
          det = self.detector.process(rgb)
      except Exception:
          return rgb, [], []

      boxes_full = []
      boxes_small = []
      if det.detections:
          for d in det.detections:
              b = d.location_data.relative_bounding_box
              x1 = max(0, int(b.xmin * w_s))
              y1 = max(0, int(b.ymin * h_s))
              bw = int(b.width * w_s)
              bh = int(b.height * h_s)
              x2 = min(w_s, x1 + bw)
              y2 = min(h_s, y1 + bh)
              if x2 - x1 < 20 or y2 - y1 < 20:
                  continue
              boxes_small.append((y1, x2, y2, x1))
              top = int(y1 * self.scale_back)
              right = int(x2 * self.scale_back)
              bottom = int(y2 * self.scale_back)
              left = int(x1 * self.scale_back)
              boxes_full.append((top, right, bottom, left))
      return rgb, boxes_small, boxes_full

  def _match_tracks(self, boxes_full, now):
      assignments = []
      used_track_ids = set()
      used_det_idx = set()

      for det_idx, det_box in enumerate(boxes_full):
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

      for det_idx, det_box in enumerate(boxes_full):
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
              rgb_small, boxes_small, boxes_full = self._detect_boxes(frame)
              if not self.running:
                  break
              now = time.time()

              with self.lock:
                  assignments = self._match_tracks(boxes_full, now)

              idents_needed = []
              idents_boxes = []
              emos_needed = []

              with self.lock:
                  for track, det_idx in assignments:
                      if track.needs_identity(now):
                          idents_needed.append(track)
                          idents_boxes.append(boxes_small[det_idx])
                      if track.needs_emotion(now) and self.emotion_sess is not None:
                          emos_needed.append((track, det_idx))

              if idents_needed:
                  encs = face_recognition.face_encodings(rgb_small, idents_boxes)
                  for track, enc in zip(idents_needed, encs):
                      distances = face_recognition.face_distance(self.known_encodings, enc)
                      if len(distances) > 0:
                          best = int(np.argmin(distances))
                          name = self.known_names[best] if distances[best] < TOLERANCE else "Unknown"
                      else:
                          name = "Unknown"
                      with self.lock:
                          track.name = name
                          track.last_identity_time = now

              for track, det_idx in emos_needed:
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
  print("Multi-Face Recognition + Emotion (HW-decoded, tracked)")
  print("=" * 60)

  if shutil.which("ffmpeg") is None:
      print("")
      print("[X] ffmpeg not found on PATH.")
      print("    1) Download: https://www.gyan.dev/ffmpeg/builds/ -> ffmpeg-release-essentials.zip")
      print("    2) Unzip to C:\\ffmpeg\\")
      print("    3) Add C:\\ffmpeg\\bin to PATH (Environment Variables)")
      print("    4) Re-open terminal and run again")
      sys.exit(1)

  known_encodings, known_names = load_known_faces(STUDENTS_DIR)
  if len(known_encodings) == 0:
      print("[!] Add student photos to students/ and re-run.")
      return

  emotion_sess = load_emotion_session()

  print("[+] Starting ffmpeg HW-decode pipeline: " + RTSP_URL)
  print("    Output: " + str(FRAME_WIDTH) + "x" + str(FRAME_HEIGHT) + " BGR24")
  camera = FFmpegCameraReader(RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT)
  camera.start()

  t0 = time.time()
  while camera.read() is None:
      if time.time() - t0 > CONNECT_TIMEOUT_SECONDS + 5:
          print("[X] No frames from ffmpeg within timeout. Exiting.")
          camera.stop()
          return
      time.sleep(0.1)

  print("[+] Stream alive. Starting tracker + recognizer...")
  recognizer = TrackedRecognizer(camera, known_encodings, known_names, emotion_sess)
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
              "FPS: " + str(round(fps_value, 1)) + "  Tracks: " + str(len(results)),
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
