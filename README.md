# Face Recognition — IP Camera

Real-time face recognition from an IP (RTSP) camera. Detects faces in the video stream and labels each known student by name inside a bounding box.

## Workflow

1. **Register students** (one-time, ~10 students) — run `register.py`, capture each student's photo from the camera
2. **Run recognition** — run `recognize.py`, see live video with names on each face

---

## How to run in Visual Studio Code (step by step)

### 1. Install Python 3.10 or 3.11

Download from [python.org](https://www.python.org/downloads/) — **check "Add Python to PATH"** during install.

> ⚠️ Use Python **3.10 or 3.11**. `face_recognition` / `dlib` may not work on 3.12+.

### 2. Install VS Code extensions

Open VS Code → Extensions (Ctrl+Shift+X) → install:
- **Python** (by Microsoft)
- **Pylance**

### 3. Clone the project

Open VS Code terminal (`Ctrl+` `) and run:

```bash
git clone https://github.com/Mirlanoff/face-recognition-camera.git
cd face-recognition-camera
code .
```

### 4. Create a virtual environment

In VS Code terminal:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

VS Code will ask "Use this environment for the workspace?" — click **Yes**.

### 5. Install system dependencies (only for `dlib`)

**Windows:** install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → select "Desktop development with C++".

**Mac:**
```bash
brew install cmake
```

**Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

### 6. Install Python packages

```bash
pip install -r requirements.txt
```

> If `dlib` fails to install — run `pip install cmake` first, then retry.

### 7. Configure the camera

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and set your real RTSP URL (with password).

### 8. Register students

```bash
python register.py
```

For each of your 10 students:
1. Type their name → press Enter
2. Camera window opens — position the student in frame (1 green box around face)
3. Press **SPACE** to capture
4. Press any key to continue → next student
5. When done, press Enter without a name to finish

All photos saved to `students/` folder.

### 9. Run recognition

```bash
python recognize.py
```

A window opens with the live camera. Each registered student gets a **green box + their name**. Unknown people get a **red box + "Unknown"**. Press **q** to quit.

---

## Files

| File | Purpose |
|---|---|
| `register.py` | Register students one-by-one from the camera |
| `recognize.py` | Live recognition — main app |
| `requirements.txt` | Python dependencies |
| `.env.example` | Camera URL template |
| `students/` | Folder with registered student photos (auto-created) |

## Configuration

Edit constants at the top of `recognize.py`:

| Variable | Default | Description |
|---|---|---|
| `TOLERANCE` | `0.55` | Lower = stricter match. Range 0.4–0.6 |
| `RESIZE_FACTOR` | `0.25` | Smaller = faster but less accurate |
| `PROCESS_EVERY_N_FRAMES` | `3` | Process every Nth frame to save CPU |

## Troubleshooting

**"Failed to open RTSP stream"** — check camera IP, credentials in `.env`, same network as camera. Try opening the URL in [VLC Player](https://www.videolan.org/vlc/) first to verify.

**`pip install` fails on `dlib`** — install C++ build tools (step 5), then `pip install cmake`, then retry.

**Slow / laggy video** — increase `PROCESS_EVERY_N_FRAMES` to 5 or `RESIZE_FACTOR` to 0.2 in `recognize.py`.

**Too many "Unknown"** — adjust `TOLERANCE` (try 0.6 for more lenient). Re-register the student with better lighting.

**Window doesn't open / camera freezes** — close, run again. If RTSP keeps dropping, `register.py` will fall back to your webcam automatically.

## License

MIT
