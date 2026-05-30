# Face Recognition — IP Camera

Real-time face recognition from an IP (RTSP) camera. Detects faces in the video stream and labels each known student by name inside a bounding box.

## Features

- Connects to IP camera via RTSP
- Recognizes registered students by name
- Draws green box + name for known faces, red box + "Unknown" for unknown
- Works on CPU (no GPU required)
- Press `q` to quit

## Quick start

### 1. Install dependencies

```bash
# System packages (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-pip cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Python packages
pip install -r requirements.txt
```

> On Windows: install Visual C++ Build Tools first. On macOS: `brew install cmake`.

### 2. Configure camera

```bash
cp .env.example .env
# Edit .env and set your RTSP_URL
```

### 3. Add student photos

Put one clear frontal photo per student in the `students/` folder. **Filename = student name.**

```
students/
├── Marlis.jpg
├── Aizhan.jpg
└── Bekzat.png
```

Tips:
- One face per photo
- Clear lighting, frontal angle
- Use the student's name in Latin or Cyrillic — it will appear on the video as-is

### 4. Run

```bash
python recognize.py
```

A window opens showing the camera feed with recognized faces labeled.

## Configuration

Edit constants at the top of `recognize.py`:

| Variable | Default | Description |
|---|---|---|
| `TOLERANCE` | `0.55` | Lower = stricter match. Range 0.4–0.6 |
| `RESIZE_FACTOR` | `0.25` | Smaller = faster but less accurate |
| `PROCESS_EVERY_N_FRAMES` | `3` | Process every Nth frame to save CPU |

## Troubleshooting

**"Failed to open RTSP stream"** — check camera IP, credentials, and that you're on the same network.

**Slow / laggy video** — increase `PROCESS_EVERY_N_FRAMES` to 5 or decrease `RESIZE_FACTOR` to 0.2.

**Wrong recognition / many "Unknown"** — adjust `TOLERANCE` (try 0.5 for stricter, 0.6 for more lenient). Use better quality photos in `students/`.

**`dlib` install fails** — install cmake first: `pip install cmake`, then retry `pip install -r requirements.txt`.

## License

MIT
