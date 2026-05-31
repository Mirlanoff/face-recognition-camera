"""
FastAPI server for the school recognition dashboard.

Routes:
HTML
  GET  /                        -> overview page
  GET  /analytics               -> analytics page (stub for now)
  GET  /alerts                  -> alerts page (stub)
  GET  /history                 -> history page (stub)

Data
  GET  /api/classes
  GET  /api/subjects
  GET  /api/students            (optional class_id)
  POST /api/students            (multipart register)
  DELETE /api/students/{id}

  GET  /api/lessons/active
  GET  /api/lessons/history
  POST /api/lessons/start       -> starts a LessonWorker
  POST /api/lessons/{id}/stop   -> stops a LessonWorker

  GET  /api/lessons/{id}/attendees -> students currently marked present
  GET  /api/lessons/{id}/emotions  -> emotion counts for this lesson

  GET  /api/overview

WebSocket
  /ws       -> live events (student_entered, emotion, alert, worker_started/stopped)
"""

import os
import asyncio
import time
from datetime import datetime
from pathlib import Path

from fastapi import (
  FastAPI, Depends, HTTPException, UploadFile, File, Form,
  Request, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import (
  init_db, get_session, SessionLocal,
  SchoolClass, Subject, Student, Lesson, Attendance, EmotionLog, Alert,
)
import recognizer_worker

STUDENTS_DIR = Path("students")
STUDENTS_DIR.mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app = FastAPI(title="Мектеп таануу системасы")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/students_photos", StaticFiles(directory="students"), name="students_photos")
templates = Jinja2Templates(directory="templates")


# ===== Event hub =====
event_queue = None
main_loop = None
connected_sockets = set()


@app.on_event("startup")
async def on_startup():
  global event_queue, main_loop
  init_db()
  main_loop = asyncio.get_event_loop()
  event_queue = asyncio.Queue()
  # Launch broadcaster
  asyncio.create_task(_broadcast_loop())
  print("[server] DB ready, event hub running.")


async def _broadcast_loop():
  while True:
      evt = await event_queue.get()
      dead = []
      for ws in list(connected_sockets):
          try:
              await ws.send_json(evt)
          except Exception:
              dead.append(ws)
      for d in dead:
          connected_sockets.discard(d)


# ===== Lazy InsightFace loader for STUDENT REGISTRATION (independent of workers) =====
_insightface_app = None


def get_insightface_for_registration():
  global _insightface_app
  if _insightface_app is None:
      from insightface.app import FaceAnalysis
      print("[server] Loading InsightFace for student registration ...")
      _insightface_app = FaceAnalysis(
          name="buffalo_l",
          providers=["CPUExecutionProvider"],
          allowed_modules=["detection", "recognition"],
      )
      _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
  return _insightface_app


# ===== HTML pages =====
@app.get("/", response_class=HTMLResponse)
def overview_page(request: Request):
  return templates.TemplateResponse(request, "overview.html", {"active_tab": "overview"})


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request):
  return templates.TemplateResponse(request, "stub.html", {
      "active_tab": "analytics",
      "title": "Аналитика",
      "message": "Бул бөлүм 3-этапта кошулат.",
  })


@app.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request):
  return templates.TemplateResponse(request, "stub.html", {
      "active_tab": "alerts",
      "title": "Эскертүүлөр",
      "message": "Бул бөлүм 4-этапта кошулат.",
  })


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
  return templates.TemplateResponse(request, "stub.html", {
      "active_tab": "history",
      "title": "Тарых",
      "message": "Бул бөлүм 5-этапта кошулат.",
  })


# ===== Classes / Subjects =====
@app.get("/api/classes")
def list_classes(db: Session = Depends(get_session)):
  classes = db.query(SchoolClass).order_by(SchoolClass.id).all()
  return [
      {"id": c.id, "name": c.name, "student_count": len(c.students)}
      for c in classes
  ]


@app.get("/api/subjects")
def list_subjects(db: Session = Depends(get_session)):
  subjects = db.query(Subject).order_by(Subject.name).all()
  return [{"id": s.id, "name": s.name} for s in subjects]


# ===== Students =====
@app.get("/api/students")
def list_students(class_id: int = None, db: Session = Depends(get_session)):
  q = db.query(Student)
  if class_id is not None:
      q = q.filter(Student.class_id == class_id)
  out = []
  for s in q.order_by(Student.full_name).all():
      out.append({
          "id": s.id,
          "full_name": s.full_name,
          "class_id": s.class_id,
          "class_name": s.school_class.name if s.school_class else "",
          "photo_url": "/students_photos/" + os.path.basename(s.photo_path),
          "created_at": s.created_at.isoformat() if s.created_at else None,
      })
  return out


@app.post("/api/students")
async def register_student(
  full_name: str = Form(...),
  class_id: int = Form(...),
  photo: UploadFile = File(...),
  db: Session = Depends(get_session),
):
  school_class = db.query(SchoolClass).filter(SchoolClass.id == class_id).first()
  if school_class is None:
      raise HTTPException(status_code=400, detail="Класс табылган жок")

  safe_name = "".join(c for c in full_name if c.isalnum() or c in (" ", "_", "-")).strip()
  if not safe_name:
      safe_name = "student"
  timestamp = str(int(time.time()))
  ext = Path(photo.filename).suffix.lower()
  if ext not in (".jpg", ".jpeg", ".png"):
      ext = ".jpg"
  filename = safe_name.replace(" ", "_") + "_" + timestamp + ext
  dest = STUDENTS_DIR / filename

  contents = await photo.read()
  with open(dest, "wb") as f:
      f.write(contents)

  import cv2
  img = cv2.imread(str(dest))
  if img is None:
      os.remove(dest)
      raise HTTPException(status_code=400, detail="Сүрөттү окуу мүмкүн эмес")

  app_if = get_insightface_for_registration()
  faces = app_if.get(img)
  if len(faces) == 0:
      os.remove(dest)
      raise HTTPException(status_code=400, detail="Сүрөттө бет табылган жок")

  face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
  emb = face.normed_embedding

  student = Student(
      full_name=full_name.strip(),
      photo_path=str(dest),
      class_id=class_id,
  )
  student.set_embedding(emb)
  db.add(student)
  db.commit()
  db.refresh(student)

  return {
      "id": student.id,
      "full_name": student.full_name,
      "class_id": student.class_id,
      "photo_url": "/students_photos/" + filename,
  }


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_session)):
  student = db.query(Student).filter(Student.id == student_id).first()
  if student is None:
      raise HTTPException(status_code=404, detail="Окуучу табылган жок")
  try:
      if student.photo_path and os.path.exists(student.photo_path):
          os.remove(student.photo_path)
  except Exception:
      pass
  db.delete(student)
  db.commit()
  return {"ok": True}


# ===== Lessons =====
@app.get("/api/lessons/active")
def active_lessons(db: Session = Depends(get_session)):
  lessons = db.query(Lesson).filter(Lesson.is_active == True).order_by(Lesson.started_at.desc()).all()
  out = []
  for l in lessons:
      out.append({
          "id": l.id,
          "subject_name": l.subject.name if l.subject else "",
          "class_name": l.school_class.name if l.school_class else "",
          "started_at": l.started_at.isoformat() if l.started_at else None,
          "students_in_class": len(l.school_class.students) if l.school_class else 0,
          "students_seen": db.query(Attendance).filter(Attendance.lesson_id == l.id).count(),
          "camera_id": l.camera_id,
          "worker_running": recognizer_worker.is_worker_running(l.id),
      })
  return out


@app.get("/api/lessons/history")
def history_lessons(db: Session = Depends(get_session)):
  lessons = db.query(Lesson).filter(Lesson.is_active == False).order_by(Lesson.ended_at.desc()).limit(100).all()
  out = []
  for l in lessons:
      duration = None
      if l.started_at and l.ended_at:
          duration = int((l.ended_at - l.started_at).total_seconds())
      out.append({
          "id": l.id,
          "subject_name": l.subject.name if l.subject else "",
          "class_name": l.school_class.name if l.school_class else "",
          "started_at": l.started_at.isoformat() if l.started_at else None,
          "ended_at": l.ended_at.isoformat() if l.ended_at else None,
          "duration_seconds": duration,
          "students_count": db.query(Attendance).filter(Attendance.lesson_id == l.id).count(),
          "alerts_count": db.query(Alert).filter(Alert.lesson_id == l.id).count(),
      })
  return out


@app.post("/api/lessons/start")
async def start_lesson(payload: dict, db: Session = Depends(get_session)):
  subject_id = payload.get("subject_id")
  class_id = payload.get("class_id")
  if not subject_id or not class_id:
      raise HTTPException(status_code=400, detail="subject_id жана class_id керек")

  subject = db.query(Subject).filter(Subject.id == subject_id).first()
  school_class = db.query(SchoolClass).filter(SchoolClass.id == class_id).first()
  if subject is None or school_class is None:
      raise HTTPException(status_code=400, detail="Сабак же класс табылган жок")

  existing = db.query(Lesson).filter(
      Lesson.class_id == class_id, Lesson.is_active == True
  ).first()
  if existing is not None:
      raise HTTPException(status_code=400, detail="Бул класста активдүү сабак бар")

  lesson = Lesson(
      subject_id=subject_id,
      class_id=class_id,
      camera_id="cam-1",
      started_at=datetime.utcnow(),
      is_active=True,
  )
  db.add(lesson)
  db.commit()
  db.refresh(lesson)

  # Spawn worker
  recognizer_worker.start_worker(lesson.id, event_queue, main_loop)

  return {
      "id": lesson.id,
      "subject_name": subject.name,
      "class_name": school_class.name,
      "started_at": lesson.started_at.isoformat(),
  }


@app.post("/api/lessons/{lesson_id}/stop")
def stop_lesson(lesson_id: int, db: Session = Depends(get_session)):
  lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
  if lesson is None:
      raise HTTPException(status_code=404, detail="Сабак табылган жок")
  recognizer_worker.stop_worker(lesson_id)
  if lesson.is_active:
      lesson.is_active = False
      lesson.ended_at = datetime.utcnow()
      db.commit()
  return {
      "ok": True,
      "ended_at": lesson.ended_at.isoformat() if lesson.ended_at else None,
  }


@app.get("/api/lessons/{lesson_id}/attendees")
def lesson_attendees(lesson_id: int, db: Session = Depends(get_session)):
  rows = db.query(Attendance).filter(Attendance.lesson_id == lesson_id).all()
  out = []
  for a in rows:
      st = a.student
      out.append({
          "student_id": a.student_id,
          "student_name": st.full_name if st else "?",
          "photo_url": ("/students_photos/" + os.path.basename(st.photo_path)) if st and st.photo_path else None,
          "entered_at": a.entered_at.isoformat() if a.entered_at else None,
          "left_at": a.left_at.isoformat() if a.left_at else None,
      })
  return out


@app.get("/api/lessons/{lesson_id}/emotions")
def lesson_emotions(lesson_id: int, db: Session = Depends(get_session)):
  rows = db.query(
      EmotionLog.emotion, func.count(EmotionLog.id)
  ).filter(EmotionLog.lesson_id == lesson_id).group_by(EmotionLog.emotion).all()
  return {emo: cnt for emo, cnt in rows}


# ===== Overview summary =====
@app.get("/api/overview")
def overview_stats(db: Session = Depends(get_session)):
  total_students = db.query(Student).count()
  total_classes = db.query(SchoolClass).count()
  active = db.query(Lesson).filter(Lesson.is_active == True).count()
  today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
  lessons_today = db.query(Lesson).filter(Lesson.started_at >= today_start).count()
  return {
      "total_students": total_students,
      "total_classes": total_classes,
      "active_lessons": active,
      "lessons_today": lessons_today,
  }


# ===== WebSocket =====
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
  await ws.accept()
  connected_sockets.add(ws)
  try:
      await ws.send_json({"type": "hello", "message": "connected"})
      while True:
          # Keep-alive: we don't expect client messages, but read so disconnects fire.
          await ws.receive_text()
  except WebSocketDisconnect:
      pass
  except Exception:
      pass
  finally:
      connected_sockets.discard(ws)


if __name__ == "__main__":
  import uvicorn
  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
