"""
FastAPI server for the school recognition dashboard.

Endpoints:
GET  /                    -> overview page (HTML)
GET  /analytics           -> analytics page (HTML, stub for now)
GET  /alerts              -> alerts page (HTML, stub)
GET  /history             -> history page (HTML, stub)

GET  /api/classes         -> list classes
GET  /api/students        -> list students (optional class_id filter)
POST /api/students        -> register student (multipart: full_name, class_id, photo)
DELETE /api/students/{id} -> delete student

GET  /api/subjects        -> list subjects
GET  /api/lessons/active  -> currently active lessons
GET  /api/lessons/history -> finished lessons
POST /api/lessons/start   -> start a lesson (json: subject_id, class_id)
POST /api/lessons/{id}/stop -> stop a lesson

GET  /api/overview        -> summary numbers for the overview page
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from models import (
    init_db, get_session, SessionLocal,
    SchoolClass, Subject, Student, Lesson, Attendance, EmotionLog, Alert,
)

STUDENTS_DIR = Path("students")
STUDENTS_DIR.mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app = FastAPI(title="Мектеп таануу системасы")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/students_photos", StaticFiles(directory="students"), name="students_photos")
templates = Jinja2Templates(directory="templates")


# ===== Startup =====
@app.on_event("startup")
def on_startup():
    init_db()
    print("[server] DB ready.")


# ===== Lazy InsightFace loader (used only when registering a student) =====
_insightface_app = None


def get_insightface():
    global _insightface_app
    if _insightface_app is None:
        from insightface.app import FaceAnalysis
        print("[server] Loading InsightFace for registration...")
        _insightface_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[server] InsightFace ready.")
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


# ===== Classes =====
@app.get("/api/classes")
def list_classes(db: Session = Depends(get_session)):
    classes = db.query(SchoolClass).order_by(SchoolClass.id).all()
    out = []
    for c in classes:
        out.append({
            "id": c.id,
            "name": c.name,
            "student_count": len(c.students),
        })
    return out


# ===== Subjects =====
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
    # Validate class
    school_class = db.query(SchoolClass).filter(SchoolClass.id == class_id).first()
    if school_class is None:
        raise HTTPException(status_code=400, detail="Класс табылган жок")

    # Save file with a unique name
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

    # Run InsightFace on the file
    import cv2
    img = cv2.imread(str(dest))
    if img is None:
        os.remove(dest)
        raise HTTPException(status_code=400, detail="Сүрөттү окуу мүмкүн эмес")

    app_if = get_insightface()
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
    # Remove the photo file
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

    # If this class already has an active lesson, refuse
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
    if not lesson.is_active:
        return {"ok": True, "already_stopped": True}
    lesson.is_active = False
    lesson.ended_at = datetime.utcnow()
    db.commit()
    return {
        "ok": True,
        "ended_at": lesson.ended_at.isoformat(),
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
