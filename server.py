"""
FastAPI server for the school recognition dashboard.

Routes:
  HTML
    GET  /                        -> overview page
    GET  /analytics               -> analytics page
    GET  /alerts                  -> alerts page
    GET  /history                 -> history page (stub)

  Data
    GET  /api/classes
    GET  /api/subjects
    GET  /api/students            (optional class_id)
    POST /api/students            (multipart register)
    DELETE /api/students/{id}

    GET  /api/lessons/active
    GET  /api/lessons/history
    POST /api/lessons/start
    POST /api/lessons/{id}/stop

    GET  /api/lessons/{id}/attendees
    GET  /api/lessons/{id}/emotions

    GET  /api/overview

  Analytics
    GET  /api/analytics/emotions_pie
    GET  /api/analytics/emotions_timeline
    GET  /api/analytics/attendance_by_class
    GET  /api/analytics/student_summary

  Alerts
    GET    /api/alerts                          -> list (filters: status, alert_type, lesson_id, student_id, limit)
    GET    /api/alerts/stats                    -> counts by status / by type
    PATCH  /api/alerts/{id}                     -> change status
    POST   /api/alerts/mark_all_seen            -> bulk mark all new as seen
    DELETE /api/alerts/{id}                     -> delete one
    DELETE /api/alerts                          -> delete all (optional status filter)

  WebSocket
    /ws       -> live events
"""

import os
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

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


# ===== Lazy InsightFace loader for STUDENT REGISTRATION =====
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
    return templates.TemplateResponse(request, "analytics.html", {"active_tab": "analytics"})


@app.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request):
    return templates.TemplateResponse(request, "alerts.html", {"active_tab": "alerts"})


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
    import cv2

    print("[register_student] full_name=" + repr(full_name)
          + " class_id=" + str(class_id)
          + " filename=" + repr(photo.filename)
          + " content_type=" + repr(photo.content_type))

    school_class = db.query(SchoolClass).filter(SchoolClass.id == class_id).first()
    if school_class is None:
        raise HTTPException(status_code=400, detail="Класс табылган жок")

    contents = await photo.read()
    print("[register_student] received bytes=" + str(len(contents)))
    if not contents:
        raise HTTPException(status_code=400, detail="Бош файл")

    # Decode from memory — cv2.imread can't handle cyrillic paths on Windows
    try:
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Сүрөттү декоддоо катасы: " + str(e))

    if img is None:
        print("[register_student] cv2.imdecode returned None; first bytes hex="
              + contents[:16].hex())
        raise HTTPException(status_code=400, detail="Сүрөттү окуу мүмкүн эмес (формат туура эмес)")
    print("[register_student] decoded shape=" + str(img.shape))

    try:
        app_if = get_insightface_for_registration()
        faces = app_if.get(img)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="InsightFace катасы: " + str(e))

    print("[register_student] faces=" + str(len(faces)))
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="Сүрөттө бет табылган жок")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding

    # Build ASCII-safe filename (Windows + cv2 friendly)
    safe = "".join(c for c in full_name if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_ascii = safe.encode("ascii", "ignore").decode("ascii").strip()
    if not safe_ascii:
        safe_ascii = "student"
    timestamp = str(int(time.time()))
    ext = Path(photo.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    filename = safe_ascii.replace(" ", "_") + "_" + timestamp + ext
    dest = STUDENTS_DIR / filename
    with open(dest, "wb") as f:
        f.write(contents)
    print("[register_student] saved to " + str(dest))

    student = Student(
        full_name=full_name.strip(),
        photo_path=str(dest),
        class_id=class_id,
    )
    student.set_embedding(emb)
    db.add(student)
    db.commit()
    db.refresh(student)
    print("[register_student] OK id=" + str(student.id))

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


# ===== Analytics =====
@app.get("/api/analytics/emotions_pie")
def analytics_emotions_pie(
    lesson_id: int = None,
    class_id: int = None,
    db: Session = Depends(get_session),
):
    q = db.query(EmotionLog.emotion, func.count(EmotionLog.id))
    if lesson_id is not None:
        q = q.filter(EmotionLog.lesson_id == lesson_id)
    elif class_id is not None:
        q = q.join(Lesson, Lesson.id == EmotionLog.lesson_id).filter(Lesson.class_id == class_id)
    rows = q.group_by(EmotionLog.emotion).all()
    result = {emo: cnt for emo, cnt in rows}
    total = sum(result.values()) if result else 0
    return {
        "counts": result,
        "total": total,
        "scope": (
            "lesson:" + str(lesson_id) if lesson_id is not None else
            ("class:" + str(class_id) if class_id is not None else "all")
        ),
    }


@app.get("/api/analytics/emotions_timeline")
def analytics_emotions_timeline(
    lesson_id: int,
    db: Session = Depends(get_session),
):
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()
    if lesson is None:
        raise HTTPException(status_code=404, detail="Сабак табылган жок")
    start = lesson.started_at
    logs = db.query(EmotionLog).filter(EmotionLog.lesson_id == lesson_id).order_by(EmotionLog.timestamp).all()
    buckets = {}
    for log in logs:
        delta = log.timestamp - start
        minute = int(delta.total_seconds() // 60)
        if minute < 0:
            minute = 0
        if minute not in buckets:
            buckets[minute] = {}
        buckets[minute][log.emotion] = buckets[minute].get(log.emotion, 0) + 1
    series = []
    for m in sorted(buckets.keys()):
        series.append({"minute": m, "counts": buckets[m]})
    return {
        "lesson_id": lesson_id,
        "started_at": start.isoformat() if start else None,
        "series": series,
    }


@app.get("/api/analytics/attendance_by_class")
def analytics_attendance_by_class(
    days: int = 30,
    db: Session = Depends(get_session),
):
    since = datetime.utcnow() - timedelta(days=days)
    classes = db.query(SchoolClass).order_by(SchoolClass.id).all()
    out = []
    for c in classes:
        lessons = db.query(Lesson).filter(
            Lesson.class_id == c.id,
            Lesson.started_at >= since,
        ).all()
        if not lessons:
            continue
        total_students = len(c.students)
        if total_students == 0:
            continue
        total_seen = 0
        for l in lessons:
            seen = db.query(Attendance).filter(Attendance.lesson_id == l.id).count()
            total_seen += seen
        avg_seen = total_seen / len(lessons)
        pct = round((avg_seen / total_students) * 100, 1) if total_students > 0 else 0
        out.append({
            "class_id": c.id,
            "class_name": c.name,
            "total_students": total_students,
            "lessons_count": len(lessons),
            "avg_attendance_pct": pct,
        })
    return {"days": days, "rows": out}


@app.get("/api/analytics/student_summary")
def analytics_student_summary(
    class_id: int = None,
    db: Session = Depends(get_session),
):
    q = db.query(Student)
    if class_id is not None:
        q = q.filter(Student.class_id == class_id)
    students = q.order_by(Student.full_name).all()
    NEG = {"sad", "angry", "fear", "disgust"}
    out = []
    for s in students:
        lessons_seen = db.query(Attendance).filter(Attendance.student_id == s.id).count()
        emo_rows = db.query(
            EmotionLog.emotion, func.count(EmotionLog.id)
        ).filter(EmotionLog.student_id == s.id).group_by(EmotionLog.emotion).all()
        emo_counts = {emo: cnt for emo, cnt in emo_rows}
        total_emo = sum(emo_counts.values())
        dominant = max(emo_counts.items(), key=lambda x: x[1])[0] if emo_counts else None
        neg_total = sum(v for k, v in emo_counts.items() if k in NEG)
        neg_pct = round((neg_total / total_emo) * 100, 1) if total_emo > 0 else 0.0
        out.append({
            "student_id": s.id,
            "student_name": s.full_name,
            "class_name": s.school_class.name if s.school_class else "",
            "lessons_seen": lessons_seen,
            "dominant_emotion": dominant,
            "negative_pct": neg_pct,
            "emotion_counts": emo_counts,
        })
    return {"rows": out}


# ===== Alerts =====
ALLOWED_ALERT_STATUSES = {"new", "seen", "resolved"}


def _serialize_alert(a, db):
    student = a.student
    lesson = a.lesson
    return {
        "id": a.id,
        "alert_type": a.alert_type,
        "message": a.message,
        "status": a.status,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "lesson_id": a.lesson_id,
        "lesson_label": (
            ((lesson.subject.name if lesson.subject else "?") + " · "
             + (lesson.school_class.name if lesson.school_class else "?"))
            if lesson else None
        ),
        "student_id": a.student_id,
        "student_name": student.full_name if student else None,
        "student_photo": (
            "/students_photos/" + os.path.basename(student.photo_path)
            if student and student.photo_path else None
        ),
        "class_name": (
            student.school_class.name if student and student.school_class else
            (lesson.school_class.name if lesson and lesson.school_class else "")
        ),
    }


@app.get("/api/alerts")
def list_alerts(
    status: str = None,
    alert_type: str = None,
    lesson_id: int = None,
    student_id: int = None,
    limit: int = 200,
    db: Session = Depends(get_session),
):
    q = db.query(Alert)
    if status:
        q = q.filter(Alert.status == status)
    if alert_type:
        q = q.filter(Alert.alert_type == alert_type)
    if lesson_id is not None:
        q = q.filter(Alert.lesson_id == lesson_id)
    if student_id is not None:
        q = q.filter(Alert.student_id == student_id)
    rows = q.order_by(Alert.created_at.desc()).limit(min(limit, 1000)).all()
    return [_serialize_alert(a, db) for a in rows]


@app.get("/api/alerts/stats")
def alerts_stats(db: Session = Depends(get_session)):
    by_status_rows = db.query(Alert.status, func.count(Alert.id)).group_by(Alert.status).all()
    by_type_rows = db.query(Alert.alert_type, func.count(Alert.id)).group_by(Alert.alert_type).all()
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(Alert).filter(Alert.created_at >= today_start).count()
    return {
        "total": db.query(Alert).count(),
        "today": today_count,
        "by_status": {s: c for s, c in by_status_rows},
        "by_type": {t: c for t, c in by_type_rows},
    }


@app.patch("/api/alerts/{alert_id}")
async def update_alert(alert_id: int, payload: dict, db: Session = Depends(get_session)):
    a = db.query(Alert).filter(Alert.id == alert_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail="Эскертүү табылган жок")
    new_status = payload.get("status")
    if new_status not in ALLOWED_ALERT_STATUSES:
        raise HTTPException(status_code=400, detail="status must be one of: " + ", ".join(ALLOWED_ALERT_STATUSES))
    a.status = new_status
    db.commit()
    db.refresh(a)
    if event_queue is not None:
        try:
            event_queue.put_nowait({"type": "alert_updated", "alert": _serialize_alert(a, db)})
        except Exception:
            pass
    return _serialize_alert(a, db)


@app.post("/api/alerts/mark_all_seen")
async def mark_all_seen(db: Session = Depends(get_session)):
    count = db.query(Alert).filter(Alert.status == "new").update({Alert.status: "seen"})
    db.commit()
    if event_queue is not None:
        try:
            event_queue.put_nowait({"type": "alerts_bulk_update", "count": count, "new_status": "seen"})
        except Exception:
            pass
    return {"ok": True, "updated": count}


@app.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_session)):
    a = db.query(Alert).filter(Alert.id == alert_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail="Эскертүү табылган жок")
    db.delete(a)
    db.commit()
    return {"ok": True}


@app.delete("/api/alerts")
def delete_alerts_bulk(status: str = None, db: Session = Depends(get_session)):
    q = db.query(Alert)
    if status:
        q = q.filter(Alert.status == status)
    count = q.count()
    q.delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": count}


# ===== WebSocket =====
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_sockets.add(ws)
    try:
        await ws.send_json({"type": "hello", "message": "connected"})
        while True:
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
