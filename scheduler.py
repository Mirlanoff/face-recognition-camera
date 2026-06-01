"""
Auto-lesson scheduler.

Runs a background thread that checks `schedule` table every 30 seconds:
- If current local day_of_week + time matches a Schedule entry AND no lesson
is currently active for that class -> auto-start a Lesson and worker.
- If a lesson with auto_started=True is past its scheduled end_time -> auto-stop.

The scheduler keeps a reference to the asyncio event_queue + main_loop so it can
push WS events ("lesson_auto_started" / "lesson_auto_stopped") to the dashboard.
"""

import threading
import time
from datetime import datetime, timedelta

from models import SessionLocal, Lesson, Schedule, SchoolClass, Subject
import recognizer_worker


_thread = None
_stop_flag = threading.Event()
_event_queue = None
_main_loop = None


def _push_event(evt: dict):
    """Push a WS event to the dashboard from a non-asyncio thread."""
    global _event_queue, _main_loop
    if _event_queue is None or _main_loop is None:
        return
    try:
        _main_loop.call_soon_threadsafe(_event_queue.put_nowait, evt)
    except Exception as e:
        print("[scheduler] push_event failed: " + str(e))


def _parse_hhmm(s):
    try:
        hh, mm = s.split(":")
        return int(hh), int(mm)
    except Exception:
        return None


def _within_window(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """True if now is between start_hhmm and end_hhmm on the same day."""
    s = _parse_hhmm(start_hhmm)
    e = _parse_hhmm(end_hhmm)
    if s is None or e is None:
        return False
    start_dt = now.replace(hour=s[0], minute=s[1], second=0, microsecond=0)
    end_dt = now.replace(hour=e[0], minute=e[1], second=0, microsecond=0)
    return start_dt <= now < end_dt


def _scheduler_loop():
    print("[scheduler] thread started")
    while not _stop_flag.is_set():
        try:
            _tick()
        except Exception as e:
            print("[scheduler] tick error: " + str(e))
        _stop_flag.wait(timeout=30)
    print("[scheduler] thread stopped")


def _tick():
    now = datetime.now()  # local time
    dow = now.weekday()   # 0=Mon ... 6=Sun

    db = SessionLocal()
    try:
        # ===== AUTO-START =====
        # Find all enabled schedules for today whose window contains now.
        entries = db.query(Schedule).filter(
            Schedule.enabled == True,
            Schedule.day_of_week == dow,
        ).all()

        for sch in entries:
            if not _within_window(now, sch.start_time, sch.end_time):
                continue
            # Is there already an active lesson for this class?
            existing = db.query(Lesson).filter(
                Lesson.class_id == sch.class_id,
                Lesson.is_active == True,
            ).first()
            if existing is not None:
                continue
            # Start a new auto-lesson
            subject = db.query(Subject).filter(Subject.id == sch.subject_id).first()
            school_class = db.query(SchoolClass).filter(SchoolClass.id == sch.class_id).first()
            if subject is None or school_class is None:
                continue
            lesson = Lesson(
                subject_id=sch.subject_id,
                class_id=sch.class_id,
                camera_id="cam-1",
                started_at=datetime.utcnow(),
                is_active=True,
                auto_started=True,
            )
            db.add(lesson)
            db.commit()
            db.refresh(lesson)
            recognizer_worker.start_worker(lesson.id, _event_queue, _main_loop)
            print("[scheduler] AUTO-STARTED lesson " + str(lesson.id)
                  + " " + subject.name + " / " + school_class.name)
            _push_event({
                "type": "lesson_auto_started",
                "lesson_id": lesson.id,
                "subject_name": subject.name,
                "class_name": school_class.name,
                "started_at": lesson.started_at.isoformat(),
            })

        # ===== AUTO-STOP =====
        # Stop auto-started lessons whose scheduled window has ended.
        auto_active = db.query(Lesson).filter(
            Lesson.is_active == True,
            Lesson.auto_started == True,
        ).all()
        for l in auto_active:
            # Find matching schedule entry to know when it ends
            sch = db.query(Schedule).filter(
                Schedule.subject_id == l.subject_id,
                Schedule.class_id == l.class_id,
                Schedule.day_of_week == dow,
                Schedule.enabled == True,
            ).first()
            if sch is None:
                continue
            if _within_window(now, sch.start_time, sch.end_time):
                continue
            # Window ended -> stop
            recognizer_worker.stop_worker(l.id)
            l.is_active = False
            l.ended_at = datetime.utcnow()
            db.commit()
            print("[scheduler] AUTO-STOPPED lesson " + str(l.id))
            _push_event({
                "type": "lesson_auto_stopped",
                "lesson_id": l.id,
                "ended_at": l.ended_at.isoformat(),
            })
    finally:
        db.close()


def start_scheduler(event_queue, main_loop):
    """Called once from server.py on startup."""
    global _thread, _event_queue, _main_loop
    if _thread is not None and _thread.is_alive():
        return
    _event_queue = event_queue
    _main_loop = main_loop
    _stop_flag.clear()
    _thread = threading.Thread(target=_scheduler_loop, daemon=True, name="auto-lesson-scheduler")
    _thread.start()


def stop_scheduler():
    _stop_flag.set()
