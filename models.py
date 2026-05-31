"""
SQLAlchemy ORM models for the school dashboard.

Tables:
- classes:      1А, 1Б, ... 11В
- subjects:     Математика, Физика, ...
- students:     name, class_id, photo_path, embedding (512-D, stored as JSON)
- lessons:      teacher-started session (subject + class + camera + start/end times)
- attendance:   student presence inside a lesson (entered_at, left_at)
- emotion_log:  raw emotion samples per (student, lesson, timestamp)
- alerts:       generated alerts (e.g. prolonged negative emotion, unknown face)
"""

import json
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()


class SchoolClass(Base):
    __tablename__ = "classes"
    id = Column(Integer, primary_key=True)
    name = Column(String(16), unique=True, nullable=False)  # "1А", "11В"
    students = relationship("Student", back_populates="school_class", cascade="all, delete-orphan")
    lessons = relationship("Lesson", back_populates="school_class")


class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)  # "Математика"
    lessons = relationship("Lesson", back_populates="subject")


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    full_name = Column(String(128), nullable=False)         # "Марлис Мирланов"
    photo_path = Column(String(256), nullable=False)        # students/Marlis.jpg
    embedding_json = Column(Text, nullable=False)           # JSON list of 512 floats
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    school_class = relationship("SchoolClass", back_populates="students")
    attendances = relationship("Attendance", back_populates="student")
    emotion_logs = relationship("EmotionLog", back_populates="student")
    alerts = relationship("Alert", back_populates="student")

    def get_embedding(self):
        return json.loads(self.embedding_json)

    def set_embedding(self, vec):
        self.embedding_json = json.dumps(list(map(float, vec)))


class Lesson(Base):
    __tablename__ = "lessons"
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    camera_id = Column(String(32), default="cam-1")
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    subject = relationship("Subject", back_populates="lessons")
    school_class = relationship("SchoolClass", back_populates="lessons")
    attendances = relationship("Attendance", back_populates="lesson", cascade="all, delete-orphan")
    emotion_logs = relationship("EmotionLog", back_populates="lesson", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="lesson", cascade="all, delete-orphan")


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    entered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    left_at = Column(DateTime, nullable=True)
    total_seconds = Column(Integer, default=0)

    lesson = relationship("Lesson", back_populates="attendances")
    student = relationship("Student", back_populates="attendances")


class EmotionLog(Base):
    __tablename__ = "emotion_log"
    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    emotion = Column(String(32), nullable=False)  # neutral, happy, sad, ...
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    lesson = relationship("Lesson", back_populates="emotion_logs")
    student = relationship("Student", back_populates="emotion_logs")


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    alert_type = Column(String(64), nullable=False)
    # alert_type values:
    #   "negative_emotion"   - sad/angry/fear sustained
    #   "student_left"       - not detected for > N minutes
    #   "unknown_face"       - face not in DB
    message = Column(String(512), nullable=False)
    status = Column(String(16), default="new")  # new | seen | resolved
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    lesson = relationship("Lesson", back_populates="alerts")
    student = relationship("Student", back_populates="alerts")


# ===== DB factory =====
DB_PATH = "schoolsystem.db"
DB_URL = "sqlite:///" + DB_PATH

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create tables and seed default classes + subjects if DB is empty."""
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        if session.query(SchoolClass).count() == 0:
            grades = list(range(1, 12))
            letters = ["А", "Б", "В"]
            for g in grades:
                for l in letters:
                    session.add(SchoolClass(name=str(g) + l))
            session.commit()
            print("[db] Seeded " + str(len(grades) * len(letters)) + " classes (1А-11В)")
        if session.query(Subject).count() == 0:
            default_subjects = [
                "Математика", "Физика", "Химия", "Биология", "География",
                "Кыргыз тили", "Орус тили", "Англис тили", "Тарых",
                "Информатика", "Адабият", "Дене тарбия", "Музыка", "Сүрөт"
            ]
            for s in default_subjects:
                session.add(Subject(name=s))
            session.commit()
            print("[db] Seeded " + str(len(default_subjects)) + " default subjects")
    finally:
        session.close()


def get_session():
    """FastAPI dependency."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
