"""
CLI tool to export and query BASIRET SQLite database.
Usage: python export_db.py <command> [args]
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from models import SessionLocal, Student, SchoolClass, Lesson, Attendance, EmotionLog, Alert
from sqlalchemy import func


def fmt_table(headers, rows, widths=None):
    if not rows:
        print("  (пусто)")
        return
    if widths is None:
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    print(sep)
    print(line(headers))
    print(sep)
    for r in rows:
        print(line(r))
    print(sep)


def cmd_tables(args):
    db = SessionLocal()
    try:
        models = [("Students", Student), ("SchoolClasses", SchoolClass),
                  ("Lessons", Lesson), ("Attendance", Attendance),
                  ("EmotionLogs", EmotionLog), ("Alerts", Alert)]
        rows = []
        for name, model in models:
            cnt = db.query(model).count()
            rows.append([name, cnt])
        fmt_table(["Таблица", "Записей"], rows, [20, 10])
    finally:
        db.close()


def cmd_students(args):
    db = SessionLocal()
    try:
        rows = []
        for s in db.query(Student).all():
            c = db.query(SchoolClass).filter(SchoolClass.id == s.class_id).first()
            rows.append([s.id, s.full_name, c.name if c else "?"])
        fmt_table(["ID", "ФИО", "Класс"], rows, [5, 30, 10])
    finally:
        db.close()


def cmd_classes(args):
    db = SessionLocal()
    try:
        rows = []
        for c in db.query(SchoolClass).all():
            cnt = db.query(Student).filter(Student.class_id == c.id).count()
            rows.append([c.id, c.name, cnt])
        fmt_table(["ID", "Класс", "Учеников"], rows, [5, 15, 10])
    finally:
        db.close()


def cmd_lessons(args):
    db = SessionLocal()
    try:
        rows = []
        for l in db.query(Lesson).all():
            c = db.query(SchoolClass).filter(SchoolClass.id == l.class_id).first()
            rows.append([l.id, c.name if c else "?", l.started_at, l.ended_at])
        fmt_table(["ID", "Класс", "Начало", "Конец"], rows, [5, 15, 22, 22])
    finally:
        db.close()


def cmd_attendance(args):
    db = SessionLocal()
    try:
        rows = []
        for a in db.query(Attendance).filter(Attendance.lesson_id == args.lesson_id).all():
            s = db.query(Student).filter(Student.id == a.student_id).first()
            rows.append([a.student_id, s.full_name if s else "?", str(a.entered_at)])
        fmt_table(["ID ученика", "ФИО", "Время входа"], rows, [12, 30, 20])
    finally:
        db.close()


def cmd_emotions(args):
    db = SessionLocal()
    try:
        q = db.query(EmotionLog).filter(EmotionLog.lesson_id == args.lesson_id)
        if args.student:
            q = q.filter(EmotionLog.student_id == args.student)
        rows = []
        for e in q.order_by(EmotionLog.timestamp).all():
            s = db.query(Student).filter(Student.id == e.student_id).first()
            rows.append([e.student_id, s.full_name if s else "?", e.emotion, str(e.timestamp)])
        fmt_table(["ID", "ФИО", "Эмоция", "Время"], rows, [5, 30, 12, 20])
    finally:
        db.close()


def cmd_alerts(args):
    db = SessionLocal()
    try:
        rows = []
        for a in db.query(Alert).filter(Alert.lesson_id == args.lesson_id).order_by(Alert.created_at).all():
            s = db.query(Student).filter(Student.id == a.student_id).first()
            rows.append([a.alert_type, s.full_name if s else "?", a.message, str(a.created_at)])
        fmt_table(["Тип", "ФИО", "Сообщение", "Время"], rows, [15, 20, 35, 20])
    finally:
        db.close()


def cmd_stats(args):
    db = SessionLocal()
    try:
        rows = db.query(EmotionLog.emotion, func.count(EmotionLog.id)).filter(
            EmotionLog.lesson_id == args.lesson_id
        ).group_by(EmotionLog.emotion).all()
        rows = [[e, c] for e, c in rows]
        fmt_table(["Эмоция", "Количество"], rows, [15, 12])
    finally:
        db.close()


def cmd_export_csv(args):
    db = SessionLocal()
    try:
        with open(args.filename, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["type", "student_id", "student_name", "emotion_or_message", "timestamp"])
            for e in db.query(EmotionLog).filter(EmotionLog.lesson_id == args.lesson_id).all():
                s = db.query(Student).filter(Student.id == e.student_id).first()
                w.writerow(["emotion", e.student_id, s.full_name if s else "?", e.emotion, e.timestamp])
            for a in db.query(Alert).filter(Alert.lesson_id == args.lesson_id).all():
                s = db.query(Student).filter(Student.id == a.student_id).first()
                w.writerow(["alert", a.student_id, s.full_name if s else "?", a.message, a.created_at])
        print("Сохранено в " + args.filename)
    finally:
        db.close()


def cmd_export_json(args):
    db = SessionLocal()
    try:
        data = {"lesson_id": args.lesson_id, "emotions": [], "alerts": [], "attendance": []}
        for e in db.query(EmotionLog).filter(EmotionLog.lesson_id == args.lesson_id).all():
            s = db.query(Student).filter(Student.id == e.student_id).first()
            data["emotions"].append({
                "student_id": e.student_id,
                "student_name": s.full_name if s else "?",
                "emotion": e.emotion,
                "timestamp": str(e.timestamp),
            })
        for a in db.query(Alert).filter(Alert.lesson_id == args.lesson_id).all():
            s = db.query(Student).filter(Student.id == a.student_id).first()
            data["alerts"].append({
                "student_id": a.student_id,
                "student_name": s.full_name if s else "?",
                "alert_type": a.alert_type,
                "message": a.message,
                "timestamp": str(a.created_at),
            })
        for a in db.query(Attendance).filter(Attendance.lesson_id == args.lesson_id).all():
            s = db.query(Student).filter(Student.id == a.student_id).first()
            data["attendance"].append({
                "student_id": a.student_id,
                "student_name": s.full_name if s else "?",
                "entered_at": str(a.entered_at),
            })
        with open(args.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Сохранено в " + args.filename)
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description="BASIRET DB exporter")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("tables", help="Показать все таблицы")
    sub.add_parser("students", help="Список учеников")
    sub.add_parser("classes", help="Список классов")
    sub.add_parser("lessons", help="Список уроков")

    pa = sub.add_parser("attendance", help="Посещаемость урока")
    pa.add_argument("lesson_id", type=int)

    pe = sub.add_parser("emotions", help="Эмоции за урок")
    pe.add_argument("lesson_id", type=int)
    pe.add_argument("--student", type=int, default=None)

    pal = sub.add_parser("alerts", help="Алерты за урок")
    pal.add_argument("lesson_id", type=int)

    ps = sub.add_parser("stats", help="Статистика эмоций")
    ps.add_argument("lesson_id", type=int)

    pc = sub.add_parser("export-csv", help="Экспорт в CSV")
    pc.add_argument("lesson_id", type=int)
    pc.add_argument("filename")

    pj = sub.add_parser("export-json", help="Экспорт в JSON")
    pj.add_argument("lesson_id", type=int)
    pj.add_argument("filename")

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    cmds = {
        "tables": cmd_tables, "students": cmd_students, "classes": cmd_classes,
        "lessons": cmd_lessons, "attendance": cmd_attendance, "emotions": cmd_emotions,
        "alerts": cmd_alerts, "stats": cmd_stats, "export-csv": cmd_export_csv,
        "export-json": cmd_export_json,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()