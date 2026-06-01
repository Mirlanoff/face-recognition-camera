"""
Idempotent seeder: create classes 1А, 1Б, 2А, 2Б, ..., 11А, 11Б (22 total).

Safe to run multiple times - existing classes are skipped.

Usage:
    python seed_classes.py
"""

from models import SessionLocal, SchoolClass


GRADES = list(range(1, 12))          # 1 .. 11
LETTERS = ["А", "Б"]                  # Cyrillic A, B


def main():
    s = SessionLocal()
    try:
        existing = {c.name for c in s.query(SchoolClass).all()}
        created = []
        skipped = []
        for grade in GRADES:
            for letter in LETTERS:
                name = str(grade) + letter
                if name in existing:
                    skipped.append(name)
                    continue
                s.add(SchoolClass(name=name))
                created.append(name)
        s.commit()

        print("=" * 50)
        print("Класстарды кошуу аякталды")
        print("=" * 50)
        print("Кошулду  (" + str(len(created)) + "): " + ", ".join(created) if created else "Жаңы класс кошулган жок")
        print("Бар экен (" + str(len(skipped)) + "): " + ", ".join(skipped) if skipped else "")
        total = s.query(SchoolClass).count()
        print("Бардыгы базада: " + str(total) + " класс")
    finally:
        s.close()


if __name__ == "__main__":
    main()
