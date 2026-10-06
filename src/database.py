"""
SQLite Database Management Module for Automated Vision Attendance
Author: Ashish Kumar Maurya (VisionAttend AI)
Handles student registry, daily attendance logging, and anti-duplicate constraints.
"""

import os
import sqlite3
import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attendance.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes SQLite tables for students and attendance logs."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER UNIQUE NOT NULL,
                roll_no TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
        """)

        # Attendance Logs Table (Enforces UNIQUE(student_id, date) to prevent double marking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                roll_no TEXT NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT DEFAULT 'Present',
                device TEXT DEFAULT 'Webcam Kiosk',
                UNIQUE(student_id, date),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
        """)
        conn.commit()


def register_student(
    student_id: int,
    name: str,
    roll_no: str,
    department: str = "Computer Science & Engineering",
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, str]:
    """Registers a new student in the database."""
    init_db(db_path)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO students (student_id, roll_no, name, department, registered_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, roll_no.strip(), name.strip(), department.strip(), now_str)
            )
            conn.commit()
            return True, f"Student '{name}' registered successfully."
    except sqlite3.IntegrityError as e:
        if "student_id" in str(e):
            return False, f"Error: Student ID {student_id} is already registered."
        elif "roll_no" in str(e):
            return False, f"Error: Roll Number '{roll_no}' already exists."
        return False, f"Integrity Error: {str(e)}"
    except Exception as e:
        return False, f"Database error: {str(e)}"


def get_all_students(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Retrieves all registered students."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, roll_no, name, department, registered_at FROM students ORDER BY student_id ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_student_by_id(student_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict]:
    """Fetches details of a specific student by their numerical student_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, roll_no, name, department, registered_at FROM students WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def mark_attendance(
    student_id: int,
    confidence: float,
    device: str = "Webcam Kiosk",
    db_path: str = DEFAULT_DB_PATH
) -> Dict:
    """
    Logs attendance for the given student.
    Prevents duplicate entries if already marked for today's date.
    """
    init_db(db_path)
    student = get_student_by_id(student_id, db_path)
    if not student:
        return {
            "success": False,
            "status": "not_registered",
            "message": f"Student ID {student_id} is not registered in the system."
        }

    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if already marked for today
        cursor.execute(
            "SELECT id, time FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, today_date)
        )
        existing = cursor.fetchone()
        if existing:
            return {
                "success": True,
                "status": "already_marked",
                "message": f"Attendance already logged for {student['name']} today at {existing['time']}.",
                "student": student,
                "date": today_date,
                "time": existing["time"]
            }

        # Insert new attendance record
        try:
            cursor.execute(
                """
                INSERT INTO attendance (student_id, roll_no, name, department, date, time, confidence, status, device)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'Present', ?)
                """,
                (
                    student_id,
                    student["roll_no"],
                    student["name"],
                    student["department"],
                    today_date,
                    current_time,
                    round(confidence, 2),
                    device
                )
            )
            conn.commit()
            return {
                "success": True,
                "status": "newly_marked",
                "message": f"Attendance successfully recorded for {student['name']}!",
                "student": student,
                "date": today_date,
                "time": current_time,
                "confidence": round(confidence, 2)
            }
        except sqlite3.IntegrityError:
            return {
                "success": True,
                "status": "already_marked",
                "message": f"Attendance already logged for {student['name']} today.",
                "student": student,
                "date": today_date,
                "time": current_time
            }


def get_today_attendance(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Retrieves all attendance records logged for today."""
    init_db(db_path)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, student_id, roll_no, name, department, date, time, confidence, status, device
            FROM attendance
            WHERE date = ?
            ORDER BY time DESC
            """,
            (today_date,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_attendance_history(
    date_filter: Optional[str] = None,
    department_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """Retrieves filtered attendance history."""
    init_db(db_path)
    query = "SELECT id, student_id, roll_no, name, department, date, time, confidence, status, device FROM attendance WHERE 1=1"
    params = []

    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    if department_filter and department_filter != "All":
        query += " AND department = ?"
        params.append(department_filter)

    query += " ORDER BY date DESC, time DESC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_attendance_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Exports all attendance records to a pandas DataFrame."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM attendance ORDER BY date DESC, time DESC", conn)
        return df


def get_analytics_summary(db_path: str = DEFAULT_DB_PATH) -> Dict:
    """Returns aggregated metrics for dashboard visualizations."""
    init_db(db_path)
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today_date,))
        today_present = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_days = cursor.fetchone()[0]

        cursor.execute("SELECT department, COUNT(*) as count FROM attendance GROUP BY department")
        dept_counts = {row["department"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT date, COUNT(*) as count FROM attendance GROUP BY date ORDER BY date DESC LIMIT 7")
        daily_trends = {row["date"]: row["count"] for row in cursor.fetchall()}

        attendance_rate = round((today_present / max(total_students, 1)) * 100, 1)

        return {
            "total_students": total_students,
            "today_present": today_present,
            "today_absent": max(0, total_students - today_present),
            "attendance_rate": attendance_rate,
            "total_days_logged": total_days,
            "dept_counts": dept_counts,
            "daily_trends": daily_trends
        }
