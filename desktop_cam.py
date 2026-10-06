"""
VisionAttend AI - Standalone Real-Time OpenCV Desktop Camera Runner
Author: Ashish Kumar Maurya | B.Tech CSE (Artificial Intelligence)
Run this script for classroom / kiosk webcam video attendance.
Controls:
  - Press 'Q' or 'ESC' to exit.
  - Press 'R' to reload the face recognition model.
"""

import sys
import os
import time
import cv2

from src.database import init_db, get_student_by_id, mark_attendance
from src.face_engine import (
    detect_faces,
    crop_and_normalize_face,
    FaceRecognizerManager,
    seed_demo_students_if_empty
)


def run_desktop_kiosk(camera_index: int = 0):
    print("=" * 60)
    print("  Starting VisionAttend AI Desktop Kiosk...")
    print("  Author: Ashish Kumar Maurya (B.Tech CSE - AI)")
    print("=" * 60)

    init_db()
    seed_demo_students_if_empty()

    recognizer = FaceRecognizerManager()
    if not recognizer.is_trained():
        print("[!] No trained face model found. Training on available samples...")
        from src.face_engine import train_face_recognizer
        train_face_recognizer()
        recognizer.load_model()

    print("[*] Opening camera feed on index", camera_index)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[!] Error: Unable to open webcam at index {camera_index}.")
        print("    If using an external webcam, try camera index 1.")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    last_feedback_text = "Ready to scan faces..."
    last_feedback_time = time.time()
    feedback_color = (255, 255, 255)

    print("[✓] Camera feed active. Press 'Q' to quit, 'R' to reload model.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab video frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max((curr_time - prev_time), 0.001)
        prev_time = curr_time

        # Detect Faces
        faces, gray_eq = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = crop_and_normalize_face(gray_eq, (x, y, w, h))
            student_id, confidence = recognizer.recognize(face_roi)

            if student_id is not None:
                student = get_student_by_id(student_id)
                if student:
                    # Recognized Student
                    label = f"{student['name']} ({confidence:.0f}%)"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                    cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 220, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    # Mark attendance in SQLite
                    res = mark_attendance(student_id, confidence, device="Desktop Kiosk")
                    if res.get("status") == "newly_marked":
                        last_feedback_text = f"ATTENDANCE MARKED: {student['name']} ({student['roll_no']})"
                        feedback_color = (0, 255, 0)
                        last_feedback_time = time.time()
                    elif res.get("status") == "already_marked":
                        last_feedback_text = f"ALREADY MARKED TODAY: {student['name']}"
                        feedback_color = (255, 200, 0)
                        last_feedback_time = time.time()
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Unrecognized
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Header HUD
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 25), cv2.FILLED)
        cv2.putText(frame, "VisionAttend AI - Automated Attendance Kiosk", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Bottom Notification HUD
        if time.time() - last_feedback_time < 3.5:
            cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (25, 25, 30), cv2.FILLED)
            cv2.putText(frame, last_feedback_text, (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, feedback_color, 2)

        cv2.imshow("VisionAttend AI - Real-Time Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
            print("[*] Exiting camera feed...")
            break
        elif key in [ord('r'), ord('R')]:
            print("[*] Reloading face recognition model...")
            recognizer.load_model()
            last_feedback_text = "Face recognition model reloaded."
            feedback_color = (0, 200, 255)
            last_feedback_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_desktop_kiosk(idx)
