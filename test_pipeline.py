"""
Automated Test Suite for VisionAttend AI
Author: Ashish Kumar Maurya | B.Tech CSE (Artificial Intelligence)
Tests:
  1. SQLite database operations, student registration, and attendance logging.
  2. Duplicate attendance prevention on the same calendar day.
  3. OpenCV Haar Cascade face detector and LBPH face recognition lifecycle.
  4. Master DataFrame and analytics reporting.
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.database import (
    init_db,
    register_student,
    get_student_by_id,
    get_all_students,
    mark_attendance,
    get_today_attendance,
    get_attendance_df,
    get_analytics_summary
)
from src.face_engine import (
    get_face_detector,
    crop_and_normalize_face,
    save_student_face_samples,
    train_face_recognizer,
    FaceRecognizerManager
)


class TestDatabasePipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_attendance.db")
        init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_student_registration(self):
        ok, msg = register_student(101, "Ashish Kumar Maurya", "23001001", "AI", self.db_path)
        self.assertTrue(ok)
        
        student = get_student_by_id(101, self.db_path)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Ashish Kumar Maurya")
        self.assertEqual(student["roll_no"], "23001001")

        # Duplicate Roll No prevention
        ok_dup, _ = register_student(102, "Duplicate User", "23001001", "CSE", self.db_path)
        self.assertFalse(ok_dup)

    def test_attendance_logging_and_anti_duplicate(self):
        register_student(101, "Ashish Kumar Maurya", "23001001", "AI", self.db_path)

        # First mark
        res1 = mark_attendance(101, 92.5, "Test Kiosk", self.db_path)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["status"], "newly_marked")

        # Second mark on same day (Anti-Duplicate Check)
        res2 = mark_attendance(101, 89.0, "Test Kiosk", self.db_path)
        self.assertTrue(res2["success"])
        self.assertEqual(res2["status"], "already_marked")

        # Verify only 1 record exists
        today_records = get_today_attendance(self.db_path)
        self.assertEqual(len(today_records), 1)

    def test_analytics_and_df_export(self):
        register_student(101, "Student One", "ROLL01", "AI", self.db_path)
        register_student(102, "Student Two", "ROLL02", "CSE", self.db_path)
        mark_attendance(101, 95.0, "Test", self.db_path)

        summary = get_analytics_summary(self.db_path)
        self.assertEqual(summary["total_students"], 2)
        self.assertEqual(summary["today_present"], 1)
        self.assertEqual(summary["today_absent"], 1)
        self.assertEqual(summary["attendance_rate"], 50.0)

        df = get_attendance_df(self.db_path)
        self.assertEqual(len(df), 1)


class TestFaceEnginePipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_face_detector_loaded(self):
        detector = get_face_detector()
        self.assertFalse(detector.empty())

    def test_crop_and_normalize(self):
        dummy_gray = np.zeros((300, 300), dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        cropped = crop_and_normalize_face(dummy_gray, bbox)
        self.assertEqual(cropped.shape, (200, 200))

    def test_training_and_recognition_lifecycle(self):
        # Generate synthetic distinct face images for 2 students with distinct spatial geometry
        for sid in [1, 2]:
            samples = []
            for i in range(4):
                img = np.zeros((200, 200), dtype=np.uint8)
                center = (60, 60) if sid == 1 else (140, 140)
                cv2.circle(img, center, 35, 180 + i * 5, -1)
                samples.append(img)
            save_student_face_samples(sid, samples, self.dataset_dir)

        # Train model
        train_res = train_face_recognizer(self.dataset_dir, self.models_dir)
        self.assertTrue(train_res["success"])
        self.assertEqual(train_res["total_students"], 2)

        # Test manager inference
        model_file = os.path.join(self.models_dir, "face_trained.yml")
        labels_file = os.path.join(self.models_dir, "student_labels.json")
        manager = FaceRecognizerManager(model_file, labels_file)
        self.assertTrue(manager.is_trained())

        # Test sample recognition on student 1 pattern
        test_face = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(test_face, (60, 60), 35, 180, -1)
        pred_sid, confidence = manager.recognize(test_face)
        self.assertIsNotNone(pred_sid)
        self.assertEqual(pred_sid, 1)


if __name__ == "__main__":
    unittest.main()
