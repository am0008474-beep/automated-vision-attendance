"""
Computer Vision Face Detection & LBPH Recognition Engine
Author: Ashish Kumar Maurya (VisionAttend AI)
Uses OpenCV Haar Cascades for facial localization and Local Binary Patterns
Histograms (LBPH) for lightweight, robust on-device face recognition.
"""

import os
import json
import glob
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(MODELS_DIR, "face_trained.yml")
LABELS_FILE = os.path.join(MODELS_DIR, "student_labels.json")

# Initialize Haar Cascade Face Detector
LOCAL_CASCADE = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
PKG_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_PATH = LOCAL_CASCADE if os.path.exists(LOCAL_CASCADE) else PKG_CASCADE
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def get_face_detector():
    return face_cascade


def detect_faces(image: np.ndarray, min_size: Tuple[int, int] = (60, 60)) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detects faces in an RGB or BGR image.
    Returns list of bounding boxes [(x, y, w, h)] and grayscale image.
    """
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Contrast adjustment via Histogram Equalization
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return list(faces), gray_eq


def crop_and_normalize_face(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """Crops a face from bounding box and normalizes to standard dimensions."""
    x, y, w, h = bbox
    face_roi = gray_image[y:y + h, x:x + w]
    face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    return face_resized


def save_student_face_samples(student_id: int, face_images: List[np.ndarray], dataset_dir: str = DATASET_DIR) -> int:
    """Saves normalized face crops to disk for training."""
    os.makedirs(dataset_dir, exist_ok=True)
    saved_count = 0
    for idx, face_img in enumerate(face_images):
        filename = f"user_{student_id}_{idx + 1}.jpg"
        filepath = os.path.join(dataset_dir, filename)
        cv2.imwrite(filepath, face_img)
        saved_count += 1
    return saved_count


def train_face_recognizer(dataset_dir: str = DATASET_DIR, models_dir: str = MODELS_DIR) -> Dict:
    """
    Reads all saved face crops from dataset directory, trains an LBPH Face Recognizer,
    and serializes the trained model to models/face_trained.yml.
    """
    os.makedirs(models_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(dataset_dir, "user_*_*.jpg"))
    if not image_paths:
        return {"success": False, "message": "No face samples found in dataset directory."}

    faces = []
    labels = []
    label_map = {}

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) >= 3:
            try:
                student_id = int(parts[1])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to guarantee standard dimension
                    img_resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
                    faces.append(img_resized)
                    labels.append(student_id)
                    label_map[student_id] = str(student_id)
            except Exception:
                continue

    if not faces:
        return {"success": False, "message": "Failed to parse valid face images."}

    # Create & Train LBPH Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    # Save trained weights & label mapping
    model_path = os.path.join(models_dir, "face_trained.yml")
    labels_path = os.path.join(models_dir, "student_labels.json")

    recognizer.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    return {
        "success": True,
        "total_samples": len(faces),
        "total_students": len(set(labels)),
        "model_file": model_path
    }


class FaceRecognizerManager:
    """Manages loaded LBPH model instance for low-latency inference."""
    def __init__(self, model_path: str = MODEL_FILE, labels_path: str = LABELS_FILE):
        self.model_path = model_path
        self.labels_path = labels_path
        self.recognizer = None
        self.labels = {}
        self.load_model()

    def load_model(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(self.model_path)
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                return True
            except Exception as e:
                print(f"Error loading face model: {e}")
                self.recognizer = None
                return False
        return False

    def is_trained(self) -> bool:
        return self.recognizer is not None

    def recognize(self, face_gray: np.ndarray, distance_threshold: float = 75.0) -> Tuple[Optional[int], float]:
        """
        Predicts student_id from a cropped grayscale face.
        In LBPH: lower distance = closer match.
        Returns: (student_id or None, confidence_percentage)
        """
        if self.recognizer is None:
            return None, 0.0

        if face_gray.shape != (200, 200):
            face_gray = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)

        try:
            student_id, distance = self.recognizer.predict(face_gray)
            
            # Convert LBPH distance (typical range 0 - 100) to confidence %
            # Distance 0 => 100% confidence, distance 75 => ~47% confidence
            confidence = max(0.0, min(100.0, round(100.0 - (distance * 0.7), 1)))

            if distance <= distance_threshold and confidence >= 45.0:
                return int(student_id), confidence
            else:
                return None, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 0.0


def seed_demo_students_if_empty(dataset_dir: str = DATASET_DIR, db_path: Optional[str] = None):
    """
    Seeds demo students into the database and dataset if empty,
    so the system is functional and verifiable immediately.
    """
    from src.database import register_student, get_all_students
    
    os.makedirs(dataset_dir, exist_ok=True)
    existing_students = get_all_students(db_path) if db_path else get_all_students()
    
    if not existing_students:
        demo_users = [
            (101, "Ashish Kumar Maurya", "23001001", "Artificial Intelligence"),
            (102, "Rahul Sharma", "23001002", "Computer Science & Engineering"),
            (103, "Priya Verma", "23001003", "Data Science & Analytics")
        ]
        for sid, name, roll, dept in demo_users:
            if db_path:
                register_student(sid, name, roll, dept, db_path)
            else:
                register_student(sid, name, roll, dept)

    # If dataset has no face samples, synthesize distinct geometric face samples for testing
    existing_samples = glob.glob(os.path.join(dataset_dir, "user_*_*.jpg"))
    if not existing_samples:
        for sid in [101, 102, 103]:
            for sample_idx in range(1, 6):
                # Create synthetic 200x200 face-like pattern with oval, eyes and mouth
                img = np.full((200, 200), 120 + (sid * 15) % 80, dtype=np.uint8)
                # Face contour
                cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, 200, -1)
                # Eyes
                cv2.circle(img, (75, 80), 12 + sample_idx, 50, -1)
                cv2.circle(img, (125, 80), 12 + sample_idx, 50, -1)
                # Nose
                cv2.line(img, (100, 85), (100, 115), 40, 3)
                # Mouth
                cv2.ellipse(img, (100, 140), (25, 12), 0, 0, 180, 50, 4)
                
                # Add minor gaussian noise for realistic training variance
                noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)

                filename = f"user_{sid}_{sample_idx}.jpg"
                cv2.imwrite(os.path.join(dataset_dir, filename), img)
        
        # Train model on these seed samples
        train_face_recognizer(dataset_dir)
