"""
Face Detection Module — 3 Detectors
Haar Cascade (OpenCV) | MTCNN | RetinaFace
"""

import cv2
import numpy as np
import time
from pathlib import Path


# ─────────────────────────────────────────────
# 1. HAAR CASCADE (OpenCV)
# ─────────────────────────────────────────────
class HaarDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.name = "Haar Cascade"

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Gagal memuat Haar Cascade XML!")

    def detect(self, image_bgr):
        """
        Returns:
            faces (list of dict): [{"bbox": [x,y,w,h], "confidence": None}]
            elapsed_ms (float)
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t0 = time.perf_counter()
        detections = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        faces = []
        if len(detections) > 0:
            for (x, y, w, h) in detections:
                faces.append({"bbox": [int(x), int(y), int(w), int(h)], "confidence": None})

        return faces, elapsed_ms


# ─────────────────────────────────────────────
# 2. MTCNN
# ─────────────────────────────────────────────
class MTCNNDetector:
    def __init__(self, min_face_size=20, thresholds=None):
        from mtcnn import MTCNN
        self.name = "MTCNN"
        self.thresholds = thresholds or [0.6, 0.7, 0.7]
        
        # Initialize with defaults to avoid keyword argument errors
        self.detector = MTCNN()

    def detect(self, image_bgr):
        """
        Returns:
            faces (list of dict): [{"bbox": [x,y,w,h], "confidence": float, "keypoints": dict}]
            elapsed_ms (float)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        detections = self.detector.detect_faces(image_rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        faces = []
        for d in detections:
            x, y, w, h = d["box"]
            # MTCNN kadang return koordinat negatif
            x, y = max(0, x), max(0, y)
            faces.append({
                "bbox": [x, y, w, h],
                "confidence": round(d["confidence"], 4),
                "keypoints": d.get("keypoints", {}),
            })

        return faces, elapsed_ms


# ─────────────────────────────────────────────
# 3. RETINAFACE
# ─────────────────────────────────────────────
class RetinaFaceDetector:
    def __init__(self, threshold=0.9, resize_to=None):
        import insightface
        from insightface.app import FaceAnalysis
        self.name = "RetinaFace"
        self.threshold = threshold
        self.resize_to = resize_to
        
        # ctx_id=0 = GPU 0 (RTX 4050), ctx_id=-1 = CPU
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image_bgr):
        import time
        original_h, original_w = image_bgr.shape[:2]

        if self.resize_to:
            rw, rh = self.resize_to
            infer_img = cv2.resize(image_bgr, (rw, rh))
            scale_x, scale_y = original_w / rw, original_h / rh
        else:
            infer_img = image_bgr
            scale_x = scale_y = 1.0

        t0 = time.perf_counter()
        # insightface pakai RGB
        img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
        detections = self.app.get(img_rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        faces = []
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
            faces.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": round(float(d.det_score), 4),
                "landmarks": {},
            })
        return faces, elapsed_ms


# ─────────────────────────────────────────────
# FACTORY — load detector by name
# ─────────────────────────────────────────────
def get_detector(name: str):
    name = name.lower()
    if name in ("haar", "haarcascade", "haar_cascade"):
        return HaarDetector()
    elif name == "mtcnn":
        return MTCNNDetector()
    elif name in ("retinaface", "retina_face"):
        return RetinaFaceDetector()
    else:
        raise ValueError(f"Detector tidak dikenal: '{name}'. Pilih: haar | mtcnn | retinaface")
