"""
Live Webcam Face Recognition — 3 Detector, 1 Kamera
=====================================================
Mendeteksi dan MENGENALI siapa yang muncul di kamera secara real-time.

PENTING: Jalankan train.py terlebih dahulu untuk membangun database .pkl
    python train.py

Lalu jalankan program ini:
    python livecam_recognition.py
    python livecam_recognition.py --camera 1
    python livecam_recognition.py --threshold 0.55
    python livecam_recognition.py --retrain   # paksa rebuild .pkl sebelum mulai

Kontrol saat kamera aktif:
    1 / 2 / 3  : Ganti detektor (Haar / MTCNN / RetinaFace)
    S          : Screenshot → results/recognition/
    Q / ESC    : Keluar
"""

import cv2
import argparse
import sys
import io
import os
import time
import pickle
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─── KONSTANTA ─────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results" / "recognition"
DB_PATH     = Path(__file__).parent / "models" / "face_db.pkl"

PEOPLE = ["gley", "gervik", "dino"]

PERSON_COLORS = {
    "gley":    (0,   200,  0),
    "gervik":  (255, 120,  0),
    "dino":    (0,   60,  255),
    "Unknown": (60,  60,  60),
}
UNKNOWN_COLOR = PERSON_COLORS["Unknown"]

DETECTOR_BADGE_COLORS = {
    "Haar Cascade": (30, 144, 255),
    "MTCNN":        (0,  200, 130),
    "RetinaFace":   (80,  30, 220),
}


# ─── HELPER ────────────────────────────────────────────────────────────────────
class _SuppressStderr:
    def __enter__(self):
        self._orig = sys.stderr
        sys.stderr = io.StringIO()
        return self
    def __exit__(self, *_):
        sys.stderr = self._orig


# ─── FACE RECOGNIZER ──────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    Load embedding database dari .pkl (hasil train.py).
    Jika .pkl tidak ada, fallback ke build dari dataset secara langsung
    dan simpan hasilnya sebagai .pkl baru.
    """

    MODEL_NAME = "ArcFace"

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.db: dict  = {}
        self._lock     = threading.Lock()

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("DEEPFACE_HOME", str(Path.home() / ".deepface"))

        self._load_or_build()

    def _load_or_build(self):
        if DB_PATH.exists():
            self._load_pkl()
        else:
            print("\n[WARN] models/face_db.pkl tidak ditemukan.")
            print("[INFO] Jalankan `python train.py` terlebih dahulu untuk hasil terbaik.")
            print("[INFO] Sekarang membangun database sementara dari dataset/ ...\n")
            self._build_and_save()

    def _load_pkl(self):
        print(f"\n[INFO] Memuat database dari {DB_PATH} ...")
        try:
            with open(DB_PATH, "rb") as f:
                payload = pickle.load(f)

            self.db = payload["db"]
            total   = payload.get("total_emb", sum(len(v) for v in self.db.values()))
            built   = payload.get("built_at", "?")

            print(f"[INFO] {total} embedding dari {len(self.db)} orang (dibuat: {built})")
            for person, embs in self.db.items():
                print(f"       * {person}: {len(embs)} embedding")
            print()

        except Exception as e:
            print(f"[ERROR] Gagal baca .pkl: {e}")
            print("[INFO] Fallback ke build dari dataset ...\n")
            self._build_and_save()

    def _build_and_save(self):
        """Fallback: build dari dataset lalu simpan .pkl."""
        from deepface import DeepFace

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        db = {}

        for person in PEOPLE:
            person_dir = DATASET_DIR / person
            if not person_dir.exists():
                continue

            images = []
            for ext in img_exts:
                images.extend(person_dir.rglob(f"*{ext}"))
            if not images:
                continue

            print(f"  Memproses {person} ({len(images)} foto)...")
            embeddings = []
            ok = fail = 0

            for img_path in images:
                try:
                    with _SuppressStderr():
                        result = DeepFace.represent(
                            img_path          = str(img_path),
                            model_name        = self.MODEL_NAME,
                            enforce_detection = False,
                            detector_backend  = "opencv",
                        )
                    embeddings.append(np.array(result[0]["embedding"], dtype=np.float32))
                    ok += 1
                except Exception as e:
                    print(f"    [!] {img_path.name}: {e}")
                    fail += 1

            if embeddings:
                db[person] = embeddings
                print(f"  [OK] {person}: {ok} embedding ({fail} gagal)")

        self.db = db

        if db:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "db":        db,
                "model":     self.MODEL_NAME,
                "people":    list(db.keys()),
                "total_emb": sum(len(v) for v in db.values()),
                "built_at":  datetime.now().isoformat(timespec="seconds"),
            }
            with open(DB_PATH, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n[INFO] Database disimpan ke {DB_PATH}")

        total = sum(len(v) for v in self.db.values())
        print(f"[INFO] Database siap -- {total} embedding dari {len(self.db)} orang.\n")

    def recognize(self, face_bgr: np.ndarray):
        if not self.db:
            return "Unknown", 0.0

        from deepface import DeepFace

        try:
            with _SuppressStderr():
                result = DeepFace.represent(
                    img_path          = face_bgr,
                    model_name        = self.MODEL_NAME,
                    enforce_detection = False,
                    detector_backend  = "skip",
                )
            query_emb = np.array(result[0]["embedding"], dtype=np.float32)
        except Exception:
            return "Unknown", 0.0

        best_name  = "Unknown"
        best_score = 0.0

        with self._lock:
            for person, emb_list in self.db.items():
                for ref_emb in emb_list:
                    sim = float(
                        np.dot(query_emb, ref_emb) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(ref_emb) + 1e-10)
                    )
                    if sim > best_score:
                        best_score = sim
                        best_name  = person

        if best_score < (1.0 - self.threshold):
            best_name = "Unknown"

        return best_name, round(best_score, 3)


# ─── ASYNC DETECTOR ───────────────────────────────────────────────────────────
class AsyncDetector:
    def __init__(self, detector):
        self.detector    = detector
        self._lock       = threading.Lock()
        self._running    = False
        self._last_faces = []
        self._last_ms    = 0.0

    def submit(self, frame):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame):
        try:
            faces, ms = self.detector.detect(frame)
            with self._lock:
                self._last_faces = faces
                self._last_ms    = ms
        finally:
            self._running = False

    @property
    def result(self):
        with self._lock:
            return list(self._last_faces), self._last_ms

    @property
    def name(self):
        return self.detector.name

    @property
    def is_busy(self):
        return self._running


# ─── DRAW ─────────────────────────────────────────────────────────────────────
def draw_recognition(frame: np.ndarray, detections: list) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        name  = det["name"]
        score = det["score"]
        color = PERSON_COLORS.get(name, UNKNOWN_COLOR)

        thick = 2 if name == "Unknown" else 3
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)

        label = f"{name}  {score:.2f}" if name != "Unknown" else "Unknown"
        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.6
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        ly = max(y - 4, th + 8)
        cv2.rectangle(out, (x, ly - th - 6), (x + tw + 8, ly + 2), color, -1)
        cv2.putText(out, label, (x + 4, ly - 2), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cs = 14
        for corner, h_end, v_end in [
            ((x,   y),   (x+cs, y),     (x,   y+cs)),
            ((x+w, y),   (x+w-cs, y),   (x+w, y+cs)),
            ((x,   y+h), (x+cs,   y+h), (x,   y+h-cs)),
            ((x+w, y+h), (x+w-cs, y+h), (x+w, y+h-cs)),
        ]:
            cv2.line(out, corner, h_end, color, 3)
            cv2.line(out, corner, v_end, color, 3)

    return out


def draw_hud(frame, detector_name, det_index, n_faces, fps, det_ms,
             names_visible, is_busy):
    h, w   = frame.shape[:2]
    badge  = DETECTOR_BADGE_COLORS.get(detector_name, (80, 80, 80))
    who    = f"  [{', '.join(sorted(set(names_visible)))}]" if names_visible else ""
    status = "processing..." if is_busy else f"det: {det_ms:.0f} ms"
    info   = (f"[{det_index}] {detector_name}  |  "
              f"Wajah: {n_faces}{who}  |  {fps:.0f} FPS  {status}")

    bar_w = min(len(info) * 9 + 20, w)
    cv2.rectangle(frame, (0, 0), (bar_w, 36), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (6, 36), badge, -1)
    cv2.putText(frame, info, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame,
                "1=Haar Cascade   2=MTCNN   3=RetinaFace   S=Screenshot   Q/ESC=Keluar",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run(camera_index: int = 0, threshold: float = 0.55, retrain: bool = False):
    from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Retrain jika diminta
    if retrain:
        print("[INFO] --retrain: menjalankan train.py ...\n")
        import subprocess
        subprocess.run([sys.executable,
                        str(Path(__file__).parent / "train.py"), "--force"],
                       check=True)

    # Load recognizer dari .pkl
    recognizer = FaceRecognizer(threshold=threshold)

    # Load detektor
    print("[INFO] Memuat semua detektor ...")
    detectors = [
        AsyncDetector(HaarDetector(scale_factor=1.1, min_neighbors=5, min_size=(50, 50))),
        AsyncDetector(MTCNNDetector()),
        AsyncDetector(RetinaFaceDetector(resize_to=(320, 240))),
    ]
    for i, d in enumerate(detectors, 1):
        print(f"  [{i}] {d.name} [OK]")
    det_idx = 0

    # Buka kamera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {camera_index} tidak bisa dibuka.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"\n[INFO] Kamera index {camera_index} aktif.")
    print("[INFO] Tekan 1/2/3 ganti detektor  |  S screenshot  |  Q/ESC keluar\n")

    screenshot_count = 0
    fps_buf          = []
    key_to_idx       = {ord("1"): 0, ord("2"): 1, ord("3"): 2}

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak bisa membaca frame dari kamera.")
            break

        async_det = detectors[det_idx]
        async_det.submit(frame)
        faces, det_ms = async_det.result

        detections = []
        h_fr, w_fr = frame.shape[:2]

        for f in faces:
            fx, fy, fw, fh = f["bbox"]
            x1 = max(0, fx);          y1 = max(0, fy)
            x2 = min(w_fr, fx + fw);  y2 = min(h_fr, fy + fh)

            if x2 - x1 < 20 or y2 - y1 < 20:
                detections.append({"bbox": f["bbox"], "name": "Unknown",
                                   "score": 0.0, "confidence": f.get("confidence")})
                continue

            name, score = recognizer.recognize(frame[y1:y2, x1:x2])
            detections.append({"bbox": f["bbox"], "name": name,
                               "score": score, "confidence": f.get("confidence")})

        annotated = draw_recognition(frame, detections)

        elapsed_total = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000 / max(elapsed_total, 1))
        if len(fps_buf) > 15:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        names_visible = [d["name"] for d in detections if d["name"] != "Unknown"]
        annotated = draw_hud(annotated, async_det.name, det_idx + 1,
                             len(faces), fps, det_ms, names_visible, async_det.is_busy)

        cv2.imshow("Live Face Recognition -- [1/2/3] Ganti Detektor", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in key_to_idx:
            det_idx = key_to_idx[key]
            fps_buf.clear()
            print(f"  [Detektor] -> {detectors[det_idx].name}")

        elif key in (ord("s"), ord("S")):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = RESULTS_DIR / (f"recog_{detectors[det_idx].name.replace(' ','_')}"
                                   f"_{ts}_{screenshot_count:03d}.jpg")
            cv2.imwrite(str(fname), annotated)
            screenshot_count += 1
            who_str = ", ".join(sorted(set(names_visible))) or "-"
            print(f"  [Screenshot] {fname.name}  wajah:{len(faces)}  dikenal:{who_str}")

        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Selesai. {screenshot_count} screenshot tersimpan di {RESULTS_DIR}/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--retrain",   action="store_true",
                        help="Rebuild .pkl dari dataset sebelum mulai")
    args = parser.parse_args()
    run(camera_index=args.camera, threshold=args.threshold, retrain=args.retrain)