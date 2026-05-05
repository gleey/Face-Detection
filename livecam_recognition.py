"""
Live Webcam Face Recognition — 3 Detector, 1 Kamera
=====================================================
Mendeteksi dan MENGENALI siapa yang muncul di kamera secara real-time.
Wajah dikenali berdasarkan foto referensi di: dataset/{orang}/frontal/

Jalankan:
    python livecam_recognition.py
    python livecam_recognition.py --camera 1          # kamera lain
    python livecam_recognition.py --threshold 0.55    # toleransi pengenalan

Kontrol saat kamera aktif:
    1          : Ganti ke Haar Cascade (cepat)
    2          : Ganti ke MTCNN (akurat)
    3          : Ganti ke RetinaFace (terbaik)
    S / s      : Screenshot → results/recognition/
    Q / ESC    : Keluar

Catatan:
  - Kamera dibuka HANYA SATU KALI (tidak restart saat ganti detektor)
  - Semua 3 detektor dimuat di awal program
  - Bounding box berlabel nama orang + confidence score
  - Wajah tidak dikenal → label "Unknown"
"""

import cv2
import argparse
import sys
import io
import os
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── pastikan src/ tersedia di path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─── KONSTANTA ─────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results" / "recognition"

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


# ─── HELPER: suppress stderr sementara ────────────────────────────────────────
class _SuppressStderr:
    """
    Context manager untuk menangkap / membuang output stderr.
    Mencegah 'ValueError: I/O operation on closed file' dari
    DeepFace / tqdm yang mencoba nulis ke stderr di Windows.
    """
    def __enter__(self):
        self._orig = sys.stderr
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *_):
        sys.stderr = self._orig


# ─── FACE RECOGNIZER ──────────────────────────────────────────────────────────
class FaceRecognizer:
    def __init__(self, dataset_dir: Path, people: list, threshold: float = 0.55):
        self.dataset_dir = dataset_dir
        self.people      = people
        self.threshold   = threshold
        self.model_name  = "ArcFace"
        self.db: dict    = {}
        self._lock       = threading.Lock()

        print("\n[INFO] Memuat referensi wajah dari seluruh folder dataset/ …")
        self._build_database()

    def _build_database(self):
        # Set env var untuk menekan log TF/DeepFace
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("DEEPFACE_HOME", str(Path.home() / ".deepface"))

        from deepface import DeepFace

        for person in self.people:
            person_dir = self.dataset_dir / person

            if not person_dir.exists():
                print(f"  [SKIP] {person_dir} tidak ditemukan.")
                continue

            images = (list(person_dir.rglob("*.jpg"))
                      + list(person_dir.rglob("*.jpeg"))
                      + list(person_dir.rglob("*.png")))

            if not images:
                print(f"  [SKIP] {person} kosong — tidak ada foto referensi.")
                continue

            embeddings = []
            ok, fail = 0, 0
            print(f"\n  Memproses {person} ({len(images)} foto)...")

            for img_path in images:
                try:
                    with _SuppressStderr():
                        result = DeepFace.represent(
                            img_path          = str(img_path),
                            model_name        = self.model_name,
                            enforce_detection = False,
                            detector_backend  = "opencv",
                        )
                    emb = np.array(result[0]["embedding"])
                    embeddings.append(emb)
                    ok += 1
                except Exception as e:
                    print(f"    [!] Gagal: {img_path.name} — {e}")
                    fail += 1

            if embeddings:
                with self._lock:
                    self.db[person] = embeddings
                print(f"  ✓ {person}: {ok} embedding dimuat ({fail} gagal)")
            else:
                print(f"  ✗ {person}: tidak ada embedding yang berhasil.")

        total = sum(len(v) for v in self.db.values())
        print(f"\n[INFO] Database siap — {total} embedding dari {len(self.db)} orang.\n")

    def recognize(self, face_bgr: np.ndarray):
        if not self.db:
            return "Unknown", 0.0

        from deepface import DeepFace

        try:
            with _SuppressStderr():
                result = DeepFace.represent(
                    img_path          = face_bgr,
                    model_name        = self.model_name,
                    enforce_detection = False,
                    detector_backend  = "skip",
                )
            query_emb = np.array(result[0]["embedding"])
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


# ─── ASYNC DETECTOR (non-blocking agar GUI tidak freeze) ──────────────────────
class AsyncDetector:
    """Jalankan detector.detect() di background thread — GUI tetap smooth."""

    def __init__(self, detector):
        self.detector     = detector
        self._lock        = threading.Lock()
        self._running     = False
        self._last_faces  = []
        self._last_ms     = 0.0

    def submit(self, frame):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._run, args=(frame.copy(),), daemon=True)
        t.start()

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


# ─── GAMBAR BOUNDING BOX + LABEL NAMA ────────────────────────────────────────
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

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y - 4, th + 8)

        cv2.rectangle(out, (x, label_y - th - 6), (x + tw + 8, label_y + 2), color, -1)
        cv2.putText(out, label, (x + 4, label_y - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # sudut dekoratif
        cs = 14
        corners = [
            ((x,   y),   (x+cs, y),   (x,   y+cs)),
            ((x+w, y),   (x+w-cs, y), (x+w, y+cs)),
            ((x,   y+h), (x+cs, y+h), (x,   y+h-cs)),
            ((x+w, y+h), (x+w-cs, y+h), (x+w, y+h-cs)),
        ]
        for corner, h_end, v_end in corners:
            cv2.line(out, corner, h_end, color, 3)
            cv2.line(out, corner, v_end, color, 3)

    return out


def draw_hud(frame, detector_name, det_index, n_faces, fps, det_ms,
             names_visible, is_busy):
    h, w = frame.shape[:2]

    badge_color = DETECTOR_BADGE_COLORS.get(detector_name, (80, 80, 80))
    unique      = sorted(set(names_visible))
    who         = f"  [{', '.join(unique)}]" if unique else ""
    status      = "processing…" if is_busy else f"det: {det_ms:.0f} ms"

    info = (f"[{det_index}] {detector_name}"
            f"  |  Wajah: {n_faces}{who}"
            f"  |  {fps:.0f} FPS  {status}")

    bar_w = min(len(info) * 9 + 20, w)
    cv2.rectangle(frame, (0, 0), (bar_w, 36), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (6, 36), badge_color, -1)
    cv2.putText(frame, info, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    hint = "1=Haar Cascade   2=MTCNN   3=RetinaFace   S=Screenshot   Q/ESC=Keluar"
    cv2.putText(frame, hint, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def run(camera_index: int = 0, threshold: float = 0.55):
    from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector

    # Tekan log TF sebelum import apapun
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Bangun database wajah ───────────────────────────────────────────────
    recognizer = FaceRecognizer(DATASET_DIR, PEOPLE, threshold=threshold)

    # ── 2. Muat detektor ──────────────────────────────────────────────────────
    print("[INFO] Memuat semua detektor …")
    raw_detectors = [
        HaarDetector(scale_factor=1.1, min_neighbors=5, min_size=(50, 50)),
        MTCNNDetector(),
        RetinaFaceDetector(resize_to=(320, 240)),  # resize untuk performa
    ]
    # Bungkus semua dengan AsyncDetector
    detectors = [AsyncDetector(d) for d in raw_detectors]

    for i, d in enumerate(detectors, 1):
        print(f"  [{i}] {d.name} ✓")

    det_idx = 0

    # ── 3. Buka kamera ────────────────────────────────────────────────────────
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

        # Kirim frame ke background thread (non-blocking)
        async_det.submit(frame)

        # Ambil hasil terakhir yang sudah selesai
        faces, det_ms = async_det.result

        # ── kenali setiap wajah ────────────────────────────────────────────────
        detections = []
        h_fr, w_fr = frame.shape[:2]

        for f in faces:
            fx, fy, fw, fh = f["bbox"]
            x1 = max(0, fx);          y1 = max(0, fy)
            x2 = min(w_fr, fx + fw);  y2 = min(h_fr, fy + fh)

            if x2 - x1 < 20 or y2 - y1 < 20:
                detections.append({
                    "bbox": f["bbox"], "name": "Unknown",
                    "score": 0.0, "confidence": f.get("confidence"),
                })
                continue

            crop        = frame[y1:y2, x1:x2]
            name, score = recognizer.recognize(crop)
            detections.append({
                "bbox": f["bbox"], "name": name,
                "score": score,    "confidence": f.get("confidence"),
            })

        # ── render ────────────────────────────────────────────────────────────
        annotated = draw_recognition(frame, detections)

        elapsed_total = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000 / max(elapsed_total, 1))
        if len(fps_buf) > 15:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        names_visible = [d["name"] for d in detections if d["name"] != "Unknown"]

        annotated = draw_hud(
            annotated,
            detector_name = async_det.name,
            det_index     = det_idx + 1,
            n_faces       = len(faces),
            fps           = fps,
            det_ms        = det_ms,
            names_visible = names_visible,
            is_busy       = async_det.is_busy,
        )

        cv2.imshow("Live Face Recognition — [1/2/3] Ganti Detektor", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key in key_to_idx:
            det_idx = key_to_idx[key]
            fps_buf.clear()
            print(f"  [Detektor] → {detectors[det_idx].name}")

        elif key in (ord("s"), ord("S")):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = RESULTS_DIR / f"recog_{detectors[det_idx].name.replace(' ','_')}_{ts}_{screenshot_count:03d}.jpg"
            cv2.imwrite(str(fname), annotated)
            screenshot_count += 1
            who_str = ", ".join(sorted(set(names_visible))) or "–"
            print(f"  [Screenshot] {fname.name}  wajah: {len(faces)}  dikenal: {who_str}")

        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Selesai. {screenshot_count} screenshot tersimpan di {RESULTS_DIR}/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live webcam face recognition — 1 kamera, 3 detektor via keyboard"
    )
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()
    run(camera_index=args.camera, threshold=args.threshold)