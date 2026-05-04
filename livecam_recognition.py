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
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── pastikan src/ tersedia di path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─── KONSTANTA ─────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results" / "recognition"

# Nama folder setiap anggota kelompok (sesuaikan jika ada perubahan)
PEOPLE = ["gley", "gervik", "dino"]

# Warna bounding box per orang (BGR)
PERSON_COLORS = {
    "gley":    (0,   200,  0),     # hijau
    "gervik":  (255, 120,  0),     # biru terang
    "dino":  (0,   60,  255),    # merah
    "Unknown": (60,  60,  60),     # abu-abu gelap
}

UNKNOWN_COLOR = PERSON_COLORS["Unknown"]

# Warna badge nama detektor (BGR)
DETECTOR_BADGE_COLORS = {
    "Haar Cascade": (30, 144, 255),   # biru dodger
    "MTCNN":        (0,  200, 130),   # hijau cyan
    "RetinaFace":   (80,  30, 220),   # ungu
}


# ─── FACE RECOGNIZER ──────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    Membangun database embedding wajah dari seluruh folder dataset,
    lalu mencocokkan setiap wajah baru terhadap database tersebut.

    Memakai DeepFace (ArcFace model) untuk embedding,
    dan cosine similarity untuk pencocokan.
    """

    def __init__(self, dataset_dir: Path, people: list, threshold: float = 0.55):
        self.dataset_dir = dataset_dir
        self.people      = people
        self.threshold   = threshold        # cosine similarity minimum
        self.model_name  = "ArcFace"
        self.db: dict    = {}               # {nama: [embedding, ...]}

        print("\n[INFO] Memuat referensi wajah dari seluruh folder dataset/ …")
        self._build_database()

    def _build_database(self):
        from deepface import DeepFace

        for person in self.people:
            # Ubah target direktori ke folder utama orang tersebut
            person_dir = self.dataset_dir / person
            
            if not person_dir.exists():
                print(f"  [SKIP] {person_dir} tidak ditemukan.")
                continue

            # Gunakan rglob() untuk mencari file gambar di semua subfolder 
            images = (list(person_dir.rglob("*.jpg"))
                      + list(person_dir.rglob("*.jpeg"))
                      + list(person_dir.rglob("*.png")))

            if not images:
                print(f"  [SKIP] {person} kosong — tidak ada foto referensi di folder manapun.")
                continue

            embeddings = []
            ok, fail = 0, 0
            print(f"\n  Memproses {person}...")
            
            for img_path in images:
                try:
                    result = DeepFace.represent(
                        img_path          = str(img_path),
                        model_name        = self.model_name,
                        enforce_detection = False,
                        detector_backend  = "opencv",  # <<--- Ganti ke opencv agar lebih stabil
                    )
                    emb = np.array(result[0]["embedding"])
                    embeddings.append(emb)
                    ok += 1
                except Exception as e:
                    # <<--- Tampilkan error aslinya agar ketahuan masalahnya
                    print(f"    [!] Gagal pada {img_path.name} | Error: {e}") 
                    fail += 1

            if embeddings:
                self.db[person] = embeddings
                print(f"  ✓ {person}: {ok} embedding dimuat ({fail} gagal)")
            else:
                print(f"  ✗ {person}: tidak ada embedding yang berhasil dimuat.")

        total = sum(len(v) for v in self.db.values())
        print(f"\n[INFO] Database siap — {total} embedding dari "
              f"{len(self.db)} orang.\n")

    def recognize(self, face_bgr: np.ndarray):
        """
        Kenali satu wajah (crop BGR).
        Returns:
            name  (str)   : nama orang atau 'Unknown'
            score (float) : cosine similarity 0–1
        """
        if not self.db:
            return "Unknown", 0.0

        from deepface import DeepFace

        try:
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
    
# ─── GAMBAR BOUNDING BOX + LABEL NAMA ────────────────────────────────────────
def draw_recognition(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Gambar bounding box berlabel nama + confidence pada frame.

    detections: list of {
        "bbox":       [x, y, w, h],
        "name":       str,
        "score":      float,     ← cosine similarity
        "confidence": float|None ← kepercayaan detektor
    }
    """
    out = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        name  = det["name"]
        score = det["score"]
        color = PERSON_COLORS.get(name, UNKNOWN_COLOR)

        # ── bounding box ──────────────────────────────────────────────────────
        thick = 2 if name == "Unknown" else 3
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)

        # ── label teks ────────────────────────────────────────────────────────
        if name != "Unknown":
            label = f"{name}  {score:.2f}"
        else:
            label = "Unknown"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y - 4, th + 8)

        # latar label
        cv2.rectangle(
            out,
            (x, label_y - th - 6),
            (x + tw + 8, label_y + 2),
            color, -1,
        )
        cv2.putText(
            out, label,
            (x + 4, label_y - 2),
            font, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # ── sudut dekoratif (corner bracket) ─────────────────────────────────
        cs = 14
        corners = [
            ((x, y),         (x+cs, y),     (x, y+cs)),
            ((x+w, y),       (x+w-cs, y),   (x+w, y+cs)),
            ((x, y+h),       (x+cs, y+h),   (x, y+h-cs)),
            ((x+w, y+h),     (x+w-cs, y+h), (x+w, y+h-cs)),
        ]
        for corner, h_end, v_end in corners:
            cv2.line(out, corner, h_end, color, 3)
            cv2.line(out, corner, v_end, color, 3)

    return out


def draw_hud(frame: np.ndarray,
             detector_name: str,
             det_index: int,
             n_faces: int,
             fps: float,
             det_ms: float,
             names_visible: list) -> np.ndarray:
    """Gambar HUD atas dan bawah pada frame."""
    h, w = frame.shape[:2]

    # ── HUD atas ──────────────────────────────────────────────────────────────
    badge_color = DETECTOR_BADGE_COLORS.get(detector_name, (80, 80, 80))
    unique      = sorted(set(names_visible))
    who         = f"  [{', '.join(unique)}]" if unique else ""

    info = (f"[{det_index}] {detector_name}"
            f"  |  Wajah: {n_faces}{who}"
            f"  |  {fps:.0f} FPS  det: {det_ms:.0f} ms")

    bar_w = min(len(info) * 9 + 20, w)
    cv2.rectangle(frame, (0, 0), (bar_w, 36), (20, 20, 20), -1)
    # badge warna detektor
    cv2.rectangle(frame, (0, 0), (6, 36), badge_color, -1)
    cv2.putText(frame, info, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    # ── HUD bawah ─────────────────────────────────────────────────────────────
    hint = "1=Haar Cascade   2=MTCNN   3=RetinaFace   S=Screenshot   Q/ESC=Keluar"
    cv2.putText(frame, hint,
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def run(camera_index: int = 0, threshold: float = 0.55):
    from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Bangun database wajah (sekali di awal) ──────────────────────────────
    recognizer = FaceRecognizer(DATASET_DIR, PEOPLE, threshold=threshold)

    # ── 2. Muat SEMUA 3 detektor di awal (tidak reload saat ganti) ────────────
    print("[INFO] Memuat semua detektor …")
    detectors = [
        HaarDetector(scale_factor=1.1, min_neighbors=5, min_size=(50, 50)),
        MTCNNDetector(),
        RetinaFaceDetector(),
    ]
    det_names = [d.name for d in detectors]
    for i, d in enumerate(detectors, 1):
        print(f"  [{i}] {d.name} ✓")

    det_idx = 0   # aktif: Haar Cascade

    # ── 3. Buka kamera SATU KALI ───────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {camera_index} tidak bisa dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)  # resolusi lebih rendah untuk FPS lebih tinggi
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"\n[INFO] Kamera index {camera_index} aktif.")
    print("[INFO] Tekan 1/2/3 ganti detektor  |  S screenshot  |  Q/ESC keluar\n")

    screenshot_count = 0
    fps_buf          = []
    det_ms_last      = 0.0

    # Mapping tombol keyboard → index detektor
    key_to_idx = {
        ord("1"): 0,
        ord("2"): 1,
        ord("3"): 2,
    }

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak bisa membaca frame dari kamera.")
            break

        detector = detectors[det_idx]

        # ── deteksi wajah ──────────────────────────────────────────────────────
        faces, det_ms_last = detector.detect(frame)

        # ── kenali setiap wajah ────────────────────────────────────────────────
        detections  = []
        h_fr, w_fr  = frame.shape[:2]

        for f in faces:
            fx, fy, fw, fh = f["bbox"]
            x1 = max(0, fx);           y1 = max(0, fy)
            x2 = min(w_fr, fx + fw);   y2 = min(h_fr, fy + fh)

            if x2 - x1 < 20 or y2 - y1 < 20:
                # wajah terlalu kecil → lewati recognition
                detections.append({
                    "bbox":       f["bbox"],
                    "name":       "Unknown",
                    "score":      0.0,
                    "confidence": f.get("confidence"),
                })
                continue

            crop         = frame[y1:y2, x1:x2]
            name, score  = recognizer.recognize(crop)
            detections.append({
                "bbox":       f["bbox"],
                "name":       name,
                "score":      score,
                "confidence": f.get("confidence"),
            })

        # ── gambar anotasi ─────────────────────────────────────────────────────
        annotated = draw_recognition(frame, detections)

        # ── hitung FPS (rolling average 15 frame) ─────────────────────────────
        elapsed_total = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000 / max(elapsed_total, 1))
        if len(fps_buf) > 15:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        names_visible = [d["name"] for d in detections if d["name"] != "Unknown"]

        # ── HUD ───────────────────────────────────────────────────────────────
        annotated = draw_hud(
            annotated,
            detector_name  = detector.name,
            det_index      = det_idx + 1,
            n_faces        = len(faces),
            fps            = fps,
            det_ms         = det_ms_last,
            names_visible  = names_visible,
        )

        cv2.imshow("Live Face Recognition — [1/2/3] Ganti Detektor", annotated)

        # ── keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in key_to_idx:
            new_idx  = key_to_idx[key]
            det_idx  = new_idx
            fps_buf.clear()      # reset FPS buffer saat ganti detektor
            print(f"  [Detektor] → {detectors[det_idx].name}")

        elif key in (ord("s"), ord("S")):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = RESULTS_DIR / f"recog_{detectors[det_idx].name.replace(' ','_')}_{ts}_{screenshot_count:03d}.jpg"
            cv2.imwrite(str(fname), annotated)
            screenshot_count += 1
            who_str = ", ".join(sorted(set(names_visible))) or "–"
            print(f"  [Screenshot] {fname.name}  wajah: {len(faces)}  dikenal: {who_str}")

        elif key in (ord("q"), ord("Q"), 27):   # Q atau ESC
            break

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Selesai. {screenshot_count} screenshot tersimpan di {RESULTS_DIR}/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live webcam face recognition — 1 kamera, 3 detektor via keyboard"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index kamera (default: 0)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help=(
            "Cosine similarity minimum agar wajah diakui (0–1, default: 0.55). "
            "Naikkan → lebih ketat (kurangi false positive). "
            "Turunkan → lebih longgar (mudah dikenali)."
        ),
    )
    args = parser.parse_args()
    run(camera_index=args.camera, threshold=args.threshold)