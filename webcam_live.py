"""
Scenario G — Webcam Real-Time Detection
Struktur simpan: dataset/{person}/webcam_live/
Jalankan: python webcam_live.py --person orang1

Kontrol:
  1 / 2 / 3  : Ganti detektor (Haar / MTCNN / RetinaFace)
  S          : Screenshot → dataset/{person}/webcam_live/
  Q / ESC    : Keluar
"""

import cv2
import argparse
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector
from visualize import draw_faces

PEOPLE = ["gley", "gervik", "dino"]

DETECTORS = [
    HaarDetector(),
    MTCNNDetector(),
    RetinaFaceDetector(),
]

KEY_LABELS = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
}


def run_webcam(person: str, camera_index: int = 0):
    save_dir = Path("dataset") / person / "webcam_live"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {camera_index} tidak tersedia.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # resolusi lebih rendah untuk FPS lebih tinggi
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    det_idx = 0
    screenshot_count = 0

    print(f"\nWebcam aktif untuk: {person}")
    print(f"Screenshot tersimpan di: {save_dir}")
    print("Tekan 1/2/3 ganti detektor | S simpan | Q keluar\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame tidak terbaca.")
            break

        detector = DETECTORS[det_idx]
        faces, elapsed_ms = detector.detect(frame)
        annotated = draw_faces(frame, faces, detector.name)

        # HUD atas
        info = (f"[{det_idx+1}] {detector.name} | "
                f"Wajah: {len(faces)} | "
                f"{elapsed_ms:.0f} ms | "
                f"Orang: {person}")
        cv2.rectangle(annotated, (0, 0), (len(info) * 9, 32), (0, 0, 0), -1)
        cv2.putText(annotated, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD bawah
        hint = "1=Haar  2=MTCNN  3=RetinaFace  S=Simpan  Q=Keluar"
        cv2.putText(annotated, hint, (8, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(f"Face Detection — {person}", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key in KEY_LABELS:
            det_idx = KEY_LABELS[key]
            print(f"  Detektor: {DETECTORS[det_idx].name}")

        elif key in (ord("s"), ord("S")):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = save_dir / f"{person}_{ts}_{screenshot_count:03d}.jpg"
            cv2.imwrite(str(fname), frame)   # simpan frame asli (tanpa anotasi)
            screenshot_count += 1
            print(f"  Screenshot: {fname.name}")

        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai. {screenshot_count} screenshot tersimpan di {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam real-time face detection")
    parser.add_argument(
        "--person", type=str, default="gley",
        choices=PEOPLE,
        help="Nama folder anggota (default: gley)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index kamera (default: 0)",
    )
    args = parser.parse_args()
    run_webcam(person=args.person, camera_index=args.camera)
