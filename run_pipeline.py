"""
Pipeline Utama — Evaluasi 3 Detektor pada Dataset
Struktur: dataset/orang_X/scenario/
Jalankan: python run_pipeline.py
"""

import cv2
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector
from visualize import draw_faces, compare_detectors, plot_summary


# ─────────────────────────────────────────────
# KONFIGURASI — ubah di sini jika perlu
# ─────────────────────────────────────────────
DATASET_ROOT = Path("dataset")
RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)

# Nama folder per anggota kelompok
PEOPLE = ["gley", "gervik", "dino"]

# Nama scenario (harus sama dengan nama subfolder)
SCENARIO_NAMES = [
    "frontal",
    "side_pose",
    "low_light",
    "occlusion",
    "multiple_faces",
    "small_face",
    "webcam_live",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Detektor yang digunakan
DETECTORS = {
    "Haar Cascade": HaarDetector(),
    "MTCNN":        MTCNNDetector(),
    "RetinaFace":   RetinaFaceDetector(),
}


# ─────────────────────────────────────────────
# FUNGSI BANTU
# ─────────────────────────────────────────────
def get_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def build_scenario_map():
    """
    Buat mapping:
      { "gley/frontal": Path("dataset/gley/frontal"), ... }
    """
    scenarios = {}
    for person in PEOPLE:
        for scenario in SCENARIO_NAMES:
            key = f"{person}/{scenario}"
            scenarios[key] = DATASET_ROOT / person / scenario
    return scenarios


def process_image(image_path: Path, detectors: dict, person: str, scenario: str,
                  save_comparison=True):
    """Proses satu gambar dengan semua detektor. Return list of result dicts."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] Tidak bisa membaca: {image_path.name}")
        return []

    results_dict = {}
    row_list = []

    for det_name, detector in detectors.items():
        faces, elapsed_ms = detector.detect(image)

        results_dict[det_name] = (faces, elapsed_ms)
        row_list.append({
            "image":      image_path.name,
            "person":     person,
            "scenario":   scenario,
            "detector":   det_name,
            "n_faces":    len(faces),
            "elapsed_ms": round(elapsed_ms, 2),
            "faces_json": json.dumps(faces),
        })

        # Simpan gambar teranotasi → results/<detector>/<person>/<scenario>/
        det_key = det_name.lower().replace(" ", "_")
        out_dir = RESULTS_ROOT / det_key / person / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = draw_faces(image, faces, det_name)
        cv2.imwrite(str(out_dir / image_path.name), annotated)

    # Simpan grid perbandingan 3 detektor
    if save_comparison:
        comp_dir = RESULTS_ROOT / "comparison" / person / scenario
        comp_dir.mkdir(parents=True, exist_ok=True)
        compare_detectors(
            image,
            results_dict,
            save_path=str(comp_dir / f"cmp_{image_path.stem}.png"),
        )

    return row_list


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print(" FACE DETECTION PIPELINE")
    print(" Haar Cascade | MTCNN | RetinaFace")
    print(f" Anggota: {', '.join(PEOPLE)}")
    print("=" * 60)

    all_rows = []
    scenarios = build_scenario_map()

    for key, folder in scenarios.items():
        person, scenario = key.split("/")
        images = get_images(folder)

        if not images:
            print(f"\n[{key}] Kosong — lewati.")
            continue

        print(f"\n[{key}] {len(images)} gambar")

        for img_path in tqdm(images, desc=f"  {key}", unit="img"):
            rows = process_image(img_path, DETECTORS, person, scenario,
                                 save_comparison=True)
            all_rows.extend(rows)

    # ── Simpan CSV ────────────────────────────
    if not all_rows:
        print("\n[INFO] Belum ada gambar. Isi folder dataset/ terlebih dahulu.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_ROOT / "detection_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV disimpan: {csv_path}")

    # ── Ringkasan per orang & skenario ───────
    print("\n── RINGKASAN PER ORANG & SKENARIO ──────────────")
    summary = df.groupby(["person", "scenario", "detector"]).agg(
        jumlah_foto  =("image",      "count"),
        rata_wajah   =("n_faces",    "mean"),
        rata_waktu_ms=("elapsed_ms", "mean"),
    ).round(2)
    print(summary.to_string())

    # ── Total per orang ───────────────────────
    print("\n── TOTAL PER ORANG ──────────────────────────────")
    per_person = df.groupby(["person", "detector"]).agg(
        total_foto =("image",      "count"),
        total_wajah=("n_faces",    "sum"),
        avg_time_ms=("elapsed_ms", "mean"),
    ).round(2)
    print(per_person.to_string())

    # ── Chart ─────────────────────────────────
    plot_summary(df, save_path=str(RESULTS_ROOT / "summary_chart.png"))


if __name__ == "__main__":
    run()
