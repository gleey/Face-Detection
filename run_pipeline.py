"""
Pipeline Utama — Evaluasi 3 Detektor pada Dataset
Struktur: dataset/orang_X/scenario/
Jalankan: python run_pipeline.py

Output:
  results/comparison/<person>/<scenario>_comparison.png  ← satu grid per skenario
  results/summary_chart.png                              ← chart ringkasan
"""

import cv2
import pandas as pd
import json
import numpy as np
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detectors import HaarDetector, MTCNNDetector, RetinaFaceDetector
from visualize import draw_faces, plot_summary


# ─────────────────────────────────────────────
# Custom JSON encoder untuk numpy types
# ─────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_ROOT = Path("dataset")
RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)

PEOPLE = ["gley", "gervik", "dino"]

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

MAX_WORKERS = 5


# ─────────────────────────────────────────────
# THREAD-LOCAL DETECTORS
# ─────────────────────────────────────────────
_thread_local = threading.local()

def get_thread_detectors():
    if not hasattr(_thread_local, "detectors"):
        _thread_local.detectors = {
            "Haar Cascade": HaarDetector(),
            "MTCNN":        MTCNNDetector(),
            "RetinaFace":   RetinaFaceDetector(),
        }
    return _thread_local.detectors


# ─────────────────────────────────────────────
# FUNGSI BANTU
# ─────────────────────────────────────────────
def get_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def build_scenario_map():
    scenarios = {}
    for person in PEOPLE:
        for scenario in SCENARIO_NAMES:
            scenarios[f"{person}/{scenario}"] = DATASET_ROOT / person / scenario
    return scenarios


def sanitize_faces(faces: list) -> list:
    clean = []
    for f in faces:
        entry = {}
        for k, v in f.items():
            if isinstance(v, np.integer):    entry[k] = int(v)
            elif isinstance(v, np.floating): entry[k] = float(v)
            elif isinstance(v, np.ndarray):  entry[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                entry[k] = [int(x) if isinstance(x, np.integer) else
                             float(x) if isinstance(x, np.floating) else x for x in v]
            elif isinstance(v, dict):
                entry[k] = {pk: (int(pv[0]), int(pv[1]))
                             if isinstance(pv, (list, tuple, np.ndarray)) else pv
                             for pk, pv in v.items()}
            else:
                entry[k] = v
        clean.append(entry)
    return clean


# ─────────────────────────────────────────────
# PROSES SATU FOTO
# ─────────────────────────────────────────────
def process_image(image_path: Path, person: str, scenario: str):
    """
    Jalankan semua detektor pada satu foto.
    Hanya return data hasil deteksi — tidak menyimpan foto ke disk.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] Tidak bisa membaca: {image_path.name}")
        return [], {}

    detectors = get_thread_detectors()
    results_dict = {}
    row_list = []

    for det_name, detector in detectors.items():
        try:
            faces, elapsed_ms = detector.detect(image)
        except Exception as e:
            print(f"  [ERROR] {det_name} @ {image_path.name}: {e}")
            faces, elapsed_ms = [], 0.0

        results_dict[det_name] = (faces, elapsed_ms)
        row_list.append({
            "image":      image_path.name,
            "person":     person,
            "scenario":   scenario,
            "detector":   det_name,
            "n_faces":    len(faces),
            "elapsed_ms": round(elapsed_ms, 2),
        })

    return row_list, results_dict


# ─────────────────────────────────────────────
# COMPARISON GRID — 1 foto representatif
# ─────────────────────────────────────────────
def build_scenario_comparison(all_results, person, scenario, save_path):
    if not all_results:
        return

    det_names = list(get_thread_detectors().keys())
    rep_path, rep_dict = all_results[len(all_results) // 2]
    image_bgr = cv2.imread(str(rep_path))
    if image_bgr is None:
        return

    # Statistik rata-rata dari seluruh foto
    stats = {d: {"n_faces": [], "elapsed_ms": []} for d in det_names}
    for _, res_dict in all_results:
        for det_name, (faces, elapsed_ms) in res_dict.items():
            stats[det_name]["n_faces"].append(len(faces))
            stats[det_name]["elapsed_ms"].append(elapsed_ms)

    avg_faces = {d: np.mean(v["n_faces"])    for d, v in stats.items()}
    avg_time  = {d: np.mean(v["elapsed_ms"]) for d, v in stats.items()}

    n_det = len(det_names)
    fig, axes = plt.subplots(1, n_det, figsize=(n_det * 4.5, 5), squeeze=False)
    fig.suptitle(
        f"Perbandingan Detektor  —  {person.upper()} / {scenario}\n"
        f"Foto sampel: {rep_path.name}   (total {len(all_results)} foto di skenario ini)",
        fontsize=12, fontweight="bold",
    )

    THUMB_W, THUMB_H = 400, 320
    for col_idx, det_name in enumerate(det_names):
        ax = axes[0][col_idx]
        faces, _ = rep_dict.get(det_name, ([], 0))
        annotated = draw_faces(image_bgr, faces, det_name)
        thumb = cv2.resize(annotated, (THUMB_W, THUMB_H))
        ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(
            f"{det_name}\n"
            f"Wajah di foto ini : {len(faces)}\n"
            f"Avg wajah (semua) : {avg_faces[det_name]:.1f}\n"
            f"Avg waktu (semua) : {avg_time[det_name]:.0f} ms",
            fontsize=9, pad=8,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparison: {save_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print(" FACE DETECTION PIPELINE  (parallel mode)")
    print(" Haar Cascade | MTCNN | RetinaFace")
    print(f" Anggota : {', '.join(PEOPLE)}")
    print(f" Workers : {MAX_WORKERS} thread")
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
        scenario_results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(process_image, img_path, person, scenario): img_path
                for img_path in images
            }
            with tqdm(total=len(images), desc=f"  {key}", unit="img") as pbar:
                for future in as_completed(future_map):
                    try:
                        rows, res_dict = future.result()
                        all_rows.extend(rows)
                        if res_dict:
                            scenario_results.append((future_map[future], res_dict))
                    except Exception as e:
                        print(f"  [ERROR] {future_map[future].name}: {e}")
                    pbar.update(1)

        scenario_results.sort(key=lambda x: x[0].name)

        if scenario_results:
            comp_path = str(
                RESULTS_ROOT / "comparison" / person / f"{scenario}_comparison.png"
            )
            build_scenario_comparison(scenario_results, person, scenario, comp_path)

    # ── Ringkasan terminal ────────────────────────────────────────────────
    if not all_rows:
        print("\n[INFO] Belum ada gambar. Isi folder dataset/ terlebih dahulu.")
        return

    df = pd.DataFrame(all_rows)

    print("\n── RINGKASAN PER ORANG & SKENARIO ──────────────")
    summary = df.groupby(["person", "scenario", "detector"]).agg(
        jumlah_foto  =("image",      "count"),
        rata_wajah   =("n_faces",    "mean"),
        rata_waktu_ms=("elapsed_ms", "mean"),
    ).round(2)
    print(summary.to_string())

    print("\n── TOTAL PER ORANG ──────────────────────────────")
    per_person = df.groupby(["person", "detector"]).agg(
        total_foto =("image",      "count"),
        total_wajah=("n_faces",    "sum"),
        avg_time_ms=("elapsed_ms", "mean"),
    ).round(2)
    print(per_person.to_string())

    plot_summary(df, save_path=str(RESULTS_ROOT / "summary_chart.png"))
    print("\n✅ Pipeline selesai!")


if __name__ == "__main__":
    run()