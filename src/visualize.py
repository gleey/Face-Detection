"""
Utilitas visualisasi hasil deteksi wajah
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# Warna per detektor (BGR untuk OpenCV)
DETECTOR_COLORS = {
    "Haar Cascade":  (0,   200, 0),    # hijau
    "MTCNN":         (255, 120, 0),    # biru-muda
    "RetinaFace":    (0,   60,  255),  # merah
}

DETECTOR_COLORS_RGB = {
    "Haar Cascade":  (0,   200, 0),
    "MTCNN":         (0,   180, 255),
    "RetinaFace":    (255, 60,  0),
}


def draw_faces(image_bgr, faces, detector_name, show_conf=True):
    """Gambar bounding box di atas image, return image baru."""
    out = image_bgr.copy()
    color = DETECTOR_COLORS.get(detector_name, (0, 255, 0))

    for f in faces:
        x, y, w, h = f["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label = detector_name
        if show_conf and f.get("confidence") is not None:
            label += f" {f['confidence']:.2f}"

        # Background teks
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Landmark MTCNN
        if "keypoints" in f:
            for pt_name, pt in f["keypoints"].items():
                cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        # Landmark RetinaFace
        if "landmarks" in f:
            for pt_name, pt in f["landmarks"].items():
                cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

    return out


def compare_detectors(image_bgr, results_dict, save_path=None):
    """
    Tampilkan perbandingan hasil 3 detektor dalam 1 figure.

    results_dict = {
        "Haar Cascade": (faces, elapsed_ms),
        "MTCNN":        (faces, elapsed_ms),
        "RetinaFace":   (faces, elapsed_ms),
    }
    """
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (det_name, (faces, elapsed_ms)) in zip(axes, results_dict.items()):
        annotated = draw_faces(image_bgr, faces, det_name)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        ax.set_title(
            f"{det_name}\n{len(faces)} wajah | {elapsed_ms:.1f} ms",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Tersimpan: {save_path}")
    plt.show()
    return fig


def plot_summary(df_results, save_path=None):
    """
    3 chart:
      (1) Rata-rata wajah per skenario × detektor
      (2) Rata-rata waktu (ms) per skenario × detektor
      (3) Total foto per orang (kontribusi dataset)
    """
    has_person = "person" in df_results.columns

    n_rows = 2 if not has_person else 3
    fig, axes = plt.subplots(1, n_rows, figsize=(7 * n_rows, 5))

    # ── Chart 1: wajah per skenario ──────────
    pivot_faces = df_results.pivot_table(
        index="scenario", columns="detector", values="n_faces", aggfunc="mean"
    )
    pivot_faces.plot(kind="bar", ax=axes[0], rot=30, colormap="tab10")
    axes[0].set_title("Rata-rata Wajah Terdeteksi\nper Skenario", fontweight="bold")
    axes[0].set_ylabel("Jumlah Wajah")
    axes[0].legend(title="Detektor", fontsize=8)
    axes[0].grid(axis="y", alpha=0.4)

    # ── Chart 2: waktu per skenario ──────────
    pivot_time = df_results.pivot_table(
        index="scenario", columns="detector", values="elapsed_ms", aggfunc="mean"
    )
    pivot_time.plot(kind="bar", ax=axes[1], rot=30, colormap="tab10")
    axes[1].set_title("Rata-rata Waktu Deteksi (ms)\nper Skenario", fontweight="bold")
    axes[1].set_ylabel("Waktu (ms)")
    axes[1].legend(title="Detektor", fontsize=8)
    axes[1].grid(axis="y", alpha=0.4)

    # ── Chart 3: kontribusi per orang ────────
    if has_person:
        pivot_person = df_results.drop_duplicates(
            subset=["image", "person", "scenario"]
        ).groupby(["person", "scenario"]).size().unstack(fill_value=0)

        pivot_person.plot(kind="bar", ax=axes[2], rot=0, colormap="Set2")
        axes[2].set_title("Jumlah Foto per Orang\nper Skenario", fontweight="bold")
        axes[2].set_ylabel("Jumlah Foto")
        axes[2].legend(title="Skenario", fontsize=7)
        axes[2].grid(axis="y", alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Chart tersimpan: {save_path}")
    plt.show()
    return fig
