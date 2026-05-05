"""
train.py — Build / Update Face Embedding Database
===================================================
Memproses semua foto di dataset/ dan menyimpan hasil embedding
ArcFace ke file: models/face_db.pkl

Jalankan SEKALI setelah dataset diisi atau diperbarui:
    python train.py

Opsi:
    python train.py --force      # paksa rebuild dari nol (hapus .pkl lama)
    python train.py --person gley  # hanya update 1 orang

Cara kerja:
  1. Baca semua foto dari dataset/{orang}/**/*.jpg|png
  2. Ekstrak embedding ArcFace via DeepFace
  3. Simpan dict { "gley": [emb1, emb2, ...], ... } ke models/face_db.pkl
  4. livecam_recognition.py akan load .pkl ini — tidak perlu proses ulang
"""

import os
import sys
import io
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Tekan log TF sebelum import DeepFace ─────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("DEEPFACE_HOME", str(Path.home() / ".deepface"))

# ─── KONSTANTA ─────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(__file__).parent / "dataset"
MODELS_DIR   = Path(__file__).parent / "models"
DB_PATH      = MODELS_DIR / "face_db.pkl"
MODEL_NAME   = "ArcFace"
PEOPLE       = ["gley", "gervik", "dino"]
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── HELPER ────────────────────────────────────────────────────────────────────
class _SuppressStderr:
    def __enter__(self):
        self._orig = sys.stderr
        sys.stderr = io.StringIO()
        return self
    def __exit__(self, *_):
        sys.stderr = self._orig


def _get_images(person_dir: Path):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(person_dir.rglob(f"*{ext}"))
        imgs.extend(person_dir.rglob(f"*{ext.upper()}"))
    # deduplicate (case-insensitive filesystem bisa dobel)
    seen = set()
    result = []
    for p in imgs:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return sorted(result)


# ─── CORE ──────────────────────────────────────────────────────────────────────
def build_database(people: list, existing_db: dict = None, force: bool = False):
    """
    Buat / perbarui database embedding.

    Parameters
    ----------
    people       : list nama orang yang akan diproses
    existing_db  : dict yang sudah ada (untuk update incremental)
    force        : jika True, proses ulang semua foto meski embedding sudah ada

    Returns
    -------
    db : dict { person_name: [np.ndarray, ...] }
    """
    from deepface import DeepFace

    db = {} if (existing_db is None or force) else dict(existing_db)

    for person in people:
        person_dir = DATASET_DIR / person

        if not person_dir.exists():
            print(f"  [SKIP] Folder tidak ditemukan: {person_dir}")
            continue

        images = _get_images(person_dir)
        if not images:
            print(f"  [SKIP] {person} — tidak ada foto.")
            continue

        # Jika update incremental: cek berapa embedding yang sudah ada
        existing_count = len(db.get(person, []))
        if not force and existing_count > 0:
            print(f"  [SKIP] {person} sudah punya {existing_count} embedding "
                  f"(gunakan --force untuk rebuild)")
            continue

        print(f"\n  ▶  Memproses {person} ({len(images)} foto) …")
        embeddings = []
        ok = fail = 0

        for img_path in images:
            try:
                with _SuppressStderr():
                    result = DeepFace.represent(
                        img_path          = str(img_path),
                        model_name        = MODEL_NAME,
                        enforce_detection = False,
                        detector_backend  = "opencv",
                    )
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                embeddings.append(emb)
                ok += 1
                # Progress sederhana setiap 5 foto
                if ok % 5 == 0:
                    print(f"     {ok}/{len(images)} foto …")
            except Exception as e:
                print(f"     [!] Gagal: {img_path.name} — {e}")
                fail += 1

        if embeddings:
            db[person] = embeddings
            print(f"  ✓  {person}: {ok} embedding berhasil, {fail} gagal")
        else:
            print(f"  ✗  {person}: tidak ada embedding yang berhasil!")

    return db


def save_db(db: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "db":         db,
        "model":      MODEL_NAME,
        "people":     list(db.keys()),
        "total_emb":  sum(len(v) for v in db.values()),
        "built_at":   datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size / 1024
    print(f"\n  💾  Disimpan ke: {path}  ({size_kb:.1f} KB)")


def load_db(path: Path):
    """Load .pkl, return (db_dict, metadata)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    db   = payload["db"]
    meta = {k: v for k, v in payload.items() if k != "db"}
    return db, meta


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build face embedding database → .pkl")
    parser.add_argument(
        "--force", action="store_true",
        help="Paksa rebuild dari nol (hapus data lama)"
    )
    parser.add_argument(
        "--person", type=str, default=None,
        help="Hanya proses 1 orang, misal: --person gley"
    )
    args = parser.parse_args()

    target_people = [args.person] if args.person else PEOPLE

    print("=" * 55)
    print("  FACE EMBEDDING TRAINER")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Output  : {DB_PATH}")
    print(f"  Target  : {', '.join(target_people)}")
    print(f"  Mode    : {'FORCE REBUILD' if args.force else 'incremental'}")
    print("=" * 55)

    # Load existing DB jika ada (untuk incremental update)
    existing_db = None
    if DB_PATH.exists() and not args.force:
        try:
            existing_db, meta = load_db(DB_PATH)
            print(f"\n[INFO] .pkl ditemukan — {meta['total_emb']} embedding "
                  f"dari {meta['built_at']}")
            print("[INFO] Mode incremental: hanya orang baru yang diproses.\n")
        except Exception as e:
            print(f"[WARN] Gagal baca .pkl lama ({e}), rebuild dari nol.")
            existing_db = None

    # Build / update
    db = build_database(target_people, existing_db=existing_db, force=args.force)

    if not db:
        print("\n[ERROR] Database kosong. Isi folder dataset/ terlebih dahulu.")
        sys.exit(1)

    # Simpan
    save_db(db, DB_PATH)

    # Ringkasan
    print("\n── RINGKASAN ─────────────────────────────────────")
    for person, embs in db.items():
        print(f"  {person:<12} : {len(embs):>3} embedding")
    print(f"  {'TOTAL':<12} : {sum(len(v) for v in db.values()):>3} embedding")
    print("\n✅  Training selesai! Jalankan livecam_recognition.py.")


if __name__ == "__main__":
    main()
