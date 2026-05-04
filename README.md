# Face Detection Project
**Tugas Visi Komputer — 3 Detector Comparison**

## Detektor
| # | Detector | Metode | Kecepatan | Akurasi |
|---|----------|--------|-----------|---------|
| 1 | Haar Cascade | HOG + sliding window | ⚡⚡⚡ Cepat | ★★☆ Dasar |
| 2 | MTCNN | Multi-task CNN | ⚡⚡ Sedang | ★★★ Baik |
| 3 | RetinaFace | ResNet backbone | ⚡ Lebih lambat | ★★★★ Terbaik |

## Struktur Folder
```
face_detection_project/
├── dataset/
│   ├── orang1/
│   │   ├── frontal/          ← Scenario A
│   │   ├── side_pose/        ← Scenario B
│   │   ├── multiple_faces/   ← Scenario C
│   │   ├── low_light/        ← Scenario D
│   │   ├── occlusion/        ← Scenario E
│   │   ├── small_face/       ← Scenario F
│   │   └── webcam_live/      ← Scenario G
│   ├── orang2/
│   │   └── (sama seperti orang1)
│   └── orang3/
│       └── (sama seperti orang1)
│
├── results/
│   ├── haar_cascade/
│   │   ├── orang1/frontal/   ← hasil anotasi per orang
│   │   ├── orang2/side_pose/
│   │   └── ...
│   ├── mtcnn/
│   ├── retinaface/
│   ├── comparison/           ← grid 3 detektor berdampingan
│   ├── detection_results.csv ← semua hasil dalam 1 file
│   └── summary_chart.png     ← 3 chart ringkasan
│
├── src/
│   ├── detectors.py          ← kelas 3 detektor
│   └── visualize.py          ← visualisasi & chart
├── run_pipeline.py           ← pipeline utama (batch)
├── webcam_live.py            ← Scenario G: real-time webcam
└── requirements.txt
```

## Instalasi
```bash
pip install -r requirements.txt
```

## Cara Pakai

### 1. Ganti Nama Folder Anggota (Opsional)
Di `run_pipeline.py` baris 22, ganti sesuai nama asli anggota:
```python
PEOPLE = ["budi", "siti", "andi"]   # ← sesuaikan
```
Nama ini harus sama persis dengan nama subfolder di `dataset/`.

### 2. Isi Dataset
Setiap anggota mengisi folder miliknya sendiri:
```
dataset/orang1/frontal/     → foto wajah lurus
dataset/orang1/side_pose/   → kepala miring kiri/kanan
dataset/orang1/low_light/   → kondisi cahaya redup
dataset/orang1/occlusion/   → wajah tertutup masker/tangan
```

### 3. Webcam Real-Time
```bash
python webcam_live.py
```
| Tombol | Fungsi |
|--------|--------|
| `1` | Ganti ke Haar Cascade |
| `2` | Ganti ke MTCNN |
| `3` | Ganti ke RetinaFace |
| `S` | Screenshot → simpan ke `dataset/orang1/webcam_live/` |
| `Q` / `ESC` | Keluar |

> Ganti `orang1` di `webcam_live.py` sesuai siapa yang sedang rekam.

## Format CSV Output
| Kolom | Keterangan |
|-------|-----------|
| `image` | Nama file |
| `person` | Nama anggota (`orang1`, `orang2`, `orang3`) |
| `scenario` | Skenario (`frontal`, `side_pose`, dst.) |
| `detector` | Nama detektor |
| `n_faces` | Jumlah wajah terdeteksi |
| `elapsed_ms` | Waktu deteksi (ms) |
| `faces_json` | Koordinat bbox & confidence |

## Mengganti Nama Dataset Root
Jika folder dataset berada di lokasi lain, ubah satu baris di `run_pipeline.py`:
```python
DATASET_ROOT = Path("dataset")           # default
DATASET_ROOT = Path("../data/kelompok")  # contoh path lain
```
