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
│   │   ├── frontal/
│   │   ├── side_pose/ 
│   │   ├── low_light/
│   │   ├── occlusion/
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
Ganti sesuai nama asli anggota:
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

> Ganti `orang1, orang2, orang3` sesuai nama folder di dataset.


## Mengganti Nama Dataset Root
Jika folder dataset berada di lokasi lain, ubah satu baris di `run_pipeline.py`:
```python
DATASET_ROOT = Path("dataset")           # default
DATASET_ROOT = Path("../data/kelompok")  # contoh path lain
```
