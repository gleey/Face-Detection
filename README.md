# Face Detection Project
**Tugas Visi Komputer — 3 Detector Comparison & Face Recognition**

Proyek ini mengimplementasikan perbandingan 3 metode face detection berbeda (Haar Cascade, MTCNN, RetinaFace) dan face recognition dengan dataset multi-pose, multi-lighting condition.

## 📊 Perbandingan Detektor
| # | Detector | Metode | Kecepatan | Akurasi |
|---|----------|--------|-----------|---------|
| 1 | Haar Cascade | HOG + sliding window | ⚡⚡⚡ Cepat | ★★☆ Dasar |
| 2 | MTCNN | Multi-task CNN | ⚡⚡ Sedang | ★★★ Baik |
| 3 | RetinaFace | ResNet backbone | ⚡ Lebih lambat | ★★★★ Terbaik |

## 📁 Struktur Proyek
```
face_detection_project/
│
├── dataset/                          # Dataset training & testing
│   ├── orang1/
│   │   ├── frontal/                  # Foto wajah langsung
│   │   ├── low_light/                # Kondisi cahaya redup
│   │   ├── occlusion/                # Wajah tertutup (masker, tangan)
│   │   └── side_pose/                # Kepala miring kiri/kanan
│   ├── orang2/
│   └── orang3/      
│
├── models/                           # Model training & weights
│
├── results/                          # Output & hasil analisis
│   └── recognition/                  # Hasil face recognition
│   └── comparison/                   # Hasil Compare Antar Model Tiap Skenario
│
├── src/                              # Source code utilities
│   ├── detectors.py                  # Kelas implementasi 3 detektor
│   └── visualize.py                  # Fungsi visualisasi & chart
│
├── livecam_recognition.py            # Real-time recognition dari kamera
├── run_pipeline.py                   # Pipeline batch processing
├── train.py                          # Training face recognition model
├── webcam_live.py                    # Real-time detection testing
├── requirements.txt                  # Dependencies
└── README.md                         # Dokumentasi ini
```

## 📋 Deskripsi File

| File | Deskripsi |
|------|-----------|
| `run_pipeline.py` | Pipeline batch utama untuk menjalankan semua detector pada dataset |
| `train.py` | Script training model face recognition |
| `webcam_live.py` | Real-time face detection dari webcam dengan switch detector |
| `livecam_recognition.py` | Real-time face recognition dengan model yang sudah trained |
| `src/detectors.py` | Implementasi kelas Haar Cascade, MTCNN, RetinaFace |
| `src/visualize.py` | Visualisasi detection results dan chart ringkasan |

## 🚀 Quick Start

### 1. Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset
Tambahkan foto ke folder dataset sesuai struktur:
```
dataset/[nama_anggota]/[pose]/image.jpg
```

Contoh:
```
dataset/orang1/frontal/photo1.jpg
dataset/orang1/side_pose/photo2.jpg
dataset/orang/low_light/photo3.jpg
```

### 3. Jalankan Pipeline Batch (Tidak wajib jika hanya ingin livecam)
```bash
python run_pipeline.py
```
Output akan tersimpan di folder `results/recognition/`

### 4. Real-Time Webcam Detection
```bash
python webcam_live.py
```

**Kontrol:**
| Tombol | Fungsi |
|--------|--------|
| `1` | Pilih Haar Cascade |
| `2` | Pilih MTCNN |
| `3` | Pilih RetinaFace |
| `S` | Screenshot (tersimpan ke dataset) |
| `Q` / `ESC` | Keluar |

### 5. Real-Time Webcam Recognition
```bash
python livecam_recognition.py
```

### 6. Train Recognition Model
```bash
python train.py
```

## 📊 Output & Hasil

Setelah menjalankan `run_pipeline.py`, hasil akan disimpan di:
- `results/recognition/` - Hasil detection dan recognition per detector
- CSV files - Data detection metrics
- Chart files - Visualisasi perbandingan antar detector

## ⚙️ Konfigurasi

### Mengubah Dataset Root
Jika dataset berada di lokasi lain, ubah di file pipeline:
```python
DATASET_ROOT = Path("dataset")           # default
DATASET_ROOT = Path("../data/kelompok")  # custom path
```

### Menambah/Mengubah Nama Anggota
Nama folder di `dataset/` akan otomatis dideteksi sebagai list anggota.
Atau set manual di config script masing-masing.

## 📝 Catatan

- Dataset minimal 10-15 foto per orang per pose untuk hasil optimal
- Gunakan RetinaFace untuk akurasi terbaik (lebih lambat)
- Gunakan Haar Cascade untuk processing cepat (akurasi rendah)
- MTCNN adalah middle ground (kecepatan & akurasi baik)
- Pastikan requirements.txt terinstall dengan benar sebelum menjalankan script
