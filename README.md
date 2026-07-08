# RF ISP Payment Delay Prediction System

## Deskripsi
Sistem prediksi keterlambatan pembayaran pelanggan ISP menggunakan **Algoritma Random Forest** (Supervised Learning).
Dibangun dengan **FastAPI** (backend) + **Vanilla JS** (dashboard web).

## Struktur Folder
```
rf_system/
├── app.py                    ← FastAPI backend
├── requirements.txt          ← Python dependencies
├── generate_sample_data.py   ← Script generate dataset
├── data/
│   └── sample_data.csv       ← Dataset contoh
├── models/                   ← Model .pkl (auto-dibuat)
├── results/                  ← Hasil prediksi CSV (auto-dibuat)
└── dashboard/
    ├── index.html
    ├── css/style.css
    └── js/app.js
```

## Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python generate_sample_data.py
```

### 3. Jalankan Server
```bash
uvicorn app:app --reload --port 8000
```

### 4. Buka Dashboard
Buka browser → http://localhost:8000/dashboard/

## Alur Kerja Sistem

1. **Upload CSV** → `/api/upload`
   - Label `latepay_30d` di-generate otomatis jika tidak ada
   - Rule: `latepay_30d = 1` jika `telat_bayar_30 > 0`

2. **Train Model** → `POST /api/train`
   - Split 80% train / 20% test
   - RandomForestClassifier (balanced class_weight)
   - Model disimpan ke `models/rf_model.pkl`
   - Evaluasi: Accuracy, Precision, Recall, F1-Score

3. **Prediksi** → `POST /api/predict`
   - Output: `proba_latepay`, `pred_label`, `risk_level`
   - Hasil disimpan ke `results/prediction_results.csv`

## Fitur API
| Endpoint | Method | Deskripsi |
|---|---|---|
| `/api/upload` | POST | Upload CSV dataset |
| `/api/train` | POST | Latih model RF |
| `/api/predict` | POST | Prediksi keterlambatan |
| `/api/model-info` | GET | Metrics & feature importance |
| `/api/results` | GET | Hasil prediksi terakhir |
| `/api/download-results` | GET | Download CSV hasil |
| `/docs` | GET | API documentation (Swagger) |

## Fitur Dataset
| Kolom | Deskripsi |
|---|---|
| `telat_bayar_30` | Jumlah keterlambatan 30 hari terakhir |
| `freq_keluhan_30` | Jumlah keluhan |
| `masa_langganan` | Masa berlangganan (hari) |
| `avg_latency_30` | Rata-rata latency (ms) |
| `downtime_30` | Downtime (jam) |
| `loss_pct_30` | Loss paket (%) |
| `p95_jitter_30` | Jitter P95 (ms) |
| `paket_speed` | Kecepatan paket (Mbps) |
| `latepay_30d` | **Target**: 0=tepat waktu, 1=telat |
