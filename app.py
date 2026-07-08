"""
app.py — FastAPI Backend
Sistem Prediksi Keterlambatan Pembayaran Pelanggan ISP
menggunakan Algoritma Random Forest

Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

import pickle
import os
import json
import io
from datetime import datetime

# =============================================================================
# Inisialisasi App
# =============================================================================
app = FastAPI(
    title="ISP Payment Delay Prediction API",
    description="Prediksi Keterlambatan Pembayaran Pelanggan ISP dengan Random Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Direktori
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Konstanta
# =============================================================================
FEATURE_COLS = [
    "telat_bayar_30", "freq_keluhan_30", "masa_langganan",
    "avg_latency_30", "downtime_30", "loss_pct_30",
    "p95_jitter_30", "paket_speed"
]
TARGET_COL = "latepay_30d"

FEATURE_LABELS = {
    "telat_bayar_30": "Keterlambatan 30 Hari",
    "freq_keluhan_30": "Frekuensi Keluhan",
    "masa_langganan":  "Masa Langganan (hari)",
    "avg_latency_30":  "Avg Latency (ms)",
    "downtime_30":     "Downtime (jam)",
    "loss_pct_30":     "Loss Paket (%)",
    "p95_jitter_30":   "Jitter P95 (ms)",
    "paket_speed":     "Kecepatan Paket (Mbps)",
}

# State in-memory
_state: dict = {
    "data": None,
    "model": None,
    "metrics": None,
    "importances": None,
}

# =============================================================================
# Helpers
# =============================================================================
def _load_model():
    path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def _risk_level(p: float) -> str:
    if p >= 0.6: return "Tinggi"
    if p >= 0.3: return "Sedang"
    return "Rendah"

def _parse_csv(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception:
        return pd.read_csv(io.StringIO(content.decode("latin-1")))

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
def root():
    return {
        "system": "ISP Payment Delay Prediction System",
        "version": "1.0.0",
        "model_trained": os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")),
        "docs": "/docs"
    }


# ── 1. Upload Dataset ──────────────────────────────────────────────────────
@app.post("/api/upload", tags=["Data"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload CSV dataset pelanggan ISP.
    Label `latepay_30d` akan di-generate otomatis jika tidak ada
    (latepay_30d = 1 jika telat_bayar_30 > 0).
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File harus berformat CSV (.csv)")

    content = await file.read()
    df = _parse_csv(content)

    # Validasi kolom fitur
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Kolom berikut tidak ditemukan: {missing}")

    # Auto-generate label jika belum ada
    label_generated = TARGET_COL not in df.columns
    if label_generated:
        df[TARGET_COL] = (df["telat_bayar_30"] > 0).astype(int)

    # Simpan ke disk
    save_path = os.path.join(DATA_DIR, "uploaded_data.csv")
    df.to_csv(save_path, index=False)
    _state["data"] = df

    dist = df[TARGET_COL].value_counts().to_dict()
    return {
        "success": True,
        "message": f"Dataset berhasil diupload ({len(df)} baris)",
        "rows": len(df),
        "columns": list(df.columns),
        "label_auto_generated": label_generated,
        "class_distribution": {
            "tidak_telat (0)": int(dist.get(0, 0)),
            "telat (1)":       int(dist.get(1, 0)),
        },
        "preview": df.head(5).to_dict(orient="records"),
    }


# ── Helper: simpan pkl di background (tidak blocking) ──────────────────────
def _save_model_background(clf, metrics: dict, importances: list):
    """
    Dipanggil di background thread.
    Menyimpan model .pkl dan metrics.json ke disk.
    Tidak memblokir response API.
    """
    try:
        with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "wb") as f:
            pickle.dump(clf, f)
        full_metrics = {**metrics, "feature_importances": importances}
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
            json.dump(full_metrics, f, indent=2)
        print("[INFO] Model .pkl dan metrics.json berhasil disimpan ke disk.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan model ke disk: {e}")


# ── 2. Train Model ─────────────────────────────────────────────────────────
@app.post("/api/train", tags=["Model"])
async def train_model(
    n_estimators: int = Query(100, ge=10, le=500, description="Jumlah pohon"),
    max_depth:    int = Query(None, ge=1, le=50, description="Kedalaman maks (None=unlimited)"),
    random_state: int = Query(42),
):
    """
    Latih model Random Forest dengan split 80:20.
    - Model langsung tersedia di memory setelah training selesai.
    - File .pkl disimpan di background (tidak memblokir response).
    - Prediksi bisa langsung dilakukan tanpa menunggu .pkl selesai disimpan.
    """
    # Load data
    if _state["data"] is None:
        path = os.path.join(DATA_DIR, "uploaded_data.csv")
        if not os.path.exists(path):
            raise HTTPException(400, "Belum ada dataset. Silakan upload CSV terlebih dahulu.")
        _state["data"] = pd.read_csv(path)

    df = _state["data"].copy()

    # Pastikan label ada
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = (df["telat_bayar_30"] > 0).astype(int)

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET_COL]

    # Split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Training RandomForest
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth else None,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluasi
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "trained_at": datetime.now().isoformat(),
    }

    # Feature importance
    importances = [
        {
            "feature": feat,
            "label":   FEATURE_LABELS.get(feat, feat),
            "importance": round(float(imp), 4),
        }
        for feat, imp in sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]

    # ✅ Simpan ke memory SEKARANG — prediksi langsung bisa dilakukan
    _state["model"]       = clf
    _state["metrics"]     = metrics
    _state["importances"] = importances

    # ✅ Simpan .pkl ke disk di background thread (tidak blocking)
    t = threading.Thread(
        target=_save_model_background,
        args=(clf, metrics, importances),
        daemon=True
    )
    t.start()

    return {
        "success": True,
        "message": "Model berhasil dilatih. File .pkl sedang disimpan di background.",
        "metrics": metrics,
        "feature_importances": importances,
    }


# ── 3. Predict ─────────────────────────────────────────────────────────────
@app.post("/api/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(None)):
    """
    Prediksi keterlambatan. Gunakan file baru atau dataset yang sudah diupload.
    Hasil disimpan ke results/prediction_results.csv.
    """
    # Load model
    model = _state["model"] or _load_model()
    if model is None:
        raise HTTPException(400, "Model belum dilatih. Silakan train terlebih dahulu.")
    _state["model"] = model

    # Load data
    if file:
        content = await file.read()
        df = _parse_csv(content)
    elif _state["data"] is not None:
        df = _state["data"].copy()
    else:
        path = os.path.join(DATA_DIR, "uploaded_data.csv")
        if not os.path.exists(path):
            raise HTTPException(400, "Tidak ada data untuk diprediksi.")
        df = pd.read_csv(path)

    # Validasi fitur
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Kolom fitur tidak lengkap: {missing}")

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    proba  = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    df["proba_latepay"] = np.round(proba, 4)
    df["pred_label"]    = labels.astype(int)
    df["risk_level"]    = [_risk_level(p) for p in proba]

    # Simpan hasil
    result_path = os.path.join(RESULTS_DIR, "prediction_results.csv")
    df.to_csv(result_path, index=False)
    _state["data"] = df

    # Ringkasan
    high   = int((proba >= 0.6).sum())
    medium = int(((proba >= 0.3) & (proba < 0.6)).sum())
    low    = int((proba < 0.3).sum())

    # Top 10 risiko tertinggi
    top_cols = ["id_pelanggan", "nama_pelanggan", "proba_latepay",
                "pred_label", "risk_level", "telat_bayar_30", "freq_keluhan_30"]
    top_cols = [c for c in top_cols if c in df.columns]
    top10 = df.nlargest(10, "proba_latepay")[top_cols].to_dict(orient="records")

    return {
        "success": True,
        "message": f"Prediksi selesai untuk {len(df)} pelanggan",
        "summary": {
            "total_customers":   len(df),
            "high_risk":         high,
            "medium_risk":       medium,
            "low_risk":          low,
            "avg_probability":   round(float(proba.mean()), 4),
        },
        "top_risk_customers": top10,
    }


# ── 4. Model Info ──────────────────────────────────────────────────────────
@app.get("/api/model-info", tags=["Model"])
def model_info():
    """Mengembalikan metrics dan feature importance model terakhir."""
    data = _state["metrics"] and {**_state["metrics"],
                                   "feature_importances": _state["importances"]}
    if data is None:
        data = _load_metrics()
    if data is None:
        return {"trained": False, "message": "Model belum dilatih."}
    return {"trained": True, **data}


# ── 5. Results ─────────────────────────────────────────────────────────────
@app.get("/api/results", tags=["Prediction"])
def get_results():
    """Mengambil hasil prediksi terakhir."""
    path = os.path.join(RESULTS_DIR, "prediction_results.csv")
    if not os.path.exists(path):
        return {"available": False, "message": "Belum ada hasil prediksi."}

    df    = pd.read_csv(path)
    proba = df["proba_latepay"] if "proba_latepay" in df.columns else pd.Series([0]*len(df))

    top_cols = ["id_pelanggan", "nama_pelanggan", "proba_latepay",
                "pred_label", "risk_level", "telat_bayar_30", "freq_keluhan_30"]
    top_cols = [c for c in top_cols if c in df.columns]

    return {
        "available": True,
        "total_customers": len(df),
        "summary": {
            "high_risk":       int((proba >= 0.6).sum()),
            "medium_risk":     int(((proba >= 0.3) & (proba < 0.6)).sum()),
            "low_risk":        int((proba < 0.3).sum()),
            "avg_probability": round(float(proba.mean()), 4),
        },
        "top_risk": df.nlargest(10, "proba_latepay")[top_cols].to_dict(orient="records"),
        "all_data": df[top_cols + ["proba_latepay"]].to_dict(orient="records")
                   if "proba_latepay" not in top_cols else
                   df[top_cols].to_dict(orient="records"),
    }


# ── 6. Download Results ────────────────────────────────────────────────────
@app.get("/api/download-results", tags=["Prediction"])
def download_results():
    """Download hasil prediksi sebagai CSV."""
    path = os.path.join(RESULTS_DIR, "prediction_results.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "Belum ada hasil prediksi.")
    return FileResponse(path, media_type="text/csv",
                        filename="prediction_results.csv")


# ── 7. Download Model PKL ──────────────────────────────────────────────────
@app.get("/api/download-model", tags=["Model"])
def download_model():
    """
    Download file model Random Forest (.pkl) yang sudah dilatih.
    File dapat dibuka dengan: pickle.load(open('rf_model.pkl','rb'))
    """
    path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if not os.path.exists(path):
        raise HTTPException(404, "Model belum dilatih. Silakan train terlebih dahulu.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="rf_model.pkl",
        headers={"Content-Disposition": "attachment; filename=rf_model.pkl"}
    )


# ── 8. Download Metrics JSON ───────────────────────────────────────────────
@app.get("/api/download-metrics", tags=["Model"])
def download_metrics():
    """Download file metrics evaluasi model (.json)."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Metrics belum tersedia. Silakan train terlebih dahulu.")
    return FileResponse(
        path,
        media_type="application/json",
        filename="rf_metrics.json"
    )



# ── Static Dashboard ───────────────────────────────────────────────────────
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True),
              name="dashboard")

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
