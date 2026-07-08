"""
generate_sample_data.py
Script untuk generate contoh dataset pelanggan ISP
Run: python generate_sample_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 150

nama_depan = ["Budi","Siti","Andi","Dewi","Rudi","Yuni","Heri","Lina","Agus","Rina",
              "Dian","Yoga","Nina","Bayu","Tari","Hendra","Sari","Eko","Wati","Joko",
              "Fitri","Tono","Mega","Doni","Ayu","Wahyu","Citra","Rizky","Maya","Iwan"]
nama_belakang = ["Santoso","Wijaya","Susilo","Pratama","Hidayat","Rahayu","Putra",
                 "Sari","Wibowo","Kurniawan","Saputra","Handoko","Nugroho","Lestari",
                 "Setiawan","Prabowo","Hakim","Utama","Maulana","Surya"]

names = [f"{np.random.choice(nama_depan)} {np.random.choice(nama_belakang)}" for _ in range(N)]
ids = [f"P{str(i+1).zfill(4)}" for i in range(N)]

# Pelanggan berisiko tinggi (40%) vs rendah (60%)
is_at_risk = np.random.choice([0, 1], size=N, p=[0.60, 0.40])

telat_bayar_30 = np.where(is_at_risk == 1,
    np.random.randint(1, 6, size=N),
    np.zeros(N, dtype=int))

freq_keluhan_30 = np.where(is_at_risk == 1,
    np.random.randint(2, 11, size=N),
    np.random.randint(0, 4, size=N))

masa_langganan = np.random.randint(30, 1826, size=N)  # 1 bulan - 5 tahun

avg_latency_30 = np.where(is_at_risk == 1,
    np.round(np.random.uniform(80, 200, size=N), 1),
    np.round(np.random.uniform(10, 80, size=N), 1))

downtime_30 = np.where(is_at_risk == 1,
    np.round(np.random.uniform(4, 24, size=N), 2),
    np.round(np.random.uniform(0, 4, size=N), 2))

loss_pct_30 = np.where(is_at_risk == 1,
    np.round(np.random.uniform(5, 30, size=N), 2),
    np.round(np.random.uniform(0, 5, size=N), 2))

p95_jitter_30 = np.where(is_at_risk == 1,
    np.round(np.random.uniform(30, 80, size=N), 1),
    np.round(np.random.uniform(5, 30, size=N), 1))

paket_speed = np.random.choice([10, 20, 50, 100], size=N, p=[0.3, 0.3, 0.3, 0.1])

latepay_30d = (telat_bayar_30 > 0).astype(int)

df = pd.DataFrame({
    "id_pelanggan": ids,
    "nama_pelanggan": names,
    "telat_bayar_30": telat_bayar_30,
    "freq_keluhan_30": freq_keluhan_30,
    "masa_langganan": masa_langganan,
    "avg_latency_30": avg_latency_30,
    "downtime_30": downtime_30,
    "loss_pct_30": loss_pct_30,
    "p95_jitter_30": p95_jitter_30,
    "paket_speed": paket_speed,
    "latepay_30d": latepay_30d
})

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "sample_data.csv")
df.to_csv(out_path, index=False)
print(f"[OK] Dataset berhasil dibuat: {out_path}")
print(f"     Total baris   : {N}")
print(f"     Pelanggan telat: {latepay_30d.sum()} ({latepay_30d.mean()*100:.1f}%)")
print(f"     Pelanggan tepat: {(latepay_30d==0).sum()}")
