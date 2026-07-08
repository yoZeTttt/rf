import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.tree import plot_tree
import os

os.makedirs("results", exist_ok=True)

# Load model
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = list(model.feature_names_in_)
class_names = ["Tepat Waktu", "Terlambat"]

# ─────────────────────────────────────────
# 1. Feature Importance
# ─────────────────────────────────────────
importances = model.feature_importances_
indices = np.argsort(importances)  # ascending, untuk horizontal bar

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_names)))

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor("#0f172a")
ax1.set_facecolor("#1e293b")

bars = ax1.barh(
    [feature_names[i] for i in indices],
    importances[indices],
    color=[colors[i] for i in range(len(indices))],
    edgecolor="none",
    height=0.6,
)

# Value labels
for bar, val in zip(bars, importances[indices]):
    ax1.text(
        val + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left",
        color="white", fontsize=9, fontweight="bold"
    )

ax1.set_title("Feature Importance — Random Forest", color="white", fontsize=14, fontweight="bold", pad=14)
ax1.set_xlabel("Importance Score", color="#94a3b8", fontsize=11)
ax1.tick_params(colors="white", labelsize=10)
ax1.spines[:].set_visible(False)
ax1.xaxis.label.set_color("#94a3b8")
ax1.grid(axis="x", color="#334155", linestyle="--", alpha=0.5)

plt.tight_layout()
fig1.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
print("✅ Tersimpan: results/feature_importance.png")

# ─────────────────────────────────────────
# 2. Visualisasi Salah Satu Pohon (Tree ke-0)
# ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(22, 10))
fig2.patch.set_facecolor("#0f172a")
ax2.set_facecolor("#0f172a")

plot_tree(
    model.estimators_[0],
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=3,           # batasi depth agar tidak terlalu penuh
    fontsize=8,
    ax=ax2,
    impurity=True,
    proportion=False,
)

ax2.set_title(
    "Visualisasi Decision Tree #1 (depth max=3) — Random Forest",
    color="white", fontsize=14, fontweight="bold", pad=14
)
plt.tight_layout()
fig2.savefig("results/decision_tree.png", dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print("✅ Tersimpan: results/decision_tree.png")

# ─────────────────────────────────────────
# 3. Distribusi Depth semua pohon
# ─────────────────────────────────────────
depths = [est.get_depth() for est in model.estimators_]

fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.patch.set_facecolor("#0f172a")
ax3.set_facecolor("#1e293b")

ax3.hist(depths, bins=20, color="#06b6d4", edgecolor="#0f172a", alpha=0.85)
ax3.axvline(np.mean(depths), color="#f59e0b", linestyle="--", linewidth=2, label=f"Rata-rata: {np.mean(depths):.1f}")
ax3.set_title("Distribusi Kedalaman 100 Pohon", color="white", fontsize=13, fontweight="bold")
ax3.set_xlabel("Depth", color="#94a3b8")
ax3.set_ylabel("Jumlah Pohon", color="#94a3b8")
ax3.tick_params(colors="white")
ax3.spines[:].set_visible(False)
ax3.legend(facecolor="#334155", labelcolor="white", fontsize=10)
ax3.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)

plt.tight_layout()
fig3.savefig("results/tree_depth_distribution.png", dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
print("✅ Tersimpan: results/tree_depth_distribution.png")

print("\n🎉 Semua visualisasi selesai! Cek folder: results/")
