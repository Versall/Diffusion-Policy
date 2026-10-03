"""
EDA — Visualisasi dataset Push-T (pusht_cchi_v7_replay.zarr).
Menghasilkan 3 figur bersih untuk bagian Data/Methodology skripsi.

  Fig 1 — Ringkasan dataset (statistik + distribusi panjang episode)
  Fig 2 — Sampel trajectory 3 episode (agent vs block)
  Fig 3 — Contoh frame observasi dari dataset
"""
from __future__ import annotations
import os, numpy as np, zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_AGENT = "#2563EB"   # biru
C_BLOCK = "#F59E0B"   # amber
C_ACCENT = "#10B981"  # hijau
C_DARK = "#1E293B"

DATAROOT = os.path.join(os.path.dirname(__file__), "pusht", "pusht_cchi_v7_replay.zarr")
OUTDIR   = os.path.join(os.path.dirname(__file__), "viz_output", "EDA")
os.makedirs(OUTDIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
root  = zarr.open(DATAROOT, mode="r")
state  = root["data/state"][:]      # (N, 5)
action = root["data/action"][:]     # (N, 2)
imgs   = root["data/img"]           # lazy (N, 96, 96, 3)
ends   = root["meta/episode_ends"][:]  # (206,)
N_TOTAL, N_EPS = len(state), len(ends)

ep_lens = np.array([int(ends[i]) - (int(ends[i-1]) if i > 0 else 0)
                     for i in range(N_EPS)])

def ep_slice(k):
    s = int(ends[k-1]) if k > 0 else 0
    return s, int(ends[k])


# ══════════════════════════════════════════════════════════════════════
# FIG 1 — Ringkasan dataset
# ══════════════════════════════════════════════════════════════════════
fig, (ax_txt, ax_hist) = plt.subplots(1, 2, figsize=(12, 4),
                                       gridspec_kw={"width_ratios": [1, 1.6]})

# Panel kiri: ringkasan teks
ax_txt.axis("off")
stats = (
    f"Dataset : pusht_cchi_v7_replay.zarr\n"
    f"Episodes: {N_EPS}\n"
    f"Frames  : {N_TOTAL:,}\n"
    f"Episode length\n"
    f"  min   : {ep_lens.min()}\n"
    f"  max   : {ep_lens.max()}\n"
    f"  mean  : {ep_lens.mean():.1f}\n"
    f"  median: {int(np.median(ep_lens))}\n"
    f"\n"
    f"Per frame:\n"
    f"  action  : {tuple(action.shape[1:])}  float32\n"
    f"  state   : {tuple(state.shape[1:])}  float32\n"
    f"  img     : (96,96,3)  float32\n"
    f"  keypoint: (9,2)      float32\n"
    f"  action range: [{action.min():.1f}, {action.max():.1f}] px\n"
    f"  state  range: [{state.min():.1f}, {state.max():.1f}] px"
)
ax_txt.text(0.05, 0.95, stats, transform=ax_txt.transAxes,
            fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

# Panel kanan: distribusi panjang episode
ax_hist.hist(ep_lens, bins=25, color=C_AGENT, edgecolor="white", linewidth=0.5, alpha=0.85)
ax_hist.axvline(np.median(ep_lens), color=C_DARK, ls="--", lw=1, label=f"median={int(np.median(ep_lens))}")
ax_hist.set_xlabel("Episode length (frame)")
ax_hist.set_ylabel("Count")
ax_hist.set_title("Distribusi panjang episode")
ax_hist.legend(frameon=False)

fig.suptitle("Ringkasan Dataset Push-T", fontsize=13, fontweight="bold", y=1.02)
p1 = os.path.join(OUTDIR, "fig1_dataset_overview.png")
fig.savefig(p1); plt.close(fig)
print(f"[OK] {p1}")


# ══════════════════════════════════════════════════════════════════════
# FIG 2 — Sampel trajectory (3 episode: pendek, sedang, panjang)
# ══════════════════════════════════════════════════════════════════════
sorted_idx = np.argsort(ep_lens)
picks = [sorted_idx[0], sorted_idx[N_EPS // 2], sorted_idx[-1]]
ep_labels = ["pendek", "sedang", "panjang"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

for i, (ep_i, label) in enumerate(zip(picks, ep_labels)):
    s, e = ep_slice(ep_i)
    ag = state[s:e, :2]
    bl = state[s:e, 2:4]
    ax = axes[i]
    ax.plot(ag[:, 0], ag[:, 1], color=C_AGENT, lw=1.2, alpha=0.85, label="Agent")
    ax.plot(bl[:, 0], bl[:, 1], color=C_BLOCK, lw=1.2, alpha=0.85, label="Block")
    ax.scatter(*ag[0],  color=C_AGENT, s=25, zorder=5, marker="o")
    ax.scatter(*ag[-1], color=C_AGENT, s=35, zorder=5, marker="x")
    ax.scatter(*bl[0],  color=C_BLOCK, s=25, zorder=5, marker="o")
    ax.scatter(*bl[-1], color=C_BLOCK, s=35, zorder=5, marker="x")
    ax.set_title(f"Ep {ep_i} — {label} ({e-s} frame)")
    ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect("equal"); ax.invert_yaxis()
    if i == 0:
        ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel("x (px)")
    if i == 0:
        ax.set_ylabel("y (px)")

fig.suptitle("Contoh Trajectory: Agent (biru) vs Block (amber)", fontsize=12, fontweight="bold")
fig.tight_layout()
p2 = os.path.join(OUTDIR, "fig2_sample_trajectories.png")
fig.savefig(p2); plt.close(fig)
print(f"[OK] {p2}")


# ══════════════════════════════════════════════════════════════════════
# FIG 3 — Contoh frame observasi (4 frame tersebar dari 1 episode)
# ══════════════════════════════════════════════════════════════════════
# Pakai episode panjang
long_ep = sorted_idx[-1]
s, e = ep_slice(long_ep)
n_f = e - s
sample_idx = np.linspace(0, n_f - 1, 4, dtype=int)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, fi in zip(axes, sample_idx):
    img = imgs[s + fi].astype(np.uint8)
    ax.imshow(img)
    ax.set_title(f"t = {fi}", fontsize=10)
    ax.axis("off")

fig.suptitle(f"Observasi RGB (96×96) — Episode {long_ep}", fontsize=12, fontweight="bold")
fig.tight_layout()
p3 = os.path.join(OUTDIR, "fig3_observation_samples.png")
fig.savefig(p3); plt.close(fig)
print(f"[OK] {p3}")

print("\n=== EDA selesai ===")
