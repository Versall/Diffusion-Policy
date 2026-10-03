"""
Tahap 3 — U-Net 1D belajar menebak noise + GIF denoising BERBASIS MODEL.

Beda dengan tahap 2 (oracle): di sini U-Net harus MENEBaK noise ε_θ dari
action chunk ber-noise. Denoising dilakukan lewat reverse DDPM loop:

    x_{t-1} = 1/sqrt(ᾱ_t) · ( x_t - β_t/sqrt(1-ᾱ_t) · ε_θ(x_t, t, cond) ) + σ_t z

Tahapan:
  1. Siapkan data train (action chunk nyata dari zarr, dinormalisasi).
  2. Latih U-Net cepat (beberapa iterasi) untuk memprediksi noise (loss MSE).
  3. Simpan checkpoint pada step 0 (random), step ~N (partially trained),
     dan step akhir (best) → bandingkan loss naik/turun.
  4. Untuk tiap checkpoint, jalankan reverse loop dari x_100 murni noise →
     x_0, hasil akhir kasus gabungkan dengan target asli di axis.
  5. GIF: lihat jalur denoise model pada checkpoint akhir, bandingkan target.

Catatan: training hanya demonstrasi singkat (bukan training penuh skripsi).
Kondisi (cond) = state 5D, digenerate acak ~N(0,1) untuk demo (dataset nyata
punya state, diambil dari zarr). action dinormalisasi mean/std dataset.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.noise_schedule import CosineNoiseSchedule
from models.unet_1d import UNet1D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAROOT = os.path.join(REPO, "data", "pusht", "pusht_cchi_v7_replay.zarr")
OUTDIR = os.path.join(REPO, "data", "viz_output", "tahap3")
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = "cpu"          # CPU-only lokal (AMD iGPU no CUDA)
T_DIFF = 100            # diffusion steps
CHUNK = 16              # T_a action chunk
BATCH = 32
STEPS_TRAIN = 600       # iterasi percobaan singkat
SEEDS_SHOW = [0, 120, 590]   # checkpoint yang ditampilkan (harus < STEPS_TRAIN)

torch.manual_seed(0)
np.random.seed(0)

# ── load nyata ──
import zarr
root = zarr.open(DATAROOT, mode="r")
action_all = root["data/action"][:]
state_all = root["data/state"][:]
ends = root["meta/episode_ends"][:]
N_TOTAL = len(action_all)
print(f"dataset frames {N_TOTAL}, episodes {len(ends)}")

# normalisasi action position -> ~N(0,1) untuk training stabil
a_mean = action_all.mean(axis=0, keepdims=True)
a_std = action_all.std(axis=0, keepdims=True) + 1e-6
def norm_a(a):
    return (a - a_mean) / a_std
def denorm_a(a):
    return a * a_std + a_mean

# state: 5D, normalisasi juga ringan
s_mean = state_all.mean(axis=0, keepdims=True)
s_std = state_all.std(axis=0, keepdims=True) + 1e-6
def norm_s(s):
    return (s - s_mean) / s_std

# ── dataset loader (batch tetangga frame & state seed) ──
rng = np.random.default_rng(0)
def sample_batch(B, T_a=CHUNK, T_len=N_TOTAL):
    # pilih B episode acak, di dalamnya mulai offset; pastikan ada T_a frame
    ep_bounds = []
    for k in range(len(ends)):
        st = int(ends[k-1]) if k > 0 else 0
        en = int(ends[k])
        if en - st >= T_a:
            ep_bounds.append((st, en))
    idxs = []
    for _ in range(B):
        st, en = ep_bounds[rng.integers(len(ep_bounds))]
        o = rng.integers(0, en - st - T_a + 1)   # memastikan ada T_a frame
        idxs.append(o + st)
    act = np.stack([action_all[o:o+T_a] for o in idxs])       # (B,T_a,2)
    stt = np.stack([state_all[o] for o in idxs])              # (B,5) di awal chunk
    return (torch.tensor(norm_a(act).astype(np.float32)),
            torch.tensor(norm_s(stt).astype(np.float32)))

# ── model ──
model = UNet1D(action_dim=2, cond_dim=5).to(DEVICE)
sched = CosineNoiseSchedule(T=T_DIFF, s=0.008)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

def train_step(x_0, cond):
    B = x_0.shape[0]
    t = torch.randint(0, T_DIFF+1, (B,), device=DEVICE)
    noise = torch.randn_like(x_0)
    x_t = sched.q_sample(x_0, t, noise)
    pred = model(x_t, t, cond)
    return F.mse_loss(pred, noise)

# ── reverse DDPM loop (deterministik, tanpa stokastik z) ──
#   x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1-alpha_t)/sqrt(1-ab_t) * eps_theta )
# alpha_t = alphas[ti-1]  (per-step),  ab_t = alphas_bar[ti], 1-ab_t = oab
def denoise_stable(model, sched, x_T, cond):
    x = x_T
    steps = T_DIFF
    for ti in range(steps, 0, -1):
        tt = torch.full((1,), ti, device=DEVICE)  # B=1
        eps = model(x, tt, cond)
        alpha_t = sched.alphas[ti-1]
        ab_t = sched.alphas_bar[ti]
        oab_t = (1.0 - ab_t).clamp_min(1e-8)
        x = (x - (1.0 - alpha_t) / oab_t.sqrt() * eps) / alpha_t.sqrt()
    return x

print("start training demo...")
history = {s: None for s in SEEDS_SHOW}
losses = []
checkpoints = {}
for it in range(STEPS_TRAIN):
    x_0, cond = sample_batch(BATCH, CHUNK)
    x_0, cond = x_0.to(DEVICE), cond.to(DEVICE)
    loss = train_step(x_0, cond)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if it in SEEDS_SHOW:
        checkpoints[it] = {k: v.detach().clone() for k, v in model.state_dict().items()}

# ── evaluasi denoising per checkpoint ──
# target: satu chunk action nyata (episode 0, 16 frame pertama)
target_act = action_all[:CHUNK]
x0_norm = torch.tensor(norm_a(target_act).astype(np.float32)).unsqueeze(0)  # (1,T,2)
cond_norm = torch.tensor(norm_s(state_all[0]).reshape(1, -1), dtype=torch.float32)  # (1,5)
x0_display = denorm_a(x0_norm[0].numpy())

# noise untuk sampling dari x_T murni
x_T = torch.randn(1, CHUNK, 2)

finals = {}
for step_i, ck in checkpoints.items():
    m = UNet1D(action_dim=2, cond_dim=5).to(DEVICE)
    m.load_state_dict(ck)
    x_rec_norm = denoise_stable(m, sched, x_T.clone(), cond_norm.clone())
    finals[step_i] = denorm_a(x_rec_norm[0].detach().numpy())

# ── plot instance: loss curve ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(losses, color="#1e88e5", lw=1.4)
ax.set_xlabel("iterasi"); ax.set_ylabel("MSE loss (predisi noise)")
ax.set_title("Training singkat U-Net 1D — prediksi noise (demonstrasi)")
ax.grid(alpha=0.3)
for s_ in SEEDS_SHOW:
    ax.axvline(s_, color="grey", ls="--", lw=0.8)
    ax.text(s_ + 3, max(losses)*0.9, f"step {s_}", fontsize=8, rotation=90)
fig.tight_layout()
p_l = os.path.join(OUTDIR, "tahap3_loss.png")
fig.savefig(p_l, dpi=150); plt.close(fig)
print("[OK]", p_l)

# ── figure: denoising hasil U-Net vs target (3 checkpoint) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Denoising dari x_100 noise murni — hasil U-Net 1D (inline demo)",
             fontsize=14, fontweight="bold")
for i, step_i in enumerate(SEEDS_SHOW):
    ax = axes[i]
    rec = finals[step_i]
    ax.plot(target_act[:, 0], target_act[:, 1], "-o", color="#2e7d32", lw=1.6, ms=5,
            label="target x_0")
    ax.plot(rec[:, 0], rec[:, 1], "-o", color="#d81b60", lw=1.6, ms=5,
            label=f"U-Net denoise (step {step_i})")
    ax.scatter(rec[:, 0], rec[:, 1], c=range(CHUNK), cmap="viridis", s=18, zorder=3)
    ax.set_title(f"checkpoint step {step_i}  (loss {losses[step_i]:.4f})")
    ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
p_d = os.path.join(OUTDIR, "tahap3_denoise_unet.png")
fig.savefig(p_d, dpi=150); plt.close(fig)
print("[OK]", p_d)

# ── GIF: denoise progressive untuk checkpoint akhir ──
m_final = UNet1D(action_dim=2, cond_dim=5).to(DEVICE)
m_final.load_state_dict(checkpoints[max(SEEDS_SHOW)])

# kumpulkan jalur per t (dari x_T, tambahkan sedikit stokastik supaya menarik)
gif_paths = []
x = x_T.clone()
for ti in range(T_DIFF, 0, -1):
    tt = torch.full((1,), ti)
    eps = m_final(x, tt, cond_norm.clone())
    alpha_t = sched.alphas[ti-1]
    ab_t = sched.alphas_bar[ti]
    oab_t = (1.0 - ab_t).clamp_min(1e-8)
    x = (x - (1.0 - alpha_t) / oab_t.sqrt() * eps) / alpha_t.sqrt()
    gif_paths.append((ti, denorm_a(x[0].detach().numpy())))

fig3, ax3 = plt.subplots(figsize=(7, 7))
line3, = ax3.plot([], [], "-o", lw=1.4, ms=5, color="#d81b60", zorder=3)
target3 = ax3.plot([], [], "--", color="#2e7d32", lw=1, alpha=0.7)[0]
txt = ax3.text(0.02, 0.95, "", transform=ax3.transAxes, fontsize=11, va="top")

def anim_init3():
    ax3.set_xlim(0,512); ax3.set_ylim(0,512); ax3.set_aspect("equal"); ax3.invert_yaxis()
    ax3.grid(alpha=0.2)
    ax3.set_title("Denoising model U-Net (final checkpoint)")
    target3.set_data(target_act[:,0], target_act[:,1])
    return line3, target3, txt

def anim_up3(fi):
    ti, pts = gif_paths[fi]
    line3.set_data(pts[:,0], pts[:,1])
    txt.set_text(f"t={ti}")
    return line3, txt

anim3 = FuncAnimation(fig3, anim_up3, frames=len(gif_paths), blit=False,
                      init_func=anim_init3)
g3 = os.path.join(OUTDIR, "tahap3_unet_denoise.gif")
anim3.save(g3, writer=PillowWriter(fps=8))
plt.close(fig3)
print("[OK]", g3)

# ── print ringkas ──
print("loss_0  =", round(losses[0], 4))
print("loss_end=", round(losses[-1], 4))
print("denoise final: head error vs target (px):",
      round(np.linalg.norm(finals[max(SEEDS_SHOW)][-1] - target_act[-1]), 2))
print("\n=== selesai ===")