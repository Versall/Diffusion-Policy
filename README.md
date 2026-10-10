# Diffusion Policy — Push-T

Implementasi **Diffusion Policy** untuk task **Push-T** (state-based, low-dim) sebagai fondasi skripsi, dengan rencana pengembangan menuju **DPPO (Diffusion Policy Policy Optimization)** dan integrasi **World Model**.

> **Tujuan akhir skripsi:** Membandingkan DPPO baseline (tanpa World Model) dengan DPPO + World Model pada metrik *inference time*, *kecepatan iterasi training*, dan *kualitas policy*.

## Status Proyek

| Fase | Deskripsi | Status |
|---|---|---|
| **Fase 1** | Reproduksi Diffusion Policy baseline (Push-T state-based) | 🟡 Sedang berjalan |
| **Fase 2** | Pengembangan DPPO (baseline skripsi) | ⏸️ Belum dimulai |
| **Fase 3** | Integrasi DPPO + World Model (kontribusi) | ⏸️ Belum dimulai |
| **Fase 4** | Analisis & penulisan skripsi | ⏸️ Belum dimulai |

### Yang Sudah Selesai (Fase 1)

- ✅ Arsitektur `ConditionalUnet1D` — **1:1 dengan [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)**
- ✅ **Reproducibility**: `set_seed()` di training loop
- ✅ **Fix data leakage**: normalisasi (min/max) dihitung dari **train set saja**
- ✅ **Conditioning `To=2`**: seluruh `obs_horizon` frame dipakai sebagai conditioning (`global_cond_dim=10`)
- ✅ **DDIM Sampler**: 10 langkah deterministik untuk inference cepat
- ✅ **Pipeline lengkap**: training, resume, evaluasi, visualisasi (GIF/MP4), animasi proses denoising
- ✅ **Setup uv**: `pyproject.toml` + `uv.lock` untuk instalasi reproducible

## Struktur Repo

```
.
├── data/
│   ├── pusht/                      # Dataset Zarr (tidak di-commit, ~3 GB)
│   │   └── pusht_cchi_v7_replay.zarr/
│   ├── manual_pusht/               # Dataset hasil koleksi manual (tidak di-commit)
│   ├── pusht_dataset.py            # Loader dataset + sliding window + normalisasi
│   ├── get_data.py                 # Script unduh dataset benchmark
│   ├── evaluate_episodes.py        # Analisis episode dataset
│   ├── read_data.py                # Inspeksi isi Zarr
│   ├── replay_episode.py           # Replay episode dari dataset
│   ├── visualize_data.py           # Visualisasi distribusi data
│   └── viz_unet_denoise.py         # Visualisasi proses denoising UNet
├── environment/
│   └── push_t.py                   # Kolektor demo manual (PyGame + mouse)
├── models/
│   ├── mlp_noise_head.py           # ConditionalUnet1D (1:1 repo asli)
│   ├── noise_schedule.py           # Cosine noise schedule (s=0.008, T=100)
│   └── diffusion_policy.py         # DDIMSampler + DiffusionPolicy wrapper
├── train_pusht.py                  # Script training utama
├── train_resume.py                 # Resume training dari checkpoint
├── eval_policy.py                  # Evaluasi success rate di environment
├── visualize_policy.py             # Simpan GIF/MP4 episode
├── visualize_diffusion_process.py  # Animasi proses denoising DDIM
├── verification.py                 # Verifikasi anti data leakage
├── verify_k2.py                    # Verifikasi conditioning To=2
├── pyproject.toml                  # Dependencies + konfigurasi CPU-only torch
├── uv.lock                         # Lockfile (WAJIB commit)
├── .python-version                 # Pin versi Python
└── .gitignore
```

## Requirements

- **Python**: 3.11–3.12 (lihat `.python-version`)
- **Hardware**: CPU-only (dioptimalkan untuk AMD iGPU tanpa CUDA)
- **Package manager**: [uv](https://docs.astral.sh/uv/)

## Instalasi

### 1. Install `uv`

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone & sync

```bash
git clone https://github.com/Versall/Diffusion-Policy.git
cd Diffusion-Policy

# Install semua dependencies ke .venv (otomatis)
uv sync
```

`uv sync` akan:
- Membuat `.venv` di root repo
- Meng-resolve semua dependencies dari `uv.lock`
- Menginstall PyTorch versi **CPU-only** (hemat ~6 GB, tanpa dependensi CUDA)

### 3. Verifikasi environment

```bash
uv run python verification.py     # Cek data leakage fix
uv run python verify_k2.py        # Cek conditioning To=2
```

## Dataset

Dataset Push-T **tidak disertakan** dalam repo karena ukurannya besar (~3 GB). Ada 2 cara mendapatkannya:

### Opsi A — Dataset benchmark resmi (206 episode)

Unduh dari sumber resmi:

```bash
# Via script bawaan
uv run python data/get_data.py
```

Atau unduh manual dari: https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip

Letakkan di:
```
data/pusht/pusht_cchi_v7_replay.zarr/
├── data/
│   ├── action      # (25650, 2)
│   ├── img         # (25650, 96, 96, 3)
│   ├── keypoint    # (25650, 9, 2)
│   ├── n_contacts  # (25650, 1)
│   └── state       # (25650, 5)
└── meta/
    └── episode_ends  # (206,)
```

### Opsi B — Koleksi demo manual

```bash
uv run python environment/push_t.py
```

Kontrol agent dengan mouse, demo tersimpan sebagai `episode_*.npz` di `data/manual_pusht/`.

## Training

### Konfigurasi Default

| Parameter | Nilai | Catatan |
|---|---|---|
| `pred_horizon` (`Ta`) | 16 | Panjang action chunk |
| `obs_horizon` (`To`) | 2 | Frame observasi untuk conditioning |
| `action_horizon` | 8 | Action dieksekusi per inference |
| `global_cond_dim` | 10 | `obs_horizon × state_dim` (K2 fix) |
| `down_dims` | (256, 512, 1024) | Arsitektur U-Net |
| `batch_size` | 64 | |
| `lr` | 1e-4 | AdamW + cosine annealing |
| `epochs` | 50 | ~3 menit/epoch di CPU |
| `num_diffusion_steps` | 100 | Jadwal noise kosinus (`s=0.008`) |

### Jalankan Training

```bash
uv run python train_pusht.py
```

**Output:**
```
checkpoints/pusht_state_diffusion/
├── best.pt           # Checkpoint val loss terbaik
├── final.pt          # Checkpoint epoch terakhir
├── epoch_0001_train_..._val_....pt
├── epoch_0002_train_..._val_....pt
└── ...
```

### Resume Training

Kalau training terputus, lanjutkan dari `best.pt`:

```bash
uv run python train_resume.py
```

Config akan **dikunci** dari checkpoint (arsitektur & normalisasi tetap sama) dan hanya `epochs` yang bisa diperpanjang.

### Reproducibility

Semua training sudah memakai `set_seed(42)`. Untuk run multi-seed (untuk mean ± std di skripsi):

```bash
# Ubah seed di train_pusht.py (main() -> TrainConfig(seed=...))
uv run python train_pusht.py   # seed 42
# Edit seed=43, jalankan lagi
# Edit seed=44, jalankan lagi
```

## Evaluasi

### Ukur Success Rate

```bash
uv run python eval_policy.py \
  --ckpt checkpoints/pusht_state_diffusion/best.pt \
  --episodes 20 \
  --inference-steps 10
```

**Threshold sukses:** coverage > 0.80 (sesuai paper Diffusion Policy).

**Output:**
```
EVALUATION SUMMARY (20 episodes)
==================================================
Success Rate: XX/20 (XX.0%)
Avg Reward: ...
Avg Steps: ...
Avg Max Cov: ...
```

### Bandingkan Inference Steps

Untuk analisis trade-off kecepatan vs kualitas:

```bash
uv run python eval_policy.py --inference-steps 5
uv run python eval_policy.py --inference-steps 10
uv run python eval_policy.py --inference-steps 20
```

## Visualisasi

### GIF/MP4 Episode

```bash
uv run python visualize_policy.py \
  --ckpt checkpoints/pusht_state_diffusion/best.pt \
  --episodes 3 \
  --format gif
```

Output: `viz_output/policy_runs/episode_*.gif`

### Animasi Proses Denoising

Menampilkan proses DDIM dari noise → action chunk, panel per panel:

```bash
uv run python visualize_diffusion_process.py \
  --ckpt checkpoints/pusht_state_diffusion/best.pt \
  --inference-steps 20 \
  --mode diffusion_only
```

Output: `viz_output/diffusion_process/ep*_seed*_diffusion_process.gif`

## Arsitektur

### ConditionalUnet1D

Mengikuti arsitektur resmi `real-stanford/diffusion_policy`:

- **Time embedding**: sinusoidal positional + 2×Linear(Mish)
- **FiLM conditioning**: `h = scale · h + bias` di setiap ResBlock
- **U-Net 1D**: 3 level down + 2 mid + 2 level up
- **2 ConditionalResidualBlock1d per level**
- **Final**: `Conv1dBlock` → `Conv1d(1×1)` ke action_dim

### DDIM Sampling

- **Training**: 100 langkah (DDPM, ε-prediction, MSE loss)
- **Inference**: 10 langkah (DDIM deterministic, `η=0`)
- **Kecepatan**: ~10× lebih cepat dari DDPM dengan kualitas hampir sama

### Conditioning (K2 Fix)

```
obs (B, To=2, state_dim=5)
    ↓ flatten
cond (B, 10)
    ↓ concat dengan time embedding
global_feature (B, 10 + 256)
    ↓ masuk ke setiap ResBlock via FiLM
```

## Referensi

- **Paper utama**: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023.
- **Repo asli**: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- **Tutorial**: [qlOoOlp/Diffusion-Policy-Tutorial](https://github.com/qlOoOlp/Diffusion-Policy-Tutorial)
- **DPPO**: Ren et al., *Diffusion Policy Policy Optimization*, ICLR 2025. Kode: [irom-lab/dppo](https://github.com/irom-lab/dppo)
- **Dataset Push-T**: [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)

## Catatan Teknis

### CPU-Only

Proyek ini dioptimalkan untuk **CPU-only** (AMD iGPU tanpa CUDA):

- `pyproject.toml` mengarahkan `torch` ke index CPU-only (`pytorch-cpu`)
- `down_dims` bisa diperkecil `(64, 128, 256)` untuk iterasi cepat
- `num_workers=0` di DataLoader (Windows-safe)
- Batch size kecil + epoch singkat

### Konvensi Tensor

- **Internal model**: `(B, C, L)` — untuk `Conv1d` di U-Net
- **Antar-muka dataset**: `(B, To, state_dim)` dan `(B, Ta, action_dim)`
- **Permute** dilakukan eksplisit di `diffusion_loss()` dan `sampler.sample()`

### File Besar Tidak di-Commit

`.gitignore` mengabaikan:
- Dataset: `data/**/*.zarr`, `data/**/*.npz`
- Checkpoint: `checkpoints/`, `*.pt`, `*.pth`
- Virtual env: `.venv/`
- Output visualisasi: `viz_output/`, `*.gif`, `*.mp4`

**Transfer dataset & checkpoint** ke PC remote dilakukan terpisah via SCP/cloud.

## Lisensi

Proyek ini dibuat untuk keperluan skripsi. Komponen arsitektur mengikuti lisensi MIT dari [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).
```

## 🎯 Cara Pasang

```cmd
:: 1. Buka README lama (kosong)
notepad README.md

:: 2. Paste seluruh isi di atas
:: 3. Simpan (Ctrl+S), tutup

:: 4. Cek hasilnya
type README.md

:: 5. Commit
git add README.md
git commit -m "docs: tambah README lengkap

- Deskripsi proyek & roadmap (DP -> DPPO -> World Model)
- Status Fase 1 (selesai: seed, data leakage, K2)
- Struktur repo aktual
- Instalasi via uv
- Panduan dataset, training, eval, visualisasi
- Arsitektur & referensi"
git push
```

## 📌 Yang Saya Sesuaikan dengan Repo Anda

| Item | Nilai Aktual dari Repo |
|---|---|
| Package name | `diffusion-policy-push-t` |
| Python version | `>=3.11,<3.13` (`.python-version` = 3.12) |
| Dependencies | 11 paket (sesuai `pyproject.toml`) |
| Torch index | CPU-only (`pytorch-cpu`) |
| File root | `train_pusht.py`, `train_resume.py`, `eval_policy.py`, `visualize_*.py`, `verification.py` |
| Folder | `data/`, `environment/`, `models/` |
| Dataset | `data/pusht/pusht_cchi_v7_replay.zarr` |
