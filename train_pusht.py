""" Training script untuk Diffusion Policy pada Push-T (state-based, low-dim).

Integrasi:
- Dataset: PushTStateDataset (data/pusht_dataset.py)
- Model: ConditionalUnet1D (models/mlp_noise_head.py)
- Noise Schedule: NoiseScheduleCosine (models/noise_schedule.py)
- Device: auto-detect via utils/device.py (CUDA > MPS > CPU)

Referensi arsitektur:
- real-stanford/diffusion_policy (conditional_unet1d.py, train_diffusion_unet_*.py)
- qlOoOlp/Diffusion-Policy-Tutorial (pusht_trainer.py)

Device-agnostic:
- `config.device = None` → auto-detect.
- Override via CLI: `--device cpu` atau `--device cuda`.
- Override via env: `FORCE_DEVICE=cpu python train_pusht.py`.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Pastikan root repo di sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.device import get_device, describe_device, is_cuda, summary
from data.pusht_dataset import (
    PushTStateDataset,
    PushTDataConfig,
    build_pusht_dataset,
    build_pusht_dataloader,
    split_train_val_dataset,
)
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int, deterministic: bool = True):
    """Set seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Konfigurasi Training
# ============================================================
@dataclass
class TrainConfig:
    # Dataset
    zarr_path: str
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    val_ratio: float = 0.1
    seed: int = 42

    # Model
    input_dim: int = 2
    global_cond_dim: int = 10          # K2: obs_horizon * state_dim = 2 * 5
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8

    # Diffusion
    num_diffusion_steps: int = 100
    noise_schedule_s: float = 0.008

    # Training
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50

    # Device: None = auto-detect (CUDA > MPS > CPU)
    # Bisa juga "cpu", "cuda", "cuda:0", dsb.
    device: Optional[str] = None

    # Logging / Checkpoint
    log_interval: int = 10
    save_interval: int = 10
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion"


# ============================================================
# Loss Function: MSE noise prediction
# ============================================================
def diffusion_loss(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Diffusion training loss (MSE antara noise prediksi vs noise asli).

    Algorithm:
    1. Sample timestep t ~ Uniform(1, T)
    2. Sample noise eps ~ N(0, I)
    3. x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    4. eps_pred = model(x_t, t, cond)
    5. Loss = MSE(eps_pred, eps)
    """
    B = x_0.shape[0]
    T = noise_sched.T

    # Sample timestep
    t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)

    # Sample noise
    noise = torch.randn_like(x_0)

    # Forward diffusion: q_sample
    x_t = noise_sched.q_sample(x_0, t, noise)

    # Model prediction
    x_t_permuted = x_t.permute(0, 2, 1)  # (B, action_dim, T_a)
    eps_pred = model(x_t_permuted, t, cond)
    loss = F.mse_loss(eps_pred.permute(0, 2, 1), noise)

    return loss


# ============================================================
# Validation Loss
# ============================================================
@torch.no_grad()
def evaluate(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Hitung rata-rata validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x_0 = batch["action"].to(device)
        # K2 FIX: flatten obs_horizon × state_dim
        cond = batch["state"].reshape(batch["state"].shape[0], -1).to(device)
        loss = diffusion_loss(model, noise_sched, x_0, cond, device)
        total_loss += loss.item() * x_0.size(0)
        total_samples += x_0.size(0)

    model.train()
    return total_loss / total_samples if total_samples > 0 else float("inf")


# ============================================================
# Checkpoint Utilities
# ============================================================
def save_checkpoint(
    model: ConditionalUnet1D,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainConfig,
    action_stats: Dict,
    state_stats: Dict,
    path: Path,
    device: Optional[torch.device] = None,
):
    """
    Simpan checkpoint lengkap.

    Catatan: model state selalu disimpan di CPU agar kompatibel
    di-load di device apa pun (CPU/CUDA/MPS).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pindah state_dict ke CPU untuk portabilitas
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Optimizer state juga — tapi ini kompleks, biarkan di device asli
    opt_state = optimizer.state_dict()

    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": opt_state,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config.__dict__,
        "action_stats": action_stats,
        "state_stats": state_stats,
    }, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(
    path: Path,
    model: ConditionalUnet1D,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load checkpoint.

    `map_location` otomatis ke device target, jadi checkpoint yang
    disimpan dari GPU bisa di-load di CPU (dan sebaliknya).
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ============================================================
# Main Training Loop
# ============================================================
def train(config: TrainConfig) -> ConditionalUnet1D:
    # --- Seed ---
    set_seed(config.seed)

    # --- Device auto-detect ---
    device = get_device(prefer=config.device)
    print(summary())
    print(f"[Train] Device: {describe_device(device)} | Seed: {config.seed}")
    print(f"[Train] Config: {config}")

    # --- Dataset ---
    print("\n[Data] Loading dataset...")
    train_ds, val_ds = split_train_val_dataset(
        zarr_path=config.zarr_path,
        val_ratio=config.val_ratio,
        seed=config.seed,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )
    train_loader = build_pusht_dataloader(
        train_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, device=device,
    )
    val_loader = build_pusht_dataloader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, device=device,
    )
    action_stats, state_stats = train_ds.get_normalization_stats()
    print(f"[Data] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"[Data] Batches/epoch: {len(train_loader)}")

    # --- Model ---
    print("\n[Model] Building ConditionalUnet1D...")
    model = ConditionalUnet1D(
        input_dim=config.input_dim,
        global_cond_dim=config.global_cond_dim,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        down_dims=list(config.down_dims),
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"[Model] global_cond_dim: {config.global_cond_dim}")

    # --- Noise Schedule ---
    noise_sched = NoiseScheduleCosine(
        T=config.num_diffusion_steps,
        s=config.noise_schedule_s,
        device=device,
    )

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # --- LR Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.01,
    )

    # --- Output dir ---
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints -> {output_dir}")

    # --- Training Loop ---
    print(f"\n[Train] Starting training for {config.epochs} epochs...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch["action"].to(device)
            # K2 FIX: flatten obs_horizon × state_dim
            cond = batch["state"].reshape(batch["state"].shape[0], -1).to(device)

            loss = diffusion_loss(model, noise_sched, x_0, cond, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss = evaluate(model, noise_sched, val_loader, device)
        history["val_loss"].append(val_loss)

        # LR step
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        if epoch % config.save_interval == 0 or val_loss < best_val_loss:
            ckpt_path = output_dir / f"epoch_{epoch:04d}_train_{avg_train_loss:.4f}_val_{val_loss:.4f}.pt"
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, val_loss, config,
                action_stats, state_stats, ckpt_path, device=device,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, epoch, avg_train_loss, val_loss, config,
                    action_stats, state_stats, best_path, device=device,
                )

    # Save final
    final_path = output_dir / "final.pt"
    save_checkpoint(
        model, optimizer, config.epochs, history["train_loss"][-1],
        history["val_loss"][-1], config, action_stats, state_stats,
        final_path, device=device,
    )

    print(f"\n[Train] Done! Best val loss: {best_val_loss:.6f}")
    print(f"[Train] Checkpoints saved to: {output_dir}")
    return model


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on Push-T")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', atau None (auto-detect).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--zarr", type=str, default=None,
        help="Path ke dataset .zarr. Default: <repo>/data/pusht/pusht_cchi_v7_replay.zarr",
    )
    args = parser.parse_args()

    # Resolve path dataset
    if args.zarr:
        zarr_path = Path(args.zarr).resolve()
    else:
        repo_root = Path(__file__).resolve().parent
        zarr_path = repo_root / "data" / "pusht" / "pusht_cchi_v7_replay.zarr"

    if not zarr_path.exists():
        print(f"[Error] Dataset not found: {zarr_path}")
        print("Jalankan: uv run python data/get_data.py")
        sys.exit(1)

    # Susun config
    config = TrainConfig(
        zarr_path=str(zarr_path),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        val_ratio=0.1,
        seed=args.seed,
        input_dim=2,
        global_cond_dim=40,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        num_diffusion_steps=100,
        noise_schedule_s=0.008,
        batch_size=args.batch_size if args.batch_size else 64,
        num_workers=0,
        lr=1e-4,
        weight_decay=1e-4,
        epochs=args.epochs,
        device=args.device,           # None = auto
        log_interval=20,
        save_interval=10,
        output_dir="checkpoints",
        run_name="pusht_state_diffusion",
    )

    train(config)


if __name__ == "__main__":
    main()