"""
Training Diffusion Policy Push-T (keypoint + agent_pos, 20D).

Mengikuti gaya repo asli `real-stanford/diffusion_policy`:
- TIDAK menggunakan validation set / val loss
- Checkpoint periodik setiap `save_interval` epoch
- Hanya track training loss (MSE noise prediction)
- EMA (Exponential Moving Average) opsional untuk stabilisasi
- Metrik sebenarnya = success rate, dihitung terpisah via eval_policy.py

Training loss (DDPM ε-prediction):
  Untuk tiap sampel di batch:
    1. Sample t ~ Uniform(1, T)
    2. Sample ε ~ N(0, I)
    3. x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
    4. loss = MSE(ε_θ(x_t, t, cond), ε)
"""
from __future__ import annotations

import argparse
import copy
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
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.device import get_device, describe_device, summary
from data.pusht_dataset import (
    PushTDataConfig,
    split_train_val_dataset,
    build_pusht_dataloader,
)
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine


# ============================================================
# Reproducibility (K3)
# ============================================================
def set_seed(seed: int, deterministic: bool = True):
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
    zarr_path: str = "data/pusht/pusht_cchi_v7_replay.zarr"
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    seed: int = 42

    # Dimensi (keypoint + agent_pos)
    obs_dim: int = 20
    input_dim: int = 2
    global_cond_dim: int = 40     # obs_horizon × obs_dim = 2 × 20

    # Model
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
    grad_clip: float = 0.0

    # EMA (opsional, seperti repo asli)
    use_ema: bool = True
    ema_decay: float = 0.999

    # Device
    device: Optional[str] = None

    # Logging / Checkpoint
    log_interval: int = 20
    save_interval: int = 10        # simpan tiap N epoch
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion"


# ============================================================
# EMA Helper (mengikuti repo asli)
# ============================================================
class EMAModel:
    """Exponential Moving Average dari model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, d: Dict):
        self.decay = d["decay"]
        self.shadow = d["shadow"]

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# ============================================================
# Diffusion Loss
# ============================================================
def diffusion_loss(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    x_0: torch.Tensor,          # (B, Ta, Da) normalized
    cond: torch.Tensor,         # (B, global_cond_dim)
    device: torch.device,
) -> torch.Tensor:
    B = x_0.shape[0]
    T = noise_sched.T

    # 1. Sample timestep acak
    t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)

    # 2. Sample noise
    noise = torch.randn_like(x_0)

    # 3. Forward diffusion
    x_t = noise_sched.q_sample(x_0, t, noise)

    # 4. Prediksi noise
    eps_pred = model(x_t.permute(0, 2, 1), t, cond)

    # 5. MSE loss
    return F.mse_loss(eps_pred.permute(0, 2, 1), noise)


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(
    model: ConditionalUnet1D,
    optimizer: optim.Optimizer,
    ema: Optional[EMAModel],
    epoch: int,
    train_loss: float,
    config: TrainConfig,
    normalizer_state: Optional[Dict],
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "config": config.__dict__,
        "normalizer_state": normalizer_state,
    }
    if ema is not None:
        # EMA shadow juga disimpan di CPU
        ema_state = {
            "decay": ema.decay,
            "shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        }
        ckpt["ema_state"] = ema_state
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved: {path}")


# ============================================================
# Training Loop
# ============================================================
def train(config: TrainConfig) -> ConditionalUnet1D:
    set_seed(config.seed)
    device = get_device(prefer=config.device)
    print(summary())
    print(f"[Train] Device: {describe_device(device)} | Seed: {config.seed}")
    print(f"[Train] Config: {config}")

    # --- Dataset (train only, no val) ---
    print("\n[Data] Loading dataset (keypoint + agent_pos, 20D)...")
    data_cfg = PushTDataConfig(
        dataset_path=config.zarr_path,
        obs_horizon=config.obs_horizon,
        pred_horizon=config.pred_horizon,
        obs_dim=config.obs_dim,
        action_dim=config.input_dim,
        val_ratio=0.0,           # tidak pakai val
        seed=config.seed,
    )
    # Pakai seluruh dataset sebagai train
    train_ds, _ = split_train_val_dataset(data_cfg)
    # Kembalikan val_ratio=0.0 agar val_idx kosong
    # (fallback: kalau split menghasilkan val tidak kosong, abaikan saja)
    print(f"[Data] Train samples: {len(train_ds)}")

    sample = train_ds[0]
    assert sample["obs"].shape == (config.obs_horizon, config.obs_dim)
    assert sample["action"].shape == (config.pred_horizon, config.input_dim)

    pin = (device.type == "cuda")
    train_loader = build_pusht_dataloader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    print(f"[Data] Batches/epoch: {len(train_loader)}")

    normalizer_state = (
        train_ds.normalizer.state_dict()
        if train_ds.normalizer is not None else None
    )

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
    print(f"[Model] Params: {total_params:,}")
    print(f"[Model] global_cond_dim = {config.global_cond_dim}")

    # --- EMA ---
    ema = EMAModel(model, decay=config.ema_decay) if config.use_ema else None
    if ema is not None:
        print(f"[EMA] Enabled, decay={config.ema_decay}")

    # --- Noise Schedule ---
    noise_sched = NoiseScheduleCosine(
        T=config.num_diffusion_steps,
        s=config.noise_schedule_s,
        device=device,
    )

    # --- Optimizer + Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.01,
    )

    # --- Output ---
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints -> {output_dir}")

    # --- Loop ---
    print(f"\n[Train] Starting {config.epochs} epochs (no validation)...")
    history = {"train_loss": [], "lr": []}
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch["action"].to(device, non_blocking=True)
            cond = batch["obs"].reshape(batch["obs"].shape[0], -1).to(device, non_blocking=True)

            loss = diffusion_loss(model, noise_sched, x_0, cond, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Update EMA setiap step (seperti repo asli)
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if num_batches % config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"time: {epoch_time:.1f}s"
        )

        # --- Checkpoint periodik (gaya repo asli) ---
        if epoch % config.save_interval == 0:
            # Simpan model reguler
            ckpt_path = output_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                model, optimizer, ema, epoch, avg_train_loss,
                config, normalizer_state, ckpt_path,
            )

            # Simpan versi EMA terpisah (biasanya lebih stabil)
            if ema is not None:
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                ema_path = output_dir / f"epoch_{epoch:04d}_ema.pt"
                save_checkpoint(
                    ema_model, optimizer, None, epoch, avg_train_loss,
                    config, normalizer_state, ema_path,
                )

        # Simpan last.pt setiap epoch (untuk resume)
        save_checkpoint(
            model, optimizer, ema, epoch, avg_train_loss,
            config, normalizer_state, output_dir / "last.pt",
        )

    # --- Simpan final ---
    final_path = output_dir / "final.pt"
    save_checkpoint(
        model, optimizer, ema, config.epochs, history["train_loss"][-1],
        config, normalizer_state, final_path,
    )

    print(f"\n[Train] Done!")
    print(f"[Train] Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"[Train] Checkpoints: {output_dir}")
    print(f"[Train] Untuk evaluasi success rate:")
    print(f"        uv run python eval_policy.py --ckpt {output_dir / 'epoch_{:04d}.pt'.format(config.epochs)} --episodes 50")

    return model


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train DP Push-T (keypoint 20D)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--zarr", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()

    if args.zarr:
        zarr_path = Path(args.zarr).resolve()
    else:
        zarr_path = Path(__file__).resolve().parent / "data" / "pusht" / "pusht_cchi_v7_replay.zarr"

    if not zarr_path.exists():
        print(f"[Error] Dataset not found: {zarr_path}")
        sys.exit(1)

    config = TrainConfig(
        zarr_path=str(zarr_path),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        seed=args.seed,
        obs_dim=20,
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
        grad_clip=1.0,
        use_ema=not args.no_ema,
        ema_decay=0.999,
        device=args.device,
        log_interval=20,
        save_interval=args.save_interval,
        output_dir="checkpoints",
        run_name="pusht_state_diffusion",
    )

    train(config)


if __name__ == "__main__":
    main()