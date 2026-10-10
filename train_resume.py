""" Resume Training dari Locked Checkpoint (best.pt)

K2 UPDATE:
    - Config `global_cond_dim` default = 10.
    - Flatten conditioning di evaluate() dan training loop.
    - CATATAN: tidak bisa resume dari checkpoint To=1 lama. Harus retrain dari awal.
"""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pusht_dataset import (
    build_pusht_dataloader, split_train_val_dataset, normalize_data,
)
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine


# ============================================================
# Load Locked Config
# ============================================================
def load_locked_config(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    config_dict = ckpt["config"]
    model_state = ckpt["model_state_dict"]
    optimizer_state = ckpt["optimizer_state_dict"]
    action_stats = ckpt["action_stats"]
    state_stats = ckpt["state_stats"]
    start_epoch = ckpt["epoch"]
    ckpt_val_loss = float(ckpt["val_loss"])

    print(f"[Locked] Loaded checkpoint: {ckpt_path}")
    print(f"[Locked] Epoch: {start_epoch}, "
          f"Train Loss: {ckpt['train_loss']:.6f}, Val Loss: {ckpt_val_loss:.6f}")
    print(f"[Locked] Config keys: {list(config_dict.keys())}")

    return (config_dict, model_state, optimizer_state,
            action_stats, state_stats, start_epoch, ckpt_val_loss)


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class LockedTrainConfig:
    zarr_path: str
    pred_horizon: int
    obs_horizon: int
    action_horizon: int
    val_ratio: float
    seed: int
    input_dim: int
    global_cond_dim: int                    # K2: dari checkpoint (10)
    diffusion_step_embed_dim: int
    down_dims: Tuple[int, ...]
    kernel_size: int
    n_groups: int
    num_diffusion_steps: int
    noise_schedule_s: float
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    device: str
    log_interval: int
    save_interval: int
    action_stats: Dict
    state_stats: Dict
    resume_epochs: int = 70
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion_locked"

    @classmethod
    def from_checkpoint(cls, config_dict, **overrides):
        if isinstance(config_dict.get("down_dims"), list):
            config_dict = {**config_dict, "down_dims": tuple(config_dict["down_dims"])}
        merged = {**config_dict, **overrides}
        return cls(**merged)

    def to_train_config(self):
        locked_fields = {k: v for k, v in asdict(self).items()
                         if k not in ("resume_epochs", "run_name")}
        return TrainConfig(**locked_fields)


@dataclass
class TrainConfig:
    zarr_path: str
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    val_ratio: float = 0.1
    seed: int = 42
    input_dim: int = 2
    global_cond_dim: int = 10                # K2
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8
    num_diffusion_steps: int = 100
    noise_schedule_s: float = 0.008
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    device: str = "cuda"
    log_interval: int = 10
    save_interval: int = 10
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion"
    action_stats: Dict = None
    state_stats: Dict = None
    finetune_lr_scale: float = 0.1
    scheduler_eta_min_scale: float = 0.01


# ============================================================
# Loss & Evaluation
# ============================================================
def diffusion_loss(model, noise_sched, x_0, cond, device):
    B = x_0.shape[0]
    T = noise_sched.T
    t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = noise_sched.q_sample(x_0, t, noise)
    x_t_permuted = x_t.permute(0, 2, 1)
    eps_pred = model(x_t_permuted, t, cond)
    loss = F.mse_loss(eps_pred.permute(0, 2, 1), noise)
    return loss


@torch.no_grad()
def evaluate(model, noise_sched, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in val_loader:
        x_0 = batch["action"].to(device)
        # K2 FIX
        cond = batch["state"].reshape(batch["state"].shape[0], -1).to(device)
        loss = diffusion_loss(model, noise_sched, x_0, cond, device)
        total_loss += loss.item() * x_0.size(0)
        total_samples += x_0.size(0)
    model.train()
    return total_loss / total_samples if total_samples > 0 else float("inf")


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                    config, action_stats, state_stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss, "val_loss": val_loss,
        "config": config.__dict__, "action_stats": action_stats,
        "state_stats": state_stats,
    }, path)
    print(f"[Checkpoint] Saved: {path}")


# ============================================================
# Main Resume Training
# ============================================================
def train_resume(locked_config, config, start_epoch,
                 model_state=None, optimizer_state=None,
                 best_val_loss_init=float("inf")):
    device = torch.device(config.device)
    print(f"\n[Resume Train] Device: {device}")
    print(f"[Resume Train] Starting from epoch {start_epoch + 1} to {config.epochs}")

    train_ds, val_ds = split_train_val_dataset(
        zarr_path=config.zarr_path, val_ratio=config.val_ratio,
        seed=config.seed, pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon, action_horizon=config.action_horizon,
    )

    train_ds.action_stats = locked_config.action_stats
    train_ds.state_stats = locked_config.state_stats
    val_ds.action_stats = locked_config.action_stats
    val_ds.state_stats = locked_config.state_stats

    train_data = {"state": train_ds.state_all.astype(np.float32),
                  "action": train_ds.action_all.astype(np.float32)}
    train_ds.normalized_train_data = {
        "state": normalize_data(train_data["state"], locked_config.state_stats).astype(np.float32),
        "action": normalize_data(train_data["action"], locked_config.action_stats).astype(np.float32),
    }

    val_data = {"state": val_ds.state_all.astype(np.float32),
                "action": val_ds.action_all.astype(np.float32)}
    val_ds.normalized_train_data = {
        "state": normalize_data(val_data["state"], locked_config.state_stats).astype(np.float32),
        "action": normalize_data(val_data["action"], locked_config.action_stats).astype(np.float32),
    }

    train_loader = build_pusht_dataloader(
        train_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, device=device)
    val_loader = build_pusht_dataloader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, device=device)
    print(f"[Data] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"[Data] Batches/epoch: {len(train_loader)}")

    model = ConditionalUnet1D(
        input_dim=config.input_dim,
        global_cond_dim=config.global_cond_dim,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        down_dims=list(config.down_dims),
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total params: {total_params:,}")
    print(f"[Model] global_cond_dim: {config.global_cond_dim}")

    noise_sched = NoiseScheduleCosine(
        T=config.num_diffusion_steps, s=config.noise_schedule_s, device=device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if model_state is not None:
        model.load_state_dict(model_state)
        print("[Resume] Model weights loaded from locked checkpoint.")
    else:
        print("[Warning] model_state is None — model starts from random init!")

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("[Resume] Optimizer state loaded from locked checkpoint.")

    remaining_epochs = max(1, config.epochs - start_epoch)
    finetune_lr = config.lr * config.finetune_lr_scale
    for g in optimizer.param_groups:
        g["lr"] = finetune_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_epochs,
        eta_min=finetune_lr * config.scheduler_eta_min_scale)

    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints -> {output_dir}")

    best_val_loss = best_val_loss_init
    print(f"[Resume] best_val_loss initialized to {best_val_loss:.6f}")

    print(f"\n[Resume Train] Continuing from epoch {start_epoch + 1} to {config.epochs}...")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch + 1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch["action"].to(device)
            # K2 FIX
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

        val_loss = evaluate(model, noise_sched, val_loader, device)
        history["val_loss"].append(val_loss)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        if epoch % config.save_interval == 0 or val_loss < best_val_loss:
            ckpt_path = output_dir / f"epoch_{epoch:04d}_train_{avg_train_loss:.4f}_val_{val_loss:.4f}.pt"
            save_checkpoint(model, optimizer, epoch, avg_train_loss, val_loss,
                           config, locked_config.action_stats,
                           locked_config.state_stats, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                save_checkpoint(model, optimizer, epoch, avg_train_loss, val_loss,
                               config, locked_config.action_stats,
                               locked_config.state_stats, best_path)
                print(f"[Best] New best val loss: {best_val_loss:.6f}")

    final_path = output_dir / "final.pt"
    save_checkpoint(model, optimizer, config.epochs, history["train_loss"][-1],
                    history["val_loss"][-1], config, locked_config.action_stats,
                    locked_config.state_stats, final_path)

    print(f"\n[Resume Train] Done! Best val loss: {best_val_loss:.6f}")
    print(f"[Resume Train] Checkpoints saved to: {output_dir}")
    return model


# ============================================================
# Entry Point
# ============================================================
def main():
    REPO_ROOT = Path(__file__).parent
    LOCKED_CKPT = REPO_ROOT / "checkpoints" / "pusht_state_diffusion" / "best.pt"

    if not LOCKED_CKPT.exists():
        print(f"[Error] Locked checkpoint not found: {LOCKED_CKPT}")
        sys.exit(1)

    (locked_config_dict, model_state, optimizer_state,
     action_stats, state_stats, start_epoch, ckpt_val_loss) = load_locked_config(LOCKED_CKPT)

    EXCLUDE_FIELDS = ("run_name", "epochs", "action_stats", "state_stats")
    checkpoint_config = {k: v for k, v in locked_config_dict.items()
                         if k not in EXCLUDE_FIELDS}
    locked_config = LockedTrainConfig.from_checkpoint(
        checkpoint_config, action_stats=action_stats, state_stats=state_stats,
        resume_epochs=80, run_name="pusht_state_diffusion")

    config = locked_config.to_train_config()
    config.epochs = locked_config.resume_epochs
    config.run_name = "pusht_state_diffusion"

    train_resume(locked_config, config, start_epoch,
                 model_state=model_state, optimizer_state=optimizer_state,
                 best_val_loss_init=ckpt_val_loss)


if __name__ == "__main__":
    main()