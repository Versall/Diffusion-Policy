"""
Resume Training dari Locked Checkpoint (best.pt)
==================================================
Menggunakan pipeline yang sama dengan train_pusht.py, namun:
- Config DIKUNCI dari checkpoint (tidak bisa diubah)
- Model & optimizer di-load dari checkpoint
- Lanjut training dari epoch berikutnya
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

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pusht_dataset import (
    build_pusht_dataloader,
    split_train_val_dataset,
    normalize_data,
)
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine


# ============================================================
# Load Locked Config dari Checkpoint
# ============================================================

def load_locked_config(
    ckpt_path: Path,
) -> Tuple[Dict[str, Any], Dict, Dict, Dict, Dict, int, float]:
    """
    Load checkpoint dan return
    (config_dict, model_state, optimizer_state, action_stats, state_stats,
     start_epoch, ckpt_val_loss).
    Config DIKUNCI - tidak boleh diubah.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config_dict = ckpt["config"]
    model_state = ckpt["model_state_dict"]
    optimizer_state = ckpt["optimizer_state_dict"]
    action_stats = ckpt["action_stats"]
    state_stats = ckpt["state_stats"]
    start_epoch = ckpt["epoch"]
    ckpt_val_loss = float(ckpt["val_loss"])

    print(f"[Locked] Loaded checkpoint: {ckpt_path}")
    print(
        f"[Locked] Epoch: {start_epoch}, "
        f"Train Loss: {ckpt['train_loss']:.6f}, "
        f"Val Loss: {ckpt_val_loss:.6f}"
    )
    print(f"[Locked] Config keys: {list(config_dict.keys())}")

    return (
        config_dict,
        model_state,
        optimizer_state,
        action_stats,
        state_stats,
        start_epoch,
        ckpt_val_loss,
    )


# ============================================================
# Locked TrainConfig (immutable dari checkpoint)
# ============================================================

@dataclass(frozen=True)
class LockedTrainConfig:
    """Config yang DIKUNCI dari checkpoint - tidak bisa diubah."""
    # Dataset (locked)
    zarr_path: str
    pred_horizon: int
    obs_horizon: int
    action_horizon: int
    val_ratio: float
    seed: int

    # Model (locked)
    input_dim: int
    global_cond_dim: int
    diffusion_step_embed_dim: int
    down_dims: Tuple[int, ...]
    kernel_size: int
    n_groups: int

    # Diffusion (locked)
    num_diffusion_steps: int
    noise_schedule_s: float

    # Training (locked - kecuali epochs & output_dir yang boleh extend)
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    device: str

    # Logging (locked)
    log_interval: int
    save_interval: int

    # Normalization stats (locked)
    action_stats: Dict
    state_stats: Dict

    # Extended untuk resume
    resume_epochs: int = 70
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion_locked"

    @classmethod
    def from_checkpoint(cls, config_dict: Dict, **overrides) -> "LockedTrainConfig":
        """Factory dari checkpoint config dict."""
        if isinstance(config_dict.get("down_dims"), list):
            config_dict = {**config_dict, "down_dims": tuple(config_dict["down_dims"])}
        merged = {**config_dict, **overrides}
        return cls(**merged)

    def to_train_config(self) -> "TrainConfig":
        """Convert ke TrainConfig mutable untuk training loop."""
        locked_fields = {
            k: v
            for k, v in asdict(self).items()
            if k not in ("resume_epochs", "run_name")
        }
        return TrainConfig(**locked_fields)


@dataclass
class TrainConfig:
    """Mutable config untuk training loop (copy dari locked)."""
    zarr_path: str
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    val_ratio: float = 0.1
    seed: int = 42
    input_dim: int = 2
    global_cond_dim: int = 5
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
    device: str = "cpu"
    log_interval: int = 10
    save_interval: int = 10
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion"
    action_stats: Dict = None
    state_stats: Dict = None
    # Fine-tuning scheduler
    finetune_lr_scale: float = 0.1   # LR awal resume = lr * scale
    scheduler_eta_min_scale: float = 0.01


# ============================================================
# Loss & Evaluation
# ============================================================

def diffusion_loss(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
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
def evaluate(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x_0 = batch["action"].to(device)
        cond = batch["state"][:, 0].to(device)

        loss = diffusion_loss(model, noise_sched, x_0, cond, device)
        total_loss += loss.item() * x_0.size(0)
        total_samples += x_0.size(0)

    model.train()
    return total_loss / total_samples if total_samples > 0 else float("inf")


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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config.__dict__,
            "action_stats": action_stats,
            "state_stats": state_stats,
        },
        path,
    )
    print(f"[Checkpoint] Saved: {path}")


# ============================================================
# Main Resume Training Loop
# ============================================================

def train_resume(
    locked_config: LockedTrainConfig,
    config: TrainConfig,
    start_epoch: int,
    model_state: Optional[Dict] = None,
    optimizer_state: Optional[Dict] = None,
    best_val_loss_init: float = float("inf"),
) -> ConditionalUnet1D:
    device = torch.device(config.device)
    print(f"\n[Resume Train] Device: {device}")
    print(f"[Resume Train] Starting from epoch {start_epoch + 1} to {config.epochs}")

    # --- Dataset (pakai locked stats) ---
    print("\n[Data] Loading dataset with locked normalization stats...")
    train_ds, val_ds = split_train_val_dataset(
        zarr_path=config.zarr_path,
        val_ratio=config.val_ratio,
        seed=config.seed,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )

    # Override stats dengan locked stats
    train_ds.action_stats = locked_config.action_stats
    train_ds.state_stats = locked_config.state_stats
    val_ds.action_stats = locked_config.action_stats
    val_ds.state_stats = locked_config.state_stats

    # Normalisasi train & val dengan LOCKED stats (masing-masing pakai datanya sendiri)
    train_data = {
        "state": train_ds.state_all.astype(np.float32),
        "action": train_ds.action_all.astype(np.float32),
    }
    train_ds.normalized_train_data = {
        "state": normalize_data(train_data["state"], locked_config.state_stats).astype(np.float32),
        "action": normalize_data(train_data["action"], locked_config.action_stats).astype(np.float32),
    }

    val_data = {
        "state": val_ds.state_all.astype(np.float32),
        "action": val_ds.action_all.astype(np.float32),
    }
    val_ds.normalized_train_data = {
        "state": normalize_data(val_data["state"], locked_config.state_stats).astype(np.float32),
        "action": normalize_data(val_data["action"], locked_config.action_stats).astype(np.float32),
    }

    train_loader = build_pusht_dataloader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    val_loader = build_pusht_dataloader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    print(f"[Data] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"[Data] Batches/epoch: {len(train_loader)}")

    # --- Model (locked architecture) ---
    print("\n[Model] Building ConditionalUnet1D with locked architecture...")
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

    # --- Noise Schedule (locked) ---
    noise_sched = NoiseScheduleCosine(
        T=config.num_diffusion_steps,
        s=config.noise_schedule_s,
        device=device,
    )

    # --- Optimizer (locked hyperparams) ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # --- Load Locked Weights & Optimizer ---
    if model_state is not None:
        model.load_state_dict(model_state)
        print("[Resume] Model weights loaded from locked checkpoint.")
    else:
        print("[Warning] model_state is None — model starts from random init!")

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # Pindahkan state tensor ke device yang benar
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("[Resume] Optimizer state loaded from locked checkpoint.")

    # ============================================================
    # FIX SCHEDULER: fine-tuning LR (bukan restart dari LR awal)
    # ============================================================
    remaining_epochs = max(1, config.epochs - start_epoch)
    finetune_lr = config.lr * config.finetune_lr_scale
    for g in optimizer.param_groups:
        g["lr"] = finetune_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs,
        eta_min=finetune_lr * config.scheduler_eta_min_scale,
    )
    print(
        f"[Scheduler] Fine-tuning LR = {finetune_lr:.2e} "
        f"(scale={config.finetune_lr_scale}), "
        f"T_max={remaining_epochs}, "
        f"eta_min={finetune_lr * config.scheduler_eta_min_scale:.2e}"
    )

    # --- Output dir ---
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints -> {output_dir}")

    # ============================================================
    # FIX best_val_loss: inisialisasi dari checkpoint (bukan inf)
    # ============================================================
    best_val_loss = best_val_loss_init
    print(f"[Resume] best_val_loss initialized to {best_val_loss:.6f}")

    # --- Training Loop (resume) ---
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
            cond = batch["state"][:, 0].to(device)

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
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if epoch % config.save_interval == 0 or val_loss < best_val_loss:
            ckpt_path = (
                output_dir
                / f"epoch_{epoch:04d}_train_{avg_train_loss:.4f}_val_{val_loss:.4f}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, val_loss,
                config, locked_config.action_stats, locked_config.state_stats, ckpt_path,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, epoch, avg_train_loss, val_loss,
                    config, locked_config.action_stats, locked_config.state_stats, best_path,
                )
                print(f"[Best] New best val loss: {best_val_loss:.6f}")

    # Save final
    final_path = output_dir / "final.pt"
    save_checkpoint(
        model, optimizer, config.epochs,
        history["train_loss"][-1], history["val_loss"][-1],
        config, locked_config.action_stats, locked_config.state_stats, final_path,
    )

    print(f"\n[Resume Train] Done! Best val loss: {best_val_loss:.6f}")
    print(f"[Resume Train] Checkpoints saved to: {output_dir}")

    return model


# ============================================================
# Entry Point
# ============================================================

def main():
    REPO_ROOT = Path(__file__).parent
    LOCKED_CKPT = (
        REPO_ROOT
        / "checkpoints"
        / "pusht_state_diffusion"
        / "best.pt"
    )

    if not LOCKED_CKPT.exists():
        print(f"[Error] Locked checkpoint not found: {LOCKED_CKPT}")
        sys.exit(1)

    # 1. Load locked config & states
    (
        locked_config_dict,
        model_state,
        optimizer_state,
        action_stats,
        state_stats,
        start_epoch,
        ckpt_val_loss,
    ) = load_locked_config(LOCKED_CKPT)

    # 2. Create locked config (immutable)
    EXCLUDE_FIELDS = ("run_name", "epochs", "action_stats", "state_stats")
    checkpoint_config = {
        k: v for k, v in locked_config_dict.items() if k not in EXCLUDE_FIELDS
    }

    locked_config = LockedTrainConfig.from_checkpoint(
        checkpoint_config,
        action_stats=action_stats,
        state_stats=state_stats,
        resume_epochs=80,
        run_name="pusht_state_diffusion",
    )

    # 3. Create mutable train config (copy dari locked)
    config = locked_config.to_train_config()
    config.epochs = locked_config.resume_epochs
    config.run_name = "pusht_state_diffusion"

    # 4. Resume training
    train_resume(
        locked_config,
        config,
        start_epoch,
        model_state=model_state,
        optimizer_state=optimizer_state,
        best_val_loss_init=ckpt_val_loss,   # <-- inisialisasi dari checkpoint
    )


if __name__ == "__main__":
    main()