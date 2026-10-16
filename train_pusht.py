"""
Training Diffusion Policy Push-T (keypoint + agent_pos, 20D).

Fitur:
- Full CLI configuration (semua parameter TrainConfig dapat di-override)
- Resume training (--resume / --auto-resume)
- EMA
- Checkpoint periodik + last + final
- Support config file YAML/JSON (--config)
- Mixed precision (--amp) dengan API torch.amp (PyTorch 2.x)

Contoh:
    # Training default
    python train_pusht.py --epochs 100

    # Training dengan konfigurasi custom
    python train_pusht.py \\
        --epochs 200 --batch-size 128 --lr 3e-4 \\
        --down-dims 256 512 1024 --num-diffusion-steps 100 \\
        --use-ema --ema-decay 0.9999

    # Resume
    python train_pusht.py --epochs 200 --auto-resume

    # Load config dari YAML
    python train_pusht.py --config configs/pusht.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
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
# Reproducibility
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
    # --- Dataset ---
    zarr_path: str = "data/pusht/pusht_cchi_v7_replay.zarr"
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    seed: int = 42

    # --- Dimensi ---
    obs_dim: int = 20
    input_dim: int = 2
    global_cond_dim: int = 40

    # --- Model ---
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 3
    n_groups: int = 8

    # --- Diffusion ---
    num_diffusion_steps: int = 100
    noise_schedule_s: float = 0.008

    # --- Optimizer ---
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip: float = 0.0
    optimizer: str = "adamw"           # "adamw" | "adam" | "sgd"
    scheduler: str = "cosine"           # "cosine" | "constant" | "step"
    eta_min_ratio: float = 0.01

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999

    # --- Device ---
    device: Optional[str] = None
    amp: bool = False

    # --- Logging / Checkpoint ---
    log_interval: int = 20
    save_interval: int = 10
    output_dir: str = "checkpoints"
    run_name: str = "pusht_state_diffusion"

    # --- Resume ---
    resume: Optional[str] = None
    auto_resume: bool = False

    # --- Debug ---
    debug: bool = False
    deterministic: bool = True


# ============================================================
# EMA
# ============================================================
class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
    x_0: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B = x_0.shape[0]
    T = noise_sched.T
    t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = noise_sched.q_sample(x_0, t, noise)
    eps_pred = model(x_t.permute(0, 2, 1), t, cond)
    return F.mse_loss(eps_pred.permute(0, 2, 1), noise)


# ============================================================
# Checkpoint Save / Load
# ============================================================
def save_checkpoint(
    model: ConditionalUnet1D,
    optimizer: optim.Optimizer,
    ema: Optional[EMAModel],
    epoch: int,
    train_loss: float,
    config,
    normalizer_state: Optional[Dict],
    path: Path,
):
    """
    Simpan checkpoint (gaya repo asli).

    - `config` bisa dataclass (TrainConfig) atau dict (dari resume)
    - `ema` opsional; kalau ada, shadow disimpan sebagai 'ema_state'
    - Model state selalu di CPU untuk portabilitas
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Handle dataclass ATAU dict
    config_dict = config.__dict__ if hasattr(config, "__dict__") else config

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "config": config_dict,
        "normalizer_state": normalizer_state,
    }

    if ema is not None:
        ema_state = {
            "decay": ema.decay,
            "shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        }
        ckpt["ema_state"] = ema_state

    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved: {path}")
    

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    ema: Optional[EMAModel],
    device: torch.device,
) -> Tuple[int, int, float]:
    print(f"[Resume] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ema is not None and "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)
    last_loss = ckpt.get("train_loss", float("nan"))
    print(
        f"[Resume] Resumed from epoch {ckpt['epoch']} | "
        f"global_step={global_step} | last_loss={last_loss:.6f}"
    )
    return start_epoch, global_step, last_loss


# ============================================================
# Optimizer / Scheduler Factory
# ============================================================
def build_optimizer(model: nn.Module, config: TrainConfig) -> optim.Optimizer:
    name = config.optimizer.lower()
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return optim.SGD(
            model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay
        )
    raise ValueError(f"Optimizer tidak dikenal: {config.optimizer}")


def build_scheduler(
    optimizer: optim.Optimizer, config: TrainConfig
) -> optim.lr_scheduler._LRScheduler:
    name = config.scheduler.lower()
    eta_min = config.lr * config.eta_min_ratio
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=eta_min
        )
    if name == "constant":
        return optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=config.epochs
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, config.epochs // 5), gamma=0.5
        )
    raise ValueError(f"Scheduler tidak dikenal: {config.scheduler}")


# ============================================================
# Normalizer State Helper
# ============================================================
def _extract_normalizer_state(dataset) -> Optional[Dict]:
    """
    Ambil state dict dari dataset.normalizer, mendukung:
      - dict {'obs': LinearNormalizer, 'action': LinearNormalizer, ...}
      - objek tunggal LinearNormalizer
      - None
    """
    normalizer = getattr(dataset, "normalizer", None)
    if normalizer is None:
        return None

    # Kasus 1: dict of LinearNormalizer
    if isinstance(normalizer, dict):
        out = {}
        for k, v in normalizer.items():
            out[k] = v.state_dict() if hasattr(v, "state_dict") else v
        return out

    # Kasus 2: objek tunggal dengan state_dict()
    if hasattr(normalizer, "state_dict"):
        return normalizer.state_dict()

    # Kasus 3: sudah berupa dict biasa
    return normalizer


# ============================================================
# Training
# ============================================================
def train(config: TrainConfig) -> ConditionalUnet1D:
    set_seed(config.seed, deterministic=config.deterministic)
    device = get_device(prefer=config.device)
    print(summary())
    print(f"[Train] Device: {describe_device(device)} | Seed: {config.seed}")
    print(f"[Train] Config: {config}")

    # --- Data ---
    print("\n[Data] Loading dataset (keypoint + agent_pos, 20D)...")
    data_cfg = PushTDataConfig(
        dataset_path=config.zarr_path,
        obs_horizon=config.obs_horizon,
        pred_horizon=config.pred_horizon,
        obs_dim=config.obs_dim,
        action_dim=config.input_dim,
        val_ratio=0.0,
        seed=config.seed,
    )
    train_ds, _ = split_train_val_dataset(data_cfg)
    print(f"[Data] Train samples: {len(train_ds)}")

    sample = train_ds[0]
    assert sample["obs"].shape == (config.obs_horizon, config.obs_dim), sample["obs"].shape
    assert sample["action"].shape == (config.pred_horizon, config.input_dim), sample["action"].shape

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

    base_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    normalizer_state = _extract_normalizer_state(base_ds)

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
        T=config.num_diffusion_steps, s=config.noise_schedule_s, device=device
    )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    print(f"[Optim] {config.optimizer} | lr={config.lr} | wd={config.weight_decay}")
    print(f"[Sched] {config.scheduler} | eta_min_ratio={config.eta_min_ratio}")

    # --- AMP (API baru PyTorch 2.x) ---
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if config.amp:
        print(f"[AMP] Enabled={use_amp} (device={device.type})")

    # --- Output ---
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Checkpoints -> {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # --- Resume ---
    start_epoch = 1
    global_step = 0
    resume_path = config.resume
    if resume_path is None and config.auto_resume:
        auto = output_dir / "last.pt"
        if auto.exists():
            resume_path = str(auto)
            print(f"[Auto-Resume] Ditemukan: {resume_path}")

    if resume_path is not None:
        start_epoch, global_step, _ = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, ema, device
        )
        if start_epoch > config.epochs:
            print(f"[Resume] Sudah selesai ({start_epoch - 1} >= {config.epochs}).")
            return model

    # --- Loop ---
    epochs_to_run = config.epochs
    if config.debug:
        epochs_to_run = 1
        print("[Debug] Mode debug: hanya 1 epoch, 3 step.")

    print(f"\n[Train] Starting epochs {start_epoch}..{epochs_to_run}...")
    history = {"train_loss": [], "lr": []}

    for epoch in range(start_epoch, epochs_to_run + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs_to_run}", leave=False)
        for batch in pbar:
            x_0 = batch["action"].to(device, non_blocking=True)
            cond = batch["obs"].reshape(batch["obs"].shape[0], -1).to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion_loss(model, noise_sched, x_0, cond, device)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if num_batches % config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            if config.debug and num_batches >= 3:
                break

        avg_train_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{epochs_to_run} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"time: {time.time() - epoch_start:.1f}s"
        )

        # Checkpoint periodik
        if epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, ema, epoch, global_step,
                avg_train_loss, config, normalizer_state,
                output_dir / f"epoch_{epoch:04d}.pt",
            )
            if ema is not None:
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                save_checkpoint(
                    ema_model, optimizer, scheduler, None, epoch, global_step,
                    avg_train_loss, config, normalizer_state,
                    output_dir / f"epoch_{epoch:04d}_ema.pt",
                )

        # last.pt
        save_checkpoint(
            model, optimizer, scheduler, ema, epoch, global_step,
            avg_train_loss, config, normalizer_state,
            output_dir / "last.pt",
        )

    # Final
    final_path = output_dir / "final.pt"
    save_checkpoint(
        model, optimizer, scheduler, ema, epochs_to_run, global_step,
        history["train_loss"][-1] if history["train_loss"] else float("nan"),
        config, normalizer_state, final_path,
    )

    print(f"\n[Train] Done! Final loss: {history['train_loss'][-1]:.6f}")
    print(f"[Train] Checkpoints: {output_dir}")
    return model


# ============================================================
# Config dari file (YAML/JSON)
# ============================================================
def _load_config_file(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("Install PyYAML: pip install pyyaml")
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    if p.suffix == ".json":
        with open(p, "r") as f:
            return json.load(f)
    raise ValueError(f"Format config tidak didukung: {p.suffix}")


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy Push-T (keypoint 20D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path ke YAML/JSON config (opsional, override default)")

    # Dataset
    g = parser.add_argument_group("Dataset")
    g.add_argument("--zarr", type=str, default=None, help="Path ke .zarr dataset")
    g.add_argument("--pred-horizon", type=int, default=None)
    g.add_argument("--obs-horizon", type=int, default=None)
    g.add_argument("--action-horizon", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--num-workers", type=int, default=None)

    # Dimensi
    g = parser.add_argument_group("Dimensions")
    g.add_argument("--obs-dim", type=int, default=None)
    g.add_argument("--input-dim", type=int, default=None)
    g.add_argument("--global-cond-dim", type=int, default=None)

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument("--diffusion-step-embed-dim", type=int, default=None)
    g.add_argument("--down-dims", type=int, nargs="+", default=None,
                   help="Channel U-Net (contoh: 256 512 1024)")
    g.add_argument("--kernel-size", type=int, default=None)
    g.add_argument("--n-groups", type=int, default=None)

    # Diffusion
    g = parser.add_argument_group("Diffusion")
    g.add_argument("--num-diffusion-steps", type=int, default=None)
    g.add_argument("--noise-schedule-s", type=float, default=None)

    # Optimizer
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--grad-clip", type=float, default=None)
    g.add_argument("--optimizer", type=str, choices=["adamw", "adam", "sgd"], default=None)
    g.add_argument("--scheduler", type=str, choices=["cosine", "constant", "step"], default=None)
    g.add_argument("--eta-min-ratio", type=float, default=None)

    # EMA
    g = parser.add_argument_group("EMA")
    g.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    g.add_argument("--no-ema", dest="use_ema", action="store_false")
    g.add_argument("--ema-decay", type=float, default=None)

    # Device / perf
    g = parser.add_argument_group("Device & Performance")
    g.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    g.add_argument("--amp", action="store_true", default=None, help="Mixed precision")

    # Logging & Checkpoint
    g = parser.add_argument_group("Logging & Checkpoint")
    g.add_argument("--log-interval", type=int, default=None)
    g.add_argument("--save-interval", type=int, default=None)
    g.add_argument("--output-dir", type=str, default=None)
    g.add_argument("--run-name", type=str, default=None)

    # Resume
    g = parser.add_argument_group("Resume")
    g.add_argument("--resume", type=str, default=None,
                   help="Path ke checkpoint .pt")
    g.add_argument("--auto-resume", action="store_true", default=None,
                   help="Resume otomatis dari <output_dir>/<run_name>/last.pt")

    # Debug
    g = parser.add_argument_group("Debug")
    g.add_argument("--debug", action="store_true", default=None,
                   help="Debug: 1 epoch, 3 step")
    g.add_argument("--no-deterministic", dest="deterministic",
                   action="store_false", default=None)

    return parser


def _override_if_not_none(cfg: TrainConfig, key: str, value):
    """Set cfg.key = value jika value bukan None."""
    if value is not None:
        setattr(cfg, key, value)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Mulai dari default
    config = TrainConfig()

    # 2. Override dari config file (jika ada)
    if args.config:
        file_cfg = _load_config_file(args.config)
        for k, v in file_cfg.items():
            if not hasattr(config, k):
                print(f"[Warn] Config file key tidak dikenal: {k}")
                continue
            if k == "down_dims" and isinstance(v, list):
                v = tuple(v)
            setattr(config, k, v)
        print(f"[Config] Loaded from: {args.config}")

    # 3. Override dari CLI (prioritas tertinggi)
    cli_map = {
        "zarr_path": args.zarr,
        "pred_horizon": args.pred_horizon,
        "obs_horizon": args.obs_horizon,
        "action_horizon": args.action_horizon,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "obs_dim": args.obs_dim,
        "input_dim": args.input_dim,
        "global_cond_dim": args.global_cond_dim,
        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "num_diffusion_steps": args.num_diffusion_steps,
        "noise_schedule_s": args.noise_schedule_s,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "eta_min_ratio": args.eta_min_ratio,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "device": args.device,
        "amp": args.amp,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "resume": args.resume,
        "auto_resume": args.auto_resume,
        "debug": args.debug,
        "deterministic": args.deterministic,
    }
    for k, v in cli_map.items():
        _override_if_not_none(config, k, v)

    # down_dims (list[int] → tuple)
    if args.down_dims is not None:
        config.down_dims = tuple(args.down_dims)

    # 4. Resolve zarr path
    if args.zarr:
        zarr_path = Path(args.zarr).resolve()
    else:
        zarr_path = Path(config.zarr_path)
        if not zarr_path.is_absolute():
            zarr_path = (Path(__file__).resolve().parent / zarr_path).resolve()
    config.zarr_path = str(zarr_path)

    if not zarr_path.exists():
        print(f"[Error] Dataset tidak ditemukan: {zarr_path}")
        sys.exit(1)

    # 5. Jalankan training
    train(config)


if __name__ == "__main__":
    main()