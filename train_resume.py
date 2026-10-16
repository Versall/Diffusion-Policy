"""
Resume training dari checkpoint (gaya repo asli).

- Load model + optimizer + EMA state
- Lanjut training dari epoch berikutnya
- Checkpoint periodik, tanpa validation
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
    LinearNormalizer,
)
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine
from train_pusht import EMAModel, diffusion_loss, save_checkpoint


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class ResumeConfig:
    ckpt_path: str
    zarr_path: str
    target_epochs: int = 70
    lr_scale: float = 0.1        # fine-tune LR = base_lr × lr_scale
    save_interval: int = 10
    device: Optional[str] = None


def train_resume(cfg: ResumeConfig):
    device = get_device(prefer=cfg.device)
    print(summary())
    print(f"[Resume] Device: {describe_device(device)}")

    # --- Load checkpoint ---
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    locked_config = ckpt["config"]
    start_epoch = ckpt["epoch"]
    normalizer_state = ckpt.get("normalizer_state", None)

    print(f"[Resume] Loaded: {ckpt_path}")
    print(f"[Resume] Epoch: {start_epoch} | Train loss: {ckpt['train_loss']:.6f}")
    print(f"[Resume] Config keys: {list(locked_config.keys())}")

    # --- Dataset (rekonstruksi normalizer dari checkpoint) ---
    data_cfg = PushTDataConfig(
        dataset_path=cfg.zarr_path,
        obs_horizon=locked_config["obs_horizon"],
        pred_horizon=locked_config["pred_horizon"],
        obs_dim=locked_config["obs_dim"],
        action_dim=locked_config["input_dim"],
        val_ratio=0.0,
        seed=locked_config["seed"],
    )

    normalizer = None
    if normalizer_state is not None:
        normalizer = LinearNormalizer(mode=normalizer_state["mode"])
        normalizer.load_state_dict(normalizer_state)
        print("[Resume] Normalizer restored from checkpoint.")

    # Rebuild dataset & pakai normalizer locked
    train_ds, _ = split_train_val_dataset(data_cfg)
    train_ds.normalizer = normalizer

    train_loader = build_pusht_dataloader(
        train_ds,
        batch_size=locked_config["batch_size"],
        shuffle=True,
        num_workers=locked_config["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Resume] Train samples: {len(train_ds)} | Batches/epoch: {len(train_loader)}")

    # --- Model ---
    model = ConditionalUnet1D(
        input_dim=locked_config["input_dim"],
        global_cond_dim=locked_config["global_cond_dim"],
        diffusion_step_embed_dim=locked_config["diffusion_step_embed_dim"],
        down_dims=list(locked_config["down_dims"]),
        kernel_size=locked_config["kernel_size"],
        n_groups=locked_config["n_groups"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[Resume] Model loaded.")

    # --- EMA ---
    ema = None
    if locked_config.get("use_ema", False):
        ema = EMAModel(model, decay=locked_config.get("ema_decay", 0.999))
        if "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            print("[Resume] EMA state loaded.")
        else:
            print("[Resume] Warning: use_ema=True tapi checkpoint tidak punya ema_state.")

    # --- Noise schedule ---
    noise_sched = NoiseScheduleCosine(
        T=locked_config["num_diffusion_steps"],
        s=locked_config["noise_schedule_s"],
        device=device,
    )

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=locked_config["lr"],
        weight_decay=locked_config["weight_decay"],
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("[Resume] Optimizer state loaded.")

    # Fine-tune LR
    remaining = max(1, cfg.target_epochs - start_epoch)
    finetune_lr = locked_config["lr"] * cfg.lr_scale
    for g in optimizer.param_groups:
        g["lr"] = finetune_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=finetune_lr * 0.01,
    )
    print(f"[Resume] Fine-tune LR: {finetune_lr:.2e}, T_max={remaining}")

    # --- Output ---
    output_dir = Path(locked_config["output_dir"]) / locked_config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Loop ---
    print(f"\n[Resume] Epoch {start_epoch + 1} → {cfg.target_epochs}")
    history = {"train_loss": [], "lr": []}

    for epoch in range(start_epoch + 1, cfg.target_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.target_epochs}", leave=False)
        for batch in pbar:
            x_0 = batch["action"].to(device, non_blocking=True)
            cond = batch["obs"].reshape(batch["obs"].shape[0], -1).to(device, non_blocking=True)

            loss = diffusion_loss(model, noise_sched, x_0, cond, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), locked_config.get("grad_clip", 1.0))
            optimizer.step()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{cfg.target_epochs} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"time: {time.time() - epoch_start:.1f}s"
        )

        # Checkpoint periodik
        if epoch % cfg.save_interval == 0:
            save_checkpoint(
                model, optimizer, ema, epoch, avg_train_loss,
                locked_config, normalizer_state,
                output_dir / f"epoch_{epoch:04d}.pt",
            )
            if ema is not None:
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                save_checkpoint(
                    ema_model, optimizer, None, epoch, avg_train_loss,
                    locked_config, normalizer_state,
                    output_dir / f"epoch_{epoch:04d}_ema.pt",
                )

        # last.pt setiap epoch
        save_checkpoint(
            model, optimizer, ema, epoch, avg_train_loss,
            locked_config, normalizer_state, output_dir / "last.pt",
        )

    save_checkpoint(
        model, optimizer, ema, cfg.target_epochs, history["train_loss"][-1],
        locked_config, normalizer_state, output_dir / "final.pt",
    )

    print(f"\n[Resume] Done! Final train loss: {history['train_loss'][-1]:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path ke checkpoint (biasanya last.pt atau epoch_NNNN.pt)")
    parser.add_argument("--target-epochs", type=int, default=70)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--zarr", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    zarr_path = (
        Path(args.zarr).resolve() if args.zarr
        else Path(ckpt["config"]["zarr_path"]).resolve()
    )

    cfg = ResumeConfig(
        ckpt_path=str(ckpt_path),
        zarr_path=str(zarr_path),
        target_epochs=args.target_epochs,
        lr_scale=args.lr_scale,
        save_interval=args.save_interval,
        device=args.device,
    )
    train_resume(cfg)


if __name__ == "__main__":
    main()