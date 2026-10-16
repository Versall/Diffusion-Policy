"""
Resume training dari checkpoint (gaya repo asli).

- Load model + optimizer + EMA state + normalizer state
- Lanjut training dari epoch berikutnya
- Checkpoint periodik, tanpa validation
- Kompatibel dengan train_pusht.py terbaru (keypoint 20D, EMA, no val)

Cara pakai:
    uv run python train_resume.py --ckpt checkpoints/pusht_state_diffusion/last.pt --target-epochs 70
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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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

# Import helper dari train_pusht.py (single source of truth)
from train_pusht import EMAModel, diffusion_loss, save_checkpoint


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
# Shim: bikin dict punya .__dict__ agar save_checkpoint kompatibel
# ============================================================
class _ConfigShim:
    """
    Wrapper tipis agar dict bisa dipakai di save_checkpoint(),
    yang mengakses `config.__dict__`.

    Dipakai karena `ckpt["config"]` adalah dict, bukan dataclass.
    """
    def __init__(self, d: dict):
        self.__dict__.update(d)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__


# ============================================================
# Konfigurasi Resume
# ============================================================
@dataclass
class ResumeConfig:
    ckpt_path: str
    zarr_path: str
    target_epochs: int = 70
    lr_scale: float = 0.1          # fine-tune LR = base_lr × lr_scale
    save_interval: int = 10
    device: Optional[str] = None


# ============================================================
# Resume Training
# ============================================================
def train_resume(cfg: ResumeConfig):
    device = get_device(prefer=cfg.device)
    print(summary())
    print(f"[Resume] Device: {describe_device(device)}")

    # ---------------------------------------------
    # 1. Load checkpoint
    # ---------------------------------------------
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    locked_config_dict = ckpt["config"]
    start_epoch = ckpt["epoch"]
    normalizer_state = ckpt.get("normalizer_state", None)

    # Bungkus agar save_checkpoint bisa akses .__dict__
    locked_config = _ConfigShim(locked_config_dict)

    print(f"[Resume] Loaded: {ckpt_path}")
    print(f"[Resume] Epoch: {start_epoch} | Train loss: {ckpt['train_loss']:.6f}")
    print(f"[Resume] Config keys: {list(locked_config_dict.keys())}")

    # ---------------------------------------------
    # 2. Dataset dengan normalizer LOCKED
    # ---------------------------------------------
    obs_dim = locked_config.get("obs_dim", 20)          # fallback 20 (keypoint)
    input_dim = locked_config.get("input_dim", 2)
    obs_horizon = locked_config.get("obs_horizon", 2)
    pred_horizon = locked_config.get("pred_horizon", 16)
    global_cond_dim = locked_config.get(
        "global_cond_dim", obs_horizon * obs_dim
    )

    data_cfg = PushTDataConfig(
        dataset_path=cfg.zarr_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        obs_dim=obs_dim,
        action_dim=input_dim,
        val_ratio=0.0,           # tidak pakai val
        seed=locked_config.get("seed", 42),
    )

    # Restore normalizer dari checkpoint
    normalizer = None
    if normalizer_state is not None:
        normalizer = LinearNormalizer(mode=normalizer_state["mode"])
        normalizer.load_state_dict(normalizer_state)
        print("[Resume] Normalizer restored from checkpoint.")
    else:
        print("[Resume] Warning: checkpoint tidak menyimpan normalizer_state.")

    # Rebuild dataset & override normalizer
    train_ds, _ = split_train_val_dataset(data_cfg)
    train_ds.normalizer = normalizer

    batch_size = locked_config.get("batch_size", 64)
    num_workers = locked_config.get("num_workers", 0)

    train_loader = build_pusht_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Resume] Train samples: {len(train_ds)} | Batches/epoch: {len(train_loader)}")

    # Sanity check dimensi
    sample = train_ds[0]
    assert sample["obs"].shape == (obs_horizon, obs_dim), \
        f"obs shape salah: {sample['obs'].shape}, expected ({obs_horizon}, {obs_dim})"
    assert sample["action"].shape == (pred_horizon, input_dim), \
        f"action shape salah: {sample['action'].shape}"

    # ---------------------------------------------
    # 3. Model (arsitektur LOCKED)
    # ---------------------------------------------
    model = ConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=locked_config.get("diffusion_step_embed_dim", 256),
        down_dims=list(locked_config.get("down_dims", (256, 512, 1024))),
        kernel_size=locked_config.get("kernel_size", 3),
        n_groups=locked_config.get("n_groups", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Resume] Model loaded. global_cond_dim = {global_cond_dim}")

    # ---------------------------------------------
    # 4. EMA (kalau ada)
    # ---------------------------------------------
    ema = None
    if locked_config.get("use_ema", False):
        ema = EMAModel(model, decay=locked_config.get("ema_decay", 0.999))
        if "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            # Pindah shadow ke device yang benar
            ema.shadow = {k: v.to(device) for k, v in ema.shadow.items()}
            print("[Resume] EMA state loaded.")
        else:
            print("[Resume] Warning: use_ema=True tapi checkpoint tidak punya ema_state.")
    else:
        print("[Resume] EMA disabled (sesuai config checkpoint).")

    # ---------------------------------------------
    # 5. Noise Schedule
    # ---------------------------------------------
    noise_sched = NoiseScheduleCosine(
        T=locked_config.get("num_diffusion_steps", 100),
        s=locked_config.get("noise_schedule_s", 0.008),
        device=device,
    )

    # ---------------------------------------------
    # 6. Optimizer + Scheduler
    # ---------------------------------------------
    base_lr = locked_config.get("lr", 1e-4)
    weight_decay = locked_config.get("weight_decay", 1e-4)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Pindah state tensor ke device yang benar
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("[Resume] Optimizer state loaded.")

    # Fine-tune LR (bukan restart dari LR awal)
    remaining = max(1, cfg.target_epochs - start_epoch)
    finetune_lr = base_lr * cfg.lr_scale
    for g in optimizer.param_groups:
        g["lr"] = finetune_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining,
        eta_min=finetune_lr * 0.01,
    )
    print(f"[Resume] Fine-tune LR: {finetune_lr:.2e} (base={base_lr:.2e}, scale={cfg.lr_scale})")
    print(f"[Resume] Scheduler: T_max={remaining}, eta_min={finetune_lr * 0.01:.2e}")

    # ---------------------------------------------
    # 7. Output dir
    # ---------------------------------------------
    output_dir = Path(locked_config.get("output_dir", "checkpoints")) / \
                 locked_config.get("run_name", "pusht_state_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Resume] Checkpoints -> {output_dir}")

    # ---------------------------------------------
    # 8. Training loop (tanpa validation)
    # ---------------------------------------------
    print(f"\n[Resume] Epoch {start_epoch + 1} → {cfg.target_epochs}")
    grad_clip = locked_config.get("grad_clip", 1.0)
    history = {"train_loss": [], "lr": []}

    for epoch in range(start_epoch + 1, cfg.target_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.target_epochs}",
            leave=False,
        )
        for batch in pbar:
            x_0 = batch["action"].to(device, non_blocking=True)
            cond = batch["obs"].reshape(batch["obs"].shape[0], -1).to(device, non_blocking=True)

            loss = diffusion_loss(model, noise_sched, x_0, cond, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Update EMA setiap step
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{cfg.target_epochs} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"time: {epoch_time:.1f}s"
        )

        # ---------------------------------------------
        # Checkpoint periodik
        # ---------------------------------------------
        if epoch % cfg.save_interval == 0:
            # Simpan model reguler
            save_checkpoint(
                model, optimizer, ema, epoch, avg_train_loss,
                locked_config, normalizer_state,
                output_dir / f"epoch_{epoch:04d}.pt",
            )

            # Simpan versi EMA (biasanya lebih stabil)
            if ema is not None:
                ema_model = copy.deepcopy(model)
                ema.copy_to(ema_model)
                save_checkpoint(
                    ema_model, optimizer, None, epoch, avg_train_loss,
                    locked_config, normalizer_state,
                    output_dir / f"epoch_{epoch:04d}_ema.pt",
                )

        # Simpan last.pt setiap epoch (untuk resume berikutnya)
        save_checkpoint(
            model, optimizer, ema, epoch, avg_train_loss,
            locked_config, normalizer_state,
            output_dir / "last.pt",
        )

    # ---------------------------------------------
    # 9. Final checkpoint
    # ---------------------------------------------
    save_checkpoint(
        model, optimizer, ema, cfg.target_epochs, history["train_loss"][-1],
        locked_config, normalizer_state,
        output_dir / "final.pt",
    )

    print(f"\n[Resume] Done! Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"[Resume] Checkpoints: {output_dir}")
    print(f"[Resume] Untuk evaluasi success rate:")
    print(f"         uv run python eval_policy.py --ckpt {output_dir / f'epoch_{cfg.target_epochs:04d}_ema.pt'} --episodes 50")

    return model


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Resume training dari checkpoint")
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path ke checkpoint (biasanya last.pt atau epoch_NNNN.pt)",
    )
    parser.add_argument(
        "--target-epochs", type=int, default=70,
        help="Total epoch target (harus > epoch di checkpoint)",
    )
    parser.add_argument(
        "--lr-scale", type=float, default=0.1,
        help="Fine-tune LR = base_lr × lr_scale. Default 0.1",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Simpan checkpoint setiap N epoch",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cpu', 'cuda', atau None (auto-detect)",
    )
    parser.add_argument(
        "--zarr", type=str, default=None,
        help="Path dataset .zarr. Default: dari config checkpoint",
    )
    args = parser.parse_args()

    # ---------------------------------------------
    # Resolve path
    # ---------------------------------------------
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint tidak ditemukan: {ckpt_path}")
        sys.exit(1)

    # Peek checkpoint (CPU) untuk ambil zarr_path default
    ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if args.zarr:
        zarr_path = Path(args.zarr).resolve()
    else:
        zarr_path = Path(ckpt_peek["config"].get(
            "zarr_path",
            "data/pusht/pusht_cchi_v7_replay.zarr",
        )).resolve()

    if not zarr_path.exists():
        print(f"[Error] Dataset tidak ditemukan: {zarr_path}")
        sys.exit(1)

    # Validasi target_epochs
    start_epoch = ckpt_peek["epoch"]
    if args.target_epochs <= start_epoch:
        print(f"[Error] target_epochs ({args.target_epochs}) harus > "
              f"epoch di checkpoint ({start_epoch})")
        sys.exit(1)

    # ---------------------------------------------
    # Jalankan resume
    # ---------------------------------------------
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