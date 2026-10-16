"""
Dataset Push-T low-dimensional untuk Diffusion Policy.

Representasi observasi: keypoint + agent_pos (sesuai repo asli)
- keypoint: (N, 9, 2) → flatten → (N, 18)
- agent_pos: state[:, :2] → (N, 2)
- obs: (N, 20)

Referensi repo asli:
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/dataset/pusht_dataset.py

Output __getitem__:
- 'obs':    (obs_horizon, 20)
- 'action': (pred_horizon, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import zarr


# =========================================================
# Konfigurasi
# =========================================================

@dataclass
class PushTDataConfig:
    """Konfigurasi dataset Push-T."""
    dataset_path: str = "data/pusht/pusht_cchi_v7_replay.zarr"

    # Horizon
    obs_horizon: int = 2      # To
    pred_horizon: int = 16    # Ta

    # Dimensi
    obs_dim: int = 20         # 18 keypoint + 2 agent_pos
    action_dim: int = 2

    # Normalisasi
    normalize: bool = True
    norm_mode: str = "min_max"

    # Split
    val_ratio: float = 0.1
    seed: int = 42


# =========================================================
# Normalizer
# =========================================================

class LinearNormalizer:
    """Normalizer linear (min_max / identity). Stats dari train only (K1)."""

    def __init__(self, mode: str = "min_max", eps: float = 1e-6):
        assert mode in ("min_max", "identity"), f"Mode tidak dikenal: {mode}"
        self.mode = mode
        self.eps = eps
        self.stats = {}

    def fit(self, data: np.ndarray, key: str):
        if self.mode == "identity":
            self.stats[key] = {
                "min": np.zeros(data.shape[-1], dtype=np.float32),
                "max": np.ones(data.shape[-1], dtype=np.float32),
            }
            return
        d_min = data.min(axis=0).astype(np.float32)
        d_max = data.max(axis=0).astype(np.float32)
        self.stats[key] = {"min": d_min, "max": d_max}

    def normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        if not self.stats:
            raise RuntimeError("Normalizer belum di-fit.")
        s = self.stats[key]
        rng = np.maximum(s["max"] - s["min"], self.eps)
        return (2.0 * (data - s["min"]) / rng - 1.0).astype(np.float32)

    def unnormalize(self, data: np.ndarray, key: str) -> np.ndarray:
        s = self.stats[key]
        rng = np.maximum(s["max"] - s["min"], self.eps)
        return ((data + 1.0) * 0.5 * rng + s["min"]).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mode": self.mode, "eps": self.eps, "stats": self.stats}

    def load_state_dict(self, d: dict):
        self.mode = d["mode"]
        self.eps = d["eps"]
        self.stats = d["stats"]


# =========================================================
# Helper: bentuk obs dari Zarr
# =========================================================

def _build_obs_from_zarr(root, indices=None):
    """
    Bentuk obs (keypoint + agent_pos) dari Zarr.

    Returns:
        obs_all: (N, 20)
        action_all: (N, 2)
    """
    state_all    = root["data"]["state"][:]       # (N, 5)
    keypoint_all = root["data"]["keypoint"][:]    # (N, 9, 2)
    action_all   = root["data"]["action"][:]      # (N, 2)

    N = state_all.shape[0]

    # keypoint: (N, 9, 2) → (N, 18)
    keypoint_flat = keypoint_all.reshape(N, -1).astype(np.float32)

    # agent_pos: state[:, :2] → (N, 2)
    agent_pos = state_all[:, :2].astype(np.float32)

    # concat → (N, 20)
    obs_all = np.concatenate([keypoint_flat, agent_pos], axis=-1)

    return obs_all, action_all.astype(np.float32)


# =========================================================
# Dataset
# =========================================================

class PushTDataset(Dataset):
    """
    Dataset Push-T low-dim (keypoint + agent_pos).

    obs: keypoint (18,) + agent_pos (2,) = (20,)
    """

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        normalize: bool = True,
        norm_mode: str = "min_max",
        normalizer: Optional[LinearNormalizer] = None,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalize = normalize
        self.norm_mode = norm_mode

        root = zarr.open(dataset_path, mode="r")
        self.obs_all, self.action_all = _build_obs_from_zarr(root)
        self.N = self.obs_all.shape[0]

        # Normalizer
        if self.normalize:
            if normalizer is not None:
                self.normalizer = normalizer
            else:
                self.normalizer = LinearNormalizer(mode=norm_mode)
                if indices is not None:
                    self.normalizer.fit(self.obs_all[indices], "obs")
                    self.normalizer.fit(self.action_all[indices], "action")
                else:
                    self.normalizer.fit(self.obs_all, "obs")
                    self.normalizer.fit(self.action_all, "action")
        else:
            self.normalizer = None

        # Index valid sliding window
        self.valid_start = np.arange(
            0, self.N - obs_horizon - pred_horizon + 1, dtype=np.int64
        )
        if indices is not None:
            self.valid_start = np.intersect1d(self.valid_start, indices)

    def __len__(self) -> int:
        return len(self.valid_start)

    def __getitem__(self, idx: int) -> dict:
        start = self.valid_start[idx]

        obs_seq = self.obs_all[start : start + self.obs_horizon]
        act_start = start + self.obs_horizon - 1
        act_seq = self.action_all[act_start : act_start + self.pred_horizon]

        if self.normalize and self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq, "obs")
            act_seq = self.normalizer.normalize(act_seq, "action")

        return {
            "obs":    torch.from_numpy(obs_seq).float(),
            "action": torch.from_numpy(act_seq).float(),
        }


# =========================================================
# Builder & split
# =========================================================

def build_pusht_dataset(cfg: PushTDataConfig) -> PushTDataset:
    return PushTDataset(
        dataset_path=cfg.dataset_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        normalize=cfg.normalize,
        norm_mode=cfg.norm_mode,
    )


def split_train_val_dataset(
    cfg: PushTDataConfig,
) -> Tuple[PushTDataset, PushTDataset]:
    """
    Split train/val pada level frame (bukan window).
    Stats normalisasi di-fit dari train only (K1).
    """
    root = zarr.open(cfg.dataset_path, mode="r")
    N = root["data"]["state"].shape[0]

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N)
    n_val = int(N * cfg.val_ratio)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    # Fit normalizer dari train only
    normalizer = None
    if cfg.normalize:
        obs_all, action_all = _build_obs_from_zarr(root)
        normalizer = LinearNormalizer(mode=cfg.norm_mode)
        normalizer.fit(obs_all[train_idx], "obs")
        normalizer.fit(action_all[train_idx], "action")

        print(
            f"[split] Train frames: {len(train_idx)} ({len(train_idx)/N:.1%}) | "
            f"Val frames: {len(val_idx)} ({len(val_idx)/N:.1%})"
        )

    train_ds = PushTDataset(
        dataset_path=cfg.dataset_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        normalize=cfg.normalize,
        norm_mode=cfg.norm_mode,
        normalizer=normalizer,
        indices=train_idx,
    )
    val_ds = PushTDataset(
        dataset_path=cfg.dataset_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        normalize=cfg.normalize,
        norm_mode=cfg.norm_mode,
        normalizer=normalizer,
        indices=val_idx,
    )
    return train_ds, val_ds


def build_pusht_dataloader(
    dataset: PushTDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# =========================================================
# Sanity check
# =========================================================

if __name__ == "__main__":
    cfg = PushTDataConfig()
    print("=" * 60)
    print("Sanity check PushTDataset (keypoint + agent_pos, 20D)")
    print("=" * 60)

    train_ds, val_ds = split_train_val_dataset(cfg)
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")

    sample = train_ds[0]
    obs, act = sample["obs"], sample["action"]
    print(f"obs shape:    {tuple(obs.shape)}   (harusnya (2, 20))")
    print(f"action shape: {tuple(act.shape)}   (harusnya (16, 2))")

    assert obs.shape == (cfg.obs_horizon, 20), f"obs shape salah: {obs.shape}"
    assert act.shape == (cfg.pred_horizon, 2), f"action shape salah: {act.shape}"

    print("\n[OK] Sanity check passed.")