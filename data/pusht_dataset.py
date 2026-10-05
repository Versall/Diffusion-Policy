"""
Push-T Dataset Loader untuk Diffusion Policy.

Dataset: pusht_cchi_v7_replay.zarr (state-based, low-dim)
Struktur data:
  - data/state: (N, 5) = [agent_x, agent_y, block_x, block_y, block_angle]
  - data/action: (N, 2) = [target_x, target_y]
  - meta/episode_ends: (E,) indeks akhir tiap episode

Referensi arsitektur:
  - real-stanford/diffusion_policy (pusht_image_dataset.py, replay_buffer.py, sampler.py)
  - qlOoOlp/Diffusion-Policy-Tutorial (pusht_dataset.py, utils.py, pusht_builder.py)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utilitas Sampling & Normalisasi (adaptasi dari tutorial)
# ============================================================

def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """
    Buat indeks sliding-window per episode dengan padding.

    Args:
        episode_ends: shape (E,) - indeks akhir tiap episode (exclusive)
        sequence_length: panjang sequence prediksi (pred_horizon)
        pad_before: padding di awal (obs_horizon - 1)
        pad_after: padding di akhir (action_horizon - 1)

    Returns:
        indices: shape (num_samples, 4) = [buffer_start, buffer_end, sample_start, sample_end]
    """
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices, dtype=np.int64)


def sample_sequence(
    train_data: Dict[str, np.ndarray],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, np.ndarray]:
    """
    Ekstrak sequence dari buffer dengan padding edge-replication.
    """
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        else:
            data = sample
        result[key] = data
    return result


def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Hitung statistik min/max per dimensi untuk normalisasi [0,1] -> [-1,1]."""
    data = data.reshape(-1, data.shape[-1])
    return {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
    }


def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalisasi ke [-1, 1]."""
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    return ndata * 2 - 1


def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Denormalisasi dari [-1, 1] ke skala asli."""
    ndata = (ndata + 1) / 2
    return ndata * (stats['max'] - stats['min'] + 1e-8) + stats['min']


# ============================================================
# Dataset Class
# ============================================================

@dataclass
class PushTDataConfig:
    """Konfigurasi dataset Push-T."""
    pred_horizon: int = 16      # panjang chunk action yang diprediksi (T_a)
    obs_horizon: int = 2        # jumlah step observasi kondisi (T_o)
    action_horizon: int = 8     # jumlah action yang dieksekusi per step inference
    # Catatan: di training, pred_horizon = sequence_length
    # obs_horizon digunakan untuk menentukan pad_before
    # action_horizon digunakan untuk pad_after (biasanya action_horizon-1)


class PushTStateDataset(Dataset):
    """
    Dataset Push-T state-based (low-dim) untuk Diffusion Policy.

    Output __getitem__:
        dict dengan keys:
        - 'state': (obs_horizon, 5) - observasi state kondisi (normalized)
        - 'action': (pred_horizon, 2) - target action chunk (normalized)

    Catatan: state[..., :2] = agent position, state[..., 2:] = block pose
    """
    def __init__(
        self,
        zarr_path: str,
        config: Optional[PushTDataConfig] = None,
        # normalisasi stats (jika None, dihitung dari data)
        action_stats: Optional[Dict[str, np.ndarray]] = None,
        state_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.config = config or PushTDataConfig()
        self.zarr_path = zarr_path

        # --- load zarr (zarr v3 API) ---
        dataset_root = zarr.open(store=zarr_path, mode='r')

        # data mentah
        self.state_all = dataset_root['data']['state'][:]      # (N, 5)
        self.action_all = dataset_root['data']['action'][:]    # (N, 2)
        self.episode_ends = dataset_root['meta']['episode_ends'][:]

        # --- buat sample indices ---
        # pad_before = obs_horizon - 1 (kondisi state dari step sebelumnya)
        # pad_after = action_horizon - 1 (action yang dieksekusi setelah prediksi)
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.config.pred_horizon,
            pad_before=self.config.obs_horizon - 1,
            pad_after=self.config.action_horizon - 1,
        )

        # --- normalisasi ---
        # state: gunakan 5D penuh (agent_pos + block_pose)
        # action: 2D (target position)
        train_data = {
            'state': self.state_all.astype(np.float32),
            'action': self.action_all.astype(np.float32),
        }

        if action_stats is None:
            action_stats = get_data_stats(train_data['action'])
        if state_stats is None:
            state_stats = get_data_stats(train_data['state'])

        self.action_stats = action_stats
        self.state_stats = state_stats

        normalized_train_data = {
            'state': normalize_data(train_data['state'], state_stats).astype(np.float32),
            'action': normalize_data(train_data['action'], action_stats).astype(np.float32),
        }

        self.normalized_train_data = normalized_train_data

        print(f"[PushTStateDataset] Loaded {len(self.action_all)} steps, "
              f"{len(self.episode_ends)} episodes, "
              f"{len(self.indices)} samples")
        print(f"  pred_horizon={self.config.pred_horizon}, "
              f"obs_horizon={self.config.obs_horizon}, "
              f"action_horizon={self.config.action_horizon}")
        print(f"  action_stats: min={action_stats['min']}, max={action_stats['max']}")
        print(f"  state_stats:  min={state_stats['min']}, max={state_stats['max']}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.config.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Hanya ambil obs_horizon pertama untuk state (kondisi)
        # action: full pred_horizon sebagai target
        result = {
            'state': torch.from_numpy(nsample['state'][:self.config.obs_horizon]),  # (T_o, 5)
            'action': torch.from_numpy(nsample['action']),                          # (T_a, 2)
        }
        return result

    def get_normalization_stats(self) -> Tuple[Dict, Dict]:
        """Return (action_stats, state_stats) untuk denormalisasi saat inferensi."""
        return self.action_stats, self.state_stats


# ============================================================
# Builder Functions
# ============================================================

def build_pusht_dataset(
    zarr_path: str,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    action_stats: Optional[Dict[str, np.ndarray]] = None,
    state_stats: Optional[Dict[str, np.ndarray]] = None,
) -> PushTStateDataset:
    """
    Factory function untuk membuat PushTStateDataset.

    Args:
        zarr_path: path ke file .zarr (mis. data/pusht/pusht_cchi_v7_replay.zarr)
        pred_horizon: panjang action chunk yang diprediksi
        obs_horizon: jumlah step observasi sebagai kondisi
        action_horizon: jumlah action dieksekusi per inference step
        action_stats, state_stats: opsional, untuk shared stats (mis. val set pakai stats train)
    """
    config = PushTDataConfig(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    return PushTStateDataset(
        zarr_path=zarr_path,
        config=config,
        action_stats=action_stats,
        state_stats=state_stats,
    )


def build_pusht_dataloader(
    dataset: PushTStateDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    device: Optional[torch.device] = None,
) -> DataLoader:
    """
    Buat DataLoader untuk training.

    Args:
        dataset: PushTStateDataset instance
        batch_size: batch size
        num_workers: num workers (0 = main process, >0 = multiprocessing)
        shuffle: shuffle data
        device: device target (untuk pin_memory)
    """
    pin_memory = device is not None and device.type == 'cuda'

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,  # biar batch size konsisten
    )


# ============================================================
# Validation Split Helper
# ============================================================

def split_train_val_dataset(
    zarr_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
) -> Tuple[PushTStateDataset, PushTStateDataset]:
    """
    Split dataset jadi train/val berdasarkan episode (bukan frame).
    
    FIX data leakage:
    - Normalisasi stats (min/max) dihitung HANYA dari episode TRAIN.
    - Stats train dipakai untuk normalize train DAN val.
    - Val set tidak pernah "melihat" statistiknya sendiri.
    """
    # ============================================================
    # 1. Load raw data + episode boundaries
    # ============================================================
    dataset_root = zarr.open(store=zarr_path, mode='r')
    episode_ends = dataset_root['meta']['episode_ends'][:]
    state_all = dataset_root['data']['state'][:].astype(np.float32)
    action_all = dataset_root['data']['action'][:].astype(np.float32)
    n_episodes = len(episode_ends)

    # ============================================================
    # 2. Split episode jadi train/val
    # ============================================================
    n_val = max(1, int(n_episodes * val_ratio))
    rng = np.random.default_rng(seed=seed)
    val_episode_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_episode_idxs] = True
    train_mask = ~val_mask

    # ============================================================
    # 3. Kumpulkan frame dari episode TRAIN saja (untuk stats)
    # ============================================================
    train_frame_mask = np.zeros(len(state_all), dtype=bool)
    for i in range(n_episodes):
        if not val_mask[i]:
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            end_idx = episode_ends[i]
            train_frame_mask[start_idx:end_idx] = True

    train_state = state_all[train_frame_mask]
    train_action = action_all[train_frame_mask]

    # ============================================================
    # 4. Hitung stats HANYA dari frame TRAIN
    # ============================================================
    action_stats = get_data_stats(train_action)
    state_stats = get_data_stats(train_state)

    print(f"[Normalization] Stats computed from TRAIN ONLY "
          f"({int(train_frame_mask.sum())}/{len(state_all)} frames)")
    print(f"  action_stats: min={action_stats['min']}, max={action_stats['max']}")
    print(f"  state_stats:  min={state_stats['min']},  max={state_stats['max']}")

    # ============================================================
    # 5. Normalize SEMUA frame pakai stats train-only
    #    (val frames dinormalisasi dengan stats train — benar)
    # ============================================================
    normalized_data = {
        'state': normalize_data(state_all, state_stats).astype(np.float32),
        'action': normalize_data(action_all, action_stats).astype(np.float32),
    }

    # ============================================================
    # 6. Build indices train & val
    # ============================================================
    train_indices = []
    val_indices = []

    for i in range(n_episodes):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -(obs_horizon - 1)
        max_start = episode_length - pred_horizon + (action_horizon - 1)

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + pred_horizon, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + pred_horizon + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = pred_horizon - end_offset

            sample_info = [buffer_start_idx, buffer_end_idx,
                           sample_start_idx, sample_end_idx]

            if val_mask[i]:
                val_indices.append(sample_info)
            else:
                train_indices.append(sample_info)

    # ============================================================
    # 7. Build datasets (pakai __new__ agar tidak recompute stats)
    # ============================================================
    config = PushTDataConfig(pred_horizon, obs_horizon, action_horizon)

    train_ds = PushTStateDataset.__new__(PushTStateDataset)
    train_ds.config = config
    train_ds.zarr_path = zarr_path
    train_ds.state_all = state_all
    train_ds.action_all = action_all
    train_ds.episode_ends = episode_ends
    train_ds.indices = np.array(train_indices, dtype=np.int64)
    train_ds.action_stats = action_stats
    train_ds.state_stats = state_stats
    train_ds.normalized_train_data = normalized_data

    val_ds = PushTStateDataset.__new__(PushTStateDataset)
    val_ds.config = config
    val_ds.zarr_path = zarr_path
    val_ds.state_all = state_all
    val_ds.action_all = action_all
    val_ds.episode_ends = episode_ends
    val_ds.indices = np.array(val_indices, dtype=np.int64)
    val_ds.action_stats = action_stats
    val_ds.state_stats = state_stats
    val_ds.normalized_train_data = normalized_data

    print(f"[Train] {len(train_ds)} samples from {int(train_mask.sum())} episodes")
    print(f"[Val]   {len(val_ds)} samples from {int(val_mask.sum())} episodes")

    return train_ds, val_ds


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    # Path dataset
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ZARR_PATH = os.path.join(REPO_ROOT, "data", "pusht", "pusht_cchi_v7_replay.zarr")

    if not os.path.exists(ZARR_PATH):
        print(f"Dataset not found at {ZARR_PATH}")
        print("Run: python data/get_data.py  (download & unzip)")
        exit(1)

    print("=" * 60)
    print("TEST PushTStateDataset")
    print("=" * 60)

    # Test basic dataset
    dataset = build_pusht_dataset(ZARR_PATH, pred_horizon=16, obs_horizon=2, action_horizon=8)
    print(f"\nDataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  state shape: {sample['state'].shape}")   # (2, 5)
    print(f"  action shape: {sample['action'].shape}") # (16, 2)
    print(f"  state[0]: {sample['state'][0]}")
    print(f"  action[0]: {sample['action'][0]}")

    # Test dataloader
    print("\n--- DataLoader test ---")
    dl = build_pusht_dataloader(dataset, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    print(f"Batch state: {batch['state'].shape}")   # (4, 2, 5)
    print(f"Batch action: {batch['action'].shape}") # (4, 16, 2)

    # Test train/val split
    print("\n--- Train/Val Split test ---")
    train_ds, val_ds = split_train_val_dataset(ZARR_PATH, val_ratio=0.1)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Train sample 0 state: {train_ds[0]['state'].shape}")
    print(f"Val sample 0 state: {val_ds[0]['state'].shape}")

    print("\n=== All tests passed ===")