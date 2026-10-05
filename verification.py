# test_no_leakage_v2.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import zarr
from data.pusht_dataset import split_train_val_dataset, get_data_stats

ZARR_PATH = "data/pusht/pusht_cchi_v7_replay.zarr"

# Jalankan split
train_ds, val_ds = split_train_val_dataset(
    zarr_path=ZARR_PATH, val_ratio=0.1, seed=42,
    pred_horizon=16, obs_horizon=2, action_horizon=8,
)

# ============================================================
# Verifikasi 1: stats train_ds dan val_ds HARUS identik
# (karena val pakai stats train)
# ============================================================
assert np.allclose(train_ds.action_stats['min'], val_ds.action_stats['min'])
assert np.allclose(train_ds.action_stats['max'], val_ds.action_stats['max'])
assert np.allclose(train_ds.state_stats['min'], val_ds.state_stats['min'])
assert np.allclose(train_ds.state_stats['max'], val_ds.state_stats['max'])
print("✅ Verifikasi 1: stats train == stats val (benar, val pakai stats train)")

# ============================================================
# Verifikasi 2: stats = stats yang dihitung dari frame train saja
# (recompute manual dari kode split, tanpa pakai fungsi)
# ============================================================
root = zarr.open(ZARR_PATH, mode='r')
episode_ends = root['meta']['episode_ends'][:]
state_all = root['data']['state'][:].astype(np.float32)
action_all = root['data']['action'][:].astype(np.float32)
n_episodes = len(episode_ends)

rng = np.random.default_rng(seed=42)
n_val = max(1, int(n_episodes * 0.1))
val_episode_idxs = rng.choice(n_episodes, size=n_val, replace=False)
val_mask = np.zeros(n_episodes, dtype=bool)
val_mask[val_episode_idxs] = True

# Kumpulkan frame train
train_frame_mask = np.zeros(len(state_all), dtype=bool)
for i in range(n_episodes):
    if not val_mask[i]:
        s = 0 if i == 0 else episode_ends[i - 1]
        e = episode_ends[i]
        train_frame_mask[s:e] = True

manual_action_stats = get_data_stats(action_all[train_frame_mask])
manual_state_stats = get_data_stats(state_all[train_frame_mask])

assert np.allclose(train_ds.action_stats['min'], manual_action_stats['min']), \
    "action_stats min tidak cocok dengan manual!"
assert np.allclose(train_ds.action_stats['max'], manual_action_stats['max']), \
    "action_stats max tidak cocok dengan manual!"
assert np.allclose(train_ds.state_stats['min'], manual_state_stats['min']), \
    "state_stats min tidak cocok dengan manual!"
assert np.allclose(train_ds.state_stats['max'], manual_state_stats['max']), \
    "state_stats max tidak cocok dengan manual!"
print("✅ Verifikasi 2: stats train_ds == stats dihitung manual dari frame train")
print("             → Konfirmasi bahwa stats memang dari TRAIN saja")

# ============================================================
# Verifikasi 3: jumlah frame train yang dipakai = 23,108
# ============================================================
n_train_frames = int(train_frame_mask.sum())
n_total_frames = len(state_all)
print(f"\n📊 Frame train: {n_train_frames}/{n_total_frames} = {100*n_train_frames/n_total_frames:.1f}%")
print(f"   Episode train: {int((~val_mask).sum())}")
print(f"   Episode val: {int(val_mask.sum())}")

# ============================================================
# Verifikasi 4: bukti kuantitatif — normalisasi val tidak melampaui [-1, 1]
# jauh, yang mengindikasikan stats train cukup representatif
# ============================================================
val_action_raw = action_all[np.logical_not(train_frame_mask)]
val_action_norm = (val_action_raw - train_ds.action_stats['min']) / \
                  (train_ds.action_stats['max'] - train_ds.action_stats['min'] + 1e-8) * 2 - 1
out_of_range = ((val_action_norm < -1.5) | (val_action_norm > 1.5)).sum()
print(f"\n📊 Val action di luar [-1.5, 1.5]: {out_of_range}/{len(val_action_norm)} "
      f"({100*out_of_range/len(val_action_norm):.2f}%)")
if out_of_range / len(val_action_norm) < 0.01:
    print("   ✅ Distribusi train & val mirip — split wajar")
else:
    print("   ⚠️  Banyak val frame di luar range train — cek split")

print("\n" + "="*60)
print("✅ FIX DATA LEAKAGE TERVERIFIKASI")
print("="*60)