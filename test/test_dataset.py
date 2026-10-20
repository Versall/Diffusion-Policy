"""
Tes B: verifikasi dataset loader menghasilkan shape yang benar.
Jalankan: uv run python test_dataset.py
"""
from data.pusht_dataset import PushTDataConfig, split_train_val_dataset


def main():
    cfg = PushTDataConfig(val_ratio=0.0)
    train_ds, _ = split_train_val_dataset(cfg)

    print(f"Dataset length: {len(train_ds)}")
    print(f"obs_all shape:    {train_ds.obs_all.shape}      (harus (N, 20))")
    print(f"action_all shape: {train_ds.action_all.shape}   (harus (N, 2))")

    s = train_ds[0]
    obs, act = s["obs"], s["action"]

    print()
    print(f"obs shape:    {tuple(obs.shape)}      (harus (2, 20))")
    print(f"action shape: {tuple(act.shape)}      (harus (16, 2))")
    print(f"obs dtype:    {obs.dtype}              (harus torch.float32)")
    print(f"action dtype: {act.dtype}              (harus torch.float32)")
    print(f"obs range:    [{obs.min().item():.3f}, {obs.max().item():.3f}]")
    print(f"action range: [{act.min().item():.3f}, {act.max().item():.3f}]")

    # Cek NaN / Inf
    import torch
    assert torch.isfinite(obs).all(), "FATAL: obs mengandung NaN/Inf"
    assert torch.isfinite(act).all(), "FATAL: action mengandung NaN/Inf"

    # Cek shape
    assert obs.shape == (2, 20), f"FATAL: obs shape {obs.shape}, harus (2, 20)"
    assert act.shape == (16, 2), f"FATAL: action shape {act.shape}, harus (16, 2)"

    # Cek normalizer ada
    assert train_ds.normalizer is not None, "FATAL: normalizer None"

    stats_obs = train_ds.normalizer.stats["obs"]
    print()
    print(f"Normalizer mode: {train_ds.normalizer.mode}")
    print(f"obs stats min (5 pertama): {stats_obs['min'][:5]}")
    print(f"obs stats max (5 pertama): {stats_obs['max'][:5]}")

    print()
    print("[OK] Dataset loader benar. Siap training.")


if __name__ == "__main__":
    main()
    