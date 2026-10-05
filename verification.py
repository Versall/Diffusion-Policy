"""
Verifikasi K2: Conditioning To=2 (flatten obs_horizon × state_dim)
===================================================================

Verifikasi ini memastikan:
  1. Dataset menghasilkan state (B, To=2, D=5) — benar
  2. Flatten state → (B, 10) — benar
  3. Model ConditionalUnet1D dengan global_cond_dim=10 menerima (B, 10)
  4. Loss bisa dihitung & backward (gradient mengalir)
  5. Checkpoint baru (jika ada) menyimpan global_cond_dim=10
  6. DiffusionPolicy inference dengan checkpoint baru berfungsi
  7. Bandingkan: model To=1 (5) vs To=2 (10) — parameter count

Cara pakai:
    python verify_k2.py                    # cek dasar tanpa checkpoint
    python verify_k2.py --ckpt path.pt     # cek dengan checkpoint tertentu
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# Konfigurasi
# ============================================================
ZARR_PATH = "data/pusht/pusht_cchi_v7_replay.zarr"
EXPECTED_STATE_DIM = 5
EXPECTED_OBS_HORIZON = 2
EXPECTED_GLOBAL_COND_DIM = EXPECTED_OBS_HORIZON * EXPECTED_STATE_DIM   # 10


# ============================================================
# Helper: print header
# ============================================================
def header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str):
    print(f"  ✅ {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def info(msg: str):
    print(f"     {msg}")


# ============================================================
# TEST 1: Dataset menghasilkan state (B, To, D) dengan To=2
# ============================================================
def test_1_dataset_shape():
    header("TEST 1: Dataset state shape (B, To, D)")

    from data.pusht_dataset import split_train_val_dataset

    train_ds, val_ds = split_train_val_dataset(
        zarr_path=ZARR_PATH, val_ratio=0.1, seed=42,
        pred_horizon=16, obs_horizon=EXPECTED_OBS_HORIZON, action_horizon=8,
    )

    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    state = batch["state"]
    action = batch["action"]

    info(f"state shape : {tuple(state.shape)}")
    info(f"action shape: {tuple(action.shape)}")

    assert state.shape == (4, EXPECTED_OBS_HORIZON, EXPECTED_STATE_DIM), \
        f"state shape salah: {state.shape}, expected (4, {EXPECTED_OBS_HORIZON}, {EXPECTED_STATE_DIM})"
    assert action.shape == (4, 16, 2), \
        f"action shape salah: {action.shape}, expected (4, 16, 2)"

    ok(f"State berbentuk (B, {EXPECTED_OBS_HORIZON}, {EXPECTED_STATE_DIM})")
    ok(f"Action berbentuk (B, 16, 2)")

    # Cek isi: dua frame state harus berbeda (kalau sama, obs_horizon tidak berfungsi)
    frame_0 = state[0, 0]
    frame_1 = state[0, 1]
    diff = (frame_0 - frame_1).abs().sum().item()
    info(f"Perbedaan frame 0 vs 1 (sample pertama): {diff:.4f}")
    if diff < 1e-6:
        print("     ⚠️  Frame 0 == Frame 1 — kemungkinan padding di awal episode")

    return batch


# ============================================================
# TEST 2: Flatten state (B, To, D) → (B, To*D)
# ============================================================
def test_2_flatten(batch):
    header("TEST 2: Flatten state → conditioning")

    state = batch["state"]                                   # (4, 2, 5)
    B = state.shape[0]

    cond = state.reshape(B, -1)                              # (4, 10)

    info(f"state shape     : {tuple(state.shape)}")
    info(f"flattened shape : {tuple(cond.shape)}")

    assert cond.shape == (B, EXPECTED_GLOBAL_COND_DIM), \
        f"flattened shape salah: {cond.shape}"

    # Verifikasi: flatten harus preserve semua nilai (tidak ada yang hilang)
    # state[0, 0] = [a, b, c, d, e]; state[0, 1] = [f, g, h, i, j]
    # cond[0]     = [a, b, c, d, e, f, g, h, i, j]
    s0 = state[0, 0].tolist()
    s1 = state[0, 1].tolist()
    c0 = cond[0].tolist()

    assert np.allclose(c0[:5], s0), "Bagian pertama flatten != frame 0"
    assert np.allclose(c0[5:], s1), "Bagian kedua flatten != frame 1"

    ok(f"Flatten (B, {EXPECTED_OBS_HORIZON}, {EXPECTED_STATE_DIM}) → "
       f"(B, {EXPECTED_GLOBAL_COND_DIM})")
    ok("Konten flatten preserve frame 0 dan frame 1")

    return cond


# ============================================================
# TEST 3: Model menerima conditioning (B, 10)
# ============================================================
def test_3_model_forward(batch, cond):
    header("TEST 3: Model forward dengan global_cond_dim=10")

    from models.mlp_noise_head import ConditionalUnet1D
    from models.noise_schedule import NoiseScheduleCosine

    device = torch.device("cpu")
    sched = NoiseScheduleCosine(T=100, s=0.008, device=device)

    model = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=EXPECTED_GLOBAL_COND_DIM,          # 10
        diffusion_step_embed_dim=256,
        down_dims=(64, 128, 256),
        kernel_size=5,
        n_groups=8,
    ).to(device)

    x_t = batch["action"].permute(0, 2, 1).to(device)      # (4, 2, 16)
    t = torch.randint(1, 101, (4,), device=device)

    info(f"x_t shape  : {tuple(x_t.shape)}")
    info(f"cond shape : {tuple(cond.shape)}")

    eps_pred = model(x_t, t, cond)
    info(f"output shape: {tuple(eps_pred.shape)}")

    assert eps_pred.shape == (4, 2, 16), \
        f"Output shape salah: {eps_pred.shape}"

    ok("Model menerima cond (B, 10) & output (B, 2, 16)")
    ok("Arsitektur ConditionalUnet1D kompatibel dengan K2")

    return model, sched, x_t, t, cond


# ============================================================
# TEST 4: Loss & gradient flow
# ============================================================
def test_4_loss_gradient(model, sched, x_t, t, cond):
    header("TEST 4: Loss & gradient flow")

    from models.mlp_noise_head import ConditionalUnet1D

    # Bangun noise schedule & q_sample
    noise = torch.randn_like(x_t)
    x_t_noisy = sched.q_sample(x_t, t, noise)

    # Forward
    eps_pred = model(x_t_noisy, t, cond)
    loss = F.mse_loss(eps_pred, noise)

    info(f"Loss value: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Cek gradient ada di parameter conditioning
    # cond_encoder adalah layer yang menerima conditioning → harus dapat gradient
    grad_count = 0
    grad_norm_total = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
            grad_norm_total += p.grad.norm().item() ** 2

    grad_norm_total = grad_norm_total ** 0.5
    info(f"Parameter dengan gradient: {grad_count}")
    info(f"Total gradient norm     : {grad_norm_total:.6f}")

    assert grad_count > 0, "Tidak ada parameter yang dapat gradient!"
    assert grad_norm_total > 0, "Gradient norm nol — tidak ada flow!"

    ok("Loss bisa backward")
    ok("Gradient mengalir ke parameter")


# ============================================================
# TEST 5: Bandingkan To=1 vs To=2 (parameter count)
# ============================================================
def test_5_compare_param_count():
    header("TEST 5: Bandingkan arsitektur To=1 vs To=2")

    from models.mlp_noise_head import ConditionalUnet1D

    common_kwargs = dict(
        input_dim=2,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),                         # pakai default paper
        kernel_size=3,
        n_groups=8,
    )

    model_to1 = ConditionalUnet1D(global_cond_dim=5, **common_kwargs)
    model_to2 = ConditionalUnet1D(global_cond_dim=10, **common_kwargs)

    n1 = sum(p.numel() for p in model_to1.parameters())
    n2 = sum(p.numel() for p in model_to2.parameters())
    diff = n2 - n1

    info(f"To=1 (global_cond_dim=5) : {n1:,} parameter")
    info(f"To=2 (global_cond_dim=10): {n2:,} parameter")
    info(f"Delta                     : +{diff:,} parameter")
    info(f"Rasio                     : {n2/n1:.4f}×")

    # Delta harus kecil — hanya Linear layer pertama di cond_encoder yang berubah
    # cond_encoder: Linear(cond_dim, out_channels*2) di setiap ResBlock
    # Total ResBlock = 2(down L1) + 2(down L2) + 2(down L3) + 2(mid) + 2(up L1) + 2(up L2) = 12
    # Setiap Linear(cond_dim, C*2) → delta = 5 * C*2
    # Tapi kita tidak hitung manual — cukup cek delta < 1% dari total
    delta_ratio = diff / n1
    info(f"Delta ratio               : {delta_ratio*100:.4f}%")

    assert delta_ratio < 0.01, \
        f"Delta terlalu besar ({delta_ratio*100:.2f}%) — mungkin ada layer lain yang berubah"

    ok("Perbedaan parameter minimal (<1%)")
    ok("K2 hanya mengubah input cond_encoder, bukan arsitektur utama")


# ============================================================
# TEST 6: Cek checkpoint (jika ada)
# ============================================================
def test_6_checkpoint(ckpt_path: str):
    header(f"TEST 6: Cek checkpoint — {ckpt_path}")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"  ⏸  Checkpoint tidak ada: {ckpt_path}")
        print(f"     Retrain dulu dengan `python train_pusht.py`")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    info(f"Epoch     : {ckpt['epoch']}")
    info(f"Val loss  : {ckpt['val_loss']:.6f}")
    info(f"Train loss: {ckpt['train_loss']:.6f}")

    global_cond_dim = config.get("global_cond_dim")
    info(f"global_cond_dim : {global_cond_dim}")

    if global_cond_dim != EXPECTED_GLOBAL_COND_DIM:
        print(f"  ❌ Checkpoint masih pakai global_cond_dim={global_cond_dim} "
              f"(expected {EXPECTED_GLOBAL_COND_DIM})")
        print(f"     → Checkpoint ini dari To=1, tidak kompatibel dengan K2")
        return None

    ok(f"Checkpoint menggunakan global_cond_dim={EXPECTED_GLOBAL_COND_DIM}")

    # Cek state_dict
    state_dict = ckpt["model_state_dict"]
    # Cari layer cond_encoder pertama
    cond_enc_keys = [k for k in state_dict.keys() if "cond_encoder" in k and "weight" in k]
    if cond_enc_keys:
        first_key = cond_enc_keys[0]
        shape = state_dict[first_key].shape
        info(f"Contoh cond_encoder weight: {first_key}")
        info(f"  shape = {tuple(shape)}")
        # Format: Linear(cond_dim, out_channels*2) → weight (out*2, cond_dim)
        # Jadi kolom terakhir = cond_dim
        expected_in = EXPECTED_GLOBAL_COND_DIM + 256   # karena ada time embedding (dsed=256)
        info(f"  in_features = {shape[1]} (expected {expected_in} = "
             f"{EXPECTED_GLOBAL_COND_DIM} + 256 time_emb)")

    return ckpt


# ============================================================
# TEST 7: Policy inference (jika checkpoint ada)
# ============================================================
def test_7_policy_inference(ckpt_path: str):
    header(f"TEST 7: Policy inference")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"  ⏸  Skip — checkpoint tidak ada")
        return

    try:
        from models.diffusion_policy import load_policy_for_eval

        policy = load_policy_for_eval(str(ckpt_file), device="cpu", num_inference_steps=5)

        # Dummy obs
        obs = np.array([200.0, 200.0, 250.0, 250.0, 0.5], dtype=np.float32)
        policy.reset()
        action = policy.get_action(obs)

        info(f"Action chunk shape: {action.shape}")
        info(f"Action range: [{action.min():.2f}, {action.max():.2f}]")

        assert action.shape == (8, 2), f"Expected (8, 2), got {action.shape}"
        assert np.all(np.isfinite(action)), "Action mengandung NaN/Inf"

        ok(f"Policy inference OK")
        ok(f"Action chunk: {action.shape}, semua finite")

        # Cek apakah obs_horizon buffer bekerja
        policy.reset()
        a1 = policy.get_action(obs)
        a2 = policy.get_action(obs)   # obs sama, tapi buffer beda
        # Karena sampler pakai random noise, hasil bisa beda — tidak diuji strict

    except Exception as e:
        print(f"  ❌ Error saat policy inference: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/pusht_state_diffusion/best.pt",
                        help="Path ke checkpoint (opsional)")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("█  VERIFIKASI K2: Conditioning To=2 (flatten obs_horizon × state_dim)")
    print("█" * 60)

    # Jalankan semua test
    batch = test_1_dataset_shape()
    cond = test_2_flatten(batch)
    model, sched, x_t, t, cond = test_3_model_forward(batch, cond)
    test_4_loss_gradient(model, sched, x_t, t, cond)
    test_5_compare_param_count()
    test_6_checkpoint(args.ckpt)
    test_7_policy_inference(args.ckpt)

    # Kesimpulan
    header("KESIMPULAN")
    print("  ✅ Kode K2 sudah terpasang dengan benar:")
    print(f"       • Dataset menghasilkan state (B, To={EXPECTED_OBS_HORIZON}, D={EXPECTED_STATE_DIM})")
    print(f"       • Flatten → conditioning (B, {EXPECTED_GLOBAL_COND_DIM})")
    print(f"       • Model menerima global_cond_dim={EXPECTED_GLOBAL_COND_DIM}")
    print(f"       • Loss & gradient flow bekerja")
    print(f"       • Dampak parameter minimal (<1%)")
    print()
    print("  Langkah selanjutnya:")
    print("       1. Hapus checkpoint lama:  rm -rf checkpoints/pusht_state_diffusion")
    print("       2. Retrain dari awal:      python train_pusht.py")
    print("       3. Eval setelah selesai:   python eval_policy.py --episodes 20")
    print()


if __name__ == "__main__":
    main()