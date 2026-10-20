"""
Tes C: verifikasi model forward, loss, dan backprop berjalan.
Jalankan: uv run python test_model.py
"""
import torch

from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine
from train_pusht import diffusion_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Config sama dengan training ---
    B, Ta, Da = 4, 16, 2
    obs_horizon, obs_dim = 2, 20
    global_cond_dim = obs_horizon * obs_dim  # 40

    # --- Model ---
    model = ConditionalUnet1D(
        input_dim=Da,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # --- Noise schedule ---
    noise_sched = NoiseScheduleCosine(T=100, s=0.008, device=device)

    # --- Cek alpha_bar index (deteksi bug torch.randint) ---
    ab = noise_sched.alpha_bar
    print(f"alpha_bar shape: {tuple(ab.shape)}")
    print(f"alpha_bar[0]  = {ab[0].item():.6f}   (harusnya ~1.0 untuk 0-indexed)")
    print(f"alpha_bar[-1] = {ab[-1].item():.6f}   (harusnya ~0.0 untuk 0-indexed)")

    # --- Forward + loss + backprop ---
    x_0 = torch.randn(B, Ta, Da, device=device)
    cond = torch.randn(B, global_cond_dim, device=device)

    loss = diffusion_loss(model, noise_sched, x_0, cond, device)
    print(f"loss: {loss.item():.6f}")

    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"params dengan gradient: {n_grad}/{n_total}")

    # --- Assertions ---
    assert torch.isfinite(loss), f"FATAL: loss NaN/Inf: {loss.item()}"
    assert n_grad > 0, "FATAL: tidak ada gradient yang mengalir"
    assert n_grad == n_total, f"FATAL: hanya {n_grad}/{n_total} params punya gradient"

    print()
    print("[OK] Model forward + loss + backprop berjalan. Siap training.")


if __name__ == "__main__":
    main()