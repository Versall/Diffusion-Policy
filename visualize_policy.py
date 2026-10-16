"""
Visualize Diffusion Policy pada Push-T (keypoint + agent_pos, 20D).

- Load checkpoint dari train_pusht.py terbaru
- Rollout di gym_pusht, render ke GIF/MP4
- DDIM sampling 10 langkah
- Receding horizon: sample pred_horizon=16, eksekusi action_horizon=8

Cara pakai:
    uv run python visualize_policy.py --ckpt checkpoints/pusht_state_diffusion/epoch_0050_ema.pt --out rollout.gif
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.device import get_device, describe_device
from data.pusht_dataset import LinearNormalizer
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine


# ============================================================
# Obs extraction (handle berbagai format gym_pusht)
# ============================================================
def extract_obs_20d(obs, env=None) -> np.ndarray:
    """
    Ubah obs dari gym_pusht menjadi vektor (20,) = keypoint(18) + agent_pos(2).

    Handle:
    - obs dict {'keypoint': (9,2) atau (18,), 'state': (5,) atau 'agent_pos': (2,)}
    - obs array (18,) → ambil agent_pos dari env.unwrapped.state
    - obs array (20,) → langsung pakai
    - obs array (5,) → state, tapi keypoint dari env.unwrapped.keypoint
    """
    # --- Case 1: dict ---
    if isinstance(obs, dict):
        kp = obs.get("keypoint", None)
        st = obs.get("state", None)
        ap = obs.get("agent_pos", None)

        if kp is None:
            raise ValueError("obs dict tidak punya key 'keypoint'")

        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 2 and kp.shape == (9, 2):
            kp = kp.reshape(-1)          # (18,)
        elif kp.ndim == 1 and kp.size == 18:
            pass
        else:
            raise ValueError(f"keypoint shape tidak dikenal: {kp.shape}")

        if st is not None:
            agent_pos = np.asarray(st, dtype=np.float32).flatten()[:2]
        elif ap is not None:
            agent_pos = np.asarray(ap, dtype=np.float32).flatten()[:2]
        else:
            raise ValueError("Tidak bisa extract agent_pos dari obs dict")

        return np.concatenate([kp, agent_pos]).astype(np.float32)  # (20,)

    # --- Case 2: array ---
    arr = np.asarray(obs, dtype=np.float32).flatten()

    if arr.size == 20:
        return arr
    if arr.size == 18:
        if env is None:
            raise ValueError("Butuh env untuk extract agent_pos")
        ap = np.asarray(env.unwrapped.state, dtype=np.float32)[:2]
        return np.concatenate([arr, ap])
    if arr.size == 5:
        if env is None:
            raise ValueError("Butuh env untuk extract keypoint")
        kp = np.asarray(env.unwrapped.keypoint, dtype=np.float32).flatten()
        if kp.size != 18:
            # kalau (9,2)
            kp = kp.reshape(-1)
        return np.concatenate([kp, arr[:2]])

    raise ValueError(f"Bentuk obs tidak dikenal: {arr.shape}")


# ============================================================
# DDIM sampling (deterministik, η=0)
# ============================================================
@torch.no_grad()
def ddim_sample(
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    cond: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_steps: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sampling DDIM: dari noise murni → action (1, Ta, Da)."""
    B = cond.shape[0]
    alpha_bar = noise_sched.alpha_bar
    T = alpha_bar.shape[0]

    # Sub-sequence timesteps descending
    step_ratio = T // num_steps
    timesteps = (torch.arange(0, num_steps) * step_ratio).long()
    timesteps = torch.flip(timesteps, dims=[0])

    x = torch.randn(B, pred_horizon, action_dim, device=device)

    for i, t in enumerate(timesteps):
        t_val = int(t.item())
        a_t = alpha_bar[t_val].to(device)

        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

        # prediksi noise
        eps = model(x.permute(0, 2, 1), t_batch, cond).permute(0, 2, 1)

        # x0_pred
        sqrt_a = torch.sqrt(a_t)
        sqrt_1ma = torch.sqrt(1.0 - a_t)
        x0 = (x - sqrt_1ma * eps) / sqrt_a
        x0 = torch.clamp(x0, -1.0, 1.0)

        if i == len(timesteps) - 1:
            x = x0
        else:
            t_next = int(timesteps[i + 1].item())
            a_next = alpha_bar[t_next].to(device)
            x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps

    return x


# ============================================================
# Load policy dari checkpoint
# ============================================================
def load_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    obs_dim = cfg.get("obs_dim", 20)
    obs_horizon = cfg.get("obs_horizon", 2)
    pred_horizon = cfg.get("pred_horizon", 16)
    input_dim = cfg.get("input_dim", 2)
    global_cond_dim = cfg.get("global_cond_dim", obs_horizon * obs_dim)

    print(f"[Policy] epoch={ckpt.get('epoch')} | "
          f"obs_dim={obs_dim} | obs_horizon={obs_horizon} | "
          f"pred_horizon={pred_horizon} | global_cond_dim={global_cond_dim}")

    model = ConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=cfg.get("diffusion_step_embed_dim", 256),
        down_dims=list(cfg.get("down_dims", (256, 512, 1024))),
        kernel_size=cfg.get("kernel_size", 3),
        n_groups=cfg.get("n_groups", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    noise_sched = NoiseScheduleCosine(
        T=cfg.get("num_diffusion_steps", 100),
        s=cfg.get("noise_schedule_s", 0.008),
        device=device,
    )

    normalizer = None
    norm_state = ckpt.get("normalizer_state", None)
    if norm_state is not None:
        normalizer = LinearNormalizer(mode=norm_state["mode"])
        normalizer.load_state_dict(norm_state)
        print(f"[Policy] Normalizer restored (mode={norm_state['mode']})")
    else:
        print("[Policy] Warning: checkpoint tidak punya normalizer_state")

    return model, noise_sched, normalizer, cfg


# ============================================================
# Rollout satu episode
# ============================================================
def run_episode(
    env,
    model: ConditionalUnet1D,
    noise_sched: NoiseScheduleCosine,
    normalizer: Optional[LinearNormalizer],
    cfg: dict,
    device: torch.device,
    seed: int,
    max_steps: int,
    num_inference_steps: int,
    action_horizon: int,
) -> Tuple[List[np.ndarray], bool]:
    obs_horizon = cfg.get("obs_horizon", 2)
    pred_horizon = cfg.get("pred_horizon", 16)
    input_dim = cfg.get("input_dim", 2)

    # Reset env
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    # Buffer obs
    obs_buffer: List[np.ndarray] = [extract_obs_20d(obs, env)]

    frames: List[np.ndarray] = []
    done = False
    step = 0

    # Render frame awal
    frame = _render(env)
    if frame is not None:
        frames.append(frame)

    while step < max_steps and not done:
        # Pastikan buffer cukup
        while len(obs_buffer) < obs_horizon:
            obs_buffer.append(obs_buffer[-1])

        obs_seq = np.stack(obs_buffer[-obs_horizon:], axis=0)  # (To, 20)

        # Normalize obs
        obs_norm = (
            normalizer.normalize(obs_seq, "obs")
            if normalizer is not None else obs_seq
        )

        # cond = flatten
        cond = torch.from_numpy(obs_norm).float().reshape(1, -1).to(device)

        # Sampling action (1, Ta, Da)
        action_full = ddim_sample(
            model, noise_sched, cond,
            pred_horizon=pred_horizon,
            action_dim=input_dim,
            num_steps=num_inference_steps,
            device=device,
        )

        # Unnormalize
        action_full_np = action_full[0].cpu().numpy()  # (Ta, 2)
        if normalizer is not None:
            action_full_np = normalizer.unnormalize(action_full_np, "action")

        # Eksekusi action_horizon langkah (receding horizon)
        exec_len = min(action_horizon, pred_horizon)
        for i in range(exec_len):
            if step >= max_steps:
                break

            action = action_full_np[i].astype(np.float32)

            result = env.step(action)
            if len(result) == 5:
                obs, _, terminated, truncated, _ = result
                done = bool(terminated or truncated)
            else:  # 4-tuple
                obs, _, done, _ = result

            frame = _render(env)
            if frame is not None:
                frames.append(frame)

            obs_buffer.append(extract_obs_20d(obs, env))
            step += 1

            if done:
                break

    return frames, done


def _render(env) -> Optional[np.ndarray]:
    """Ambil frame RGB dari env."""
    try:
        frame = env.render()
    except Exception:
        frame = None
    if frame is None:
        try:
            frame = env.render(mode="rgb_array")
        except Exception:
            return None
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


# ============================================================
# Save frames → GIF / MP4
# ============================================================
def save_frames(frames: List[np.ndarray], path: str, fps: int = 10):
    if not frames:
        print("[Warn] Tidak ada frame untuk disimpan")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".gif":
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            str(path),
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    elif suffix in (".mp4", ".avi", ".mkv"):
        try:
            import imageio
            imageio.mimsave(str(path), frames, fps=fps)
        except ImportError:
            print("[Warn] imageio tidak terpasang, fallback ke GIF")
            path = path.with_suffix(".gif")
            images = [Image.fromarray(f) for f in frames]
            images[0].save(
                str(path), save_all=True, append_images=images[1:],
                duration=int(1000 / fps), loop=0,
            )
    else:
        raise ValueError(f"Format tidak didukung: {suffix}")

    print(f"[OK] Saved {len(frames)} frames → {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Diffusion Policy Push-T (keypoint 20D)"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path ke checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="visualizations/rollout.gif",
                        help="Output: .gif atau .mp4")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Jumlah episode yang dijalankan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="Langkah DDIM (default 10)")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="Jumlah action yang dieksekusi per sampling")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    print(f"[Device] {describe_device(device)}")

    # Load policy
    model, noise_sched, normalizer, cfg = load_policy(args.ckpt, device)

    # Env
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except ImportError:
        import gym
        import gym_pusht  # noqa: F401

    # Coba obs_type="keypoint" dulu, fallback ke default
    env = None
    for obs_type in ("keypoint", None):
        try:
            if obs_type is None:
                env = gym.make("gym_pusht/PushT-v0")
            else:
                env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type)
            print(f"[Env] gym_pusht/PushT-v0 (obs_type={obs_type})")
            break
        except Exception as e:
            print(f"[Env] Gagal buat env obs_type={obs_type}: {e}")
            continue

    if env is None:
        print("[Error] Tidak bisa membuat gym_pusht env")
        sys.exit(1)

    all_frames: List[np.ndarray] = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n[Episode {ep+1}/{args.episodes}] seed={seed}")
        t0 = time.time()
        frames, done = run_episode(
            env, model, noise_sched, normalizer, cfg,
            device=device,
            seed=seed,
            max_steps=args.max_steps,
            num_inference_steps=args.num_inference_steps,
            action_horizon=args.action_horizon,
        )
        elapsed = time.time() - t0
        print(f"  frames={len(frames)} | done={done} | time={elapsed:.1f}s")
        all_frames.extend(frames)

    env.close()

    # Simpan
    save_frames(all_frames, args.out, fps=args.fps)


if __name__ == "__main__":
    main()