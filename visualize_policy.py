""" Visualisasi Episode: Simpan GIF/MP4 trajectory model di env.
=============================================================
Load model terlatih → run di env → simpan frames → export GIF/MP4.

Device-agnostic:
- `--device` default None → auto-detect (CUDA > MPS > CPU).
- Override via CLI: `--device cpu` / `--device cuda`.

Output:
- GIF: `viz_output/policy_runs/episode_<N>_seed<SEED>.gif`
- MP4: `viz_output/policy_runs/episode_<N>_seed<SEED>.mp4`
- Plot: `viz_output/policy_runs/episode_<N>_trajectory.png`
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Pastikan root repo di sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.device import get_device, describe_device, summary

import gymnasium as gym
import gym_pusht
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from data.pusht_dataset import normalize_data, unnormalize_data
from models.diffusion_policy import DiffusionPolicy, load_policy_for_eval


# ============================================================
# Run Episode & Save Frames
# ============================================================
def run_episode(
    policy: DiffusionPolicy,
    env: gym.Env,
    max_steps: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Jalankan 1 episode, return frames + stats.

    Returns:
        frames: list of np.ndarray (H, W, 3) — frame environment.
        stats: dict(steps, total_reward, max_coverage, final_coverage, success, coverage_history)
    """
    policy.reset()

    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    frames: List[np.ndarray] = [env.render()]
    episode_reward = 0.0
    coverage_history: List[float] = []
    terminated = False
    truncated = False
    step = 0
    action_queue: List[np.ndarray] = []

    while not (terminated or truncated) and step < max_steps:
        if len(action_queue) == 0:
            actions_pred = policy.get_action(obs)
            action_queue = list(actions_pred)

        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        coverage_history.append(float(info.get("coverage", 0.0)))
        frames.append(env.render())
        step += 1

    max_coverage = max(coverage_history) if coverage_history else 0.0
    final_coverage = coverage_history[-1] if coverage_history else 0.0
    success = max_coverage > 0.80

    stats = {
        "steps": step,
        "total_reward": episode_reward,
        "max_coverage": max_coverage,
        "final_coverage": final_coverage,
        "success": success,
        "coverage_history": coverage_history,
    }
    return frames, stats


# ============================================================
# Visualization helpers
# ============================================================
def save_gif(frames: List[np.ndarray], out_path: str,
             fps: int = 10, title: str = "") -> None:
    """Simpan frames sebagai GIF."""
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(frames[0])
    ax.axis("off")
    title_obj = ax.set_title(title, color="white", fontsize=12)
    fig.patch.set_facecolor("black")

    def update(idx: int):
        img.set_data(frames[idx])
        if title:
            title_obj.set_text(f"{title} — frame {idx}/{len(frames)}")
        return [img, title_obj]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[OK] GIF saved: {out_path} ({size_kb:.0f} KB)")


def save_mp4(frames: List[np.ndarray], out_path: str,
             fps: int = 10, title: str = "") -> None:
    """Simpan frames sebagai MP4 (butuh ffmpeg)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(frames[0])
    ax.axis("off")
    title_obj = ax.set_title(title, color="white", fontsize=12)
    fig.patch.set_facecolor("black")

    def update(idx: int):
        img.set_data(frames[idx])
        if title:
            title_obj.set_text(f"{title} — frame {idx}/{len(frames)}")
        return [img, title_obj]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps), dpi=100)
        plt.close(fig)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"[OK] MP4 saved: {out_path} ({size_kb:.0f} KB)")
    except Exception as e:
        plt.close(fig)
        print(f"[Error] MP4 save failed (ffmpeg not installed?): {e}")
        print("  Falling back to GIF...")
        save_gif(frames, out_path.replace(".mp4", ".gif"), fps, title)


def plot_trajectory(frames: List[np.ndarray], stats: Dict, out_path: str) -> None:
    """Plot coverage over time + final frame."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Panel 1: Coverage over time
    cov = stats["coverage_history"]
    ax1.plot(cov, color="#10B981", linewidth=2)
    ax1.axhline(0.80, color="#EF5350", linestyle="--", linewidth=1.5,
                label="Success threshold (0.80)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Coverage")
    ax1.set_title(
        f"Coverage over Time "
        f"(Max: {stats['max_coverage']:.3f}, "
        f"Success: {stats['success']})"
    )
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Final frame
    if frames:
        final_frame = frames[-1].copy()
        ax2.imshow(final_frame)
        ax2.set_title(f"Final Frame (Step {stats['steps']})")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Trajectory plot: {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Diffusion Policy on PushT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="checkpoints/pusht_state_diffusion/final.pt",
        help="Path ke checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Jumlah episode yang direkam.")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Batas langkah per episode.")
    parser.add_argument("--inference-steps", type=int, default=20,
                        help="Jumlah langkah DDIM.")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', atau None (auto-detect).",
    )
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS output GIF/MP4.")
    parser.add_argument("--format", type=str,
                        choices=["gif", "mp4", "both"], default="gif",
                        help="Format output.")
    parser.add_argument("--outdir", type=str, default="viz_output/policy_runs",
                        help="Folder output.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed untuk pemilihan episode seed.")
    args = parser.parse_args()

    # --- Resolve path ---
    repo_root = Path(__file__).resolve().parent
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Device auto-detect ---
    device = get_device(prefer=args.device)
    print(summary())
    print(f"\n[Visualize] Device: {describe_device(device)}")
    print(f"[Visualize] Checkpoint: {ckpt_path}")

    # --- Buat environment ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )
    # Override success threshold env (default 0.95 → 0.80)
    env.unwrapped.success_threshold = 0.80

    # --- Load policy ---
    policy = load_policy_for_eval(
        ckpt_path=str(ckpt_path),
        device=str(device),
        num_inference_steps=args.inference_steps,
    )

    # --- Visualize ---
    print(f"\n[Visualize] Recording {args.episodes} episodes...")
    print(f"[Visualize] Output dir: {outdir}")
    print(f"[Visualize] Format: {args.format}, FPS: {args.fps}")

    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))
        print(f"\n--- Episode {ep + 1}/{args.episodes} (seed={ep_seed}) ---")

        frames, stats = run_episode(
            policy, env,
            max_steps=args.max_steps,
            seed=ep_seed,
        )

        print(f"  Steps: {stats['steps']}, Reward: {stats['total_reward']:.1f}")
        print(f"  Max Coverage: {stats['max_coverage']:.3f}, "
              f"Success: {stats['success']}")

        title = (
            f"Ep {ep} | Cov: {stats['max_coverage']:.3f} | "
            f"{'SUCCESS' if stats['success'] else 'FAIL'}"
        )

        if args.format in ["gif", "both"]:
            gif_path = outdir / f"episode_{ep}_seed{ep_seed}.gif"
            save_gif(frames, str(gif_path), fps=args.fps, title=title)

        if args.format in ["mp4", "both"]:
            mp4_path = outdir / f"episode_{ep}_seed{ep_seed}.mp4"
            save_mp4(frames, str(mp4_path), fps=args.fps, title=title)

        plot_path = outdir / f"episode_{ep}_trajectory.png"
        plot_trajectory(frames, stats, str(plot_path))

    env.close()
    print(f"\n=== Done! Files in {outdir} ===")


if __name__ == "__main__":
    main()