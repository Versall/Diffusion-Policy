""" Visualisasi Proses Diffusion: DDIM Denoising Steps → Action Chunk → Eksekusi
=============================================================================
Menampilkan animasi lengkap:
1. DDIM sampling: noise → denoised action chunk (per timestep)
2. Final action chunk yang dieksekusi
3. Environment execution (agent + block trajectory) — mode 'full'

Device-agnostic:
- `--device` default None → auto-detect (CUDA > MPS > CPU).
- Override via CLI: `--device cpu` / `--device cuda`.

Output:
- diffusion_only: `viz_output/diffusion_process/ep<N>_seed<SEED>_diffusion_process.gif`
- full          : `viz_output/diffusion_process/ep<N>_seed<SEED>_full.gif`
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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from data.pusht_dataset import normalize_data, unnormalize_data
from models.diffusion_policy import (
    DiffusionPolicy,
    DDIMSampler,
    DDIMSamplerWithHistory,
    load_policy_for_visualization,
)


# ============================================================
# Run Episode dengan Diffusion History
# ============================================================
def run_episode_with_diffusion(
    policy: DiffusionPolicy,
    env: gym.Env,
    max_steps: int = 500,
    seed: Optional[int] = None,
    record_diffusion_every: int = 1,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Jalankan episode sambil merekam history denoising tiap chunk.

    Returns:
        diffusion_history_buffer: list dict per-chunk (step, obs, diffusion)
        step_data: list dict per-step env
        episode_stats: dict ringkasan
    """
    policy.reset()

    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    step_data: List[Dict] = []
    episode_reward = 0.0
    coverage_history: List[float] = []
    terminated = False
    truncated = False
    step = 0
    action_queue: List[np.ndarray] = []
    diffusion_history_buffer: List[Dict] = []

    while not (terminated or truncated) and step < max_steps:
        if len(action_queue) == 0:
            # Ambil action + history denoising
            diffusion_data = policy.get_action_with_diffusion_history(obs)
            action_queue = list(
                diffusion_data["final_actions"][: policy.action_horizon]
            )
            diffusion_history_buffer.append({
                "step": step,
                "obs": obs.copy(),
                "diffusion": diffusion_data,
            })

        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        coverage_history.append(float(info.get("coverage", 0.0)))
        step += 1

        step_data.append({
            "step": step,
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": reward,
            "coverage": float(info.get("coverage", 0.0)),
            "terminated": terminated,
            "truncated": truncated,
        })

    max_coverage = max(coverage_history) if coverage_history else 0.0
    final_coverage = coverage_history[-1] if coverage_history else 0.0
    success = max_coverage > 0.80

    episode_stats = {
        "steps": step,
        "total_reward": episode_reward,
        "max_coverage": max_coverage,
        "final_coverage": final_coverage,
        "success": success,
        "coverage_history": coverage_history,
    }

    return diffusion_history_buffer, step_data, episode_stats


# ============================================================
# Animation 1: Diffusion Process Only (2x2 grid)
# ============================================================
def create_diffusion_process_gif(
    diffusion_data: Dict,
    out_path: str,
    fps: int = 5,
    title: str = "DDIM Denoising Process",
) -> None:
    """
    Animasi denoising untuk SATU action chunk.

    Panel:
        1. X dimension (action step vs value)
        2. Y dimension
        3. 2D action trajectory (X vs Y)
        4. Jadwal timestep DDIM
    """
    x_history = diffusion_data["x_history"]         # list of (T_a, 2)
    pred_x0_history = diffusion_data["pred_x0_history"]
    timesteps = diffusion_data["timesteps"]
    final_actions = diffusion_data["final_actions"]
    n_steps = len(x_history)
    pred_horizon = x_history[0].shape[0]
    x_indices = np.arange(pred_horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    C_X = "#EF4444"
    C_Y = "#8B5CF6"

    def update(frame: int):
        for ax in axes.flat:
            ax.clear()

        step = min(frame, n_steps - 1)
        t = timesteps[step] if step < len(timesteps) else 0

        # Panel 1: X dimension
        for i in range(step + 1):
            alpha = 0.15 + 0.7 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax1.plot(x_indices, x_history[i][:, 0],
                     color=color, alpha=alpha, linewidth=1)
            if i < len(pred_x0_history):
                ax1.plot(x_indices, pred_x0_history[i][:, 0],
                         color=color, alpha=alpha, linewidth=1.5,
                         linestyle="--")
        ax1.plot(x_indices, x_history[step][:, 0],
                 color=C_X, linewidth=3, label=f"x_t (t={t})")
        if step < len(pred_x0_history):
            ax1.plot(x_indices, pred_x0_history[step][:, 0],
                     color=C_X, linewidth=3, linestyle="--",
                     label="pred x0")
        ax1.set_xlim(-0.5, pred_horizon - 0.5)
        ax1.set_ylim(-50, 560)
        ax1.set_xlabel("Action Step")
        ax1.set_ylabel("Action X (px)")
        ax1.set_title(f"X Dimension — Step {step}/{n_steps - 1}")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Y dimension
        for i in range(step + 1):
            alpha = 0.15 + 0.7 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax2.plot(x_indices, x_history[i][:, 1],
                     color=color, alpha=alpha, linewidth=1)
            if i < len(pred_x0_history):
                ax2.plot(x_indices, pred_x0_history[i][:, 1],
                         color=color, alpha=alpha, linewidth=1.5,
                         linestyle="--")
        ax2.plot(x_indices, x_history[step][:, 1],
                 color=C_Y, linewidth=3, label=f"x_t (t={t})")
        if step < len(pred_x0_history):
            ax2.plot(x_indices, pred_x0_history[step][:, 1],
                     color=C_Y, linewidth=3, linestyle="--",
                     label="pred x0")
        ax2.set_xlim(-0.5, pred_horizon - 0.5)
        ax2.set_ylim(-50, 560)
        ax2.set_xlabel("Action Step")
        ax2.set_ylabel("Action Y (px)")
        ax2.set_title(f"Y Dimension — Step {step}/{n_steps - 1}")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: 2D trajectory
        for i in range(step + 1):
            alpha = 0.1 + 0.5 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax3.plot(x_history[i][:, 0], x_history[i][:, 1],
                     "o-", color=color, alpha=alpha, markersize=3)
            if i < len(pred_x0_history):
                ax3.plot(pred_x0_history[i][:, 0], pred_x0_history[i][:, 1],
                         "s--", color=color, alpha=alpha, markersize=3)
        ax3.plot(final_actions[:, 0], final_actions[:, 1],
                 "o-", color="green", linewidth=2, markersize=4,
                 label="Final (denorm)", alpha=0.8)
        ax3.set_xlim(-50, 560)
        ax3.set_ylim(-50, 560)
        ax3.set_aspect("equal")
        ax3.set_xlabel("Action X (px)")
        ax3.set_ylabel("Action Y (px)")
        ax3.set_title(f"Action Chunk Trajectory — Step {step}")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Timestep schedule
        ax4.plot(range(len(timesteps)), timesteps,
                 "o-", color="gray", alpha=0.5)
        ax4.axvline(step, color="red", linestyle="--",
                    linewidth=2, label=f"Current: t={t}")
        ax4.fill_between(range(len(timesteps)), 0, timesteps,
                         alpha=0.1, color="blue")
        ax4.set_xlabel("Denoising Step Index")
        ax4.set_ylabel("Timestep t")
        ax4.set_title("DDIM Timestep Schedule (T→1)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=n_steps,
                         interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"[OK] Diffusion process GIF: {out_path}")


# ============================================================
# Animation 2: Full (Denoising + Environment) — 3x4 grid
# ============================================================
def create_diffusion_animation(
    diffusion_history_buffer: List[Dict],
    step_data: List[Dict],
    episode_stats: Dict,
    policy: DiffusionPolicy,
    out_path: str,
    fps: int = 10,
) -> None:
    """
    Animasi komprehensif:
    - Panel kiri (2x2): DDIM denoising process
    - Panel kanan atas: action chunk yang dieksekusi
    - Panel kanan tengah: timestep progress
    - Panel bawah: environment (agent + block trajectory)
    """
    if not diffusion_history_buffer:
        print("[Warn] diffusion_history_buffer kosong, skip animasi.")
        return

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    ax_diff = fig.add_subplot(gs[0:2, 0:2])
    ax_chunk = fig.add_subplot(gs[0, 2:])
    ax_timestep = fig.add_subplot(gs[1, 2:])
    ax_env = fig.add_subplot(gs[2, :])

    C_AGENT = "#2563EB"
    C_BLOCK = "#F59E0B"
    C_GOAL = "#10B981"
    C_ACTION_X = "#EF4444"
    C_ACTION_Y = "#8B5CF6"

    n_env_steps = len(step_data)
    pred_horizon = 16
    x_indices = np.arange(pred_horizon)

    agent_traj: List[np.ndarray] = []
    block_traj: List[np.ndarray] = []

    state = {
        "diff_idx": 0,
        "denoise_step": 0,
        "env_step": 0,
        "action_exec_idx": 0,
    }

    def get_current_diff():
        if state["diff_idx"] < len(diffusion_history_buffer):
            return diffusion_history_buffer[state["diff_idx"]]["diffusion"]
        return None

    def update(frame: int):
        diff_data = get_current_diff()

        # Advance state machine
        if diff_data is not None:
            n_denoise = len(diff_data["x_history"]) - 1
            if state["denoise_step"] < n_denoise:
                state["denoise_step"] += 1
            elif state["action_exec_idx"] < policy.action_horizon:
                if state["env_step"] < n_env_steps:
                    state["env_step"] += 1
                    state["action_exec_idx"] += 1
                else:
                    state["action_exec_idx"] = policy.action_horizon
            else:
                state["diff_idx"] += 1
                state["denoise_step"] = 0
                state["action_exec_idx"] = 0
                diff_data = get_current_diff()

        # Clear panels
        for ax in (ax_diff, ax_chunk, ax_timestep, ax_env):
            ax.clear()

        if diff_data is None:
            return []

        x_history = diff_data["x_history"]
        pred_x0_history = diff_data["pred_x0_history"]
        timesteps = diff_data["timesteps"]
        final_actions = diff_data["final_actions"]
        n_denoise = len(x_history) - 1
        step_to_show = min(state["denoise_step"], n_denoise)

        # ----- Panel 1: Diffusion denoising -----
        for i in range(min(step_to_show + 1, n_denoise + 1)):
            alpha = 0.2 + 0.6 * (i / max(1, n_denoise))
            color = plt.cm.viridis(i / max(1, n_denoise))
            ax_diff.plot(x_indices, x_history[i][:, 0],
                         color=color, alpha=alpha, linewidth=1,
                         linestyle="--")
            ax_diff.plot(x_indices, x_history[i][:, 1],
                         color=color, alpha=alpha, linewidth=1,
                         linestyle=":")
            if i < len(pred_x0_history):
                ax_diff.plot(x_indices, pred_x0_history[i][:, 0],
                             color=color, alpha=alpha + 0.2, linewidth=1.5)
                ax_diff.plot(x_indices, pred_x0_history[i][:, 1],
                             color=color, alpha=alpha + 0.2, linewidth=1.5)

        x_curr = x_history[step_to_show]
        px0_curr = (pred_x0_history[step_to_show]
                    if step_to_show < len(pred_x0_history) else x_curr)
        ax_diff.plot(x_indices, x_curr[:, 0], color=C_ACTION_X, linewidth=3,
                     label=f"x_t X (t={timesteps[step_to_show] if step_to_show < len(timesteps) else 0})")
        ax_diff.plot(x_indices, x_curr[:, 1], color=C_ACTION_Y,
                     linewidth=3, label="x_t Y")
        ax_diff.plot(x_indices, px0_curr[:, 0], color=C_ACTION_X,
                     linewidth=3, linestyle="--", label="pred x0 X")
        ax_diff.plot(x_indices, px0_curr[:, 1], color=C_ACTION_Y,
                     linewidth=3, linestyle="--", label="pred x0 Y")
        ax_diff.set_xlim(-0.5, pred_horizon - 0.5)
        ax_diff.set_ylim(-50, 560)
        ax_diff.set_xlabel("Action Step")
        ax_diff.set_ylabel("Action Value (px)")
        ax_diff.set_title(
            f"DDIM Denoising: Step {step_to_show}/{n_denoise}"
        )
        ax_diff.legend(loc="upper right", fontsize=8)
        ax_diff.grid(True, alpha=0.3)

        # ----- Panel 2: Action chunk -----
        exec_idx = min(state["action_exec_idx"], policy.action_horizon - 1)
        ax_chunk.bar(
            [0, 1],
            [final_actions[exec_idx, 0], final_actions[exec_idx, 1]],
            color=[C_ACTION_X, C_ACTION_Y], alpha=0.8, width=0.6,
            label=f"Exec #{exec_idx + 1}/{policy.action_horizon}",
        )
        ax_chunk.set_xlim(-0.5, 1.5)
        ax_chunk.set_ylim(-50, 560)
        ax_chunk.set_xticks([0, 1])
        ax_chunk.set_xticklabels(["X", "Y"])
        ax_chunk.set_title(f"Action Chunk: #{exec_idx + 1}/{policy.action_horizon}")
        ax_chunk.legend(fontsize=8)
        ax_chunk.grid(True, alpha=0.3)

        # ----- Panel 3: Timestep progress -----
        denoise_progress = list(range(n_denoise + 1))
        t_values = [timesteps[i] if i < len(timesteps) else 0
                    for i in denoise_progress]
        ax_timestep.plot(denoise_progress, t_values,
                         "o-", color="gray", alpha=0.5)
        if state["denoise_step"] <= n_denoise:
            ax_timestep.plot([state["denoise_step"]],
                             [t_values[state["denoise_step"]]],
                             "o", color="red", markersize=10)
        ax_timestep.axvspan(0, n_denoise, alpha=0.1,
                            color="blue", label="Denoising")
        ax_timestep.axvspan(n_denoise,
                            n_denoise + policy.action_horizon,
                            alpha=0.1, color="green", label="Execution")
        ax_timestep.set_xlim(-0.5, n_denoise + policy.action_horizon + 0.5)
        ax_timestep.set_ylim(-5, 105)
        ax_timestep.set_xlabel("Step Index")
        ax_timestep.set_ylabel("Timestep t")
        ax_timestep.set_title("DDIM: Denoising → Execution")
        ax_timestep.legend(fontsize=8)
        ax_timestep.grid(True, alpha=0.3)

        # ----- Panel 4: Environment -----
        if 0 < state["env_step"] <= n_env_steps:
            sd = step_data[state["env_step"] - 1]
            agent_traj.append(sd["obs"][:2])
            block_traj.append(sd["obs"][2:4])

        if len(agent_traj) > 1:
            a = np.array(agent_traj)
            b = np.array(block_traj)
            ax_env.plot(a[:, 0], a[:, 1], "-", color=C_AGENT,
                        linewidth=2, alpha=0.7, label="Agent")
            ax_env.plot(b[:, 0], b[:, 1], "-", color=C_BLOCK,
                        linewidth=2, alpha=0.7, label="Block")

        if 0 < state["env_step"] <= n_env_steps:
            sd = step_data[state["env_step"] - 1]
            ax_env.scatter(*sd["obs"][:2], color=C_AGENT, s=100,
                           zorder=5, edgecolor="white", linewidth=2)
            ax_env.scatter(*sd["obs"][2:4], color=C_BLOCK, s=100,
                           zorder=5, edgecolor="white", linewidth=2)

        ax_env.scatter(256, 256, color=C_GOAL, s=200,
                       marker="*", zorder=5, label="Goal")
        ax_env.set_xlim(0, 512)
        ax_env.set_ylim(512, 0)
        ax_env.set_aspect("equal")
        ax_env.legend(loc="upper right", fontsize=8)
        ax_env.set_title(
            f"Env Step {state['env_step']}/{n_env_steps} | "
            f"Coverage: "
            f"{episode_stats['coverage_history'][state['env_step'] - 1] if 0 < state['env_step'] <= len(episode_stats['coverage_history']) else 0:.3f}"
        )

        fig.suptitle(
            f"Diffusion Policy | Success={episode_stats['success']} | "
            f"MaxCov={episode_stats['max_coverage']:.3f} | "
            f"Reward={episode_stats['total_reward']:.1f}",
            fontsize=14, fontweight="bold",
        )
        return []

    # Hitung total frame
    total_frames = 0
    for d in diffusion_history_buffer:
        n_d = len(d["diffusion"]["x_history"]) - 1
        total_frames += n_d + 1 + policy.action_horizon
    total_frames = min(total_frames, 300)

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[OK] Full animation: {out_path} ({size_mb:.1f} MB)")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Diffusion Process + Policy Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="checkpoints/pusht_state_diffusion/best.pt",
        help="Path ke checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=1,
                        help="Jumlah episode.")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Batas langkah per episode.")
    parser.add_argument("--inference-steps", type=int, default=20,
                        help="Jumlah langkah DDIM.")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', atau None (auto-detect).",
    )
    parser.add_argument("--fps", type=int, default=8,
                        help="FPS output GIF.")
    parser.add_argument("--outdir", type=str,
                        default="viz_output/diffusion_process",
                        help="Folder output.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument(
        "--mode", type=str,
        choices=["diffusion_only", "full"],
        default="diffusion_only",
        help="Mode: 'diffusion_only' (denoising saja) atau "
             "'full' (denoising + env).",
    )
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
    print(f"[Visualize] Mode: {args.mode}")

    # --- Buat env ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )
    env.unwrapped.success_threshold = 0.80

    # --- Load policy dengan history sampler ---
    policy = load_policy_for_visualization(
        ckpt_path=str(ckpt_path),
        device=str(device),
        num_inference_steps=args.inference_steps,
    )

    print(f"\n[Visualize] Episodes: {args.episodes}, "
          f"DDIM steps: {args.inference_steps}")
    print(f"[Visualize] Output: {outdir}")

    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))
        print(f"\n--- Episode {ep + 1}/{args.episodes} (seed={ep_seed}) ---")

        diff_buf, step_data, ep_stats = run_episode_with_diffusion(
            policy, env,
            max_steps=args.max_steps,
            seed=ep_seed,
            record_diffusion_every=1,
        )

        print(f"  Steps: {ep_stats['steps']}, "
              f"Reward: {ep_stats['total_reward']:.1f}")
        print(f"  Max Coverage: {ep_stats['max_coverage']:.3f}, "
              f"Success: {ep_stats['success']}")
        print(f"  Diffusion chunks recorded: {len(diff_buf)}")

        if args.mode == "diffusion_only":
            if diff_buf:
                diff_data = diff_buf[0]["diffusion"]
                gif_path = outdir / f"ep{ep}_seed{ep_seed}_diffusion_process.gif"
                create_diffusion_process_gif(
                    diff_data, str(gif_path), fps=args.fps,
                    title=(f"Ep {ep} | Seed {ep_seed} | "
                           f"DDIM ({args.inference_steps} steps)"),
                )
        else:
            gif_path = outdir / f"ep{ep}_seed{ep_seed}_full.gif"
            create_diffusion_animation(
                diff_buf, step_data, ep_stats, policy,
                str(gif_path), fps=args.fps,
            )

    env.close()
    print(f"\n=== Done! Files in {outdir} ===")


if __name__ == "__main__":
    main()