""" Diffusion Policy Inference di PushT Environment
================================================
Load model terlatih → run DDIM sampling → execute di env → ukur success rate.

Device-agnostic:
- `--device` default None → auto-detect (CUDA > MPS > CPU).
- Override via CLI: `--device cpu` atau `--device cuda`.
- Override via env: `FORCE_DEVICE=cpu python eval_policy.py`.

Fitur:
- Multi-episode evaluation dengan random seed per episode.
- Metrik: success rate, avg reward, avg steps, avg max coverage.
- Opsi render ke layar (butuh display).
- Bisa simpan hasil ke JSON untuk analisis lanjutan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Pastikan root repo di sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.device import get_device, describe_device, summary, is_cuda

import gymnasium as gym
import gym_pusht
from tqdm.auto import tqdm

from data.pusht_dataset import normalize_data, unnormalize_data
from models.diffusion_policy import DiffusionPolicy, load_policy_for_eval


# ============================================================
# Evaluation Loop
# ============================================================
def evaluate_policy(
    policy: DiffusionPolicy,
    env: gym.Env,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    seed: int = 42,
    success_threshold: float = 0.80,
    verbose: bool = True,
) -> Dict:
    """
    Evaluasi policy di environment Push-T.

    Args:
        policy: DiffusionPolicy yang sudah di-load.
        env: gym.Env Push-T.
        num_episodes: jumlah episode.
        max_steps: batas langkah per episode.
        render: tampilkan ke layar.
        seed: seed untuk reproducibility.
        success_threshold: coverage minimum untuk sukses (default 0.80).
        verbose: print per-episode.

    Returns:
        dict dengan:
            results: list per-episode (steps, reward, coverage, success, ...)
            success_rate: float 0..1
            avg_reward, avg_steps, avg_max_coverage
    """
    results: List[Dict] = []
    rng = np.random.default_rng(seed)

    for ep in tqdm(range(num_episodes), desc="Evaluating", disable=not verbose):
        policy.reset()
        ep_seed = int(rng.integers(0, 2**31))
        obs, info = env.reset(seed=ep_seed)

        episode_reward = 0.0
        episode_coverage: List[float] = []
        terminated = False
        truncated = False
        step = 0
        action_queue: List[np.ndarray] = []

        while not (terminated or truncated) and step < max_steps:
            # Kalau queue kosong, prediksi action chunk baru
            if len(action_queue) == 0:
                actions_pred = policy.get_action(obs)   # (action_horizon, 2)
                action_queue = list(actions_pred)

            # Ambil action dari queue
            action = action_queue.pop(0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_coverage.append(float(info.get("coverage", 0.0)))
            step += 1

            if render:
                env.render()
                time.sleep(0.01)

        # Statistik episode
        max_coverage = max(episode_coverage) if episode_coverage else 0.0
        final_coverage = episode_coverage[-1] if episode_coverage else 0.0
        success = max_coverage > success_threshold

        results.append({
            "episode": ep,
            "seed": ep_seed,
            "steps": step,
            "total_reward": episode_reward,
            "max_coverage": max_coverage,
            "final_coverage": final_coverage,
            "success": success,
        })

        if verbose:
            print(
                f"  Ep {ep:3d}: steps={step:4d}, reward={episode_reward:7.2f}, "
                f"max_cov={max_coverage:.3f}, success={success}"
            )

    # Summary
    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / num_episodes
    avg_reward = float(np.mean([r["total_reward"] for r in results]))
    avg_steps = float(np.mean([r["steps"] for r in results]))
    avg_max_cov = float(np.mean([r["max_coverage"] for r in results]))
    std_max_cov = float(np.std([r["max_coverage"] for r in results]))

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
        print(f"{'=' * 50}")
        print(f"Success Rate : {n_success}/{num_episodes} ({100 * success_rate:.1f}%)")
        print(f"Avg Reward   : {avg_reward:.2f}")
        print(f"Avg Steps    : {avg_steps:.1f}")
        print(f"Avg Max Cov  : {avg_max_cov:.3f} ± {std_max_cov:.3f}")
        print(f"{'=' * 50}")

    return {
        "results": results,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_max_coverage": avg_max_cov,
        "std_max_coverage": std_max_cov,
        "num_episodes": num_episodes,
        "success_threshold": success_threshold,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Diffusion Policy Evaluation on PushT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="checkpoints/pusht_state_diffusion/best.pt",
        help="Path ke checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=10,
                        help="Jumlah episode (default: 10).")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Batas langkah per episode.")
    parser.add_argument("--inference-steps", type=int, default=10,
                        help="Jumlah langkah DDIM.")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', atau None (auto-detect).",
    )
    parser.add_argument("--render", action="store_true",
                        help="Render environment ke layar.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed untuk eval.")
    parser.add_argument("--success-threshold", type=float, default=0.80,
                        help="Coverage minimum untuk sukses (default: 0.80).")
    parser.add_argument("--out", type=str, default=None,
                        help="Path JSON untuk simpan hasil (opsional).")
    parser.add_argument("--quiet", action="store_true",
                        help="Sembunyikan per-episode print.")
    args = parser.parse_args()

    # --- Resolve path ---
    repo_root = Path(__file__).resolve().parent
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # --- Device auto-detect ---
    device = get_device(prefer=args.device)
    print(summary())
    print(f"\n[Eval] Device: {describe_device(device)}")
    print(f"[Eval] Checkpoint: {ckpt_path}")

    # --- Buat environment ---
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(
        "gym_pusht/PushT-v0",
        max_episode_steps=args.max_steps,
        render_mode=render_mode,
    )
    # Override success threshold env (default 0.95 → 0.80)
    env.unwrapped.success_threshold = args.success_threshold

    # --- Load policy ---
    policy = load_policy_for_eval(
        ckpt_path=str(ckpt_path),
        device=str(device),
        num_inference_steps=args.inference_steps,
    )

    # --- Evaluation ---
    print(
        f"\n[Eval] Starting: {args.episodes} episodes, "
        f"{args.inference_steps} DDIM steps, "
        f"device={device.type}"
    )
    t0 = time.time()
    results = evaluate_policy(
        policy, env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        seed=args.seed,
        success_threshold=args.success_threshold,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0
    print(f"\n[Eval] Total time: {elapsed:.1f}s ({elapsed / args.episodes:.1f}s/episode)")
    env.close()

    # --- Simpan hasil ---
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(ckpt_path),
            "device": str(device),
            "num_episodes": args.episodes,
            "inference_steps": args.inference_steps,
            "success_threshold": args.success_threshold,
            "success_rate": results["success_rate"],
            "avg_reward": results["avg_reward"],
            "avg_steps": results["avg_steps"],
            "avg_max_coverage": results["avg_max_coverage"],
            "std_max_coverage": results["std_max_coverage"],
            "episodes": results["results"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Eval] Hasil disimpan: {out_path}")


if __name__ == "__main__":
    main()