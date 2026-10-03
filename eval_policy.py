"""
Diffusion Policy Inference di PushT Environment
================================================
Load model terlatih (pusht_state_diffusion/best.pt) → run DDPM sampling → execute di env.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import gym_pusht
from tqdm.auto import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pusht_dataset import PushTDataConfig, PushTStateDataset, normalize_data, unnormalize_data
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
) -> Dict:
    """Evaluasi policy di env."""
    
    results = []
    rng = np.random.default_rng(seed)
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        policy.reset()
        
        # Reset env dengan random initial state
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        
        episode_reward = 0.0
        episode_coverage = []
        terminated = False
        truncated = False
        step = 0
        
        # Action queue (karena action_horizon > 1)
        action_queue = []
        
        while not (terminated or truncated) and step < max_steps:
            # Jika queue kosong, prediksi action chunk baru
            if len(action_queue) == 0:
                # obs dari env: [agent_x, agent_y, block_x, block_y, block_angle]
                actions_pred = policy.get_action(obs)  # (action_horizon, 2)
                action_queue = list(actions_pred)
            
            # Ambil action dari queue
            action = action_queue.pop(0)
            
            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_coverage.append(info.get('coverage', 0.0))
            step += 1
            
            if render:
                env.render()
                time.sleep(0.01)
        
        # Stats
        max_coverage = max(episode_coverage) if episode_coverage else 0.0
        final_coverage = episode_coverage[-1] if episode_coverage else 0.0
        success = max_coverage > 0.80  # PushT success threshold
        
        results.append({
            'episode': ep,
            'steps': step,
            'total_reward': episode_reward,
            'max_coverage': max_coverage,
            'final_coverage': final_coverage,
            'success': success,
        })
        
        print(f"  Ep {ep}: steps={step}, reward={episode_reward:.2f}, "
              f"max_cov={max_coverage:.3f}, success={success}")
    
    # Summary
    n_success = sum(1 for r in results if r['success'])
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_max_cov = np.mean([r['max_coverage'] for r in results])
    
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Success Rate: {n_success}/{num_episodes} ({100*n_success/num_episodes:.1f}%)")
    print(f"Avg Reward:   {avg_reward:.2f}")
    print(f"Avg Steps:    {avg_steps:.1f}")
    print(f"Avg Max Cov:  {avg_max_cov:.3f}")
    print(f"{'='*50}")
    
    return {
        'results': results,
        'success_rate': n_success / num_episodes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_max_coverage': avg_max_cov,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Evaluation on PushT")
    parser.add_argument("--ckpt", type=str, default="checkpoints/pusht_state_diffusion/best.pt",
                        help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--inference-steps", type=int, default=10, help="DDIM inference steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--render", action="store_true", help="Render env")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Resolve checkpoint path
    repo_root = Path(__file__).parent
    ckpt_path = repo_root / args.ckpt
    
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    # Create env
    env = gym.make("gym_pusht/PushT-v0", max_episode_steps=1000, render_mode="human" if args.render else "rgb_array")
    # Override success threshold to 0.80 (env default is 0.95)
    env.unwrapped.success_threshold = 0.80
    
    # Load policy
    policy = load_policy_for_eval(
        ckpt_path=str(ckpt_path),
        device=args.device,
        num_inference_steps=args.inference_steps,
    )
    
    # Evaluate
    print(f"\n[Eval] Starting evaluation: {args.episodes} episodes, {args.inference_steps} DDIM steps")
    evaluate_policy(
        policy, env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        seed=args.seed,
    )
    
    env.close()


if __name__ == "__main__":
    main()