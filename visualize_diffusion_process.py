"""
Visualisasi Proses Diffusion: DDIM Denoising Steps → Action Chunk → Eksekusi
=============================================================================
Menampilkan animasi lengkap:
1. DDIM sampling: noise → denoised action chunk (per timestep)
2. Final action chunk yang dieksekusi
3. Environment execution (agent + block trajectory)
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym
import gym_pusht
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.pusht_dataset import normalize_data, unnormalize_data
from models.diffusion_policy import DiffusionPolicy, DDIMSampler, load_policy_for_visualization


# ============================================================
# Run Episode dengan Diffusion History
# ============================================================

def run_episode_with_diffusion(
    policy: DiffusionPolicy,
    env: gym.Env,
    max_steps: int = 500,
    seed: int = None,
    record_diffusion_every: int = 1,  # Record diffusion process every N steps
) -> Tuple[List[Dict], Dict]:
    """
    Returns:
        step_data: list of dict per step dengan diffusion history
        episode_stats: summary stats
    """
    policy.reset()
    
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    
    step_data = []
    episode_reward = 0.0
    coverage_history = []
    terminated = False
    truncated = False
    step = 0
    action_queue = []
    diffusion_history_buffer = []  # Store diffusion viz data
    
    while not (terminated or truncated) and step < max_steps:
        if len(action_queue) == 0:
            # Get action + diffusion history
            diffusion_data = policy.get_action_with_diffusion_history(obs)
            action_queue = list(diffusion_data['final_actions'][:policy.action_horizon])
            diffusion_history_buffer.append({
                'step': step,
                'obs': obs.copy(),
                'diffusion': diffusion_data,
            })
        
        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        coverage_history.append(info.get('coverage', 0.0))
        step += 1
        
        step_data.append({
            'step': step,
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'coverage': info.get('coverage', 0.0),
            'terminated': terminated,
            'truncated': truncated,
        })
    
    max_coverage = max(coverage_history) if coverage_history else 0.0
    final_coverage = coverage_history[-1] if coverage_history else 0.0
    success = max_coverage > 0.95
    
    episode_stats = {
        'steps': step,
        'total_reward': episode_reward,
        'max_coverage': max_coverage,
        'final_coverage': final_coverage,
        'success': success,
        'coverage_history': coverage_history,
    }
    
    return diffusion_history_buffer, step_data, episode_stats


# ============================================================
# Animation: Diffusion Process + Environment
# ============================================================

def create_diffusion_animation(
    diffusion_history_buffer: List[Dict],
    step_data: List[Dict],
    episode_stats: Dict,
    out_path: str,
    fps: int = 10,
    max_diffusion_steps_to_show: int = None,
):
    """
    Buat animasi komprehensif:
    - Panel kiri: DDIM denoising process (noise → action chunk)
    - Panel tengah: Action chunk yang dieksekusi (highlight step saat ini)
    - Panel kanan: Environment state (agent + block trajectory)
    """
    n_diffusion_steps = len(diffusion_history_buffer[0]['diffusion']['x_history']) if diffusion_history_buffer else 0
    if max_diffusion_steps_to_show is None:
        max_diffusion_steps_to_show = n_diffusion_steps
    
    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Diffusion denoising (top-left, span 2 cols)
    ax_diff = fig.add_subplot(gs[0:2, 0:2])
    ax_diff.set_title("DDIM Denoising Process: Noise → Action Chunk", fontsize=14, fontweight='bold')
    ax_diff.set_xlabel("Action Step (0-15)", fontsize=11)
    ax_diff.set_ylabel("Action Value (pixels)", fontsize=11)
    ax_diff.set_xlim(-0.5, 15.5)
    ax_diff.set_ylim(-50, 560)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_diff.axhline(512, color='gray', linestyle=':', alpha=0.5)
    
    # Panel 2: Current action chunk being executed (top-right)
    ax_chunk = fig.add_subplot(gs[0, 2:])
    ax_chunk.set_title("Action Chunk (16 steps) - Current Exec: Step 0", fontsize=12)
    ax_chunk.set_xlabel("Action Dim", fontsize=10)
    ax_chunk.set_ylabel("Value (px)", fontsize=10)
    ax_chunk.set_xlim(-0.5, 1.5)
    ax_chunk.set_ylim(-50, 560)
    ax_chunk.set_xticks([0, 1])
    ax_chunk.set_xticklabels(['X', 'Y'])
    ax_chunk.grid(True, alpha=0.3)
    
    # Panel 3: Timestep progress bar (middle-right)
    ax_timestep = fig.add_subplot(gs[1, 2:])
    ax_timestep.set_title("DDIM Timestep Progress", fontsize=12)
    ax_timestep.set_xlabel("Denoising Step", fontsize=10)
    ax_timestep.set_ylabel("Timestep t", fontsize=10)
    ax_timestep.grid(True, alpha=0.3)
    
    # Panel 4: Environment trajectory (bottom, span all)
    ax_env = fig.add_subplot(gs[2, :])
    ax_env.set_title("PushT Environment: Agent (blue) + Block (orange) + Goal (green)", fontsize=12)
    ax_env.set_xlim(0, 512)
    ax_env.set_ylim(512, 0)  # Invert Y
    ax_env.set_aspect('equal')
    ax_env.set_xlabel("X (px)")
    ax_env.set_ylabel("Y (px)")
    ax_env.grid(True, alpha=0.3)
    
    # Colors
    C_AGENT = '#2563EB'
    C_BLOCK = '#F59E0B'
    C_GOAL = '#10B981'
    C_ACTION_X = '#EF4444'
    C_ACTION_Y = '#8B5CF6'
    
    # Prepare data
    n_env_steps = len(step_data)
    goal_pos = None
    if step_data:
        # Get goal from env (fixed at 256, 256)
        goal_pos = (256, 256)
    
    # Trajectory storage
    agent_traj = []
    block_traj = []
    
    # Diffusion lines (will be updated)
    diff_lines_x = []
    diff_lines_y = []
    pred_x0_lines_x = []
    pred_x0_lines_y = []
    
    # Initialize diffusion plot
    pred_horizon = 16
    x_indices = np.arange(pred_horizon)
    
    # For each diffusion history entry, we'll animate through denoising steps
    current_diff_idx = 0
    current_denoise_step = 0
    current_env_step = 0
    action_exec_idx = 0
    
    def get_current_diffusion_data():
        """Get diffusion data for current execution step."""
        if current_diff_idx < len(diffusion_history_buffer):
            return diffusion_history_buffer[current_diff_idx]['diffusion']
        return None
    
    def update(frame):
        nonlocal current_diff_idx, current_denoise_step, current_env_step, action_exec_idx
        
        # Determine which phase we're in
        # Phase 1: Show diffusion denoising for current action chunk
        # Phase 2: Show environment step execution
        
        # We'll interleave: show full denoising for chunk, then execute actions
        diffusion_data = get_current_diffusion_data()
        
        if diffusion_data is not None:
            x_history = diffusion_data['x_history']
            pred_x0_history = diffusion_data['pred_x0_history']
            timesteps = diffusion_data['timesteps']
            n_denoise = len(x_history) - 1
            
            # Animate through denoising steps
            if current_denoise_step < n_denoise:
                # Show denoising progress
                current_denoise_step += 1
            elif current_denoise_step == n_denoise and action_exec_idx < policy.action_horizon:
                # Finished denoising, now execute actions one by one
                current_denoise_step += 1  # Move to execution phase
                action_exec_idx = 0
            elif action_exec_idx < policy.action_horizon:
                # Execute actions
                if current_env_step < n_env_steps:
                    current_env_step += 1
                    action_exec_idx += 1
                else:
                    action_exec_idx = policy.action_horizon  # Done
            else:
                # Move to next diffusion chunk
                current_diff_idx += 1
                current_denoise_step = 0
                action_exec_idx = 0
                diffusion_data = get_current_diffusion_data()
        
        # Clear plots
        ax_diff.clear()
        ax_chunk.clear()
        ax_timestep.clear()
        ax_env.clear()
        
        # ===== Panel 1: Diffusion Denoising =====
        if diffusion_data is not None:
            x_history = diffusion_data['x_history']
            pred_x0_history = diffusion_data['pred_x0_history']
            timesteps = diffusion_data['timesteps']
            n_denoise = len(x_history) - 1
            
            step_to_show = min(current_denoise_step, n_denoise)
            
            # Plot all previous denoising steps as faded lines
            for i in range(min(step_to_show + 1, n_denoise + 1)):
                alpha = 0.2 + 0.6 * (i / max(1, n_denoise))
                color = plt.cm.viridis(i / max(1, n_denoise))
                
                x_vals = x_history[i][:, 0]
                y_vals = x_history[i][:, 1]
                
                ax_diff.plot(x_indices, x_vals, color=color, alpha=alpha, linewidth=1, linestyle='--')
                ax_diff.plot(x_indices, y_vals, color=color, alpha=alpha, linewidth=1, linestyle=':')
                
                px0_x = pred_x0_history[i][:, 0]
                px0_y = pred_x0_history[i][:, 1]
                ax_diff.plot(x_indices, px0_x, color=color, alpha=alpha+0.2, linewidth=1.5)
                ax_diff.plot(x_indices, px0_y, color=color, alpha=alpha+0.2, linewidth=1.5)
            
            # Highlight current step
            if step_to_show <= n_denoise:
                x_curr = x_history[step_to_show]
                px0_curr = pred_x0_history[step_to_show] if step_to_show < len(pred_x0_history) else x_curr
                
                ax_diff.plot(x_indices, x_curr[:, 0], color=C_ACTION_X, linewidth=3, label=f'x_t (X) t={timesteps[step_to_show] if step_to_show < len(timesteps) else 0}')
                ax_diff.plot(x_indices, x_curr[:, 1], color=C_ACTION_Y, linewidth=3, label=f'x_t (Y)')
                ax_diff.plot(x_indices, px0_curr[:, 0], color=C_ACTION_X, linewidth=3, linestyle='--', label='pred x0 (X)')
                ax_diff.plot(x_indices, px0_curr[:, 1], color=C_ACTION_Y, linewidth=3, linestyle='--', label='pred x0 (Y)')
            
            ax_diff.set_xlim(-0.5, 15.5)
            ax_diff.set_ylim(-50, 560)
            ax_diff.set_xlabel("Action Step (0-15)")
            ax_diff.set_ylabel("Action Value (px)")
            ax_diff.set_title(f"DDIM Denoising: Step {step_to_show}/{n_denoise} (t={timesteps[step_to_show] if step_to_show < len(timesteps) else 'done'})")
            ax_diff.legend(loc='upper right', fontsize=8)
            ax_diff.grid(True, alpha=0.3)
            
            # ===== Panel 2: Action Chunk =====
            final_actions = diffusion_data['final_actions']  # Already denormalized? No, need to check
            # Actually final_actions from get_action_with_diffusion_history is already denormalized
            # Wait, let me check... it returns denormalized final_actions
            
            ax_chunk.bar([0, 1], [final_actions[action_exec_idx, 0], final_actions[action_exec_idx, 1]], 
                        color=[C_ACTION_X, C_ACTION_Y], alpha=0.8, width=0.6,
                        label=f'Executing Step {action_exec_idx}/{policy.action_horizon}')
            
            # Show all actions in chunk as reference
            ax_chunk.plot([0]*pred_horizon, final_actions[:, 0], 'o-', color=C_ACTION_X, alpha=0.3, label='All X')
            ax_chunk.plot([1]*pred_horizon, final_actions[:, 1], 's-', color=C_ACTION_Y, alpha=0.3, label='All Y')
            
            # Highlight current executing action
            ax_chunk.plot([0], [final_actions[action_exec_idx, 0]], 'o', color=C_ACTION_X, markersize=12, markeredgecolor='white', markeredgewidth=2)
            ax_chunk.plot([1], [final_actions[action_exec_idx, 1]], 's', color=C_ACTION_Y, markersize=12, markeredgecolor='white', markeredgewidth=2)
            
            ax_chunk.set_xlim(-0.5, 1.5)
            ax_chunk.set_ylim(-50, 560)
            ax_chunk.set_xticks([0, 1])
            ax_chunk.set_xticklabels(['X', 'Y'])
            ax_chunk.set_title(f"Action Chunk: Executing #{action_exec_idx+1}/{policy.action_horizon}")
            ax_chunk.legend(fontsize=8)
            ax_chunk.grid(True, alpha=0.3)
            
            # ===== Panel 3: Timestep Progress =====
            denoise_progress = list(range(n_denoise + 1))
            t_values = [timesteps[i] if i < len(timesteps) else 0 for i in denoise_progress]
            
            ax_timestep.plot(denoise_progress, t_values, 'o-', color='gray', alpha=0.5)
            if current_denoise_step <= n_denoise:
                ax_timestep.plot([current_denoise_step], [t_values[current_denoise_step]], 'o', color='red', markersize=10)
            ax_timestep.axvspan(0, n_denoise, alpha=0.1, color='blue', label='Denoising')
            ax_timestep.axvspan(n_denoise, n_denoise + policy.action_horizon, alpha=0.1, color='green', label='Execution')
            ax_timestep.set_xlim(-0.5, n_denoise + policy.action_horizon + 0.5)
            ax_timestep.set_ylim(-5, 105)
            ax_timestep.set_xlabel("Step Index")
            ax_timestep.set_ylabel("Timestep t")
            ax_timestep.set_title("DDIM Progress: Denoising → Execution")
            ax_timestep.legend(fontsize=8)
            ax_timestep.grid(True, alpha=0.3)
        
        # ===== Panel 4: Environment =====
        # Update trajectory
        if current_env_step > 0 and current_env_step <= n_env_steps:
            # Get state from step_data
            sd = step_data[current_env_step - 1]
            agent_pos = sd['obs'][:2]
            block_pos = sd['obs'][2:4]
            agent_traj.append(agent_pos)
            block_traj.append(block_pos)
        
        # Plot trajectories
        if len(agent_traj) > 1:
            agent_traj_arr = np.array(agent_traj)
            block_traj_arr = np.array(block_traj)
            ax_env.plot(agent_traj_arr[:, 0], agent_traj_arr[:, 1], '-', color=C_AGENT, linewidth=2, alpha=0.7, label='Agent')
            ax_env.plot(block_traj_arr[:, 0], block_traj_arr[:, 1], '-', color=C_BLOCK, linewidth=2, alpha=0.7, label='Block')
        
        # Current positions
        if current_env_step > 0 and current_env_step <= n_env_steps:
            sd = step_data[current_env_step - 1]
            agent_pos = sd['obs'][:2]
            block_pos = sd['obs'][2:4]
            ax_env.scatter(*agent_pos, color=C_AGENT, s=100, zorder=5, edgecolor='white', linewidth=2)
            ax_env.scatter(*block_pos, color=C_BLOCK, s=100, zorder=5, edgecolor='white', linewidth=2)
        
        # Start positions
        if len(agent_traj) > 0:
            ax_env.scatter(*agent_traj[0], color=C_AGENT, s=60, marker='o', zorder=5, label='Agent Start')
            ax_env.scatter(*block_traj[0], color=C_BLOCK, s=60, marker='o', zorder=5, label='Block Start')
        
        # Goal
        if goal_pos:
            goal_circle = plt.Circle(goal_pos, 30, color=C_GOAL, alpha=0.2, label='Goal Area')
            ax_env.add_patch(goal_circle)
            ax_env.scatter(*goal_pos, color=C_GOAL, s=150, marker='*', zorder=5, label='Goal')
        
        # Current action arrow
        if current_env_step > 0 and current_env_step <= n_env_steps and action_exec_idx > 0 and action_exec_idx <= policy.action_horizon:
            sd = step_data[current_env_step - 1]
            action = sd['action']
            agent_pos = sd['obs'][:2]
            # Draw action as arrow from agent
            ax_env.annotate('', xy=(agent_pos[0] + action[0] - 256, agent_pos[1] + action[1] - 256), 
                          xytext=agent_pos,
                          arrowprops=dict(arrowstyle='->', color='red', lw=3),
                          zorder=10)
        
        ax_env.set_xlim(0, 512)
        ax_env.set_ylim(512, 0)
        ax_env.set_aspect('equal')
        ax_env.legend(loc='upper right', fontsize=8)
        ax_env.set_title(f"Env Step {current_env_step}/{n_env_steps} | Coverage: {episode_stats['coverage_history'][current_env_step-1] if current_env_step > 0 and current_env_step <= len(episode_stats['coverage_history']) else 0:.3f}")
        
        # Overall title
        fig.suptitle(f"Diffusion Policy Visualization | Episode: Success={episode_stats['success']} | MaxCov={episode_stats['max_coverage']:.3f} | Reward={episode_stats['total_reward']:.1f}", 
                     fontsize=14, fontweight='bold')
        
        return []
    
    # Create animation
    total_frames = 0
    for diff_data in diffusion_history_buffer:
        n_denoise = len(diff_data['diffusion']['x_history']) - 1
        total_frames += n_denoise + 1 + policy.action_horizon  # denoising + transition + execution
    
    # Cap frames
    total_frames = min(total_frames, 300)
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000//fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"[OK] Diffusion animation saved: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


# ============================================================
# Simplified Version: Just Diffusion Process Animation
# ============================================================

def create_diffusion_process_gif(
    diffusion_data: Dict,
    out_path: str,
    fps: int = 5,
    title: str = "DDIM Denoising Process",
):
    """
    Simpler animation: just show the diffusion denoising process for ONE action chunk.
    """
    x_history = diffusion_data['x_history']          # list of (16, 2)
    pred_x0_history = diffusion_data['pred_x0_history']
    timesteps = diffusion_data['timesteps']
    final_actions = diffusion_data['final_actions']
    
    n_steps = len(x_history)
    pred_horizon = x_history[0].shape[0]
    x_indices = np.arange(pred_horizon)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Top-left: X dimension over action steps
    ax1 = axes[0, 0]
    # Top-right: Y dimension over action steps
    ax2 = axes[0, 1]
    # Bottom-left: 2D trajectory of action chunk (X vs Y)
    ax3 = axes[1, 0]
    # Bottom-right: Timestep schedule
    ax4 = axes[1, 1]
    
    C_X = '#EF4444'
    C_Y = '#8B5CF6'
    
    def update(frame):
        for ax in axes.flat:
            ax.clear()
        
        step = min(frame, n_steps - 1)
        t = timesteps[step] if step < len(timesteps) else 0
        
        # Panel 1: X dimension
        for i in range(step + 1):
            alpha = 0.15 + 0.7 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax1.plot(x_indices, x_history[i][:, 0], color=color, alpha=alpha, linewidth=1)
            ax1.plot(x_indices, pred_x0_history[i][:, 0] if i < len(pred_x0_history) else x_history[i][:, 0], 
                    color=color, alpha=alpha, linewidth=1.5, linestyle='--')
        
        # Highlight current
        ax1.plot(x_indices, x_history[step][:, 0], color=C_X, linewidth=3, label=f'x_t (t={t})')
        if step < len(pred_x0_history):
            ax1.plot(x_indices, pred_x0_history[step][:, 0], color=C_X, linewidth=3, linestyle='--', label='pred x0')
        ax1.set_xlim(-0.5, pred_horizon - 0.5)
        ax1.set_ylim(-50, 560)
        ax1.set_xlabel('Action Step')
        ax1.set_ylabel('Action X (px)')
        ax1.set_title(f'X Dimension - Denoising Step {step}/{n_steps-1}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Y dimension
        for i in range(step + 1):
            alpha = 0.15 + 0.7 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax2.plot(x_indices, x_history[i][:, 1], color=color, alpha=alpha, linewidth=1)
            ax2.plot(x_indices, pred_x0_history[i][:, 1] if i < len(pred_x0_history) else x_history[i][:, 1], 
                    color=color, alpha=alpha, linewidth=1.5, linestyle='--')
        
        ax2.plot(x_indices, x_history[step][:, 1], color=C_Y, linewidth=3, label=f'x_t (t={t})')
        if step < len(pred_x0_history):
            ax2.plot(x_indices, pred_x0_history[step][:, 1], color=C_Y, linewidth=3, linestyle='--', label='pred x0')
        ax2.set_xlim(-0.5, pred_horizon - 0.5)
        ax2.set_ylim(-50, 560)
        ax2.set_xlabel('Action Step')
        ax2.set_ylabel('Action Y (px)')
        ax2.set_title(f'Y Dimension - Denoising Step {step}/{n_steps-1}')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: 2D Action Chunk (X vs Y)
        for i in range(step + 1):
            alpha = 0.1 + 0.5 * (i / max(1, n_steps - 1))
            color = plt.cm.plasma(i / max(1, n_steps - 1))
            ax3.plot(x_history[i][:, 0], x_history[i][:, 1], 'o-', color=color, alpha=alpha, markersize=3)
            if i < len(pred_x0_history):
                ax3.plot(pred_x0_history[i][:, 0], pred_x0_history[i][:, 1], 's--', color=color, alpha=alpha, markersize=3)
        
        # Final actions as target
        ax3.plot(final_actions[:, 0], final_actions[:, 1], 'o-', color='green', linewidth=2, markersize=4, 
                label='Final (denorm)', alpha=0.8)
        
        ax3.set_xlim(-50, 560)
        ax3.set_ylim(-50, 560)
        ax3.set_aspect('equal')
        ax3.set_xlabel('Action X (px)')
        ax3.set_ylabel('Action Y (px)')
        ax3.set_title(f'Action Chunk Trajectory (X vs Y) - Step {step}')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Timestep schedule
        ax4.plot(range(len(timesteps)), timesteps, 'o-', color='gray', alpha=0.5)
        ax4.axvline(step, color='red', linestyle='--', linewidth=2, label=f'Current: t={t}')
        ax4.fill_between(range(len(timesteps)), 0, timesteps, alpha=0.1, color='blue')
        ax4.set_xlabel('Denoising Step Index')
        ax4.set_ylabel('Timestep t')
        ax4.set_title('DDIM Timestep Schedule (T→1)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000//fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"[OK] Diffusion process GIF: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Diffusion Process + Policy Execution")
    parser.add_argument("--ckpt", type=str, default="checkpoints/pusht_state_diffusion/best.pt")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to visualize")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=20, help="DDIM steps")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=8, help="Animation FPS")
    parser.add_argument("--outdir", type=str, default="viz_output/diffusion_process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=['diffusion_only', 'full'], default='diffusion_only',
                        help="diffusion_only: just denoising process; full: denoising + env")
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent
    ckpt_path = repo_root / args.ckpt
    outdir = repo_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    # Create env
    env = gym.make("gym_pusht/PushT-v0", max_episode_steps=1000, render_mode="rgb_array")
    # Override success threshold to 0.80 (env default is 0.95)
    env.unwrapped.success_threshold = 0.80
    
    # Load policy
    global policy  # For animation access
    policy = load_policy_for_visualization(
        ckpt_path=str(ckpt_path),
        device=args.device,
        num_inference_steps=args.inference_steps,
    )
    
    print(f"\n[Visualize] Mode: {args.mode}")
    print(f"[Visualize] Episodes: {args.episodes}, DDIM steps: {args.inference_steps}")
    print(f"[Visualize] Output: {outdir}")
    
    rng = np.random.default_rng(args.seed)
    
    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))
        print(f"\n--- Episode {ep+1}/{args.episodes} (seed={ep_seed}) ---")
        
        diffusion_history_buffer, step_data, episode_stats = run_episode_with_diffusion(
            policy, env,
            max_steps=args.max_steps,
            seed=ep_seed,
            record_diffusion_every=1,
        )
        
        print(f"  Steps: {episode_stats['steps']}, Reward: {episode_stats['total_reward']:.1f}")
        print(f"  Max Coverage: {episode_stats['max_coverage']:.3f}, Success: {episode_stats['success']}")
        print(f"  Diffusion chunks recorded: {len(diffusion_history_buffer)}")
        
        if args.mode == 'diffusion_only':
            # Just visualize first diffusion chunk denoising process
            if diffusion_history_buffer:
                diff_data = diffusion_history_buffer[0]['diffusion']
                gif_path = outdir / f"ep{ep}_seed{ep_seed}_diffusion_process.gif"
                create_diffusion_process_gif(
                    diff_data, str(gif_path), fps=args.fps,
                    title=f"Ep {ep} | Seed {ep_seed} | DDIM Denoising ({args.inference_steps} steps)"
                )
        else:
            # Full animation (denoising + env) - more complex, may be slow
            gif_path = outdir / f"ep{ep}_seed{ep_seed}_full.gif"
            create_diffusion_animation(
                diffusion_history_buffer, step_data, episode_stats,
                str(gif_path), fps=args.fps
            )
    
    env.close()
    print(f"\n=== Done! Files in {outdir} ===")


if __name__ == "__main__":
    main()