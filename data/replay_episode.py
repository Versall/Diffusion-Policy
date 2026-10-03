"""
Replay akurat: set initial state langsung ke pymunk bodies → step action → render.

Setiap episode punya initial state (agent & block) BERBEDA dari data.
Goal SELALU sama: (256, 256, 45°) di tengah workspace.

_set_state() punya bug: physics step menggeser block karena CoM offset.
Solusi: set posisi agent/block langsung setelah reset, TANPA _set_state.

Usage:
  dp_pusht/Scripts/python.exe data/replay_accurate.py --ep 0
  dp_pusht/Scripts/python.exe data/replay_accurate.py --ep 174 --fps 15
"""
import argparse, zarr, numpy as np, os
import gymnasium as gym
import gym_pusht
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--ep", type=int, default=0, help="Episode index")
parser.add_argument("--fps", type=int, default=10, help="FPS")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

# ── Load data ──
DATAROOT = os.path.join(os.path.dirname(__file__), "pusht", "pusht_cchi_v7_replay.zarr")
OUTDIR = os.path.join(os.path.dirname(__file__), "viz_output")
os.makedirs(OUTDIR, exist_ok=True)

root = zarr.open(DATAROOT, mode="r")
ends = root["meta/episode_ends"][:]
state_all = root["data/state"][:]
action_all = root["data/action"][:]

k = args.ep
s = int(ends[k-1]) if k > 0 else 0
e = int(ends[k])
actions = action_all[s:e]
init_state = state_all[s]
n_frames = len(actions)
print(f"Episode {k}: {n_frames} frames")
print(f"  Data init: agent=({init_state[0]:.1f},{init_state[1]:.1f}) "
      f"block=({init_state[2]:.1f},{init_state[3]:.1f}) angle={np.degrees(init_state[4]):.1f}°")

# ── Create env, set initial state langsung ke pymunk bodies ──
env = gym.make("gym_pusht/PushT-v0", max_episode_steps=1000, render_mode="rgb_array")
u = env.unwrapped
env.reset()

# Set posisi langsung (bypass _set_state yang punya bug CoM)
# URUTAN PENTING: angle DULU, baru position — karena rotasi pymunk dihitung dari CoM
u.block.angle = init_state[4]
u.block.position = list(init_state[2:4])
u.agent.position = list(init_state[:2])

# Verifikasi
obs = u.get_obs()
print(f"  After direct set: agent=({obs[0]:.1f},{obs[1]:.1f}) "
      f"block=({obs[2]:.1f},{obs[3]:.1f}) angle={np.degrees(obs[4]):.1f}°")
match_agent = np.allclose(obs[:2], init_state[:2], atol=1)
match_block = np.allclose(obs[2:4], init_state[2:4], atol=1)
print(f"  Match data: agent={'✓' if match_agent else '✗'} block={'✓' if match_block else '✗'}")

# Goal
goal = u.goal_pose
print(f"  Goal: ({goal[0]:.1f}, {goal[1]:.1f}, {np.degrees(goal[2]):.1f}°) — SELALU SAMA")

# ── Replay ──
frames = [env.render()]
for i in range(n_frames):
    obs, reward, terminated, truncated, info = env.step(actions[i])
    frames.append(env.render())
    if i % 50 == 0:
        bl = obs[2:4]
        dist = np.sqrt((bl[0] - goal[0])**2 + (bl[1] - goal[1])**2)
        print(f"  frame {i}: block=({bl[0]:.1f},{bl[1]:.1f}) dist_to_goal={dist:.1f}")
    if terminated or truncated:
        print(f"  → terminated/truncated at frame {i}")
        break

env.close()
print(f"Collected {len(frames)} frames")

# ── Save GIF ──
fig, ax = plt.subplots(figsize=(5, 5))
img = ax.imshow(frames[0])
ax.axis("off")
title = ax.set_title(f"Ep {k} — frame 0/{len(frames)}", color="white", fontsize=11)
fig.patch.set_facecolor("black")

def update(idx):
    img.set_data(frames[idx])
    title.set_text(f"Ep {k} — frame {idx}/{len(frames)}")
    return [img, title]

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps, blit=False)

out_path = args.out or os.path.join(OUTDIR, f"replay_ep{k}.gif")
anim.save(out_path, writer=PillowWriter(fps=args.fps), dpi=72)
plt.close(fig)
print(f"[OK] {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
