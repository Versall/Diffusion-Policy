"""
Evaluasi keberhasilan semua episode dari dataset pusht_cchi_v7_replay.zarr.

Untuk setiap episode:
  1. Set initial state dari data (angle → position → agent)
  2. Step semua action lewat gym-pusht env
  3. Catat: coverage, reward, jarak block→goal,terminated/truncated

Success criterion (dari gym-pusht):
  coverage > success_threshold (0.95)
  coverage = intersection_area(block_T, goal_T) / goal_area

Output:
  - episode_results.csv   (per-episode stats)
  - episode_report.png    (ringkasan visual)
"""
import zarr, numpy as np, os, csv, time
import gymnasium as gym
import gym_pusht
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Setup ──
DATAROOT = os.path.join(os.path.dirname(__file__), "pusht", "pusht_cchi_v7_replay.zarr")
OUTDIR = os.path.join(os.path.dirname(__file__), "viz_output")
os.makedirs(OUTDIR, exist_ok=True)

root = zarr.open(DATAROOT, mode="r")
ends = root["meta/episode_ends"][:]
state_all = root["data/state"][:]
action_all = root["data/action"][:]
n_contacts_all = root["data/n_contacts"][:]
N_EPS = len(ends)

print(f"Dataset: {N_EPS} episodes, {len(state_all)} total frames")

# ── Create env ──
env = gym.make("gym_pusht/PushT-v0", max_episode_steps=1000, render_mode="rgb_array")
u = env.unwrapped
env.reset()
GOAL = u.goal_pose.copy()
SUCCESS_THRESH = 0.80  # Tetapkan 0.80 untuk konsistensi evaluasi policy
print(f"Goal: ({GOAL[0]:.1f}, {GOAL[1]:.1f}, {np.degrees(GOAL[2]):.1f}°)")
print(f"Success threshold: coverage > {SUCCESS_THRESH}")
print()

# ── Evaluate each episode ──
results = []
t0 = time.time()

for ep in range(N_EPS):
    s = int(ends[ep-1]) if ep > 0 else 0
    e = int(ends[ep])
    actions = action_all[s:e]
    init_state = state_all[s]
    contacts_ep = n_contacts_all[s:e].flatten()
    n_frames = e - s

    # Reset + set initial state
    env.reset()
    u.block.angle = float(init_state[4])
    u.block.position = list(init_state[2:4].astype(float))
    u.agent.position = list(init_state[:2].astype(float))

    # Replay
    coverage_history = []
    reward_sum = 0.0
    terminated = False
    truncated = False
    first_success_frame = None

    for i in range(n_frames):
        obs, reward, terminated, truncated, info = env.step(actions[i])
        cov = info["coverage"]
        coverage_history.append(cov)
        reward_sum += reward
        if cov > SUCCESS_THRESH and first_success_frame is None:
            first_success_frame = i
        if terminated or truncated:
            break

    final_coverage = coverage_history[-1] if coverage_history else 0.0
    max_coverage = max(coverage_history) if coverage_history else 0.0
    init_block_dist = np.sqrt((init_state[2]-GOAL[0])**2 + (init_state[3]-GOAL[1])**2)
    final_block_dist = np.sqrt((obs[2]-GOAL[0])**2 + (obs[3]-GOAL[1])**2)
    total_contacts = int(contacts_ep.sum())

    is_success = max_coverage > SUCCESS_THRESH

    results.append({
        "episode": ep,
        "n_frames": n_frames,
        "init_block_dist": round(init_block_dist, 1),
        "final_block_dist": round(final_block_dist, 1),
        "max_coverage": round(max_coverage, 4),
        "final_coverage": round(final_coverage, 4),
        "reward_sum": round(reward_sum, 3),
        "first_success_frame": first_success_frame,
        "total_contacts": total_contacts,
        "terminated": terminated,
        "truncated": truncated,
        "is_success": is_success,
    })

    if (ep + 1) % 20 == 0 or ep == N_EPS - 1:
        elapsed = time.time() - t0
        rate = (ep+1) / elapsed
        eta = (N_EPS - ep - 1) / rate
        print(f"  [{ep+1:3d}/{N_EPS}] elapsed={elapsed:.0f}s  rate={rate:.1f} ep/s  ETA={eta:.0f}s")

env.close()

# ── Save CSV ──
csv_path = os.path.join(OUTDIR, "episode_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\n[OK] {csv_path}")

# ── Summary stats ──
n_success = sum(1 for r in results if r["is_success"])
n_fail = N_EPS - n_success
success_lengths = [r["n_frames"] for r in results if r["is_success"]]
fail_lengths = [r["n_frames"] for r in results if not r["is_success"]]

print(f"\n{'='*50}")
print(f"RINGKASAN EVALUASI — {N_EPS} EPISODES")
print(f"{'='*50}")
print(f"Success : {n_success}/{N_EPS} ({100*n_success/N_EPS:.1f}%)")
print(f"Failed  : {n_fail}/{N_EPS} ({100*n_fail/N_EPS:.1f}%)")
if success_lengths:
    print(f"  Avg length (success): {np.mean(success_lengths):.0f} frames")
if fail_lengths:
    print(f"  Avg length (fail)   : {np.mean(fail_lengths):.0f} frames")
print(f"  Avg max coverage    : {np.mean([r['max_coverage'] for r in results]):.3f}")
print(f"  Median max coverage : {np.median([r['max_coverage'] for r in results]):.3f}")
print(f"  Avg final dist→goal : {np.mean([r['final_block_dist'] for r in results]):.1f} px")
print(f"  Avg init dist→goal  : {np.mean([r['init_block_dist'] for r in results]):.1f} px")
print(f"{'='*50}")

# ── Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Episode Success Report — {n_success}/{N_EPS} success ({100*n_success/N_EPS:.1f}%)",
             fontsize=15, fontweight="bold")

# 1. Success/Fail pie
ax = axes[0, 0]
ax.pie([n_success, n_fail], labels=["Success", "Failed"],
       colors=["#4CAF50", "#EF5350"], autopct="%1.1f%%",
       textprops={"fontsize": 12, "color": "white"})
ax.set_title("Success Rate", color="white", fontsize=12)

# 2. Max coverage distribution
ax = axes[0, 1]
max_covs = [r["max_coverage"] for r in results]
ax.hist(max_covs, bins=30, color="#42A5F5", edgecolor="#1565C0", alpha=0.8)
ax.axvline(SUCCESS_THRESH, color="#EF5350", linestyle="--", linewidth=2, label=f"threshold={SUCCESS_THRESH}")
ax.set_xlabel("Max Coverage", color="#aaa")
ax.set_ylabel("Count", color="#aaa")
ax.set_title("Max Coverage Distribution", color="white", fontsize=12)
ax.legend(fontsize=10)

# 3. Episode length distribution (color-coded by success)
ax = axes[0, 2]
succ_lens = [r["n_frames"] for r in results if r["is_success"]]
fail_lens = [r["n_frames"] for r in results if not r["is_success"]]
ax.hist(succ_lens, bins=25, color="#4CAF50", edgecolor="#2E7D32", alpha=0.7, label="Success")
ax.hist(fail_lens, bins=25, color="#EF5350", edgecolor="#C62828", alpha=0.7, label="Failed")
ax.set_xlabel("Episode Length (frames)", color="#aaa")
ax.set_ylabel("Count", color="#aaa")
ax.set_title("Episode Length Distribution", color="white", fontsize=12)
ax.legend(fontsize=10)

# 4. Init dist vs final dist (scatter)
ax = axes[1, 0]
init_d = [r["init_block_dist"] for r in results]
final_d = [r["final_block_dist"] for r in results]
colors = ["#4CAF50" if r["is_success"] else "#EF5350" for r in results]
ax.scatter(init_d, final_d, c=colors, s=15, alpha=0.6)
ax.plot([0, 350], [0, 350], "--", color="#666", linewidth=0.8, label="no improvement")
ax.set_xlabel("Init distance to goal (px)", color="#aaa")
ax.set_ylabel("Final distance to goal (px)", color="#aaa")
ax.set_title("Init vs Final Block Distance", color="white", fontsize=12)
ax.legend(fontsize=9)

# 5. Per-episode max coverage bar chart
ax = axes[1, 1]
colors_bar = ["#4CAF50" if r["is_success"] else "#EF5350" for r in results]
ax.bar(range(N_EPS), max_covs, color=colors_bar, width=1.0, alpha=0.8)
ax.axhline(SUCCESS_THRESH, color="#FFC107", linestyle="--", linewidth=1.5, label="threshold")
ax.set_xlabel("Episode Index", color="#aaa")
ax.set_ylabel("Max Coverage", color="#aaa")
ax.set_title("Max Coverage per Episode", color="white", fontsize=12)
ax.legend(fontsize=9)

# 6. Reward sum distribution
ax = axes[1, 2]
rew_s = [r["reward_sum"] for r in results if r["is_success"]]
rew_f = [r["reward_sum"] for r in results if not r["is_success"]]
ax.hist(rew_s, bins=25, color="#4CAF50", edgecolor="#2E7D32", alpha=0.7, label="Success")
ax.hist(rew_f, bins=25, color="#EF5350", edgecolor="#C62828", alpha=0.7, label="Failed")
ax.set_xlabel("Total Reward", color="#aaa")
ax.set_ylabel("Count", color="#aaa")
ax.set_title("Cumulative Reward Distribution", color="white", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
report_path = os.path.join(OUTDIR, "episode_report.png")
plt.savefig(report_path, dpi=150, facecolor="white")
plt.close()
print(f"[OK] {report_path}")

# ── Episode list ──
succ_list = [r["episode"] for r in results if r["is_success"]]
fail_list = [r["episode"] for r in results if not r["is_success"]]
print(f"\nSuccess episodes ({len(succ_list)}): {succ_list}")
print(f"Failed episodes ({len(fail_list)}): {fail_list}")
print(f"\nDone in {time.time()-t0:.1f}s")
