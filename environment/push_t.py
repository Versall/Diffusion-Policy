import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
import time
import os

DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATADIR, exist_ok=True)
EPISODE_LIMIT = 120
episode_count = 33

env = gym.make("gym_pusht/PushT-v0", 
               obs_type="pixels", 
               render_mode="human",
               max_episode_steps=1000)
obs, info = env.reset()

def random_block_outside(
    agent_pos, 
    r_min=120
):
  while True:
      b_pos = np.random.uniform(128, 384, size=2) 
      if np.linalg.norm(b_pos - agent_pos) > r_min:
          break
  return b_pos

def reset_env(env):
    mouse_pos = pygame.mouse.get_pos()
    agent_pos = np.array(mouse_pos)
    b_pos = random_block_outside(agent_pos)
    b_angle = np.random.uniform(0, 2 * np.pi)
    goal_pos = np.random.uniform(128, 384, size=2)
    goal_angle = np.random.uniform(0, 2 * np.pi)
    obs, info = env.reset(options={"reset_to_state": [*agent_pos, *b_pos, b_angle]})
    env.unwrapped.goal_pose = np.array([*goal_pos, goal_angle])
    return obs, info

obs, info = reset_env(env)

ep_obs = []
ep_target = []

while True:
    mouse_pos = pygame.mouse.get_pos()
    target = np.array(mouse_pos, dtype=np.float64)
    
    ep_obs.append(obs)
    ep_target.append(target)
    
    obs, reward, terminated, truncated, info = env.step(target)
    env.render()
    
    time.sleep(0.01)
    
    if truncated or terminated:
        if terminated:
            n = episode_count
            np.savez(os.path.join(DATADIR, f"episode_{n}.npz"), 
                    np.array(ep_obs),
                    np.array(ep_target))
            episode_count += 1
            print(f"[SAVED] episode_{n}.npz w/ {len(ep_obs)} steps")
            
        ep_obs = []
        ep_target = []
            
        obs, info = reset_env(env)
        
        if episode_count >= EPISODE_LIMIT:
            print(f"[INFO] Reached episode limit of {EPISODE_LIMIT}.\nExiting...")
            break

env.close()
