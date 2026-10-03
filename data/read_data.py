import zarr
import os

DATAROOT = os.path.join(os.path.dirname(__file__), "pusht", "pusht_cchi_v7_replay.zarr")

root = zarr.open(DATAROOT, mode="r")
ends = root["meta/episode_ends"][:]

def load_episode(k):
    start = ends[k-1] if k > 0 else 0
    stop = ends[k]
    
    obs = root["data/state"][start:stop]
    action = root["data/action"][start:stop]
    return obs, action

obs, action = load_episode(0)

print(obs)
print(action)