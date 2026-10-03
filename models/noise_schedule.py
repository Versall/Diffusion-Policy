from __future__ import annotations
import numpy as np 
import torch 

def alpha_bar_cosine(t: float, T: float, s: float = 0.008) -> float:
    return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

class NoiseScheduleCosine:
    def __init__(self, T: float, s: float = 0.008, clip_min: float = 1e-3, device: str | torch.device = "cpu"):
        self.T = T
        self.s = s
        self.clip_min = clip_min
        self.device = device
        
        t = np.arange(0, T + 1)
        self.alpha_bar = np.clip(alpha_bar_cosine(t, T, s), self.clip_min, None)
        
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.beta = 1 - self.alpha
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(np.clip(1 - self.alpha_bar, self.clip_min, None))
        
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(self.beta, dtype=torch.float32, device=self.device)
        self.sqrt_alpha_bar = torch.tensor(self.sqrt_alpha_bar, dtype=torch.float32, device=self.device)
        self.sqrt_one_minus_alpha_bar = torch.tensor(self.sqrt_one_minus_alpha_bar, dtype=torch.float32, device=self.device)
    
    def q_sample(self, x_0, t: float, noise=None):
        
        if noise is None:
            noise = torch.randn_like(x_0)
        s_alpha_bar = self.sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        s_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        return s_alpha_bar * x_0 + s_one_minus_alpha_bar * noise