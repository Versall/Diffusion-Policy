"""
Shared Diffusion Policy Components
===================================
Common classes used by train_pusht.py, eval_policy.py, visualize_diffusion_process.py, visualize_policy.py
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

# Local imports
from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine
from data.pusht_dataset import normalize_data, unnormalize_data


# ============================================================
# DDIM Sampler (Base)
# ============================================================

class DDIMSampler:
    """
    DDIM sampling untuk diffusion policy.
    Deterministik, lebih cepat dari DDPM (num_inference_steps << training steps).
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_sched: NoiseScheduleCosine,
        num_inference_steps: int = 10,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.noise_sched = noise_sched
        self.num_inference_steps = num_inference_steps
        self.device = device

        # Timesteps untuk inference: T -> 1 (noisy -> clean)
        self.timesteps = np.linspace(
            noise_sched.T, 1, num_inference_steps, dtype=int
        ).copy()
        self.timesteps_tensor = torch.tensor(self.timesteps, device=device, dtype=torch.long)

        # Pre-compute alpha_bar untuk timesteps ini
        self.alpha_bar = noise_sched.alpha_bar[self.timesteps].cpu().numpy()
        # alpha_bar_prev: untuk step i, t_current = timesteps[i], t_next = timesteps[i+1] (lower t)
        # Last step goes to t=0 (alpha_bar=1.0)
        self.alpha_bar_prev = np.concatenate([
            self.alpha_bar[1:], noise_sched.alpha_bar[0:1].cpu().numpy()
        ])

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,           # (B, global_cond_dim) - state conditioning
        action_dim: int = 2,
        pred_horizon: int = 16,
    ) -> torch.Tensor:
        """
        Sample action chunk dari noise.
        Returns: (B, pred_horizon, action_dim) - denormalized actions
        """
        B = cond.shape[0]
        self.model.eval()

        # Start from noise
        x = torch.randn(B, action_dim, pred_horizon, device=self.device)

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Predict noise
            eps_pred = self.model(x, t_batch, cond)  # (B, action_dim, pred_horizon)

            # DDIM update
            alpha_t = self.alpha_bar[i]
            alpha_prev = self.alpha_bar_prev[i]

            # Predicted x_0
            pred_x0 = (x - np.sqrt(1 - alpha_t) * eps_pred) / np.sqrt(alpha_t)

            # Direction to x_t
            dir_xt = np.sqrt(1 - alpha_prev) * eps_pred

            # Next x
            if i == len(self.timesteps) - 1:
                # Last step: x_0 langsung
                x = pred_x0
            else:
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt

        # Return (B, pred_horizon, action_dim)
        return x.permute(0, 2, 1)


# ============================================================
# DDIM Sampler dengan History (untuk visualisasi proses denoising)
# ============================================================

class DDIMSamplerWithHistory(DDIMSampler):
    """
    Extended DDIM sampler yang menyimpan history setiap step untuk visualisasi.
    Inherits from DDIMSampler, overrides sample method to return history.
    """

    @torch.no_grad()
    def sample_with_history(
        self,
        cond: torch.Tensor,           # (B, global_cond_dim)
        action_dim: int = 2,
        pred_horizon: int = 16,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            final_actions: (B, pred_horizon, action_dim)
            x_history: list of x at each step (noisy -> clean), each (B, action_dim, pred_horizon)
            pred_x0_history: list of predicted x0 at each step, each (B, action_dim, pred_horizon)
        """
        B = cond.shape[0]
        self.model.eval()

        x = torch.randn(B, action_dim, pred_horizon, device=self.device)

        x_history = [x.clone().cpu()]  # Initial noise
        pred_x0_history = []

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_batch, cond)

            alpha_t = self.alpha_bar[i]
            alpha_prev = self.alpha_bar_prev[i]

            # Predicted x0 (denoised)
            pred_x0 = (x - np.sqrt(1 - alpha_t) * eps_pred) / np.sqrt(alpha_t)
            pred_x0_history.append(pred_x0.clone().cpu())

            dir_xt = np.sqrt(1 - alpha_prev) * eps_pred

            if i == len(self.timesteps) - 1:
                x = pred_x0
            else:
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt

            x_history.append(x.clone().cpu())

        # Final: (B, pred_horizon, action_dim)
        final_actions = x.permute(0, 2, 1)

        return final_actions, x_history, pred_x0_history


# ============================================================
# Diffusion Policy Wrapper
# ============================================================

class DiffusionPolicy:
    """
    Wrapper untuk inference diffusion policy di env.
    Load checkpoint, handle normalization, action horizon queue.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        num_inference_steps: int = 10,
        use_history_sampler: bool = False,
    ):
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt['config']
        action_stats = ckpt['action_stats']
        state_stats = ckpt['state_stats']

        print(f"[Policy] Loaded: {ckpt_path}")
        print(f"[Policy] Epoch: {ckpt['epoch']}, Val loss: {ckpt['val_loss']:.6f}")
        print(f"[Policy] Config: pred_horizon={config['pred_horizon']}, obs_horizon={config['obs_horizon']}, action_horizon={config['action_horizon']}")

        # Build model
        self.model = ConditionalUnet1D(
            input_dim=config['input_dim'],
            global_cond_dim=config['global_cond_dim'],
            diffusion_step_embed_dim=config['diffusion_step_embed_dim'],
            down_dims=config['down_dims'],
            kernel_size=config['kernel_size'],
            n_groups=config['n_groups'],
        ).to(self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # Noise schedule
        self.noise_sched = NoiseScheduleCosine(
            T=config['num_diffusion_steps'],
            s=config['noise_schedule_s'],
            device=self.device,
        )

        # Sampler (base or with history)
        if use_history_sampler:
            self.sampler = DDIMSamplerWithHistory(
                self.model, self.noise_sched,
                num_inference_steps=num_inference_steps,
                device=self.device,
            )
        else:
            self.sampler = DDIMSampler(
                self.model, self.noise_sched,
                num_inference_steps=num_inference_steps,
                device=self.device,
            )

        # Config
        self.pred_horizon = config['pred_horizon']
        self.obs_horizon = config['obs_horizon']
        self.action_horizon = config['action_horizon']
        self.action_stats = action_stats
        self.state_stats = state_stats
        self.config = config

        # Obs buffer untuk obs_horizon
        self.obs_buffer = []

    def reset(self):
        """Reset obs buffer."""
        self.obs_buffer = []

    def _init_buffer(self, obs: np.ndarray):
        """Initialize buffer with replicated first obs (like training padding)."""
        obs_norm = normalize_data(obs.reshape(1, -1).astype(np.float32), self.state_stats).squeeze(0)
        self.obs_buffer = [obs_norm] * self.obs_horizon

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (5,) state = [agent_x, agent_y, block_x, block_y, block_angle]
        return: (action_horizon, 2) actions to execute
        """
        # Initialize buffer on first call
        if len(self.obs_buffer) == 0:
            self._init_buffer(obs)

        # Normalize obs
        obs_norm = normalize_data(obs.reshape(1, -1).astype(np.float32), self.state_stats).squeeze(0)

        # Update buffer (shift + append)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs_norm)

        # Stack: (obs_horizon, 5) -> (1, obs_horizon, 5)
        cond = torch.from_numpy(np.stack(self.obs_buffer)).unsqueeze(0).to(self.device)  # (1, T_o, 5)

        # Match training: gunakan step pertama saja sebagai kondisi (state[:, 0])
        cond = cond[:, 0]  # (1, 5)

        # Sample action chunk
        action_chunk_norm = self.sampler.sample(
            cond=cond,
            action_dim=2,
            pred_horizon=self.pred_horizon,
        )  # (1, pred_horizon, 2)

        # Denormalize
        action_chunk = unnormalize_data(
            action_chunk_norm.cpu().numpy().reshape(-1, 2),
            self.action_stats
        ).reshape(self.pred_horizon, 2)

        # Return first action_horizon actions
        return action_chunk[:self.action_horizon]

    @torch.no_grad()
    def get_action_with_diffusion_history(self, obs: np.ndarray) -> dict:
        """
        Get action + full diffusion history for visualization.
        Requires use_history_sampler=True at init.
        Returns dict with:
            - final_actions: (pred_horizon, 2) denormalized
            - x_history: list of (pred_horizon, 2) denormalized
            - pred_x0_history: list of (pred_horizon, 2) denormalized
            - timesteps: array of timesteps
            - cond: (5,) normalized condition
        """
        if not isinstance(self.sampler, DDIMSamplerWithHistory):
            raise RuntimeError("Need use_history_sampler=True at init")

        if len(self.obs_buffer) == 0:
            self._init_buffer(obs)

        obs_norm = normalize_data(obs.reshape(1, -1).astype(np.float32), self.state_stats).squeeze(0)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs_norm)

        cond = torch.from_numpy(np.stack(self.obs_buffer)).unsqueeze(0).to(self.device)
        cond = cond[:, 0]  # (1, 5)

        # Sample dengan history
        final_actions_norm, x_history, pred_x0_history = self.sampler.sample_with_history(
            cond=cond,
            action_dim=2,
            pred_horizon=self.pred_horizon,
        )

        # Denormalize final actions
        final_actions = unnormalize_data(
            final_actions_norm.cpu().numpy().reshape(-1, 2),
            self.action_stats
        ).reshape(self.pred_horizon, 2)

        # Denormalize history untuk visualisasi
        x_history_denorm = []
        pred_x0_history_denorm = []

        for x_h in x_history:
            x_denorm = unnormalize_data(
                x_h.permute(0, 2, 1).cpu().numpy().reshape(-1, 2),
                self.action_stats
            ).reshape(1, self.pred_horizon, 2)
            x_history_denorm.append(x_denorm[0])

        for px0_h in pred_x0_history:
            px0_denorm = unnormalize_data(
                px0_h.permute(0, 2, 1).cpu().numpy().reshape(-1, 2),
                self.action_stats
            ).reshape(1, self.pred_horizon, 2)
            pred_x0_history_denorm.append(px0_denorm[0])

        return {
            'final_actions': final_actions,           # (pred_horizon, 2)
            'x_history': x_history_denorm,            # list of (pred_horizon, 2)
            'pred_x0_history': pred_x0_history_denorm, # list of (pred_horizon, 2)
            'timesteps': self.sampler.timesteps.copy(),
            'cond': cond.cpu().numpy()[0],            # (5,)
        }


# ============================================================
# Factory Functions
# ============================================================

def load_policy(
    ckpt_path: str,
    device: str = "cpu",
    num_inference_steps: int = 10,
    use_history_sampler: bool = False,
) -> DiffusionPolicy:
    """Convenience factory to load a policy."""
    return DiffusionPolicy(
        ckpt_path=ckpt_path,
        device=device,
        num_inference_steps=num_inference_steps,
        use_history_sampler=use_history_sampler,
    )


def load_policy_for_eval(
    ckpt_path: str,
    device: str = "cpu",
    num_inference_steps: int = 10,
) -> DiffusionPolicy:
    """Load policy for evaluation (base sampler)."""
    return load_policy(ckpt_path, device, num_inference_steps, use_history_sampler=False)


def load_policy_for_visualization(
    ckpt_path: str,
    device: str = "cpu",
    num_inference_steps: int = 20,
) -> DiffusionPolicy:
    """Load policy for diffusion process visualization (history sampler)."""
    return load_policy(ckpt_path, device, num_inference_steps, use_history_sampler=True)