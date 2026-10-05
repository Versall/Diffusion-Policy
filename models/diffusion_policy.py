"""
Diffusion Policy — Inference Wrapper & DDIM Sampler
====================================================

File ini berisi semua komponen untuk MENJALANKAN (inference) Diffusion Policy:
    1. DDIMSampler              — sampler cepat & deterministik (10 langkah)
    2. DDIMSamplerWithHistory   — varian yang menyimpan history (untuk visualisasi)
    3. DiffusionPolicy          — wrapper siap pakai di environment
    4. Factory functions        — load_policy, load_policy_for_eval, dll.

K2 UPDATE:
    - Conditioning sekarang memakai obs_horizon=2 frame (flatten).
    - global_cond_dim = obs_horizon * state_dim = 2 * 5 = 10.
    - Identik dengan pendekatan repo asli (real-stanford/diffusion_policy):
        global_cond = obs_features.reshape(B, -1)
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from models.mlp_noise_head import ConditionalUnet1D
from models.noise_schedule import NoiseScheduleCosine
from data.pusht_dataset import normalize_data, unnormalize_data


# ============================================================
#  BAGIAN 1: DDIM SAMPLER
# ============================================================
class DDIMSampler:
    """
    Sampler DDIM (Denoising Diffusion Implicit Model).

    Prinsip:
        1. Pre-compute α_bar untuk 10 timestep inference (dari T=100 → t=1).
        2. Iterasi t=100 → 1:
              a. Model prediksi noise ε dari (x_t, t, cond).
              b. Estimasi x_0 dari x_t dan ε.
              c. Hitung x_{t-1} dari x_0_estimasi dan ε.
        3. Hasil akhir = x_0 = action chunk.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_sched: NoiseScheduleCosine,
        num_inference_steps: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.noise_sched = noise_sched
        self.num_inference_steps = num_inference_steps
        self.device = device

        self.timesteps = np.linspace(
            noise_sched.T, 1, num_inference_steps, dtype=int
        ).copy()
        self.alpha_bar = noise_sched.alpha_bar[self.timesteps].cpu().numpy()
        self.alpha_bar_prev = np.concatenate([
            self.alpha_bar[1:],
            noise_sched.alpha_bar[0:1].cpu().numpy(),
        ])

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        action_dim: int = 2,
        pred_horizon: int = 16,
    ) -> torch.Tensor:
        B = cond.shape[0]
        x = torch.randn(B, action_dim, pred_horizon, device=self.device)

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_batch, cond)

            alpha_t = self.alpha_bar[i]
            alpha_prev = self.alpha_bar_prev[i]

            pred_x0 = (x - np.sqrt(1 - alpha_t) * eps_pred) / np.sqrt(alpha_t)
            dir_xt = np.sqrt(1 - alpha_prev) * eps_pred

            if i == len(self.timesteps) - 1:
                x = pred_x0
            else:
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt

        return x.permute(0, 2, 1)


# ============================================================
#  BAGIAN 2: DDIM SAMPLER DENGAN HISTORY
# ============================================================
class DDIMSamplerWithHistory(DDIMSampler):
    """Varian yang menyimpan history untuk visualisasi denoising."""

    @torch.no_grad()
    def sample_with_history(
        self,
        cond: torch.Tensor,
        action_dim: int = 2,
        pred_horizon: int = 16,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        B = cond.shape[0]
        x = torch.randn(B, action_dim, pred_horizon, device=self.device)
        x_history = [x.clone().cpu()]
        pred_x0_history = []

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_batch, cond)

            alpha_t = self.alpha_bar[i]
            alpha_prev = self.alpha_bar_prev[i]

            pred_x0 = (x - np.sqrt(1 - alpha_t) * eps_pred) / np.sqrt(alpha_t)
            pred_x0_history.append(pred_x0.clone().cpu())

            dir_xt = np.sqrt(1 - alpha_prev) * eps_pred

            if i == len(self.timesteps) - 1:
                x = pred_x0
            else:
                x = np.sqrt(alpha_prev) * pred_x0 + dir_xt

            x_history.append(x.clone().cpu())

        final_actions = x.permute(0, 2, 1)
        return final_actions, x_history, pred_x0_history


# ============================================================
#  BAGIAN 3: DIFFUSION POLICY WRAPPER
# ============================================================
class DiffusionPolicy:
    """
    Wrapper policy — jembatan antara environment dan model Diffusion.

    K2 UPDATE:
        Conditioning sekarang memakai SELURUH obs_horizon frame (2 frame),
        bukan hanya frame terakhir. Ini memberi model informasi temporal
        (kecepatan agent & block) yang dibutuhkan untuk prediksi akurat.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        num_inference_steps: int = 10,
        use_history_sampler: bool = False,
    ):
        self.device = torch.device(device)

        # --- 1. Load checkpoint ---
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        action_stats = ckpt["action_stats"]
        state_stats = ckpt["state_stats"]

        print(f"[Policy] Loaded: {ckpt_path}")
        print(f"[Policy] Epoch: {ckpt['epoch']}, Val loss: {ckpt['val_loss']:.6f}")
        print(
            f"[Policy] Config: pred_horizon={config['pred_horizon']}, "
            f"obs_horizon={config['obs_horizon']}, "
            f"action_horizon={config['action_horizon']}, "
            f"global_cond_dim={config['global_cond_dim']}"
        )

        # --- 2. Bangun model dari config ---
        self.model = ConditionalUnet1D(
            input_dim=config["input_dim"],
            global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config["diffusion_step_embed_dim"],
            down_dims=config["down_dims"],
            kernel_size=config["kernel_size"],
            n_groups=config["n_groups"],
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # --- 3. Noise schedule ---
        self.noise_sched = NoiseScheduleCosine(
            T=config["num_diffusion_steps"],
            s=config["noise_schedule_s"],
            device=self.device,
        )

        # --- 4. Sampler ---
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

        # --- 5. Config & stats ---
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config["obs_horizon"]
        self.action_horizon = config["action_horizon"]
        self.action_stats = action_stats
        self.state_stats = state_stats
        self.config = config

        # --- 6. Obs buffer ---
        self.obs_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------
    #  Buffer management
    # ------------------------------------------------------------
    def reset(self):
        """Reset obs buffer. Panggil di awal setiap episode."""
        self.obs_buffer = []

    def _init_buffer(self, obs: np.ndarray):
        """Duplikasi obs pertama untuk mengisi buffer awal (obs_horizon frame)."""
        obs_norm = normalize_data(
            obs.reshape(1, -1).astype(np.float32), self.state_stats
        ).squeeze(0)
        self.obs_buffer = [obs_norm] * self.obs_horizon

    def _update_buffer_and_get_cond(self, obs: np.ndarray) -> torch.Tensor:
        """
        Shift buffer dengan obs baru, return conditioning tensor.

        K2 FIX:
            Sebelumnya: cond = stack(buffer)[:, 0]     → (1, 5)  — hanya frame terakhir
            Sekarang:   cond = stack(buffer).reshape(B, -1) → (1, 10) — semua frame

        Shape:
            obs              (5,)
            buffer           list of 2 × (5,)
            stack(buffer)    (2, 5)
            unsqueeze(0)     (1, 2, 5)
            reshape(1, -1)   (1, 10)   ← conditioning
        """
        if len(self.obs_buffer) == 0:
            self._init_buffer(obs)

        # Normalisasi obs baru → [-1, 1]
        obs_norm = normalize_data(
            obs.reshape(1, -1).astype(np.float32), self.state_stats
        ).squeeze(0)

        # Shift buffer (buang paling lama, tambah baru)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs_norm)

        # Stack jadi (1, obs_horizon, 5), lalu flatten jadi (1, 10)
        cond = torch.from_numpy(
            np.stack(self.obs_buffer)
        ).unsqueeze(0).to(self.device)                # (1, 2, 5)

        # K2 FIX: flatten obs_horizon × state_dim
        cond = cond.reshape(cond.shape[0], -1)        # (1, 10)

        return cond

    # ------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Prediksi action chunk dari observasi."""
        cond = self._update_buffer_and_get_cond(obs)

        action_chunk_norm = self.sampler.sample(
            cond=cond,
            action_dim=2,
            pred_horizon=self.pred_horizon,
        )

        action_chunk = unnormalize_data(
            action_chunk_norm.cpu().numpy().reshape(-1, 2),
            self.action_stats,
        ).reshape(self.pred_horizon, 2)

        return action_chunk[: self.action_horizon]

    @torch.no_grad()
    def get_action_with_diffusion_history(self, obs: np.ndarray) -> Dict:
        """Sama seperti get_action, tapi dengan history denoising."""
        if not isinstance(self.sampler, DDIMSamplerWithHistory):
            raise RuntimeError("Butuh use_history_sampler=True saat __init__")

        cond = self._update_buffer_and_get_cond(obs)

        final_actions_norm, x_history, pred_x0_history = \
            self.sampler.sample_with_history(
                cond=cond,
                action_dim=2,
                pred_horizon=self.pred_horizon,
            )

        final_actions = unnormalize_data(
            final_actions_norm.cpu().numpy().reshape(-1, 2),
            self.action_stats,
        ).reshape(self.pred_horizon, 2)

        x_history_denorm: List[np.ndarray] = []
        for x_h in x_history:
            arr = x_h.permute(0, 2, 1).cpu().numpy().reshape(-1, 2)
            arr = unnormalize_data(arr, self.action_stats).reshape(
                1, self.pred_horizon, 2
            )
            x_history_denorm.append(arr[0])

        pred_x0_history_denorm: List[np.ndarray] = []
        for px0_h in pred_x0_history:
            arr = px0_h.permute(0, 2, 1).cpu().numpy().reshape(-1, 2)
            arr = unnormalize_data(arr, self.action_stats).reshape(
                1, self.pred_horizon, 2
            )
            pred_x0_history_denorm.append(arr[0])

        return {
            "final_actions": final_actions,
            "x_history": x_history_denorm,
            "pred_x0_history": pred_x0_history_denorm,
            "timesteps": self.sampler.timesteps.copy(),
            "cond": cond.cpu().numpy()[0],
        }


# ============================================================
#  BAGIAN 4: FACTORY FUNCTIONS
# ============================================================
def load_policy(
    ckpt_path: str,
    device: str = "cpu",
    num_inference_steps: int = 10,
    use_history_sampler: bool = False,
) -> DiffusionPolicy:
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
    return load_policy(
        ckpt_path, device, num_inference_steps,
        use_history_sampler=False,
    )


def load_policy_for_visualization(
    ckpt_path: str,
    device: str = "cpu",
    num_inference_steps: int = 20,
) -> DiffusionPolicy:
    return load_policy(
        ckpt_path, device, num_inference_steps,
        use_history_sampler=True,
    )