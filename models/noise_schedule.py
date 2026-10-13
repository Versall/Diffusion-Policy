"""
noise_schedule.py — Cosine noise schedule untuk Diffusion Policy.

Formula (Nichol & Dhariwal 2021, "Improved Denoising Diffusion Probabilistic Models"):
    ᾱ(t) = cos²(π/2 · (t/T + s) / (1 + s))

    dengan s = 0.008 (offset kecil untuk stabilitas di t=0).

Nilai default (mengikuti paper Diffusion Policy):
    T = 100 timesteps
    s = 0.008

Device-agnostic:
    - `device=None` → auto-detect via utils.device.get_device()
    - Semua tensor internal (ᾱ, α, β, √ᾱ, √(1-ᾱ)) di self.device
    - Method `to(device)` untuk pindah device tanpa rebuild

Contoh:
    >>> from models.noise_schedule import NoiseScheduleCosine
    >>> sched = NoiseScheduleCosine(T=100)   # auto-detect device
    >>> sched = NoiseScheduleCosine(T=100, device="cuda")  # paksa CUDA
    >>> x_t = sched.q_sample(x_0, t, noise)  # forward diffusion
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Pastikan root repo ada di sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device import get_device


# ============================================================
#  Fungsi alpha_bar
# ============================================================
def alpha_bar_cosine(t: Union[float, np.ndarray],
                     T: float,
                     s: float = 0.008) -> Union[float, np.ndarray]:
    """
    Hitung ᾱ(t) dengan jadwal kosinus.

    Args:
        t: Timestep (0..T). Bisa scalar atau array.
        T: Total timesteps (mis. 100).
        s: Offset kecil untuk hindari ᾱ terlalu cepat turun di awal.

    Returns:
        ᾱ(t) — nilai antara 0 dan 1. ᾱ(0) ≈ 1 (bersih), ᾱ(T) ≈ 0 (full noise).
    """
    return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2


# ============================================================
#  Noise Schedule Cosine
# ============================================================
class NoiseScheduleCosine:
    """
    Cosine noise schedule untuk diffusion.

    Atribut (setelah __init__):
        T:                      int    — total timesteps (100)
        s:                      float  — cosine offset (0.008)
        clip_min:               float  — minimum ᾱ (1e-3, hindari log(0))
        device:                 torch.device — device tensor disimpan
        alpha_bar:              (T+1,)  — cumulative ᾱ dari t=0..T
        alpha:                  (T,)    — α_t = ᾱ_t / ᾱ_{t-1}
        beta:                   (T,)    — β_t = 1 - α_t
        sqrt_alpha_bar:         (T+1,)  — √ᾱ
        sqrt_one_minus_alpha_bar: (T+1,) — √(1-ᾱ)
    """

    def __init__(
        self,
        T: float,
        s: float = 0.008,
        clip_min: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # --- Device auto-detect ---
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.T = int(T)
        self.s = s
        self.clip_min = clip_min

        # ============================================================
        # 1. Hitung ᾱ untuk t = 0, 1, ..., T (numpy dulu)
        # ============================================================
        t = np.arange(0, self.T + 1)
        alpha_bar = np.clip(
            alpha_bar_cosine(t, self.T, s),
            self.clip_min, None,
        )

        # ============================================================
        # 2. Turunkan α, β dari ᾱ
        #    α_t = ᾱ_t / ᾱ_{t-1}       (untuk t = 1..T)
        #    β_t = 1 - α_t
        # ============================================================
        alpha = alpha_bar[1:] / alpha_bar[:-1]
        beta = 1 - alpha

        # Pre-compute sqrt
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = np.sqrt(
            np.clip(1 - alpha_bar, self.clip_min, None)
        )

        # ============================================================
        # 3. Konversi ke torch tensor & pindah ke device
        # ============================================================
        self.alpha_bar = torch.tensor(
            alpha_bar, dtype=torch.float32, device=self.device,
        )
        self.alpha = torch.tensor(
            alpha, dtype=torch.float32, device=self.device,
        )
        self.beta = torch.tensor(
            beta, dtype=torch.float32, device=self.device,
        )
        self.sqrt_alpha_bar = torch.tensor(
            sqrt_alpha_bar, dtype=torch.float32, device=self.device,
        )
        self.sqrt_one_minus_alpha_bar = torch.tensor(
            sqrt_one_minus_alpha_bar, dtype=torch.float32, device=self.device,
        )

    # ------------------------------------------------------------
    #  Forward diffusion: q_sample
    # ------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: Union[torch.Tensor, int],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        Args:
            x_0:   (B, ...) — data bersih.
            t:     (B,) atau scalar — timestep.
            noise: (B, ...) — noise ε. Kalau None, akan di-sample.

        Returns:
            x_t: (B, ...) — data yang sudah dinoisekan.
        """
        # Sample noise kalau tidak diberikan
        if noise is None:
            noise = torch.randn_like(x_0)

        # Pastikan t di device yang sama dengan tensor schedule
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.long)
        else:
            t = t.to(self.device)

        # Pastikan x_0 dan noise di device yang sama dengan schedule
        x_0 = x_0.to(self.device)
        noise = noise.to(self.device)

        # Ambil √ᾱ_t dan √(1-ᾱ_t) untuk t (batch)
        # Broadcast ke bentuk x_0: (B, 1, 1, ...) untuk dimensi > 1
        s_alpha_bar = self.sqrt_alpha_bar[t]
        s_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]

        # Tambah dimensi trailing agar broadcasting cocok
        while s_alpha_bar.dim() < x_0.dim():
            s_alpha_bar = s_alpha_bar.unsqueeze(-1)
            s_one_minus_alpha_bar = s_one_minus_alpha_bar.unsqueeze(-1)

        return s_alpha_bar * x_0 + s_one_minus_alpha_bar * noise

    # ------------------------------------------------------------
    #  Utility: pindah device
    # ------------------------------------------------------------
    def to(self, device: Union[str, torch.device]) -> "NoiseScheduleCosine":
        """
        Pindah semua tensor internal ke device baru.

        Contoh:
            sched = NoiseScheduleCosine(T=100)      # di CPU
            sched = sched.to("cuda")                # pindah ke GPU
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    # ------------------------------------------------------------
    #  Debug helper
    # ------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"NoiseScheduleCosine(T={self.T}, s={self.s}, "
            f"device={self.device})"
        )