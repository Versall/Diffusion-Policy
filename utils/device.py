"""
utils/device.py — Deteksi device otomatis (CUDA / CPU / MPS) & helper.

Semua modul di repo ini WAJIB menggunakan `get_device()` daripada
hardcode `torch.device("cuda")` atau `torch.device("cpu")`.

Prinsip:
    - Deteksi otomatis: CUDA > MPS > CPU
    - Bisa di-override via argumen `prefer`
    - Bisa dipaksa via environment variable FORCE_DEVICE
    - Semua tensor/model pindah pakai `.to(device)`, bukan `.cuda()`/`.cpu()`

Contoh:
    >>> from utils.device import get_device, describe_device
    >>> device = get_device()
    >>> print(describe_device(device))
    cuda:0 (NVIDIA GeForce RTX 5060 Ti, 16.0 GB VRAM, CUDA 12.8)

    >>> device = get_device(prefer="cpu")
    >>> print(describe_device(device))
    cpu

    # Override via env:
    # FORCE_DEVICE=cpu python train_pusht.py
"""
from __future__ import annotations

import os
from typing import Optional

import torch


# ============================================================
# Deteksi device
# ============================================================
def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Return torch.device terbaik yang tersedia.

    Prioritas:
        1. Environment variable `FORCE_DEVICE` (jika di-set)
        2. Argumen `prefer` (jika diberikan & valid/tersedia)
        3. CUDA (jika `torch.cuda.is_available()`)
        4. MPS  (Apple Silicon, jika tersedia)
        5. CPU  (fallback terakhir)

    Args:
        prefer: Nama device yang diinginkan, mis. "cuda", "cpu", "cuda:1".
                Kalau device ini tidak tersedia, akan di-fallback otomatis.

    Returns:
        torch.device yang siap dipakai.

    Raises:
        RuntimeError: Kalau FORCE_DEVICE=cuda tapi CUDA tidak tersedia.
    """
    # ----- 1. Cek environment override -----
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced:
        if forced == "cpu":
            return torch.device("cpu")
        if forced == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FORCE_DEVICE=cuda tapi CUDA tidak tersedia. "
                    "Cek instalasi PyTorch CUDA (uv sync --extra cuXXX)."
                )
            return torch.device("cuda")
        if forced == "mps":
            if not _mps_available():
                raise RuntimeError("FORCE_DEVICE=mps tapi MPS tidak tersedia.")
            return torch.device("mps")
        # fallback: coba parse device string apa pun
        try:
            return torch.device(forced)
        except Exception as e:
            raise RuntimeError(f"FORCE_DEVICE={forced} tidak valid: {e}")

    # ----- 2. Cek prefer -----
    if prefer:
        prefer = prefer.strip().lower()
        try:
            dev = torch.device(prefer)
        except Exception:
            dev = None

        if dev is not None:
            if dev.type == "cuda":
                if torch.cuda.is_available():
                    return dev
                # fallback ke CPU
            elif dev.type == "mps":
                if _mps_available():
                    return dev
            elif dev.type == "cpu":
                return dev
            else:
                # device type tidak dikenal — coba langsung
                return dev

    # ----- 3. Auto-detect -----
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    """Cek MPS (Apple Silicon) dengan aman."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


# ============================================================
# Helper untuk logging & pengecekan
# ============================================================
def describe_device(device: torch.device) -> str:
    """
    Return string ringkas untuk logging.

    Contoh:
        "cuda:0 (NVIDIA GeForce RTX 5060 Ti, 16.0 GB VRAM, CUDA 12.8)"
        "mps (Apple Silicon)"
        "cpu"
    """
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / 1e9
            cuda_ver = torch.version.cuda or "?"
            return f"cuda:{idx} ({name}, {vram_gb:.1f} GB VRAM, CUDA {cuda_ver})"
        except Exception:
            return f"cuda:{device.index}"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return str(device)


def is_cuda(device: torch.device) -> bool:
    """Cek apakah device adalah CUDA."""
    return device.type == "cuda"


def is_mps(device: torch.device) -> bool:
    """Cek apakah device adalah MPS."""
    return device.type == "mps"


def is_cpu(device: torch.device) -> bool:
    """Cek apakah device adalah CPU."""
    return device.type == "cpu"


def summary() -> str:
    """
    Ringkasan environment device untuk logging awal.
    Berguna diletakkan di baris pertama `train()` atau `main()`.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"PyTorch version : {torch.__version__}")
    lines.append(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"CUDA version    : {torch.version.cuda}")
        lines.append(f"GPU count       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            lines.append(f"  [{i}] {name}  ({vram:.1f} GB)")
    lines.append(f"MPS available   : {_mps_available()}")
    lines.append(f"Selected device : {describe_device(get_device())}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================
# Self-test (opsional)
# ============================================================
if __name__ == "__main__":
    print(summary())
    print()
    print("get_device()          →", describe_device(get_device()))
    print("get_device('cpu')     →", describe_device(get_device("cpu")))
    print("get_device('cuda')    →", describe_device(get_device("cuda")))