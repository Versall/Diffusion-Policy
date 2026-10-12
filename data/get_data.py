"""
get_data.py — Unduh & siapkan dataset Push-T untuk pelatihan Diffusion Policy.

Dataset:
    pusht_cchi_v7_replay.zarr — 206 episode demonstrasi Push-T
    Ukuran zip: ~30 MB (setelah ekstrak: ~200 MB)
    Sumber: https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip

Struktur yang diharapkan setelah selesai:
    data/
    └── pusht/
        └── pusht_cchi_v7_replay.zarr/
            ├── data/
            │   ├── action   (25650, 2)  float32
            │   ├── state    (25650, 5)  float32
            │   ├── img      (25650, 96, 96, 3)  uint8
            │   ├── keypoint (25650, 9, 2)  float32
            │   └── n_contacts (25650, 1) float32
            └── meta/
                └── episode_ends (206,) int64

Cara pakai:
    uv run python data/get_data.py
    uv run python data/get_data.py --force      # paksa unduh ulang
    uv run python data/get_data.py --dest data  # ubah folder tujuan

Referensi:
    - real-stanford/diffusion_policy: https://github.com/real-stanford/diffusion_policy
    - Dataset resmi: https://diffusion-policy.cs.columbia.edu/
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# ============================================================
# Konfigurasi
# ============================================================
PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
PUSHT_ZIP_NAME = "pusht.zip"
PUSHT_ZARR_NAME = "pusht_cchi_v7_replay.zarr"

# Ukuran minimum yang diharapkan (byte) — toleransi jika ada perubahan
MIN_ZIP_SIZE = 10 * 1024 * 1024      # 10 MB
MIN_ZARR_SIZE = 50 * 1024 * 1024     # 50 MB setelah ekstrak

# Hash opsional: jika diisi, akan diverifikasi. Biarkan None untuk skip.
EXPECTED_MD5: Optional[str] = None


# ============================================================
# Utilitas
# ============================================================
def fmt_size(n: int) -> str:
    """Format byte ke string manusiawi."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hitung MD5 file secara streaming (hemat memori)."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[get_data] {msg}")


# ============================================================
# Download dengan progress bar
# ============================================================
def download_with_progress(url: str, dest: Path) -> None:
    """Unduh URL ke dest dengan progress bar sederhana."""

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100 / total_size)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{fmt_size(downloaded)} / {fmt_size(total_size)}"
            )
        else:
            sys.stdout.write(f"\r  {fmt_size(downloaded)} diunduh...")
        sys.stdout.flush()

    log(f"Mengunduh dari {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gagal mengunduh: {e}") from e
    finally:
        sys.stdout.write("\n")


# ============================================================
# Ekstraksi
# ============================================================
def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Ekstrak zip ke dest_dir dengan aman (mencegah path traversal)."""
    log(f"Mengekstrak {zip_path.name} → {dest_dir}/")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Cegah Zip Slip (path traversal)
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Zip berisi path berbahaya: {member}")
        zf.extractall(dest_dir)

    log("Ekstraksi selesai.")


# ============================================================
# Validasi Zarr
# ============================================================
def validate_zarr(zarr_path: Path) -> bool:
    """
    Validasi struktur Zarr: pastikan ada data/state, data/action,
    dan meta/episode_ends dengan shape yang diharapkan.
    """
    log(f"Memvalidasi struktur Zarr: {zarr_path}")

    try:
        import zarr
    except ImportError:
        log("⚠  Library 'zarr' belum terinstall. Lewati validasi.")
        log("   Install dengan: uv sync")
        return True

    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception as e:
        log(f"✗ Gagal membuka Zarr: {e}")
        return False

    required = {
        "data/state": 2,
        "data/action": 2,
        "meta/episode_ends": 1,
    }

    ok = True
    for key, ndim in required.items():
        try:
            arr = root[key]
            shape = tuple(arr.shape)
            log(f"  ✓ {key:<20} shape={shape}  dtype={arr.dtype}")
            if len(shape) != ndim:
                log(f"    ⚠ Diharapkan {ndim} dimensi, dapat {len(shape)}")
                ok = False
        except KeyError:
            log(f"  ✗ {key:<20} TIDAK DITEMUKAN")
            ok = False

    # Cek opsional
    for key in ("data/img", "data/keypoint", "data/n_contacts"):
        try:
            arr = root[key]
            log(f"  · {key:<20} shape={tuple(arr.shape)}  dtype={arr.dtype}")
        except KeyError:
            log(f"  · {key:<20} (tidak ada — opsional untuk state-based)")

    return ok


# ============================================================
# Main
# ============================================================
def prepare_dataset(
    dest_dir: Path,
    force: bool = False,
    keep_zip: bool = False,
    validate: bool = True,
) -> Path:
    """
    Siapkan dataset Push-T di dest_dir/pusht/pusht_cchi_v7_replay.zarr.

    Returns:
        Path ke folder Zarr.
    """
    dest_dir = dest_dir.resolve()
    pusht_dir = dest_dir / "pusht"
    zarr_path = pusht_dir / PUSHT_ZARR_NAME
    zip_path = dest_dir / PUSHT_ZIP_NAME

    # ---- Cek apakah sudah ada ----
    if zarr_path.exists() and not force:
        log(f"Dataset sudah ada: {zarr_path}")
        log("Gunakan --force untuk mengunduh ulang.")
        if validate:
            validate_zarr(zarr_path)
        return zarr_path

    if force and zarr_path.exists():
        log(f"--force: menghapus {zarr_path}")
        shutil.rmtree(zarr_path)

    # ---- Buat folder ----
    dest_dir.mkdir(parents=True, exist_ok=True)
    pusht_dir.mkdir(parents=True, exist_ok=True)

    # ---- Unduh ----
    if not zip_path.exists() or force:
        download_with_progress(PUSHT_URL, zip_path)
    else:
        log(f"Menggunakan zip yang sudah ada: {zip_path}")

    # ---- Verifikasi ukuran ----
    zip_size = zip_path.stat().st_size
    log(f"Ukuran zip: {fmt_size(zip_size)}")
    if zip_size < MIN_ZIP_SIZE:
        raise RuntimeError(
            f"Zip terlalu kecil ({fmt_size(zip_size)}). "
            f"Kemungkinan unduhan gagal atau rusak."
        )

    # ---- Verifikasi MD5 (opsional) ----
    if EXPECTED_MD5:
        log("Menghitung MD5...")
        actual = md5sum(zip_path)
        if actual != EXPECTED_MD5:
            raise RuntimeError(
                f"MD5 tidak cocok!\n"
                f"  diharapkan: {EXPECTED_MD5}\n"
                f"  aktual    : {actual}"
            )
        log("✓ MD5 cocok.")

    # ---- Ekstrak ----
    extract_zip(zip_path, pusht_dir)

    # ---- Cari folder Zarr setelah ekstrak ----
    if not zarr_path.exists():
        # Mungkin zip mengekstrak ke subfolder lain
        candidates = list(pusht_dir.glob("*.zarr"))
        if len(candidates) == 1:
            candidates[0].rename(zarr_path)
            log(f"Memindahkan {candidates[0].name} → {zarr_path.name}")
        elif len(candidates) == 0:
            raise RuntimeError(
                f"Tidak menemukan folder .zarr di {pusht_dir}. "
                f"Isi: {list(pusht_dir.iterdir())}"
            )
        else:
            raise RuntimeError(
                f"Ada {len(candidates)} folder .zarr — tidak jelas mana yang dipakai."
            )

    # ---- Verifikasi ukuran Zarr ----
    zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
    log(f"Ukuran Zarr setelah ekstrak: {fmt_size(zarr_size)}")
    if zarr_size < MIN_ZARR_SIZE:
        log(f"⚠ Ukuran Zarr lebih kecil dari perkiraan ({fmt_size(MIN_ZARR_SIZE)}).")

    # ---- Hapus zip (opsional) ----
    if not keep_zip:
        log(f"Menghapus zip: {zip_path}")
        zip_path.unlink()
    else:
        log(f"Zip disimpan: {zip_path}")

    # ---- Validasi ----
    if validate:
        ok = validate_zarr(zarr_path)
        if not ok:
            raise RuntimeError("Validasi Zarr gagal. Cek struktur dataset.")

    log("=" * 60)
    log(f"✓ Dataset siap: {zarr_path}")
    log("=" * 60)
    return zarr_path


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unduh & siapkan dataset Push-T untuk Diffusion Policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dest", type=str, default="data",
        help="Folder tujuan (default: data). Dataset akan di data/pusht/.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Paksa unduh ulang meski dataset sudah ada.",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Jangan hapus file zip setelah ekstrak.",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Lewati validasi struktur Zarr.",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    if not dest.is_absolute():
        # Resolve relatif ke root repo (parent dari folder script ini)
        repo_root = Path(__file__).resolve().parent.parent
        dest = (repo_root / dest).resolve()

    t0 = time.time()
    try:
        prepare_dataset(
            dest_dir=dest,
            force=args.force,
            keep_zip=args.keep_zip,
            validate=not args.no_validate,
        )
    except Exception as e:
        log(f"✗ ERROR: {e}")
        sys.exit(1)

    log(f"Selesai dalam {time.time() - t0:.1f} detik.")


if __name__ == "__main__":
    main()