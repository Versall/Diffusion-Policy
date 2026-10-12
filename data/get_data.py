"""
get_data.py — Unduh & siapkan dataset Push-T untuk pelatihan Diffusion Policy.

Lokasi target (WAJIB sesuai dengan yang dibaca train_pusht.py):
    <repo_root>/data/pusht/pusht_cchi_v7_replay.zarr/

Struktur setelah selesai:
    diffusion-policy-push-t/
    └── data/
        └── pusht/
            └── pusht_cchi_v7_replay.zarr/
                ├── data/
                │   ├── action    (25650, 2)  float32
                │   ├── state     (25650, 5)  float32
                │   ├── img       (25650, 96, 96, 3)  uint8
                │   ├── keypoint  (25650, 9, 2)  float32
                │   └── n_contacts (25650, 1) float32
                └── meta/
                    └── episode_ends  (206,) int64

Cara pakai:
    # Dari root repo
    uv run python data/get_data.py

    # Paksa unduh ulang
    uv run python data/get_data.py --force

    # Override folder data (default: <repo_root>/data)
    uv run python data/get_data.py --dest D:/datasets/diffusion

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

MIN_ZIP_SIZE = 10 * 1024 * 1024      # 10 MB
MIN_ZARR_SIZE = 50 * 1024 * 1024     # 50 MB setelah ekstrak

EXPECTED_MD5: Optional[str] = None    # set jika ingin verifikasi hash


# ============================================================
# Utilitas
# ============================================================
def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
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


def resolve_repo_root() -> Path:
    """
    Cari root repo dari lokasi file ini.

    File ini harus ada di: <repo_root>/data/get_data.py
    Sehingga: __file__ = <repo_root>/data/get_data.py
              parent  = <repo_root>/data
              parent.parent = <repo_root>
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    return repo_root


# ============================================================
# Download
# ============================================================
def download_with_progress(url: str, dest: Path) -> None:
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

    log(f"Mengunduh: {url}")
    log(f"  → {dest}")
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
    log(f"Mengekstrak {zip_path.name} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Zip berisi path berbahaya: {member}")
        zf.extractall(dest_dir)

    log("Ekstraksi selesai.")


# ============================================================
# Normalisasi struktur hasil ekstrak
# ============================================================
def _is_valid_zarr(p: Path) -> bool:
    """Cek apakah folder adalah Zarr dengan struktur minimal yang valid."""
    return (
        p.is_dir()
        and (p / "data").is_dir()
        and (p / "meta").is_dir()
        and (p / "meta" / "episode_ends").exists()
    )


def normalize_extracted_layout(pusht_dir: Path, target: Path) -> None:
    """
    Setelah ekstrak ke pusht_dir, pastikan hasil akhirnya di `target`
    (pusht_dir/pusht_cchi_v7_replay.zarr).

    Menangani beberapa kemungkinan struktur zip:
    A. pusht_dir/pusht_cchi_v7_replay.zarr/         → sudah benar
    B. pusht_dir/<nama_lain>.zarr/                  → rename
    C. pusht_dir/pusht_cchi_v7_replay.zarr/<sub>/   → bukan valid Zarr
    D. pusht_dir/<nested>/pusht_cchi_v7_replay.zarr/ → pindahkan ke atas
    """
    # --- Kasus A: sudah di lokasi yang benar ---
    if target.exists() and _is_valid_zarr(target):
        log(f"Struktur sesuai: {target.name}")
        return

    # --- Kasus D: zarr berada di subfolder bersarang ---
    nested = list(pusht_dir.rglob(PUSHT_ZARR_NAME))
    nested = [p for p in nested if _is_valid_zarr(p)]
    if len(nested) == 1:
        src = nested[0]
        if src != target:
            log(f"Memindahkan: {src.relative_to(pusht_dir)} → {target.name}")
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(src), str(target))
            # Bersihkan folder kosong sisa
            for parent in src.parents:
                if parent == pusht_dir:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        return

    # --- Kasus B: ada satu folder .zarr dengan nama berbeda ---
    candidates = [p for p in pusht_dir.glob("*.zarr") if _is_valid_zarr(p)]
    if len(candidates) == 1:
        src = candidates[0]
        log(f"Rename: {src.name} → {target.name}")
        if target.exists():
            shutil.rmtree(target)
        src.rename(target)
        return

    # --- Gagal: struktur tidak dikenali ---
    listing = "\n  ".join(str(p.relative_to(pusht_dir)) for p in pusht_dir.iterdir())
    raise RuntimeError(
        f"Struktur zip tidak dikenali.\n"
        f"Isi {pusht_dir}:\n  {listing}\n"
        f"Diharapkan salah satu:\n"
        f"  - {target.name}/data + meta\n"
        f"  - folder *.zarr lain di {pusht_dir}"
    )


# ============================================================
# Validasi Zarr
# ============================================================
def validate_zarr(zarr_path: Path) -> bool:
    log(f"Memvalidasi Zarr: {zarr_path}")

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

    for key in ("data/img", "data/keypoint", "data/n_contacts"):
        try:
            arr = root[key]
            log(f"  · {key:<20} shape={tuple(arr.shape)}  dtype={arr.dtype}")
        except KeyError:
            log(f"  · {key:<20} (tidak ada — opsional untuk state-based)")

    return ok


# ============================================================
# Main: siapkan dataset
# ============================================================
def prepare_dataset(
    dest_root: Path,
    force: bool = False,
    keep_zip: bool = False,
    validate: bool = True,
) -> Path:
    """
    Siapkan dataset Push-T di:
        dest_root/pusht/pusht_cchi_v7_replay.zarr

    Args:
        dest_root: Folder data (biasanya <repo_root>/data).
                   Di dalamnya akan dibuat subfolder `pusht/`.
        force:     Hapus & unduh ulang.
        keep_zip:  Jangan hapus zip setelah ekstrak.
        validate:  Validasi struktur Zarr setelah selesai.

    Returns:
        Path ke folder Zarr final.
    """
    dest_root = dest_root.resolve()
    pusht_dir = dest_root / "pusht"
    zarr_path = pusht_dir / PUSHT_ZARR_NAME
    zip_path = dest_root / PUSHT_ZIP_NAME

    log("=" * 60)
    log(f"Root data   : {dest_root}")
    log(f"Target Zarr : {zarr_path}")
    log("=" * 60)

    # --- Cek sudah ada ---
    if zarr_path.exists() and not force:
        log(f"Dataset sudah ada: {zarr_path}")
        log("Gunakan --force untuk unduh ulang.")
        if validate:
            validate_zarr(zarr_path)
        return zarr_path

    if force and zarr_path.exists():
        log(f"--force: menghapus {zarr_path}")
        shutil.rmtree(zarr_path)

    # --- Buat folder ---
    dest_root.mkdir(parents=True, exist_ok=True)
    pusht_dir.mkdir(parents=True, exist_ok=True)

    # --- Unduh ---
    if not zip_path.exists() or force:
        if force and zip_path.exists():
            zip_path.unlink()
        download_with_progress(PUSHT_URL, zip_path)
    else:
        log(f"Pakai zip yang sudah ada: {zip_path}")

    # --- Verifikasi ukuran ---
    zip_size = zip_path.stat().st_size
    log(f"Ukuran zip: {fmt_size(zip_size)}")
    if zip_size < MIN_ZIP_SIZE:
        raise RuntimeError(
            f"Zip terlalu kecil ({fmt_size(zip_size)}). "
            f"Kemungkinan unduhan gagal."
        )

    # --- Verifikasi MD5 (opsional) ---
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

    # --- Ekstrak ke data/pusht/ ---
    extract_zip(zip_path, pusht_dir)

    # --- Normalisasi struktur hasil ekstrak ---
    normalize_extracted_layout(pusht_dir, zarr_path)

    # --- Verifikasi ukuran Zarr ---
    if not zarr_path.exists():
        raise RuntimeError(f"Zarr tidak ditemukan di {zarr_path} setelah ekstrak.")
    zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
    log(f"Ukuran Zarr: {fmt_size(zarr_size)}")
    if zarr_size < MIN_ZARR_SIZE:
        log(f"⚠ Ukuran Zarr lebih kecil dari perkiraan ({fmt_size(MIN_ZARR_SIZE)}).")

    # --- Hapus zip ---
    if not keep_zip:
        log(f"Menghapus zip: {zip_path}")
        zip_path.unlink()
    else:
        log(f"Zip disimpan: {zip_path}")

    # --- Validasi final ---
    if validate:
        ok = validate_zarr(zarr_path)
        if not ok:
            raise RuntimeError("Validasi Zarr gagal.")

    log("=" * 60)
    log(f"✓ SELESAI — dataset siap di:")
    log(f"  {zarr_path}")
    log("=" * 60)
    log("Langkah berikutnya:")
    log("  uv run python train_pusht.py")
    return zarr_path


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unduh & siapkan dataset Push-T.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help="Folder data (default: <repo_root>/data). "
             "Dataset akan berada di <dest>/pusht/pusht_cchi_v7_replay.zarr.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Paksa unduh ulang.")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Simpan zip setelah ekstrak.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Lewati validasi Zarr.")
    args = parser.parse_args()

    # Resolusi path
    repo_root = resolve_repo_root()
    if args.dest is None:
        dest_root = repo_root / "data"
    else:
        dest_root = Path(args.dest)
        if not dest_root.is_absolute():
            dest_root = (repo_root / dest_root).resolve()

    t0 = time.time()
    try:
        prepare_dataset(
            dest_root=dest_root,
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