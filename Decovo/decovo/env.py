# -*- coding: utf-8 -*-
"""Environment helpers: Colab detection, RAM-aware process count, work directories, CUDA check."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("decovo")


def in_colab() -> bool:
    """Return True if running inside a Google Colab runtime."""
    if "google.colab" in sys.modules:
        return True
    if os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG") is not None:
        return True
    return Path("/content").exists() and "COLAB" in " ".join(os.environ.keys())


def get_system_ram_gb() -> float:
    """Return total system RAM in GB (fallback 8.0 if psutil unavailable)."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 8.0


def get_available_ram_gb() -> float:
    """Return available system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 4.0


def auto_n_processes(max_gb_per_proc: float = 3.0) -> int:
    """Compute a safe number of worker processes based on CPU cores and total RAM.

    On free Google Colab (~12.7 GB RAM, 2 vCPUs), this returns 2-4 instead of crashing.
    """
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 2
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        safe_procs = max(1, int(ram_gb / max_gb_per_proc))
        return max(1, min(cpu_count, safe_procs))
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def default_workdir(input_dir: Optional[str | Path] = None, results_folder: str = "results_decovo") -> Path:
    """Return default working/results directory.

    On Google Colab, if the input is on Google Drive (`/content/drive/...`),
    memmaps must live on the local VM SSD (`/content/decovo_work/<session_name>`)
    because FUSE mounts are slow and unreliable for `numpy.memmap`.
    Locally, returns `<input_dir>/<results_folder>`.
    """
    if input_dir is None:
        if in_colab():
            return Path("/content/decovo_work")
        return Path.cwd() / results_folder

    input_path = Path(input_dir).resolve()
    if in_colab() and str(input_path).startswith("/content/drive"):
        import hashlib
        parent_prefix = f"{input_path.parent.name}_" if input_path.parent.name else ""
        path_hash = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()[:6]
        session_slug = f"{parent_prefix}{input_path.name or 'session'}_{path_hash}"
        return Path("/content/decovo_work") / session_slug

    return input_path / results_folder


def configure_caiman_tempdir(work_dir: str | Path) -> Path:
    """Ensure `work_dir` exists and set `CAIMAN_TEMP` to it so memmaps do not land on Drive FUSE."""
    path = Path(work_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = str(path)
    return path


def detect_cuda() -> bool:
    """Explicit opt-in check for CUDA-accelerated libraries (not called implicitly in __post_init__)."""
    try:
        import cv2
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("CUDA available via OpenCV: %d GPU(s)", cv2.cuda.getCudaEnabledDeviceCount())
            return True
    except Exception:
        pass

    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            logger.info("CuPy CUDA backend available")
            return True
    except Exception:
        pass

    return False


def get_gpu_memory_mb() -> float:
    """Return available GPU memory in MB if CuPy is installed, else 0.0."""
    try:
        import cupy as cp
        free, _total = cp.cuda.runtime.memGetInfo()
        return free / (1024 * 1024)
    except Exception:
        return 0.0
