# -*- coding: utf-8 -*-
"""Core processing stages for Decovo: prepare, motion correction, summary images, CNMF-E, evaluation."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
from scipy import ndimage

from .config import PipelineConfig
from .io import InputSpec, iter_frames_chunks
from .params_auto import CancelledError

logger = logging.getLogger("decovo")


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Pipeline execution cancelled by user.")


class ProgressReporter:
    """Reports pipeline progress to console and optional callback."""

    def __init__(self, callback: Optional[Callable[..., None]] = None):
        self.callback = callback
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.current_stage: Optional[str] = None

    def start_stage(self, name: str, total_steps: int = 0) -> None:
        self.current_stage = name
        self.stages[name] = {
            "start": time.time(),
            "total": total_steps,
            "current": 0,
            "status": "running",
        }
        logger.info("▶ %s...", name)
        if self.callback:
            self.callback("stage_start", name, 0, total_steps)

    def update(self, step: int = 0, message: str = "") -> None:
        if self.current_stage and self.current_stage in self.stages:
            self.stages[self.current_stage]["current"] = step
        if message:
            logger.info("  %s", message)
        if self.callback and self.current_stage:
            total = self.stages.get(self.current_stage, {}).get("total", 0)
            self.callback("progress", self.current_stage, step, total)

    def skip_stage(self, name: str, message: str = "Cached — skipped") -> None:
        self.stages[name] = {
            "start": time.time(),
            "total": 1,
            "current": 1,
            "status": "done",
            "elapsed": 0.0,
            "cached": True,
        }
        logger.info("⏭ %s (%s)", name, message)
        if self.callback:
            self.callback("stage_start", name, 0, 1)
            self.callback("stage_end", name, 0.0, 1)

    def end_stage(self, name: Optional[str] = None) -> float:
        name = name or self.current_stage
        if name and name in self.stages:
            elapsed = time.time() - self.stages[name]["start"]
            self.stages[name]["status"] = "done"
            self.stages[name]["elapsed"] = elapsed
            logger.info("✓ %s completed in %.1fs", name, elapsed)
            if self.callback:
                self.callback("stage_end", name, elapsed, 0)
            return elapsed
        return 0.0

    def get_timing(self) -> Dict[str, float]:
        return {k: float(v.get("elapsed", 0.0)) for k, v in self.stages.items()}


class ClusterManager:
    """Context manager for CaImAn multiprocessing cluster — ensures clean startup & teardown."""

    def __init__(self, n_processes: Optional[int] = None):
        self.n_processes = n_processes
        self.c = None
        self.dview = None
        self.actual_n_processes: Optional[int] = None

    def __enter__(self) -> "ClusterManager":
        import caiman as cm

        if self.n_processes == 1:
            self.c = None
            self.dview = None
            self.actual_n_processes = 1
            logger.info("Running single-process mode (n_processes=1, dview=None)")
            return self

        self.c, self.dview, self.actual_n_processes = cm.cluster.setup_cluster(
            backend="local",
            n_processes=self.n_processes,
            single_thread=False,
        )
        logger.info("Cluster started with %s processes", self.actual_n_processes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.dview is not None:
            try:
                import caiman as cm

                cm.stop_server(dview=self.dview)
                logger.info("Cluster stopped")
            except Exception as e:
                logger.warning("Error stopping cluster: %s", e)
            finally:
                self.dview = None
                self.c = None
        return False


def _resize_block_cpu(
    block: np.ndarray,
    spatial_ds: float,
    temporal_ds: float = 1.0,
) -> np.ndarray:
    """Vectorized CPU block resize using cv2.INTER_AREA (honest fast path, no GPU<->CPU thrashing)."""
    n, h, w = block.shape
    out = block

    if temporal_ds != 1.0 and temporal_ds > 0:
        if temporal_ds < 1.0:
            new_n = max(1, int(round(n * temporal_ds)))
            idx = np.linspace(0, n - 1, new_n, dtype=int)
            out = out[idx]

    if spatial_ds != 1.0:
        new_h = max(1, int(round(h * spatial_ds)))
        new_w = max(1, int(round(w * spatial_ds)))
        resized = np.empty((out.shape[0], new_h, new_w), dtype=np.float32)
        for i in range(out.shape[0]):
            resized[i] = cv2.resize(
                out[i].astype(np.float32, copy=False),
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
        out = resized
    else:
        out = out.astype(np.float32, copy=False)

    return out


def stage_prepare(
    spec: InputSpec,
    config: PipelineConfig,
    work_dir: Path,
    progress: ProgressReporter,
    cancel_event: Optional[threading.Event] = None,
) -> List[str]:
    """Unified streaming preprocessing for both TIFF stacks and AVI chunks.

    Pass 1 (when `remove_background=True`): stream chunks, resize, accumulate global min projection
    on downsampled frames, smooth with uniform_filter.
    Pass 2: stream chunks, resize, subtract background template, write `prep_000.tif ...` chunks.
    Inputs are never modified in place.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(config.chunk_frames))
    n_chunks_est = max(1, int(np.ceil(spec.n_frames / chunk_size)))
    total_steps = n_chunks_est * (2 if config.remove_background else 1)

    progress.start_stage("Preprocessing (Downsample & Background)", total_steps)

    # Copy any .dat timestamp files if present in the input folder
    if spec.files:
        input_parent = Path(spec.files[0]).parent
        if input_parent.exists():
            for f in os.listdir(input_parent):
                if f.lower().endswith(".dat"):
                    src = input_parent / f
                    dst = work_dir / f
                    if src.is_file() and src.resolve() != dst.resolve():
                        shutil.copyfile(src, dst)

    template_smooth: Optional[np.ndarray] = None
    step_idx = 0

    if config.remove_background:
        running_min: Optional[np.ndarray] = None
        for i, block in enumerate(iter_frames_chunks(spec, chunk_size=chunk_size)):
            _check_cancel(cancel_event)
            resized = _resize_block_cpu(block, config.spatial_downsample, config.temporal_downsample)
            block_min = resized.min(axis=0)
            if running_min is None:
                running_min = block_min.copy()
            else:
                np.minimum(running_min, block_min, out=running_min)
            step_idx += 1
            progress.update(step_idx, f"Pass 1/2: background min-projection (chunk {i + 1}/{n_chunks_est})")

        if running_min is not None:
            template_smooth = (
                float(config.bg_scale)
                * ndimage.uniform_filter(running_min, size=int(config.bg_filter_size))
            ).astype(np.float32)

    # Remove any stale prep_*.tif files before writing new ones
    for old_prep in work_dir.glob("prep_*.tif"):
        try:
            old_prep.unlink()
        except OSError:
            pass

    prep_files: List[str] = []
    for i, block in enumerate(iter_frames_chunks(spec, chunk_size=chunk_size)):
        _check_cancel(cancel_event)
        resized = _resize_block_cpu(block, config.spatial_downsample, config.temporal_downsample)
        if template_smooth is not None:
            resized = resized - template_smooth
        out_path = work_dir / f"prep_{i:03d}.tif"
        tifffile.imwrite(str(out_path), resized.astype(np.float32, copy=False))
        prep_files.append(str(out_path))
        step_idx += 1
        progress.update(step_idx, f"Saved preprocessed chunk {out_path.name} ({resized.shape})")

    progress.end_stage()
    return prep_files


def _save_motion_figure(
    shifts: np.ndarray,
    template: Optional[np.ndarray],
    pw_rigid: bool,
    out_path: Path,
) -> None:
    """Save motion correction shifts and template figure for both rigid and pw_rigid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        title_kind = "Piecewise-rigid (mean patch shifts)" if pw_rigid else "Rigid shifts"
        axes[0].set_title(title_kind)
        if shifts.ndim == 2 and shifts.shape[1] >= 2:
            axes[0].plot(shifts[:, 0], label="x shifts")
            axes[0].plot(shifts[:, 1], label="y shifts")
            axes[0].legend()
        else:
            axes[0].plot(shifts)
        axes[0].set_xlabel("frames")
        axes[0].set_ylabel("pixels")

        axes[1].set_title("Corrected template")
        if template is not None:
            axes[1].imshow(template, cmap="gray")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning("Could not save motion correction figure: %s", e)


def stage_motion(
    prep_files: List[str],
    config: PipelineConfig,
    work_dir: Path,
    dview,
    progress: ProgressReporter,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[str, int, List[str]]:
    """Run NoRMCorre motion correction (or direct memmap) and return `(fname_memmap, bord_px, extra_outputs)`."""
    import caiman as cm

    _check_cancel(cancel_event)
    base_name = str(work_dir / "memmap_")
    extra_outputs: List[str] = []

    if not config.motion_correct:
        progress.start_stage("Memory Mapping (no motion correction)")
        fname_new = cm.save_memmap(
            prep_files,
            base_name=base_name,
            order="C",
            border_to_0=0,
            dview=dview,
        )
        progress.end_stage()
        return str(fname_new), 0, extra_outputs

    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params

    progress.start_stage("Motion Correction")

    mc_dict = {
        "fnames": prep_files,
        "fr": config.frate,
        "niter_rig": 1,
        "splits_rig": max(1, min(20, len(prep_files))),
        "num_splits_to_process_rig": None,
        "decay_time": config.decay_time,
        "pw_rigid": config.pw_rigid,
        "max_shifts": config.max_shifts,
        "gSig_filt": config.gSig_filt,
        "strides": config.strides,
        "overlaps": config.overlaps,
        "max_deviation_rigid": config.max_deviation_rigid,
        "border_nan": config.border_nan,
        "use_cuda": config.use_cuda,
    }

    opts = params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect(prep_files, dview=dview, **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)

    _check_cancel(cancel_event)

    fname_mc = mc.fname_tot_els if config.pw_rigid else mc.fname_tot_rig

    if config.pw_rigid:
        raw_bord = int(
            np.ceil(
                np.maximum(
                    np.max(np.abs(mc.x_shifts_els)),
                    np.max(np.abs(mc.y_shifts_els)),
                )
            )
        )
        shifts_arr = np.column_stack(
            [
                np.mean(mc.x_shifts_els, axis=1) if np.ndim(mc.x_shifts_els) > 1 else np.asarray(mc.x_shifts_els),
                np.mean(mc.y_shifts_els, axis=1) if np.ndim(mc.y_shifts_els) > 1 else np.asarray(mc.y_shifts_els),
            ]
        )
        template = getattr(mc, "total_template_els", getattr(mc, "total_template_rig", None))
    else:
        raw_bord = int(np.ceil(np.max(np.abs(mc.shifts_rig))))
        shifts_arr = np.asarray(mc.shifts_rig)
        template = getattr(mc, "total_template_rig", None)

    bord_px = 0 if config.border_nan == "copy" else raw_bord

    shifts_path = work_dir / "motion_shifts.npy"
    np.save(str(shifts_path), shifts_arr)
    extra_outputs.append(str(shifts_path))

    fname_new = cm.save_memmap(
        fname_mc,
        base_name=base_name,
        order="C",
        border_to_0=bord_px,
        dview=dview,
    )

    # Remove intermediate motion-corrected mmap files (`fname_mc`) now that the C-order memmap is saved
    for f_mc in (fname_mc if isinstance(fname_mc, list) else [fname_mc]):
        if f_mc and os.path.abspath(str(f_mc)) != os.path.abspath(str(fname_new)):
            try:
                if os.path.exists(str(f_mc)):
                    os.remove(str(f_mc))
            except OSError:
                pass

    if config.save_figures:
        fig_path = work_dir / "motion_correction.png"
        _save_motion_figure(shifts_arr, template, config.pw_rigid, fig_path)
        if fig_path.exists():
            extra_outputs.append(str(fig_path))

    if config.save_motion_corrected:
        try:
            mc_movie = cm.load(str(fname_new))
            mc_avi_path = work_dir / "motion_corrected.avi"
            mc_movie.save(str(mc_avi_path))
            extra_outputs.append(str(mc_avi_path))
        except Exception as e:
            logger.warning("Could not save motion_corrected.avi: %s", e)

    progress.end_stage()
    return str(fname_new), int(bord_px), extra_outputs


def load_memmap_movie(fname_new: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], int]:
    """Load CaImAn memmap once and return `(Yr, images, dims, T)`."""
    import caiman as cm

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + tuple(dims), order="F")
    return Yr, images, (int(dims[0]), int(dims[1])), int(T)


def stage_summary(
    images: np.ndarray,
    config: PipelineConfig,
    work_dir: Path,
    progress: ProgressReporter,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Compute correlation (Cn) and PNR summary images from already-mapped `images` and save `.npy` files."""
    import caiman as cm

    _check_cancel(cancel_event)
    progress.start_stage("Summary Images")

    T = images.shape[0]
    target_frames = 500
    if T > target_frames:
        indices = np.linspace(0, T - 1, target_frames, dtype=int)
        sample = np.asarray(images[indices])
    else:
        sample = np.asarray(images)

    cn_filter, pnr = cm.summary_images.correlation_pnr(
        sample,
        gSig=config.gSig,
        swap_dim=False,
    )

    cn_path = work_dir / "Cn.npy"
    pnr_path = work_dir / "PNR.npy"
    np.save(str(cn_path), cn_filter)
    np.save(str(pnr_path), pnr)

    progress.end_stage()
    return cn_filter, pnr, str(cn_path), str(pnr_path)


def stage_extract_and_evaluate(
    images: np.ndarray,
    fname_new: str,
    cn_filter: np.ndarray,
    dims: Tuple[int, int],
    bord_px: int,
    min_corr: float,
    min_pnr: float,
    config: PipelineConfig,
    dview,
    progress: ProgressReporter,
    cancel_event: Optional[threading.Event] = None,
):
    """Run CNMF-E source extraction + OASIS deconvolution + component evaluation.

    Sets `cnm.estimates.Cn = cn_filter` and `cnm.estimates.dims = dims` for Cell Curator GUI compatibility.
    """
    from caiman.source_extraction.cnmf import cnmf as cnmf_module
    from caiman.source_extraction.cnmf import params

    _check_cancel(cancel_event)
    progress.start_stage("CNMF-E Source Extraction")

    min_dim = min(dims)
    if min_dim <= 2 * int(config.rf) + int(config.stride):
        eff_rf = None
        eff_stride = None
    else:
        eff_rf = int(config.rf)
        eff_stride = int(config.stride)

    cnmfe_dict = {
        "fnames": [fname_new] if isinstance(fname_new, str) else fname_new,
        "fr": config.frate,
        "decay_time": config.decay_time,
        "pw_rigid": config.pw_rigid,
        "max_shifts": config.max_shifts,
        "gSig_filt": config.gSig_filt,
        "strides": config.strides,
        "overlaps": config.overlaps,
        "max_deviation_rigid": config.max_deviation_rigid,
        "border_nan": config.border_nan,
        "method_init": "corr_pnr",
        "K": None,
        "gSig": config.gSig_tuple,
        "gSiz": config.gSiz_tuple,
        "merge_thr": config.merge_thr,
        "p": config.p,
        "tsub": config.tsub,
        "ssub": config.ssub,
        "rf": eff_rf,
        "stride": eff_stride,
        "only_init": True,
        "nb": config.nb,
        "nb_patch": 0,
        "method_deconvolution": config.method_deconvolution,
        "low_rank_background": None,
        "update_background_components": True,
        "min_corr": float(min_corr),
        "min_pnr": float(min_pnr),
        "normalize_init": False,
        "center_psf": True,
        "ssub_B": config.ssub_B,
        "ring_size_factor": config.ring_size_factor,
        "del_duplicates": True,
        "border_pix": int(bord_px),
    }

    opts = params.CNMFParams(params_dict=cnmfe_dict)
    cnm = cnmf_module.CNMF(
        n_processes=config.n_processes,
        dview=dview,
        Ain=None,
        params=opts,
    )
    cnm.fit(images)
    progress.end_stage()

    _check_cancel(cancel_event)
    progress.start_stage("Component Evaluation")

    cnm.params.set(
        "quality",
        {
            "min_SNR": config.min_SNR,
            "rval_thr": config.rval_thr,
            "use_cnn": config.use_cnn,
        },
    )

    n_total = len(cnm.estimates.C) if (cnm.estimates.C is not None and len(cnm.estimates.C) > 0) else 0
    if n_total > 0:
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
        n_accepted = len(cnm.estimates.idx_components) if cnm.estimates.idx_components is not None else 0
        n_rejected = len(cnm.estimates.idx_components_bad) if cnm.estimates.idx_components_bad is not None else 0
    else:
        cnm.estimates.idx_components = np.array([], dtype=int)
        cnm.estimates.idx_components_bad = np.array([], dtype=int)
        n_accepted = 0
        n_rejected = 0

    # Detach any ndarray views into the underlying np.memmap (e.g. estimates.b) so Windows file locks are released
    for attr_k, attr_v in list(vars(cnm.estimates).items()):
        if isinstance(attr_v, np.ndarray) and attr_v.base is not None:
            setattr(cnm.estimates, attr_k, np.array(attr_v, copy=True))
    if hasattr(cnm, "mmap_file"):
        cnm.mmap_file = None

    # Attach Cn and dims directly on estimates for CleanMiniscope Cell Curator GUI
    cnm.estimates.Cn = cn_filter
    cnm.estimates.dims = tuple(dims)

    logger.info("Components: %d total, %d accepted, %d rejected", n_total, n_accepted, n_rejected)
    progress.end_stage()
    return cnm
