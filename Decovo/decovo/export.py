# -*- coding: utf-8 -*-
"""Exporting CNMF-E results: Cell Curator HDF5 (with Cn & dims), MATLAB ms.mat/SFP.mat, and figures."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import scipy.sparse as sp

from .config import PipelineConfig
from .stages import ProgressReporter

logger = logging.getLogger("decovo")


def save_hdf5_for_curator(
    cnm,
    cn_filter: np.ndarray,
    pnr: Optional[np.ndarray],
    dims: Tuple[int, int],
    hdf5_path: str | Path,
) -> str:
    """Save `cnm` to HDF5 and guarantee `estimates/Cn` and `estimates/dims` exist for Cell Curator GUI."""
    out_path = str(Path(hdf5_path).resolve())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cnm.estimates.Cn = cn_filter
    cnm.estimates.dims = tuple(dims)
    cnm.dims = tuple(dims)
    cnm.save(out_path)

    # Guarantee Cn, PNR, and dims are explicitly stored inside /estimates in the HDF5 file
    with h5py.File(out_path, "a") as f:
        est_grp = f.require_group("estimates")
        if "Cn" in est_grp:
            del est_grp["Cn"]
        est_grp.create_dataset("Cn", data=np.asarray(cn_filter, dtype=np.float32))
        if pnr is not None:
            if "PNR" in est_grp:
                del est_grp["PNR"]
            est_grp.create_dataset("PNR", data=np.asarray(pnr, dtype=np.float32))
        if "dims" in est_grp:
            del est_grp["dims"]
        est_grp.create_dataset("dims", data=np.asarray(dims, dtype=np.int64))
        if "dims" in f:
            del f["dims"]
        f.create_dataset("dims", data=np.asarray(dims, dtype=np.int64))

    logger.info("Saved Curator-compatible HDF5: %s", out_path)
    return out_path


def _save_summary_figure(
    cnm,
    cn_filter: np.ndarray,
    pnr: np.ndarray,
    SFP_array: np.ndarray,
    dims: Tuple[int, int],
    idx_good: np.ndarray,
    work_dir: Path,
) -> List[str]:
    """Generate and save summary figure (PNG + SVG)."""
    saved: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        raw_traces = cnm.estimates.C if cnm.estimates.C is not None else np.zeros((0, 0))
        deconv_traces = cnm.estimates.S if cnm.estimates.S is not None else np.zeros((0, 0))
        n_neurons = len(idx_good) if idx_good is not None else 0
        n_plot = min(100, n_neurons)
        max_val = float(np.amax(raw_traces)) if raw_traces.size > 0 else 1.0
        if max_val <= 0:
            max_val = 1.0

        fig = plt.figure(figsize=(30, 15))
        gs = fig.add_gridspec(6, 2)

        ax1 = fig.add_subplot(gs[0:2, 0])
        im1 = ax1.imshow(cn_filter, cmap="viridis")
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Correlation projection (Cn)")

        ax2 = fig.add_subplot(gs[2:4, 0])
        im2 = ax2.imshow(pnr, cmap="viridis")
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("Peak-to-Noise Ratio (PNR)")

        ax3 = fig.add_subplot(gs[4:6, 0])
        sfp_proj = np.amax(SFP_array, axis=2) if SFP_array.shape[2] > 0 else np.zeros(dims, dtype=np.float32)
        im3 = ax3.imshow(sfp_proj, cmap="hot")
        plt.colorbar(im3, ax=ax3)
        ax3.set_title("Spatial footprints")

        ax4 = fig.add_subplot(gs[0:3, 1])
        ax4.set_title(f"Raw traces (first {n_plot} of {n_neurons} accepted neurons)")
        plot_gain = 5.0
        for i in range(n_plot):
            offset = max_val * i / plot_gain if i > 0 else 0.0
            ax4.plot(raw_traces[idx_good[i], :] + offset, "k", linewidth=0.3)

        ax5 = fig.add_subplot(gs[3:6, 1])
        ax5.set_title(f"Deconvolved traces (first {n_plot} of {n_neurons} accepted neurons)")
        plot_gain = 20.0
        for i in range(n_plot):
            offset = max_val * i / plot_gain if i > 0 else 0.0
            ax5.plot(deconv_traces[idx_good[i], :] + offset, "k", linewidth=0.3)

        plt.tight_layout()
        png_path = work_dir / "summary_figure.png"
        svg_path = work_dir / "summary_figure.svg"
        plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.savefig(str(svg_path), format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.extend([str(png_path), str(svg_path)])
        logger.info("Saved summary figures (PNG + SVG)")
    except Exception as e:
        logger.warning("Could not save summary figure: %s", e)
    return saved


def export_results(
    cnm,
    cn_filter: np.ndarray,
    pnr: np.ndarray,
    dims: Tuple[int, int],
    config: PipelineConfig,
    min_corr: float,
    min_pnr: float,
    work_dir: Path,
    progress: ProgressReporter,
    input_type: str = "avi",
) -> Tuple[List[str], np.ndarray]:
    """Export MAT files (`ms.mat`, `SFP.mat`) and summary figures.

    Note: Both `RawTraces` and `DeconvTraces` in `ms.mat` are oriented as `time × neurons` (`T × N`).
    """
    progress.start_stage("Exporting Results")
    work_dir.mkdir(parents=True, exist_ok=True)
    exported_files: List[str] = []

    idx_good = (
        np.asarray(cnm.estimates.idx_components, dtype=int)
        if getattr(cnm.estimates, "idx_components", None) is not None
        else np.array([], dtype=int)
    )
    raw_traces = (
        np.asarray(cnm.estimates.C)
        if getattr(cnm.estimates, "C", None) is not None and len(cnm.estimates.C) > 0
        else np.zeros((0, 0), dtype=np.float32)
    )
    deconv_traces = (
        np.asarray(cnm.estimates.S)
        if getattr(cnm.estimates, "S", None) is not None and len(cnm.estimates.S) > 0
        else np.zeros_like(raw_traces)
    )

    A = getattr(cnm.estimates, "A", None)
    if A is not None and A.shape[1] > 0:
        A_dense = A.toarray() if sp.issparse(A) else np.asarray(A)
        SFP_dims = [int(dims[0]), int(dims[1]), int(A_dense.shape[1])]
        SFP_array = np.reshape(A_dense, SFP_dims, order="F")
    else:
        SFP_array = np.zeros((int(dims[0]), int(dims[1]), 0), dtype=np.float32)

    if config.save_mat:
        from scipy.io import savemat

        frames_num = int(raw_traces.shape[1]) if raw_traces.ndim == 2 else 0
        results_dict = {
            "dirName": str(config.path_to_analyze),
            "numFiles": config.max_files or "all",
            "framesNum": frames_num,
            "height": int(dims[0]),
            "width": int(dims[1]),
            "ds": float(config.spatial_downsample),
            "frameRate": float(config.frame_rate),
            "inputType": str(input_type),
            "configYaml": config.to_yaml_string(),
            "CorrProj": cn_filter,
            "PeakToNoiseProj": pnr,
            "RawTraces": raw_traces.T,        # time × neurons (T × N)
            "DeconvTraces": deconv_traces.T,  # time × neurons (T × N)
            "SFP": SFP_array,
            "numNeurons": int(len(idx_good)),
            "idx_good": idx_good,
            "min_corr": float(min_corr),
            "min_pnr": float(min_pnr),
            "auto_params": bool(config.auto_params),
            "auto_method": str(config.auto_method),
        }

        ms_mat_path = work_dir / "ms.mat"
        sfp_mat_path = work_dir / "SFP.mat"
        savemat(str(ms_mat_path), {"ms": results_dict})
        SFP_perm = np.transpose(SFP_array, [2, 0, 1])
        savemat(str(sfp_mat_path), {"SFP": SFP_perm})
        exported_files.extend([str(ms_mat_path), str(sfp_mat_path)])
        logger.info("Saved MAT files: ms.mat, SFP.mat")

    if config.save_figures:
        fig_files = _save_summary_figure(cnm, cn_filter, pnr, SFP_array, dims, idx_good, work_dir)
        exported_files.extend(fig_files)

    progress.end_stage()
    return exported_files, SFP_array


def cleanup_mmap_files(
    config: PipelineConfig,
    fname_new: str,
    work_dir: Path,
    progress: ProgressReporter,
) -> None:
    """Optionally convert final memmap to `.npy` and remove temporary `.mmap` files."""
    import caiman as cm

    progress.start_stage("Memmap Cleanup")
    results_dir = str(work_dir)

    if config.convert_mmap_to_npy and fname_new and os.path.exists(fname_new):
        Yr, _dims, _T = cm.load_memmap(fname_new)
        npy_path = os.path.join(results_dir, "data_memmap.npy")
        np.save(npy_path, np.array(Yr))
        del Yr
        progress.update(0, f"Converted memmap to: {npy_path}")
        logger.info("Converted memmap -> %s", npy_path)

    if config.cleanup_mmap:
        deleted_count = 0
        search_dirs = {results_dir}
        try:
            search_dirs.add(cm.paths.get_tempdir())
        except Exception:
            pass
        for sdir in search_dirs:
            for f in glob.glob(os.path.join(sdir, "*.mmap")):
                if not config.convert_mmap_to_npy and fname_new and os.path.abspath(f) == os.path.abspath(fname_new):
                    continue
                try:
                    os.remove(f)
                    deleted_count += 1
                except OSError as e:
                    logger.warning("Could not delete %s: %s", f, e)
        progress.update(0, f"Cleaned up {deleted_count} .mmap files")
        logger.info("Cleaned up %d temporary .mmap files", deleted_count)

    progress.end_stage()
