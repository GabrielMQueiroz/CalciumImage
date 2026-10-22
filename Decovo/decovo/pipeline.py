# -*- coding: utf-8 -*-
"""Two-stage resumable pipeline API: `preprocess()`, `extract()`, `run()`, `run_batch()`."""

from __future__ import annotations

import gc
import glob
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .cache import StageCache, compute_hash, input_files_signature
from .config import PipelineConfig, load_config
from .env import configure_caiman_tempdir
from .export import cleanup_mmap_files, export_results, save_hdf5_for_curator
from .io import InputSpec, discover_inputs
from .params_auto import estimate_params_heuristic, select_thresholds
from .stages import (
    ClusterManager,
    ProgressReporter,
    load_memmap_movie,
    stage_extract_and_evaluate,
    stage_motion,
    stage_prepare,
    stage_summary,
)

logger = logging.getLogger("decovo")


@dataclass
class Session:
    """State produced by Stage A (`preprocess`), reusable across multiple `extract()` calls."""

    input_dir: str
    work_dir: Path
    config: PipelineConfig
    spec: InputSpec
    cache: StageCache
    fname_memmap: str
    bord_px: int
    cn_filter: np.ndarray
    pnr: np.ndarray
    dims: Tuple[int, int]
    n_frames: int
    prepare_hash: str
    motion_hash: str
    summary_hash: str
    timing: Dict[str, float] = field(default_factory=dict)
    skipped_stages: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Container for full pipeline outputs (returned by `extract()` and `run()`)."""

    cnm: Any = None
    cn_filter: Optional[np.ndarray] = None
    pnr: Optional[np.ndarray] = None
    dims: Optional[Tuple[int, int]] = None
    idx_good: Optional[np.ndarray] = None
    raw_traces: Optional[np.ndarray] = None
    deconv_traces: Optional[np.ndarray] = None
    spatial_footprints: Optional[np.ndarray] = None
    min_corr_used: float = 0.0
    min_pnr_used: float = 0.0
    n_neurons: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    config: Optional[PipelineConfig] = None
    grid: Optional[Dict[Tuple[float, float], dict]] = None
    session: Optional[Session] = None
    skipped_stages: List[str] = field(default_factory=list)
    hdf5_path: Optional[str] = None


# Alias matching implementation_plan.md
Result = PipelineResult


def _resolve_config_and_dirs(
    input_dir: Optional[str | Path],
    config: Optional[PipelineConfig | str | Path],
) -> Tuple[PipelineConfig, Path, Path]:
    if config is None:
        cfg = load_config()
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    elif isinstance(config, PipelineConfig):
        cfg = config.copy()
    else:
        raise TypeError(f"Expected PipelineConfig or path, got {type(config)}")

    if input_dir is not None:
        cfg.path_to_analyze = str(input_dir)
    if not cfg.path_to_analyze:
        raise ValueError("input_dir (or config.path_to_analyze) must be specified.")

    in_path = Path(cfg.path_to_analyze).resolve()
    base_dir = in_path.parent if in_path.is_file() else in_path
    if os.path.isabs(cfg.results_folder):
        work_dir = Path(cfg.results_folder).resolve()
    else:
        work_dir = (base_dir / cfg.results_folder).resolve()

    return cfg, in_path, work_dir


def preprocess(
    input_dir: Optional[str | Path] = None,
    config: Optional[PipelineConfig | str | Path] = None,
    force: bool = False,
    progress: Optional[ProgressReporter | Callable[..., None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Session:
    """Stage A: Input discovery -> streaming downsample & bg removal -> motion correction -> Cn/PNR.

    Cached via `decovo_manifest.json` in `work_dir`. Skips already-completed stages if hashes match.
    """
    cfg, in_path, work_dir = _resolve_config_and_dirs(input_dir, config)
    configure_caiman_tempdir(work_dir)
    logging.getLogger("caiman").setLevel(logging.WARNING)

    if isinstance(progress, ProgressReporter):
        reporter = progress
    else:
        reporter = ProgressReporter(callback=progress)

    cache = StageCache(work_dir)
    if force:
        cache.clear()

    spec = discover_inputs(in_path, pattern=cfg.file_pattern, max_files=cfg.max_files)

    # Check metadata FPS vs user-selected frame_rate
    if spec.fps_from_metadata is not None:
        if abs(spec.fps_from_metadata - cfg.frame_rate) > 0.5:
            logger.warning(
                "Metadata frame rate is %.2f Hz, while configured frame_rate is %.2f Hz. "
                "Using configured frame_rate=%.2f Hz (verify by hand if needed).",
                spec.fps_from_metadata,
                cfg.frame_rate,
                cfg.frame_rate,
            )
        else:
            logger.info("Metadata frame rate (%.2f Hz) matches configured frame_rate.", spec.fps_from_metadata)

    skipped: List[str] = []

    # --- Stage 1: Prepare (downsample + background subtraction into prep_*.tif chunks) ---
    prep_hash = compute_hash(input_files_signature(spec), cfg.stage_params("prepare"))
    if not force and cache.is_valid("prepare", prep_hash):
        prep_files = cache.get_outputs("prepare")
        reporter.skip_stage("Preprocessing (Downsample & Background)")
        skipped.append("prepare")
    else:
        prep_files = stage_prepare(spec, cfg, work_dir, reporter, cancel_event=cancel_event)
        cache.record("prepare", prep_hash, prep_files)

    # --- Stage 2: Motion Correction + Memory Mapping ---
    mot_hash = compute_hash(prep_hash, cfg.stage_params("motion"))
    motion_valid = not force and cache.is_valid("motion", mot_hash)

    fname_memmap: str = ""
    bord_px: int = 0

    if motion_valid:
        mot_outputs = cache.get_outputs("motion")
        fname_memmap = mot_outputs[0]
        bord_px = int(cache.get_meta("motion").get("bord_px", 0))
        reporter.skip_stage("Motion Correction" if cfg.motion_correct else "Memory Mapping (no motion correction)")
        skipped.append("motion")
    else:
        with ClusterManager(cfg.n_processes) as cluster:
            fname_memmap, bord_px, extra_mot = stage_motion(
                prep_files, cfg, work_dir, cluster.dview, reporter, cancel_event=cancel_event
            )
        cache.record("motion", mot_hash, [fname_memmap] + extra_mot, meta={"bord_px": bord_px})

    # --- Stage 3: Summary Images (Cn & PNR) ---
    sum_hash = compute_hash(mot_hash, cfg.stage_params("summary"))
    summary_valid = not force and cache.is_valid("summary", sum_hash)

    if summary_valid:
        sum_outputs = cache.get_outputs("summary")
        cn_filter = np.load(sum_outputs[0])
        pnr = np.load(sum_outputs[1])
        sum_meta = cache.get_meta("summary")
        dims = (int(sum_meta.get("height", cn_filter.shape[0])), int(sum_meta.get("width", cn_filter.shape[1])))
        n_frames = int(sum_meta.get("n_frames", spec.n_frames))
        reporter.skip_stage("Summary Images")
        skipped.append("summary")
    else:
        Yr = None
        images = None
        try:
            Yr, images, dims, n_frames = load_memmap_movie(fname_memmap)
            cn_filter, pnr, cn_path, pnr_path = stage_summary(
                images, cfg, work_dir, reporter, cancel_event=cancel_event
            )
            cache.record(
                "summary",
                sum_hash,
                [cn_path, pnr_path],
                meta={"height": dims[0], "width": dims[1], "n_frames": n_frames},
            )
        finally:
            del images, Yr
            gc.collect()

    # Save resolved config for reproducibility
    cfg.save(work_dir / "config_used.yaml")

    return Session(
        input_dir=str(in_path),
        work_dir=work_dir,
        config=cfg,
        spec=spec,
        cache=cache,
        fname_memmap=fname_memmap,
        bord_px=bord_px,
        cn_filter=cn_filter,
        pnr=pnr,
        dims=dims,
        n_frames=n_frames,
        prepare_hash=prep_hash,
        motion_hash=mot_hash,
        summary_hash=sum_hash,
        timing=reporter.get_timing(),
        skipped_stages=skipped,
    )


def extract(
    session: Session,
    min_corr: Optional[float] = None,
    min_pnr: Optional[float] = None,
    force: bool = False,
    progress: Optional[ProgressReporter | Callable[..., None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineResult:
    """Stage B: Threshold selection -> CNMF-E + OASIS -> Component evaluation -> Export.

    Can be called repeatedly with different `min_corr` / `min_pnr` without redoing Stage A.
    """
    cfg = session.config.copy()
    if min_corr is not None:
        cfg.min_corr = float(min_corr)
    if min_pnr is not None:
        cfg.min_pnr = float(min_pnr)
    if min_corr is not None and min_pnr is not None:
        cfg.threshold_mode = "manual"

    work_dir = session.work_dir
    configure_caiman_tempdir(work_dir)
    logging.getLogger("caiman").setLevel(logging.WARNING)

    if isinstance(progress, ProgressReporter):
        reporter = progress
    else:
        reporter = ProgressReporter(callback=progress)

    skipped = list(session.skipped_stages)
    grid_results: Optional[Dict[Tuple[float, float], dict]] = None

    # Can we resolve min_corr / min_pnr without loading the memmap (heuristic, manual, or cached grid)?
    pre_resolved_corr: Optional[float] = None
    pre_resolved_pnr: Optional[float] = None
    grid_hash = compute_hash(session.summary_hash, cfg.stage_params("grid"), cfg.threshold_mode)

    if cfg.min_corr is not None and cfg.min_pnr is not None:
        pre_resolved_corr = round(float(cfg.min_corr), 3)
        pre_resolved_pnr = round(float(cfg.min_pnr), 2)
    elif cfg.threshold_mode in ("auto_heuristic", "heuristic"):
        h_c, h_p = estimate_params_heuristic(session.cn_filter, session.pnr)
        pre_resolved_corr = round(float(cfg.min_corr), 3) if cfg.min_corr is not None else h_c
        pre_resolved_pnr = round(float(cfg.min_pnr), 2) if cfg.min_pnr is not None else h_p
    elif cfg.threshold_mode in ("auto_grid", "grid_search", "auto_both", "both"):
        if not force and session.cache.is_valid("grid", grid_hash):
            grid_meta = session.cache.get_meta("grid")
            if "min_corr" in grid_meta and "min_pnr" in grid_meta:
                pre_resolved_corr = round(float(grid_meta["min_corr"]), 3)
                pre_resolved_pnr = round(float(grid_meta["min_pnr"]), 2)
                entries = grid_meta.get("grid_entries", [])
                if entries:
                    grid_results = {
                        (float(e["corr"]), float(e["pnr"])): dict(e["stats"])
                        for e in entries
                    }
                reporter.skip_stage("Grid Search Parameter Optimization")
                skipped.append("grid")

    # Build extract stage params for hashing (independent of how min_corr/min_pnr were chosen)
    def _make_extract_hash(mc_val: float, mp_val: float) -> str:
        ext_params = cfg.stage_params("extract")
        ext_params["min_corr"] = round(float(mc_val), 3)
        ext_params["min_pnr"] = round(float(mp_val), 2)
        return compute_hash(session.summary_hash, ext_params)

    cnm = None
    hdf5_path = str(work_dir / "analysis_results.hdf5")
    extract_cached = False

    if pre_resolved_corr is not None and pre_resolved_pnr is not None and cfg.save_hdf5:
        ext_hash = _make_extract_hash(pre_resolved_corr, pre_resolved_pnr)
        if not force and session.cache.is_valid("extract", ext_hash):
            from caiman.source_extraction.cnmf import cnmf as cnmf_module

            cached_h5 = session.cache.get_outputs("extract")[0]
            cnm = cnmf_module.load_CNMF(cached_h5)
            cnm.estimates.Cn = session.cn_filter
            cnm.estimates.dims = session.dims
            min_corr_used = pre_resolved_corr
            min_pnr_used = pre_resolved_pnr
            hdf5_path = cached_h5
            reporter.skip_stage("CNMF-E Source Extraction")
            reporter.skip_stage("Component Evaluation")
            skipped.append("extract")
            extract_cached = True

    if not extract_cached:
        Yr = None
        images = None
        try:
            Yr, images, dims, _T = load_memmap_movie(session.fname_memmap)
            with ClusterManager(cfg.n_processes) as cluster:
                dview = cluster.dview
                min_corr_used, min_pnr_used, new_grid_results = select_thresholds(
                    images,
                    session.fname_memmap,
                    session.cn_filter,
                    session.pnr,
                    cfg,
                    dview,
                    reporter,
                    bord_px=session.bord_px,
                    min_corr=pre_resolved_corr,
                    min_pnr=pre_resolved_pnr,
                    cancel_event=cancel_event,
                )
                if new_grid_results is not None:
                    grid_results = new_grid_results
                    sum_outs = session.cache.get_outputs("summary") or [str(work_dir / "Cn.npy")]
                    session.cache.record(
                        "grid",
                        grid_hash,
                        sum_outs,
                        meta={
                            "min_corr": min_corr_used,
                            "min_pnr": min_pnr_used,
                            "grid_entries": [
                                {"corr": float(k[0]), "pnr": float(k[1]), "stats": dict(v)}
                                for k, v in grid_results.items()
                            ],
                        },
                    )

                ext_hash = _make_extract_hash(min_corr_used, min_pnr_used)
                if not force and cfg.save_hdf5 and session.cache.is_valid("extract", ext_hash):
                    from caiman.source_extraction.cnmf import cnmf as cnmf_module

                    cached_h5 = session.cache.get_outputs("extract")[0]
                    cnm = cnmf_module.load_CNMF(cached_h5)
                    cnm.estimates.Cn = session.cn_filter
                    cnm.estimates.dims = session.dims
                    hdf5_path = cached_h5
                    reporter.skip_stage("CNMF-E Source Extraction")
                    reporter.skip_stage("Component Evaluation")
                    skipped.append("extract")
                    extract_cached = True
                else:
                    cnm = stage_extract_and_evaluate(
                        images,
                        session.fname_memmap,
                        session.cn_filter,
                        dims,
                        session.bord_px,
                        min_corr_used,
                        min_pnr_used,
                        cfg,
                        dview,
                        reporter,
                        cancel_event=cancel_event,
                    )
            if not extract_cached:
                if cfg.save_hdf5:
                    save_hdf5_for_curator(cnm, session.cn_filter, session.pnr, session.dims, hdf5_path)
                    session.cache.record(
                        "extract",
                        ext_hash,
                        [hdf5_path],
                        meta={"min_corr": min_corr_used, "min_pnr": min_pnr_used},
                    )
                else:
                    if os.path.exists(hdf5_path):
                        try:
                            os.remove(hdf5_path)
                        except OSError:
                            pass
        finally:
            del images, Yr
            gc.collect()

    # --- Stage 5: Export (MAT files + Figures) ---
    exp_hash = compute_hash(ext_hash, cfg.stage_params("export"))
    if not force and session.cache.is_valid("export", exp_hash):
        reporter.skip_stage("Exporting Results")
        skipped.append("export")
        A = getattr(cnm.estimates, "A", None)
        if A is not None and A.shape[1] > 0:
            A_dense = A.toarray() if sp.issparse(A) else np.asarray(A)
            SFP_array = np.reshape(A_dense, [session.dims[0], session.dims[1], A_dense.shape[1]], order="F")
        else:
            SFP_array = np.zeros((session.dims[0], session.dims[1], 0), dtype=np.float32)
    else:
        exported_files, SFP_array = export_results(
            cnm,
            session.cn_filter,
            session.pnr,
            session.dims,
            cfg,
            min_corr_used,
            min_pnr_used,
            work_dir,
            reporter,
            input_type=session.spec.kind,
        )
        if not cfg.save_hdf5 and os.path.exists(hdf5_path):
            try:
                os.remove(hdf5_path)
            except OSError:
                pass
        rec_outputs = list(exported_files)
        if cfg.save_hdf5 and os.path.exists(hdf5_path):
            rec_outputs.append(hdf5_path)
        if not rec_outputs:
            # Record config_used.yaml so cache knows export stage ran even when all flags are False
            cfg_path = work_dir / "config_used.yaml"
            cfg.save(cfg_path)
            rec_outputs.append(str(cfg_path))
        session.cache.record("export", exp_hash, rec_outputs)

    # Save resolved config with the exact min_corr and min_pnr used
    cfg_resolved = cfg.copy(min_corr=min_corr_used, min_pnr=min_pnr_used)
    cfg_resolved.save(work_dir / "config_used.yaml")

    if cfg.cleanup_mmap or cfg.convert_mmap_to_npy:
        cleanup_mmap_files(cfg, session.fname_memmap, work_dir, reporter)

    idx_good = (
        np.asarray(cnm.estimates.idx_components, dtype=int)
        if getattr(cnm.estimates, "idx_components", None) is not None
        else np.array([], dtype=int)
    )
    raw_traces = getattr(cnm.estimates, "C", None)
    deconv_traces = getattr(cnm.estimates, "S", None)

    combined_timing = dict(session.timing)
    combined_timing.update(reporter.get_timing())
    combined_timing["total"] = sum(v for k, v in combined_timing.items() if k != "total")

    return PipelineResult(
        cnm=cnm,
        cn_filter=session.cn_filter,
        pnr=session.pnr,
        dims=session.dims,
        idx_good=idx_good,
        raw_traces=raw_traces,
        deconv_traces=deconv_traces,
        spatial_footprints=SFP_array,
        min_corr_used=min_corr_used,
        min_pnr_used=min_pnr_used,
        n_neurons=len(idx_good),
        timing=combined_timing,
        config=cfg_resolved,
        grid=grid_results,
        session=session,
        skipped_stages=skipped,
        hdf5_path=hdf5_path if (cfg.save_hdf5 and os.path.exists(hdf5_path)) else None,
    )


def run(
    input_dir: Optional[str | Path] = None,
    config: Optional[PipelineConfig | str | Path] = None,
    force: bool = False,
    min_corr: Optional[float] = None,
    min_pnr: Optional[float] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineResult:
    """Run both Stage A (`preprocess`) and Stage B (`extract`) end-to-end with caching."""
    t0 = time.time()
    reporter = ProgressReporter(callback=progress_callback)
    session = preprocess(
        input_dir=input_dir,
        config=config,
        force=force,
        progress=reporter,
        cancel_event=cancel_event,
    )
    result = extract(
        session,
        min_corr=min_corr,
        min_pnr=min_pnr,
        force=force,
        progress=reporter,
        cancel_event=cancel_event,
    )
    result.timing["total"] = time.time() - t0
    return result


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Optional[Callable[..., None]] = None,
    force: bool = False,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineResult:
    """Backward-compatible entry point matching `caiman_auto_pipeline.run_pipeline`."""
    return run(
        input_dir=config.path_to_analyze,
        config=config,
        force=force,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


def discover_batch_sessions(config: PipelineConfig) -> List[str]:
    """Discover multiple session directories matching `config.batch_pattern` (supports both AVI and TIFF)."""
    base_path = config.path_to_analyze
    sessions: List[str] = []
    patterns = [p.strip() for p in config.batch_pattern.split(",") if p.strip()]

    for pattern in patterns:
        matches = glob.glob(os.path.join(base_path, pattern))
        for match in sorted(matches):
            if os.path.isdir(match):
                try:
                    discover_inputs(match, pattern=config.file_pattern, max_files=1)
                    sessions.append(match)
                except Exception:
                    pass

    if not sessions:
        sessions = [base_path]

    logger.info("Discovered %d session(s) for batch processing", len(sessions))
    return sessions


def run_batch(
    config_or_base_dir: PipelineConfig | str | Path,
    patterns: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    force: bool = False,
) -> List[PipelineResult]:
    """Run pipeline across multiple session directories."""
    if isinstance(config_or_base_dir, PipelineConfig):
        cfg = config_or_base_dir.copy()
    else:
        cfg = (config.copy() if config is not None else load_config())
        cfg.path_to_analyze = str(config_or_base_dir)
        if patterns is not None:
            cfg.batch_pattern = patterns

    sessions = discover_batch_sessions(cfg)
    results: List[PipelineResult] = []

    for i, session_path in enumerate(sessions):
        logger.info("\n%s", "#" * 60)
        logger.info("BATCH [%d/%d]: %s", i + 1, len(sessions), session_path)
        logger.info("%s", "#" * 60)

        session_cfg = cfg.copy(path_to_analyze=session_path)
        try:
            res = run(
                input_dir=session_path,
                config=session_cfg,
                force=force,
                progress_callback=progress_callback,
            )
            results.append(res)
        except Exception as e:
            logger.error("Session %s failed: %s", session_path, e)
            results.append(PipelineResult(config=session_cfg))

    if cfg.register_batch and len(results) > 1:
        _register_multisession_batch(cfg, sessions, results)

    return results


def _register_multisession_batch(
    config: PipelineConfig,
    sessions: List[str],
    results: List[PipelineResult],
) -> None:
    """Perform cross-session registration across batch results when `config.register_batch=True`."""
    try:
        import h5py
        from caiman.base.rois import register_multisession
        from scipy.io import savemat

        spatial_footprints = []
        templates = []
        traces_C = []
        traces_F = []
        valid_sessions = []

        for session_path, result in zip(sessions, results):
            if result.cnm is not None and result.n_neurons > 0:
                idx_good = (
                    np.asarray(result.idx_good, dtype=int)
                    if result.idx_good is not None
                    else np.asarray(result.cnm.estimates.idx_components, dtype=int)
                )
                if len(idx_good) == 0:
                    continue
                A_raw = result.cnm.estimates.A
                A_csc = A_raw.tocsc() if sp.issparse(A_raw) else sp.csc_matrix(A_raw)
                spatial_footprints.append(A_csc[:, idx_good])

                # Copy template because CaImAn register_multisession normalizes templates in-place
                tmpl = np.nan_to_num(result.cn_filter, nan=0.0).astype(np.float32).copy()
                if float(tmpl.max() - tmpl.min()) <= 1e-8:
                    tmpl = tmpl + np.linspace(0.0, 1e-3, tmpl.size, dtype=np.float32).reshape(tmpl.shape)
                templates.append(tmpl)

                C_arr = np.asarray(result.cnm.estimates.C)
                traces_C.append(C_arr[idx_good] if C_arr.ndim == 2 and C_arr.shape[0] > max(idx_good) else C_arr)
                F_dff = getattr(result.cnm.estimates, "F_dff", None)
                if F_dff is not None:
                    F_arr = np.asarray(F_dff)
                    traces_F.append(F_arr[idx_good] if F_arr.ndim == 2 and F_arr.shape[0] > max(idx_good) else F_arr)
                else:
                    traces_F.append(None)
                valid_sessions.append(session_path)

        if len(spatial_footprints) > 1:
            dims = templates[0].shape
            spatial_union, assignments, matchings = register_multisession(
                A=spatial_footprints,
                dims=dims,
                templates=templates,
                max_dist=config.register_max_dist,
            )

            base_abs = os.path.abspath(config.path_to_analyze) if config.path_to_analyze else ""
            session_labels = []
            for s in valid_sessions:
                s_abs = os.path.abspath(s)
                try:
                    if base_abs and os.path.commonpath([s_abs, base_abs]) == base_abs and s_abs != base_abs:
                        session_labels.append(os.path.relpath(s_abs, base_abs).replace("\\", "/"))
                    else:
                        session_labels.append(os.path.basename(s_abs))
                except ValueError:
                    session_labels.append(os.path.basename(s_abs))

            # Convert inhomogeneous per-session lists to object arrays for scipy.io.savemat
            matchings_obj = np.empty((len(matchings),), dtype=object)
            for i, m in enumerate(matchings):
                matchings_obj[i] = np.asarray(m, dtype=np.int64)

            traces_C_obj = np.empty((len(traces_C),), dtype=object)
            traces_F_obj = np.empty((len(traces_F),), dtype=object)
            for i, C_tr in enumerate(traces_C):
                traces_C_obj[i] = C_tr
            for i, F_tr in enumerate(traces_F):
                traces_F_obj[i] = F_tr if F_tr is not None else np.array([])

            sessions_obj = np.empty((len(session_labels),), dtype=object)
            for i, lbl in enumerate(session_labels):
                sessions_obj[i] = lbl

            reg_results = {
                "sessions": sessions_obj,
                "assignments": np.asarray(assignments, dtype=np.float64),
                "spatial_union": spatial_union,
                "matchings": matchings_obj,
                "max_dist": float(config.register_max_dist),
                "traces_C": traces_C_obj,
                "traces_F_dff": traces_F_obj,
            }

            if config.save_mat:
                save_path_mat = os.path.join(config.path_to_analyze, "batch_registration.mat")
                savemat(save_path_mat, {"registration": reg_results})

            if config.save_hdf5:
                save_path_h5 = os.path.join(config.path_to_analyze, "batch_registration.hdf5")
                with h5py.File(save_path_h5, "w") as f:
                    f.create_dataset("sessions", data=[lbl.encode("utf-8") for lbl in session_labels])
                    f.create_dataset("assignments", data=np.asarray(assignments, dtype=np.float64))
                    f.create_dataset("max_dist", data=float(config.register_max_dist))
                    m_grp = f.create_group("matchings")
                    for i, m in enumerate(matchings):
                        m_grp.create_dataset(f"session_{i}", data=np.asarray(m, dtype=np.int64))
                    if sp.issparse(spatial_union):
                        g = f.create_group("spatial_union")
                        g.create_dataset("data", data=spatial_union.data)
                        g.create_dataset("indices", data=spatial_union.indices)
                        g.create_dataset("indptr", data=spatial_union.indptr)
                        g.create_dataset("shape", data=spatial_union.shape)
                    else:
                        f.create_dataset("spatial_union", data=spatial_union)
    except Exception as e:
        logger.error("Multisession registration failed: %s", e, exc_info=True)
