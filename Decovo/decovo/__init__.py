# -*- coding: utf-8 -*-
"""Decovo v2: Unified CaImAn CNMF-E 1-photon pipeline for AVI & multi-page TIFF recordings."""

from .config import PipelineConfig, load_config
from .env import auto_n_processes, configure_caiman_tempdir, default_workdir, detect_cuda, in_colab
from .io import InputSpec, discover_inputs, iter_frames_chunks
from .params_auto import CancelledError, estimate_params_heuristic, run_grid_search, select_thresholds
from .pipeline import (
    PipelineResult,
    Result,
    Session,
    discover_batch_sessions,
    extract,
    preprocess,
    run,
    run_batch,
    run_pipeline,
)
from .stages import ClusterManager, ProgressReporter

__version__ = "2.0.0"

__all__ = [
    "PipelineConfig",
    "load_config",
    "InputSpec",
    "discover_inputs",
    "iter_frames_chunks",
    "Session",
    "PipelineResult",
    "Result",
    "preprocess",
    "extract",
    "run",
    "run_pipeline",
    "run_batch",
    "discover_batch_sessions",
    "estimate_params_heuristic",
    "run_grid_search",
    "select_thresholds",
    "ClusterManager",
    "ProgressReporter",
    "CancelledError",
    "in_colab",
    "default_workdir",
    "auto_n_processes",
    "configure_caiman_tempdir",
    "detect_cuda",
]
