#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible shim for `caiman_auto_pipeline.py` -> delegates to `decovo` package."""

import sys

from decovo.cli import build_parser, config_from_args, main
from decovo.config import PipelineConfig, load_config
from decovo.env import detect_cuda, get_gpu_memory_mb
from decovo.export import cleanup_mmap_files, export_results
from decovo.io import InputSpec, discover_inputs
from decovo.params_auto import estimate_params_heuristic, run_grid_search
from decovo.pipeline import (
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
from decovo.stages import ClusterManager, ProgressReporter


def discover_files(config: PipelineConfig):
    """Backward-compatible wrapper returning filenames discovered in `config.path_to_analyze`."""
    spec = discover_inputs(config.path_to_analyze, pattern=config.file_pattern, max_files=config.max_files)
    return spec.files


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "Result",
    "Session",
    "load_config",
    "preprocess",
    "extract",
    "run",
    "run_pipeline",
    "run_batch",
    "discover_files",
    "discover_inputs",
    "discover_batch_sessions",
    "estimate_params_heuristic",
    "run_grid_search",
    "ClusterManager",
    "ProgressReporter",
    "export_results",
    "cleanup_mmap_files",
    "detect_cuda",
    "get_gpu_memory_mb",
    "build_parser",
    "config_from_args",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
