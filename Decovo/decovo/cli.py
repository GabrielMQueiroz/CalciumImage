# -*- coding: utf-8 -*-
"""Command-line interface for Decovo (supports AVI chunks and multi-page TIFF stacks)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import PipelineConfig, load_config
from .io import discover_inputs
from .pipeline import run, run_batch

logger = logging.getLogger("decovo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decovo v2: CaImAn CNMF-E 1-Photon Pipeline (AVI & Multi-Page TIFF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single session (auto-detects msCam*.avi OR *.tif/*.tiff)
  python caiman_auto_pipeline.py --path "D:/data/Animal1/pre"

  # Using a custom YAML preset and forcing a fresh run (ignoring cache)
  python caiman_auto_pipeline.py --path "D:/data/session.tif" --config presets/miniscope_1p_default.yaml --force

  # Manual min_corr / min_pnr thresholds (skips Stage A if already cached!)
  python caiman_auto_pipeline.py --path "D:/data/Animal1/pre" --min-corr 0.84 --min-pnr 8.0

  # Batch mode over multiple sessions
  python caiman_auto_pipeline.py --path "D:/data" --batch --batch-pattern "Animal*/pre,Animal*/post/d*"
        """,
    )

    parser.add_argument("--path", required=True, help="Path to session directory or .tif/.avi file")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML preset file")
    parser.add_argument("--force", action="store_true", default=False, help="Ignore cached stages and start fresh")

    # Parameter selection
    param_group = parser.add_argument_group("Parameter Selection")
    param_group.add_argument("--auto-params", action="store_true", default=None, help="Auto-select min_corr/min_pnr")
    param_group.add_argument("--no-auto-params", dest="auto_params", action="store_false", help="Disable auto parameter selection")
    param_group.add_argument(
        "--auto-method",
        choices=["heuristic", "grid_search", "both"],
        default=None,
        help="Auto-parameter method",
    )
    param_group.add_argument("--min-corr", type=float, default=None, help="Manual min correlation threshold")
    param_group.add_argument("--min-pnr", type=float, default=None, help="Manual min PNR threshold")

    # Preprocessing
    pre_group = parser.add_argument_group("Preprocessing")
    pre_group.add_argument("--spatial-ds", type=float, default=None, help="Spatial downsample ratio (default: 0.6)")
    pre_group.add_argument("--temporal-ds", type=float, default=None, help="Temporal downsample ratio (default: 1.0)")
    pre_group.add_argument("--frame-rate", type=float, default=None, help="Frame rate in Hz (default: metadata or 20.0)")
    pre_group.add_argument("--decay-time", type=float, default=None, help="Transient decay time in s (default: 0.4)")
    pre_group.add_argument("--no-bg-removal", dest="remove_bg", action="store_false", default=None, help="Skip background removal")
    pre_group.add_argument("--no-motion-correct", dest="motion_correct", action="store_false", default=None, help="Skip motion correction")

    # Source extraction
    src_group = parser.add_argument_group("Source Extraction")
    src_group.add_argument("--gsig", type=float, default=None, help="Gaussian width gSig (default: 3.0)")
    src_group.add_argument("--gsiz", type=int, default=None, help="Neuron bounding box gSiz (default: 4*gSig+1 = 13)")
    src_group.add_argument("--rf", type=int, default=None, help="Patch half-size rf (default: 40)")
    src_group.add_argument("--stride", type=int, default=None, help="Patch overlap stride (default: 20)")
    src_group.add_argument("--merge-thr", type=float, default=None, help="Merge threshold (default: 0.7)")
    src_group.add_argument("--min-snr", type=float, default=None, help="Min SNR for evaluation (default: 3.0)")
    src_group.add_argument("--r-values-min", type=float, default=None, help="Min spatial consistency rval_thr (default: 0.95)")

    # Performance
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--n-processes", type=int, default=None, help="Number of worker processes")
    perf_group.add_argument("--cuda", action="store_true", default=None, help="Enable CUDA for motion correction")
    perf_group.add_argument("--max-files", type=int, default=None, help="Limit number of input files")

    # Batch
    batch_group = parser.add_argument_group("Batch Processing")
    batch_group.add_argument("--batch", action="store_true", default=False, help="Enable batch mode")
    batch_group.add_argument("--batch-pattern", type=str, default=None, help="Comma-separated glob patterns")
    batch_group.add_argument("--register-batch", action="store_true", default=False, help="Cross-session registration")
    batch_group.add_argument("--register-max-dist", type=float, default=None, help="Max registration distance in px")

    # Output & Cleanup
    out_group = parser.add_argument_group("Output & Cleanup")
    out_group.add_argument("--results-folder", type=str, default=None, help="Name of results subdirectory")
    out_group.add_argument("--no-hdf5", dest="save_hdf5", action="store_false", default=None)
    out_group.add_argument("--no-mat", dest="save_mat", action="store_false", default=None)
    out_group.add_argument("--save-mc-movie", action="store_true", default=None, help="Save motion-corrected movie")
    out_group.add_argument("--cleanup-mmap", dest="cleanup_mmap", action="store_true", default=None, help="Delete intermediate .mmap files")
    out_group.add_argument("--no-cleanup-mmap", dest="cleanup_mmap", action="store_false", help="Keep .mmap files (default for resume)")
    out_group.add_argument("--convert-npy", dest="convert_mmap_to_npy", action="store_true", default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = load_config(args.config)
    cfg.path_to_analyze = args.path

    if args.results_folder is not None:
        cfg.results_folder = args.results_folder
    if args.spatial_ds is not None:
        cfg.spatial_downsample = args.spatial_ds
    if args.temporal_ds is not None:
        cfg.temporal_downsample = args.temporal_ds

    # Frame rate: if user didn't pass --frame-rate on CLI, check if metadata provides FPS
    if args.frame_rate is not None:
        cfg.frame_rate = args.frame_rate
    elif not args.batch:
        try:
            spec = discover_inputs(args.path, pattern=cfg.file_pattern, max_files=1)
            if spec.fps_from_metadata is not None and spec.fps_from_metadata > 0:
                logger.info(
                    "Detected frame rate %.2f Hz from file metadata (pass --frame-rate to override).",
                    spec.fps_from_metadata,
                )
                cfg.frame_rate = spec.fps_from_metadata
        except Exception:
            pass

    if args.decay_time is not None:
        cfg.decay_time = args.decay_time
    if args.remove_bg is not None:
        cfg.remove_background = args.remove_bg
    if args.motion_correct is not None:
        cfg.motion_correct = args.motion_correct
    if args.gsig is not None:
        cfg.gSig = args.gsig
        if args.gsiz is None:
            cfg.gSiz = int(round(4 * cfg.gSig + 1))
    if args.gsiz is not None:
        cfg.gSiz = args.gsiz
    if args.rf is not None:
        cfg.rf = args.rf
    if args.stride is not None:
        cfg.stride = args.stride
    if args.merge_thr is not None:
        cfg.merge_thr = args.merge_thr
    if args.min_snr is not None:
        cfg.min_SNR = args.min_snr
    if args.r_values_min is not None:
        cfg.rval_thr = args.r_values_min
    if args.auto_method is not None:
        cfg.auto_method = args.auto_method
    if args.auto_params is not None:
        cfg.auto_params = args.auto_params
    if args.min_corr is not None:
        cfg.min_corr = args.min_corr
    if args.min_pnr is not None:
        cfg.min_pnr = args.min_pnr
    if args.min_corr is not None and args.min_pnr is not None:
        cfg.threshold_mode = "manual"
    if args.n_processes is not None:
        cfg.n_processes = args.n_processes
    if args.cuda is not None:
        cfg.use_cuda = args.cuda
    if args.max_files is not None:
        cfg.max_files = args.max_files
    if args.save_hdf5 is not None:
        cfg.save_hdf5 = args.save_hdf5
    if args.save_mat is not None:
        cfg.save_mat = args.save_mat
    if args.save_mc_movie is not None:
        cfg.save_motion_corrected = args.save_mc_movie
    if args.cleanup_mmap is not None:
        cfg.cleanup_mmap = args.cleanup_mmap
    if args.convert_mmap_to_npy is not None:
        cfg.convert_mmap_to_npy = args.convert_mmap_to_npy

    cfg.batch_mode = bool(args.batch)
    if args.batch_pattern is not None:
        cfg.batch_pattern = args.batch_pattern
    cfg.register_batch = bool(args.register_batch)
    if args.register_max_dist is not None:
        cfg.register_max_dist = args.register_max_dist

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        if config.batch_mode:
            results = run_batch(config, force=args.force)
            n_success = sum(1 for r in results if r.n_neurons > 0)
            logger.info("Batch complete: %d/%d sessions successful", n_success, len(results))
        else:
            result = run(input_dir=config.path_to_analyze, config=config, force=args.force)
            logger.info("Pipeline complete: %d neurons found (saved in %s)", result.n_neurons, result.session.work_dir)
        return 0
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
