# -*- coding: utf-8 -*-
"""Automatic threshold selection (heuristic and grid search) for CNMF-E initialization."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger("decovo")


class CancelledError(RuntimeError):
    """Raised when a pipeline stage or grid search is cancelled via `cancel_event`."""


def estimate_params_heuristic(
    cn_filter: np.ndarray,
    pnr: np.ndarray,
) -> Tuple[float, float]:
    """Fast heuristic estimate of `min_corr` and `min_pnr` from summary image distributions."""
    corr_mean = float(np.nanmean(cn_filter))
    corr_std = float(np.nanstd(cn_filter))
    min_corr = float(np.clip(corr_mean + 0.5 * corr_std, 0.4, 0.95))

    pnr_mean = float(np.nanmean(pnr))
    pnr_std = float(np.nanstd(pnr))
    min_pnr = float(np.clip(pnr_mean + 1.0 * pnr_std, 3.0, 20.0))

    corr_p90 = float(np.nanpercentile(cn_filter, 90))
    pnr_p90 = float(np.nanpercentile(pnr, 90))

    min_corr = round(float(0.6 * min_corr + 0.4 * corr_p90), 3)
    min_pnr = round(float(0.6 * min_pnr + 0.4 * pnr_p90), 2)

    logger.info("Heuristic estimate: min_corr=%.3f, min_pnr=%.2f", min_corr, min_pnr)
    logger.info("  (corr stats: mean=%.3f, std=%.3f, p90=%.3f)", corr_mean, corr_std, corr_p90)
    logger.info("  (pnr  stats: mean=%.2f,  std=%.2f,  p90=%.2f)", pnr_mean, pnr_std, pnr_p90)

    return min_corr, min_pnr


def run_grid_search(
    images: np.ndarray,
    fname_new: str,
    cn_filter: np.ndarray,
    pnr: np.ndarray,
    config: PipelineConfig,
    dview,
    progress,
    bord_px: int = 0,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[float, float, Dict[Tuple[float, float], dict]]:
    """Grid search over `min_corr` and `min_pnr` on a subset of frames.

    Returns:
        `(best_corr, best_pnr, results_grid)`
    """
    from caiman.source_extraction.cnmf import cnmf as cnmf_module
    from caiman.source_extraction.cnmf import params

    progress.start_stage("Grid Search Parameter Optimization")

    init_corr, init_pnr = estimate_params_heuristic(cn_filter, pnr)

    corr_range = np.linspace(
        max(0.3, init_corr - 0.15),
        min(0.98, init_corr + 0.15),
        config.grid_corr_steps,
    )
    pnr_range = np.linspace(
        max(2.0, init_pnr - 3.0),
        min(25.0, init_pnr + 3.0),
        config.grid_pnr_steps,
    )

    n_total = images.shape[0]
    n_use = min(config.grid_n_frames, n_total)
    if n_use < n_total:
        # Contiguous slice preserves np.memmap and C-order Yr layout required by CNMF.fit
        images_sub = images[:n_use]
        eff_rf = None
        eff_stride = None
    else:
        images_sub = images
        dims = images.shape[1:]
        min_dim = min(dims)
        if min_dim <= 2 * int(config.rf) + int(config.stride):
            eff_rf = None
            eff_stride = None
        else:
            eff_rf = int(config.rf)
            eff_stride = int(config.stride)

    total_combos = len(corr_range) * len(pnr_range)
    results: Dict[Tuple[float, float], dict] = {}
    best_score = -np.inf
    best_corr, best_pnr = init_corr, init_pnr

    logger.info("Testing %d parameter combinations on %d frames...", total_combos, n_use)

    combo_idx = 0
    for mc in corr_range:
        for mp in pnr_range:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Grid search cancelled by user.")

            combo_idx += 1
            mc_round = round(float(mc), 3)
            mp_round = round(float(mp), 2)

            try:
                cnmfe_dict = {
                    "fnames": [fname_new] if isinstance(fname_new, str) else fname_new,
                    "fr": config.frate,
                    "decay_time": config.decay_time,
                    "method_init": "corr_pnr",
                    "K": None,
                    "gSig": config.gSig_tuple,
                    "gSiz": config.gSiz_tuple,
                    "merge_thr": config.merge_thr,
                    "p": 0,
                    "tsub": max(config.tsub, 2),
                    "ssub": max(config.ssub, 2),
                    "rf": eff_rf,
                    "stride": eff_stride,
                    "only_init": True,
                    "nb": config.nb,
                    "nb_patch": 0,
                    "method_deconvolution": config.method_deconvolution,
                    "low_rank_background": None,
                    "update_background_components": True,
                    "min_corr": mc_round,
                    "min_pnr": mp_round,
                    "normalize_init": False,
                    "center_psf": True,
                    "ssub_B": config.ssub_B,
                    "ring_size_factor": config.ring_size_factor,
                    "del_duplicates": True,
                    "border_pix": int(bord_px),
                }

                opts_gs = params.CNMFParams(params_dict=cnmfe_dict)
                cnm_gs = cnmf_module.CNMF(
                    n_processes=config.n_processes,
                    dview=dview,
                    Ain=None,
                    params=opts_gs,
                )
                cnm_gs.fit(images_sub)

                n_total_comp = len(cnm_gs.estimates.C) if cnm_gs.estimates.C is not None else 0
                if n_total_comp > 0:
                    cnm_gs.params.set(
                        "quality",
                        {
                            "min_SNR": config.min_SNR,
                            "rval_thr": config.rval_thr,
                            "use_cnn": False,
                        },
                    )
                    cnm_gs.estimates.evaluate_components(images_sub, cnm_gs.params, dview=dview)
                    n_accepted = len(cnm_gs.estimates.idx_components)
                else:
                    n_accepted = 0

                if n_total_comp == 0:
                    score = -100.0
                else:
                    acceptance_ratio = n_accepted / n_total_comp
                    count_score = -abs(np.log(max(n_accepted, 1)) - np.log(50))
                    ratio_score = acceptance_ratio * 10.0
                    score = float(count_score + ratio_score)

                results[(mc_round, mp_round)] = {
                    "n_total": int(n_total_comp),
                    "n_accepted": int(n_accepted),
                    "acceptance_ratio": round(n_accepted / max(n_total_comp, 1), 3),
                    "score": round(score, 3),
                }

                if score > best_score:
                    best_score = score
                    best_corr = mc_round
                    best_pnr = mp_round

                logger.info(
                    "  [%d/%d] corr=%.3f, pnr=%.2f -> %d/%d neurons (score=%.2f)",
                    combo_idx, total_combos, mc_round, mp_round, n_accepted, n_total_comp, score
                )
                del cnm_gs
            except Exception as e:
                logger.warning(
                    "  [%d/%d] corr=%.3f, pnr=%.2f -> FAILED: %s",
                    combo_idx, total_combos, mc_round, mp_round, e
                )
                results[(mc_round, mp_round)] = {
                    "n_total": 0,
                    "n_accepted": 0,
                    "acceptance_ratio": 0.0,
                    "score": -999.0,
                }

            progress.update(combo_idx)

    del images_sub
    logger.info("Grid search best: min_corr=%.3f, min_pnr=%.2f (score=%.2f)", best_corr, best_pnr, best_score)
    progress.end_stage()
    return best_corr, best_pnr, results


def select_thresholds(
    images: np.ndarray,
    fname_new: str,
    cn_filter: np.ndarray,
    pnr: np.ndarray,
    config: PipelineConfig,
    dview,
    progress,
    bord_px: int = 0,
    min_corr: Optional[float] = None,
    min_pnr: Optional[float] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[float, float, Optional[Dict[Tuple[float, float], dict]]]:
    """Resolve `(min_corr, min_pnr, grid_results)` according to manual overrides or `config.threshold_mode`."""
    if min_corr is not None and min_pnr is not None:
        return round(float(min_corr), 3), round(float(min_pnr), 2), None

    mode = config.threshold_mode
    if mode == "manual":
        if config.min_corr is None or config.min_pnr is None:
            raise ValueError("min_corr and min_pnr must be provided when threshold_mode='manual'")
        return round(float(config.min_corr), 3), round(float(config.min_pnr), 2), None

    if mode in ("auto_heuristic", "heuristic"):
        c, p = estimate_params_heuristic(cn_filter, pnr)
        return (
            round(float(min_corr), 3) if min_corr is not None else c,
            round(float(min_pnr), 2) if min_pnr is not None else p,
            None,
        )

    if mode in ("auto_grid", "grid_search"):
        c, p, grid = run_grid_search(
            images, fname_new, cn_filter, pnr, config, dview, progress,
            bord_px=bord_px, cancel_event=cancel_event
        )
        return c, p, grid

    if mode in ("auto_both", "both"):
        h_corr, h_pnr = estimate_params_heuristic(cn_filter, pnr)
        best_corr, best_pnr, grid = run_grid_search(
            images, fname_new, cn_filter, pnr, config, dview, progress,
            bord_px=bord_px, cancel_event=cancel_event
        )
        h_key = (round(h_corr, 3), round(h_pnr, 2))
        g_key = (best_corr, best_pnr)
        if h_key in grid and g_key in grid:
            h_score = grid.get(h_key, {}).get("score", -999.0)
            g_score = grid.get(g_key, {}).get("score", -999.0)
            if h_score >= g_score * 0.9:
                return h_corr, h_pnr, grid
        return best_corr, best_pnr, grid

    raise ValueError(f"Unknown threshold_mode: {mode!r}")
