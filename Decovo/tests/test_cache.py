# -*- coding: utf-8 -*-
"""Tests for stage caching and selective re-execution (`decovo.cache` & `decovo.pipeline`)."""

import tifffile

from decovo.config import load_config
from decovo.pipeline import extract, preprocess, run
from tests.conftest import make_synthetic_1p_movie


def test_cache_skips_completed_stages_and_invalidates_downstream(tmp_path):
    movie, _gt = make_synthetic_1p_movie(n_frames=160, shape=(80, 80), n_neurons=6, gSig=2.5, seed=11)
    tif_path = tmp_path / "session.tif"
    tifffile.imwrite(str(tif_path), movie)

    cfg = load_config(
        path_to_analyze=str(tmp_path),
        spatial_downsample=1.0,
        frame_rate=20.0,
        gSig=2.5,
        n_processes=1,
        save_figures=False,
    )

    # 1. First full run -> nothing skipped
    res1 = run(tmp_path, config=cfg)
    assert res1.skipped_stages == []

    # 2. Second identical run -> all 5 stages skipped!
    res2 = run(tmp_path, config=cfg)
    assert set(res2.skipped_stages) == {"prepare", "motion", "summary", "extract", "export"}

    # 3. Change min_pnr -> prepare, motion, summary skipped; extract & export re-run!
    new_pnr = round(res1.min_pnr_used + 0.5, 2)
    res3 = run(tmp_path, config=cfg, min_corr=res1.min_corr_used, min_pnr=new_pnr)
    assert set(res3.skipped_stages) == {"prepare", "motion", "summary"}

    # 4. Change gSig -> prepare and motion skipped; summary, extract, export re-run!
    cfg_gsig = cfg.copy(gSig=3.0)
    res4 = run(tmp_path, config=cfg_gsig)
    assert set(res4.skipped_stages) == {"prepare", "motion"}

    # 5. Re-calling extract(session, ...) with the SAME thresholds hits cache (even though mode becomes 'manual')
    sess = preprocess(tmp_path, config=cfg_gsig)
    assert set(sess.skipped_stages) == {"prepare", "motion", "summary"}
    res_same = extract(sess, min_corr=res4.min_corr_used, min_pnr=res4.min_pnr_used)
    assert "extract" in res_same.skipped_stages

    # 6. Re-calling extract(session, ...) with DIFFERENT thresholds re-runs extract
    res5 = extract(sess, min_corr=res4.min_corr_used, min_pnr=round(res4.min_pnr_used + 0.4, 2))
    assert "extract" not in res5.skipped_stages


def test_grid_search_cache_and_no_hdf5_flag(tmp_path):
    movie, _gt = make_synthetic_1p_movie(n_frames=120, shape=(72, 72), n_neurons=5, gSig=2.5, seed=22)
    tif_path = tmp_path / "session_grid.tif"
    tifffile.imwrite(str(tif_path), movie)

    cfg_grid = load_config(
        path_to_analyze=str(tmp_path),
        spatial_downsample=1.0,
        frame_rate=20.0,
        gSig=2.5,
        threshold_mode="auto_grid",
        grid_n_frames=80,
        grid_corr_steps=2,
        grid_pnr_steps=2,
        n_processes=1,
        save_figures=False,
    )

    # First grid run executes grid + extract
    r1 = run(tmp_path, config=cfg_grid)
    assert r1.grid is not None and len(r1.grid) == 4
    assert "grid" not in r1.skipped_stages
    assert "extract" not in r1.skipped_stages

    # Second identical grid run skips prepare, motion, summary, grid, extract, and export
    r2 = run(tmp_path, config=cfg_grid)
    assert {"prepare", "motion", "summary", "grid", "extract", "export"}.issubset(set(r2.skipped_stages))
    assert r2.grid is not None and len(r2.grid) == 4

    # Running with save_hdf5=False removes analysis_results.hdf5 and sets hdf5_path=None
    cfg_no_h5 = cfg_grid.copy(save_hdf5=False, threshold_mode="auto_heuristic")
    r3 = run(tmp_path, config=cfg_no_h5, force=True)
    assert r3.hdf5_path is None
    assert not (r3.session.work_dir / "analysis_results.hdf5").exists()

