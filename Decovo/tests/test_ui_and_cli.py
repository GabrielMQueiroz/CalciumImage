# -*- coding: utf-8 -*-
"""Tests for CLI argument parsing, metadata FPS detection, cancellation, and dashboard instantiation."""

import threading

import numpy as np
import pytest
import tifffile

from decovo.cli import build_parser, config_from_args
from decovo.config import load_config
from decovo.notebook_ui import inspect_input_banner
from decovo.params_auto import CancelledError
from decovo.pipeline import preprocess


def test_cli_reads_tiff_metadata_fps_and_respects_user_override(tmp_path):
    data = np.zeros((20, 32, 32), dtype=np.uint16)
    tif_path = tmp_path / "rec.tif"
    tifffile.imwrite(
        str(tif_path),
        data,
        imagej=True,
        metadata={"axes": "TYX", "finterval": 1.0 / 12.0},
    )

    parser = build_parser()

    # 1. When --frame-rate is not passed, CLI picks up 12.0 Hz from TIFF metadata
    args_auto = parser.parse_args(["--path", str(tmp_path)])
    cfg_auto = config_from_args(args_auto)
    assert cfg_auto.frame_rate == pytest.approx(12.0, rel=1e-2)
    assert cfg_auto.gSiz_px == 13
    assert cfg_auto.rf == 40
    assert cfg.stride if (cfg := cfg_auto) else True
    assert cfg_auto.stride == 20
    assert cfg_auto.merge_thr == pytest.approx(0.7)
    assert cfg_auto.min_SNR == pytest.approx(3.0)
    assert cfg_auto.rval_thr == pytest.approx(0.95)
    assert cfg_auto.max_shifts == (4, 4)
    assert cfg_auto.spatial_downsample == pytest.approx(0.6)

    # 2. When --frame-rate is explicitly passed by user, user selection wins
    args_manual = parser.parse_args(["--path", str(tmp_path), "--frame-rate", "25.0"])
    cfg_manual = config_from_args(args_manual)
    assert cfg_manual.frame_rate == pytest.approx(25.0)


def test_inspect_input_banner_reports_metadata_fps(tmp_path, capsys):
    data = np.zeros((30, 40, 50), dtype=np.uint16)
    tifffile.imwrite(
        str(tmp_path / "session.tif"),
        data,
        imagej=True,
        metadata={"axes": "TYX", "fps": 15.0},
    )
    spec = inspect_input_banner(tmp_path, user_frame_rate=20.0, spatial_downsample=0.6)
    captured = capsys.readouterr().out
    assert spec.kind == "tiff"
    assert spec.fps_from_metadata == pytest.approx(15.0)
    assert "15.00 Hz" in captured
    assert "Differs from your form value" in captured


def test_cancel_event_aborts_preprocess(tmp_path):
    data = np.ones((25, 32, 32), dtype=np.uint16) * 100
    tifffile.imwrite(str(tmp_path / "session.tif"), data)
    cfg = load_config(path_to_analyze=str(tmp_path), chunk_frames=5, n_processes=1)

    cancel_ev = threading.Event()
    cancel_ev.set()

    with pytest.raises(CancelledError):
        preprocess(tmp_path, config=cfg, cancel_event=cancel_ev)


def test_dashboard_creation_succeeds():
    from decovo.dashboard import _PANEL_OK, create_dashboard

    if not _PANEL_OK:
        pytest.skip("Panel not installed")
    tmpl = create_dashboard()
    assert tmpl is not None
