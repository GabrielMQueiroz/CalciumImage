# -*- coding: utf-8 -*-
"""Tests for `decovo.config` and `presets/miniscope_1p_default.yaml`."""

from unittest.mock import patch

import pytest

from decovo.config import PipelineConfig, load_config


def test_default_preset_matches_validated_user_choices():
    cfg = load_config()
    assert cfg.spatial_downsample == pytest.approx(0.6)
    assert cfg.gSig == pytest.approx(3.0)
    assert cfg.gSiz_px == 13  # 4 * gSig + 1
    assert cfg.gSiz_tuple == (13, 13)
    assert cfg.rf == 40
    assert cfg.stride == 20
    assert cfg.merge_thr == pytest.approx(0.7)
    assert cfg.min_SNR == pytest.approx(3.0)
    assert cfg.rval_thr == pytest.approx(0.95)
    assert cfg.max_shifts == (4, 4)
    assert cfg.gSig_filt == (3, 3)
    assert cfg.p == 1
    assert cfg.nb == 0
    assert cfg.ring_size_factor == pytest.approx(1.4)
    assert cfg.use_cnn is False


def test_gsig_override_recomputes_gsiz_4x_plus_1():
    cfg = load_config(gSig=4.0)
    assert cfg.gSig == pytest.approx(4.0)
    assert cfg.gSiz_px == 17  # 4 * 4 + 1
    assert cfg.gSiz_tuple == (17, 17)

    cfg_explicit = load_config(gSig=4.0, gSiz=15)
    assert cfg_explicit.gSiz_px == 15


def test_yaml_round_trip(tmp_path):
    cfg = load_config(
        frame_rate=15.0,
        spatial_downsample=0.5,
        gSig=3.0,
        min_corr=0.82,
        min_pnr=9.5,
        threshold_mode="manual",
    )
    yaml_path = tmp_path / "custom_preset.yaml"
    cfg.save(yaml_path)

    reloaded = load_config(yaml_path)
    assert reloaded.frame_rate == pytest.approx(15.0)
    assert reloaded.spatial_downsample == pytest.approx(0.5)
    assert reloaded.gSig == pytest.approx(3.0)
    assert reloaded.gSiz_px == 13
    assert reloaded.rf == 40
    assert reloaded.stride == 20
    assert reloaded.merge_thr == pytest.approx(0.7)
    assert reloaded.min_SNR == pytest.approx(3.0)
    assert reloaded.rval_thr == pytest.approx(0.95)
    assert reloaded.max_shifts == (4, 4)
    assert reloaded.min_corr == pytest.approx(0.82)
    assert reloaded.min_pnr == pytest.approx(9.5)
    assert reloaded.threshold_mode == "manual"


def test_legacy_aliases_and_no_cuda_side_effect():
    with patch("decovo.env.detect_cuda") as mock_cuda:
        cfg = PipelineConfig(
            stride_cnmf=25,
            gnb=1,
            r_values_min=0.92,
            auto_params=False,
        )
        mock_cuda.assert_not_called()

    assert cfg.stride == 25
    assert cfg.stride_cnmf == 25
    assert cfg.nb == 1
    assert cfg.gnb == 1
    assert cfg.rval_thr == pytest.approx(0.92)
    assert cfg.r_values_min == pytest.approx(0.92)
    assert cfg.auto_params is False
    assert cfg.threshold_mode == "manual"
