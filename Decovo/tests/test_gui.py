# -*- coding: utf-8 -*-
"""Unit tests for Decovo native Tkinter desktop GUI (decovo/gui.py)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure Windows Anaconda Tcl/Tk path is found reliably across repeated Tk instances
conda_prefix = sys.prefix
tcl_dir = os.path.join(conda_prefix, "Library", "lib", "tcl8.6")
tk_dir = os.path.join(conda_prefix, "Library", "lib", "tk8.6")
if os.path.exists(tcl_dir) and "TCL_LIBRARY" not in os.environ:
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir) and "TK_LIBRARY" not in os.environ:
    os.environ["TK_LIBRARY"] = tk_dir

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import scipy.sparse as sp

import tkinter as tk
from decovo.config import PipelineConfig
from decovo.gui import DecovoApp, fix_dims, get_contours


@pytest.fixture
def mock_cnm():
    """Generates a synthetic CNMF-like object for testing GUI curation."""
    n_cells = 6
    T = 200
    dims = (40, 50)
    pixel_count = dims[0] * dims[1]

    # Sparse spatial footprints
    A = sp.lil_matrix((pixel_count, n_cells), dtype=np.float32)
    # Temporal components
    C = np.zeros((n_cells, T), dtype=np.float32)

    for i in range(n_cells):
        center_y = 10 + (i // 3) * 15
        center_x = 10 + (i % 3) * 15
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dy * dy + dx * dx <= 9:
                    y = np.clip(center_y + dy, 0, dims[0] - 1)
                    x = np.clip(center_x + dx, 0, dims[1] - 1)
                    idx = y + x * dims[0]
                    A[idx, i] = np.exp(-(dy * dy + dx * dx) / 4.0)

        # Baseline + calcium spikes
        t = np.arange(T)
        C[i] = np.sin(t / (10.0 + i * 2)) * 0.5 + 0.5
        C[i, [20, 60, 110, 150]] += 3.0  # Spikes

    A = A.tocsc()
    YrA = np.random.randn(n_cells, T).astype(np.float32) * 0.05
    Cn = np.random.rand(*dims).astype(np.float32)

    estimates = SimpleNamespace(
        A=A,
        C=C,
        YrA=YrA,
        Cn=Cn,
        dims=list(dims),
        idx_components=np.array([0, 1, 2, 3]),
        idx_components_bad=np.array([4, 5]),
        SNR_comp=np.array([4.2, 5.1, 3.8, 6.0, 1.2, 1.5]),
        r_values=np.array([0.96, 0.94, 0.98, 0.92, 0.65, 0.70]),
    )
    return SimpleNamespace(estimates=estimates)


def test_fix_dims():
    A = sp.csr_matrix((100, 5))
    assert fix_dims(A, (10, 10)) == (10, 10)
    assert fix_dims(A, (20, 5)) == (20, 5)


def test_fallback_get_contours(mock_cnm):
    est = mock_cnm.estimates
    cnts = get_contours(est.A, tuple(est.dims), thr=0.8)
    assert len(cnts) == 6
    assert any(len(c["coordinates"]) > 0 for c in cnts)


def test_gui_instantiation_and_config():
    """Test that DecovoApp builds all UI panels without crashing."""
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in current environment")

    app = DecovoApp()
    try:
        # Check tabs exist
        assert app.notebook.tabs()
        assert len(app.notebook.tabs()) == 3

        # Check default config gather
        cfg = app._gather_config_from_ui()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.gSig == 3.0
        assert cfg.gSiz == 13
        assert cfg.rf == 40
        assert cfg.stride == 20
        assert cfg.merge_thr == 0.70
        assert cfg.min_SNR == 3.0
        assert cfg.spatial_downsample == 0.60
    finally:
        app.destroy()


def test_gui_seed_preview():
    """Test interactive seed preview on mock Session."""
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in current environment")

    app = DecovoApp()
    try:
        dims = (60, 80)
        cn = np.zeros(dims, dtype=np.float32)
        pnr = np.zeros(dims, dtype=np.float32)

        # Plant 3 distinct bright spots
        cn[15, 20] = 0.95
        pnr[15, 20] = 18.0
        cn[30, 40] = 0.90
        pnr[30, 40] = 14.0
        cn[45, 60] = 0.88
        pnr[45, 60] = 12.0

        app.session = SimpleNamespace(cn=cn, pnr=pnr)
        app.slider_corr.set(0.85)
        app.slider_pnr.set(10.0)

        app._update_seed_preview()
        assert len(app._seed_coords) == 3
        assert "Seeds: 3" in app.lbl_seed_count.cget("text")

        # Also test with cn_filter as created by actual pipeline Session
        app.session = SimpleNamespace(cn_filter=cn, pnr=pnr)
        app._update_seed_preview()
        assert len(app._seed_coords) == 3
        assert "Seeds: 3" in app.lbl_seed_count.cget("text")

        # Test interactive click on Correlation histogram updates min_corr
        event_corr = SimpleNamespace(xdata=0.92, ydata=10, inaxes=app.ax_corr_hist, button=1)
        app._on_seed_canvas_click(event_corr)
        assert abs(float(app.slider_corr.get()) - 0.92) < 1e-3

        # Test interactive click on PNR histogram updates min_pnr
        event_pnr = SimpleNamespace(xdata=15.0, ydata=5, inaxes=app.ax_pnr_hist, button=1)
        app._on_seed_canvas_click(event_pnr)
        assert abs(float(app.slider_pnr.get()) - 15.0) < 1e-3

        # Test non-hardcoded threshold entry submit (arbitrary unconstrained values)
        app.ent_corr.delete(0, tk.END)
        app.ent_corr.insert(0, "0.35")
        app._on_corr_entry_submit()
        assert abs(float(app.slider_corr.get()) - 0.35) < 1e-3
        assert float(app.slider_corr.cget("from")) <= 0.35

        app.ent_pnr.delete(0, tk.END)
        app.ent_pnr.insert(0, "48.5")
        app._on_pnr_entry_submit()
        assert abs(float(app.slider_pnr.get()) - 48.5) < 1e-3
        assert float(app.slider_pnr.cget("to")) >= 48.5
    finally:
        app.destroy()


def test_gui_curator_navigation(mock_cnm):
    """Test neuron curator navigation and threshold filtering."""
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in current environment")

    app = DecovoApp()
    try:
        app._load_from_cnm(mock_cnm, fov_background=mock_cnm.estimates.Cn)
        assert app.n_cells == 6
        assert app.current_idx == 0
        assert np.sum(app.accepted_mask) == 4

        # Next / Prev
        app.on_next_cell()
        assert app.current_idx == 1
        app.on_prev_cell()
        assert app.current_idx == 0

        # Toggle accept
        initial_status = app.accepted_mask[0]
        app.on_toggle_accepted()
        assert app.accepted_mask[0] != initial_status

        # Set specific status
        app.on_set_accepted(True)
        assert bool(app.accepted_mask[0]) is True

        # Test filter pass criteria
        app.spn_filter_peaks.delete(0, tk.END)
        app.spn_filter_peaks.insert(0, "2")
        app.spn_filter_snr.delete(0, tk.END)
        app.spn_filter_snr.insert(0, "3.5")
        app.spn_filter_rval.delete(0, tk.END)
        app.spn_filter_rval.insert(0, "0.80")
        app._on_filter_changed()
        # Neurons with peaks>=2, SNR>=3.5, rval>=0.80 should pass
        assert len(app.passing_indices) >= 3

        # Accept all passing
        app.on_accept_all_passing()
        for idx in app.passing_indices:
            assert bool(app.accepted_mask[idx]) is True
    finally:
        app.destroy()


def test_gui_theme_toggle():
    """Test switching between Day and Night themes."""
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in current environment")

    app = DecovoApp()
    try:
        assert app.is_dark_mode is True
        assert app.BG_MAIN == "#1e1e1e"

        # Toggle to Day mode
        app.toggle_theme()
        assert app.is_dark_mode is False
        assert app.BG_MAIN == "#f1f5f9"

        # Toggle back to Night mode
        app.toggle_theme()
        assert app.is_dark_mode is True
        assert app.BG_MAIN == "#1e1e1e"
    finally:
        app.destroy()


def test_gui_auto_load_existing_stage_a(tmp_path):
    """Test auto-loading existing Cn.npy and PNR.npy from results_decovo directory."""
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("Tkinter display not available in current environment")

    rec_file = tmp_path / "recording.tif"
    rec_file.write_bytes(b"dummy")
    res_dir = tmp_path / "results_decovo"
    res_dir.mkdir()

    cn_data = np.random.rand(40, 50).astype(np.float32)
    pnr_data = np.random.rand(40, 50).astype(np.float32) * 12.0
    np.save(res_dir / "Cn.npy", cn_data)
    np.save(res_dir / "PNR.npy", pnr_data)

    app = DecovoApp()
    try:
        app.set_input_path(rec_file)
        assert app.session is not None
        assert app.session_cn is not None
        assert app.session_pnr is not None
        assert np.array_equal(app.session_cn, cn_data)
        assert np.array_equal(app.session_pnr, pnr_data)
    finally:
        app.destroy()


