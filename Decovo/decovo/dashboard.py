# -*- coding: utf-8 -*-
"""Interactive Panel/Bokeh dashboard for the Decovo CaImAn CNMF-E pipeline."""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import PipelineConfig, load_config
from .env import auto_n_processes, detect_cuda
from .io import InputSpec, discover_inputs
from .params_auto import CancelledError, estimate_params_heuristic
from .pipeline import (
    PipelineResult,
    Session,
    discover_batch_sessions,
    extract,
    preprocess,
    run_batch,
)

logger = logging.getLogger("decovo.dashboard")

_PANEL_OK = False
_BOKEH_OK = False

try:
    import panel as pn

    pn.extension("bokeh", sizing_mode="stretch_width", notifications=True)
    _PANEL_OK = True
except Exception as e:
    logger.debug("Panel not available: %s", e)

try:
    from bokeh.models import ColumnDataSource, HoverTool, Span
    from bokeh.palettes import Inferno256, Viridis256
    from bokeh.plotting import figure as bk_figure

    _BOKEH_OK = True
except Exception as e:
    logger.debug("Bokeh not available: %s", e)


ACCENT_COLOR = "#3b82f6"
SUCCESS_COLOR = "#10b981"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#ef4444"

CSS = """
:root {
    --bg-main: #0f172a;
    --bg-card: rgba(30, 41, 59, 0.7);
    --bg-card-hover: rgba(51, 65, 85, 0.8);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-primary: #0ea5e9;
    --accent-hover: #38bdf8;
    --border-color: rgba(255, 255, 255, 0.1);
    --color-success: #10b981;
    --color-warning: #f59e0b;
    --color-error: #ef4444;
    --color-pending: #64748b;
    --gap-sm: 8px;
    --gap-md: 16px;
    --gap-lg: 24px;
    --radius-md: 12px;
    --blur-effect: blur(10px);
}
:root[data-theme="light"] {
    --bg-main: #f8fafc;
    --bg-card: rgba(255, 255, 255, 0.9);
    --bg-card-hover: rgba(241, 245, 249, 0.95);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --accent-primary: #0284c7;
    --accent-hover: #0369a1;
    --border-color: rgba(0, 0, 0, 0.1);
}
body {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.glass-card {
    background: var(--bg-card);
    backdrop-filter: var(--blur-effect);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--gap-md);
}
.dashboard-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-primary);
    margin: 0 0 var(--gap-sm) 0;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 600;
}
.stepper-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    width: 100%;
    padding: var(--gap-md) 0;
    position: relative;
    overflow-x: auto;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 1;
    width: 105px;
    text-align: center;
}
.step-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: var(--gap-sm);
}
.step-item.done .step-circle {
    background: var(--color-success);
    border-color: var(--color-success);
    color: white;
}
.step-item.running .step-circle {
    background: var(--color-warning);
    border-color: var(--color-warning);
    color: white;
}
.step-item.error .step-circle {
    background: var(--color-error);
    border-color: var(--color-error);
    color: white;
}
.log-viewer {
    background: var(--bg-main);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--gap-sm);
    height: 180px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85rem;
}
"""

if _PANEL_OK:
    if CSS not in pn.config.raw_css:
        pn.config.raw_css.append(CSS)


def _ui_execute(fn: Callable[[], Any]) -> None:
    """Thread-safe execution of UI widget updates on the Bokeh document event loop."""
    if _PANEL_OK:
        try:
            if getattr(pn.state, "curdoc", None) is not None and pn.state.curdoc.session_context is not None:
                pn.state.execute(fn)
                return
        except Exception:
            pass
    fn()


def _notify(kind: str, msg: str) -> None:
    def _do():
        if _PANEL_OK and getattr(pn.state, "notifications", None) is not None:
            try:
                getattr(pn.state.notifications, kind)(msg)
            except Exception:
                pass

    _ui_execute(_do)


def _has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class DashboardState:
    """Shared state for all dashboard components."""

    def __init__(self):
        self.config: Optional[PipelineConfig] = None
        self.session: Optional[Session] = None
        self.result: Optional[PipelineResult] = None
        self.spec: Optional[InputSpec] = None
        self.cn_filter: Optional[np.ndarray] = None
        self.pnr: Optional[np.ndarray] = None
        self.dims: Optional[tuple] = None
        self.grid_results: Optional[dict] = None
        self.pipeline_running: bool = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.cancel_event: threading.Event = threading.Event()
        self.stages: Dict[str, str] = {}
        self.stage_times: Dict[str, float] = {}
        self.current_stage: str = ""
        self.log_messages: List[str] = []
        self.theme: str = "dark"


def create_config_panel(state: DashboardState):
    """Build the configuration sidebar panel (single definition, no duplicate widgets)."""
    defaults = load_config()

    path_input = pn.widgets.TextInput(
        name="📂 Data Directory or TIFF/AVI File",
        value=r"D:/Master_Gabriel/Progaming/Calciumdata/Jess/vid",
        placeholder="Path to folder with msCam*.avi or *.tif stack",
        sizing_mode="stretch_width",
    )
    browse_btn = pn.widgets.Button(
        name="Browse",
        width=75,
        button_type="primary",
        align="end",
        disabled=not _has_display(),
    )
    input_info_banner = pn.pane.HTML(
        "<small style='color:var(--text-secondary);'>Enter a path to inspect format & metadata FPS.</small>",
        sizing_mode="stretch_width",
    )

    frame_rate = pn.widgets.FloatInput(
        name="Frame Rate (Hz) [Verify]",
        value=defaults.frame_rate,
        step=1.0,
        start=0.5,
        end=200.0,
        width=155,
    )
    decay_time = pn.widgets.FloatInput(
        name="Decay Time (s)",
        value=defaults.decay_time,
        step=0.05,
        start=0.05,
        end=5.0,
        width=140,
    )

    def _inspect_path(*_args):
        p = path_input.value.strip()
        if not p or not os.path.exists(p):
            input_info_banner.object = (
                "<small style='color:var(--color-warning);'>⚠️ Path not found yet.</small>"
            )
            return
        try:
            spec = discover_inputs(p)
            state.spec = spec
            fps_note = (
                f"Metadata FPS: <b>{spec.fps_from_metadata:.2f} Hz</b>"
                if spec.fps_from_metadata
                else "Metadata FPS: <i>not in header (set manually)</i>"
            )
            if spec.fps_from_metadata and spec.fps_from_metadata > 0:
                frame_rate.value = float(spec.fps_from_metadata)
            input_info_banner.object = (
                f"<div style='padding:6px 8px; background:var(--bg-card-hover); border-radius:6px; font-size:0.82em;'>"
                f"✅ <b>{spec.kind.upper()}</b> | {spec.n_frames:,} frames | "
                f"{spec.frame_shape[0]}×{spec.frame_shape[1]} px ({spec.dtype})<br>{fps_note}</div>"
            )
        except Exception as e:
            input_info_banner.object = (
                f"<div style='color:var(--color-error); font-size:0.82em;'>⚠️ {e}</div>"
            )

    path_input.param.watch(_inspect_path, "value")

    def _browse_folder(_event):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.attributes("-topmost", True)
            root.withdraw()
            folder = filedialog.askdirectory(initialdir=path_input.value, title="Select Session Directory")
            root.destroy()
            if folder:
                path_input.value = folder
        except Exception as e:
            _notify("warning", f"Folder dialog unavailable: {e}")

    browse_btn.on_click(_browse_folder)
    path_row = pn.Row(path_input, browse_btn, sizing_mode="stretch_width")

    results_input = pn.widgets.TextInput(
        name="Results Folder", value=defaults.results_folder, sizing_mode="stretch_width"
    )
    force_checkbox = pn.widgets.Checkbox(name="Start Fresh (ignore cached stages)", value=False)

    # Preprocessing
    spatial_ds = pn.widgets.FloatSlider(
        name="Spatial Downsample",
        start=0.2,
        end=1.0,
        step=0.05,
        value=defaults.spatial_downsample,
        width=300,
    )
    temporal_ds = pn.widgets.FloatSlider(
        name="Temporal Downsample",
        start=0.25,
        end=1.0,
        step=0.25,
        value=defaults.temporal_downsample,
        width=300,
    )
    remove_bg = pn.widgets.Checkbox(name="Remove Background", value=defaults.remove_background)
    motion_correct = pn.widgets.Checkbox(name="Motion Correction", value=defaults.motion_correct)

    # Source Extraction (validated defaults: gSig=3, gSiz=13, rf=40, stride=20, merge=0.7, SNR=3.0, rval=0.95)
    gsig = pn.widgets.FloatSlider(
        name="gSig (neuron width)",
        start=1.0,
        end=10.0,
        step=0.5,
        value=defaults.gSig,
        width=300,
    )
    min_snr = pn.widgets.FloatSlider(
        name="Min SNR",
        start=1.0,
        end=15.0,
        step=0.5,
        value=defaults.min_SNR,
        width=300,
    )
    r_values = pn.widgets.FloatSlider(
        name="Min Spatial Consistency (rval)",
        start=0.5,
        end=1.0,
        step=0.01,
        value=defaults.rval_thr,
        width=300,
    )

    # Parameter Selection (single definition!)
    auto_params = pn.widgets.Checkbox(name="Auto-Select Parameters (Heuristic / Grid)", value=True)
    auto_method = pn.widgets.Select(
        name="Auto Method",
        options=["heuristic", "grid_search", "both"],
        value="heuristic",
        width=200,
    )
    manual_corr = pn.widgets.FloatInput(
        name="Manual min_corr",
        value=0.80,
        step=0.01,
        start=0.1,
        end=0.99,
        width=140,
        disabled=True,
    )
    manual_pnr = pn.widgets.FloatInput(
        name="Manual min_pnr",
        value=8.0,
        step=0.5,
        start=1.0,
        end=30.0,
        width=140,
        disabled=True,
    )

    def _toggle_manual(event):
        manual_corr.disabled = bool(event.new)
        manual_pnr.disabled = bool(event.new)

    auto_params.param.watch(_toggle_manual, "value")

    # Performance
    n_procs = pn.widgets.IntInput(
        name="Processes",
        value=auto_n_processes(),
        start=1,
        end=os.cpu_count() or 32,
        width=100,
    )
    use_cuda = pn.widgets.Checkbox(name="CUDA Acceleration (Motion)", value=False)
    max_files = pn.widgets.IntInput(name="Max Files (0=all)", value=0, start=0, end=200, width=110)

    # Batch
    batch_mode = pn.widgets.Checkbox(name="Batch Mode", value=False)
    batch_pattern = pn.widgets.TextInput(
        name="Batch Pattern",
        value="Animal*/pre,Animal*/post/d*",
        width=300,
        disabled=True,
    )
    batch_mode.param.watch(lambda e: setattr(batch_pattern, "disabled", not e.new), "value")

    # Export
    save_hdf5 = pn.widgets.Checkbox(name="Save HDF5 (Curator)", value=True)
    save_mat = pn.widgets.Checkbox(name="Save MAT (ms.mat)", value=True)
    save_figs = pn.widgets.Checkbox(name="Save Figures", value=True)

    def build_config() -> PipelineConfig:
        cfg = load_config(
            path_to_analyze=path_input.value.strip(),
            results_folder=results_input.value.strip() or "results_decovo",
            spatial_downsample=spatial_ds.value,
            temporal_downsample=temporal_ds.value,
            frame_rate=frame_rate.value,
            decay_time=decay_time.value,
            remove_background=remove_bg.value,
            motion_correct=motion_correct.value,
            gSig=gsig.value,
            min_SNR=min_snr.value,
            rval_thr=r_values.value,
            auto_params=auto_params.value,
            auto_method=auto_method.value,
            min_corr=None if auto_params.value else manual_corr.value,
            min_pnr=None if auto_params.value else manual_pnr.value,
            n_processes=n_procs.value,
            use_cuda=use_cuda.value,
            max_files=max_files.value if max_files.value > 0 else None,
            save_hdf5=save_hdf5.value,
            save_mat=save_mat.value,
            save_figures=save_figs.value,
            batch_mode=batch_mode.value,
            batch_pattern=batch_pattern.value,
        )
        state.config = cfg
        return cfg

    _inspect_path()

    sidebar = pn.Column(
        pn.pane.HTML('<p class="dashboard-title">⚙️ Configuration</p>'),
        pn.Card(
            path_row,
            input_info_banner,
            results_input,
            force_checkbox,
            title="📁 Input / Output",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            pn.Row(frame_rate, decay_time),
            spatial_ds,
            temporal_ds,
            remove_bg,
            motion_correct,
            title="🔬 Preprocessing",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            gsig,
            min_snr,
            r_values,
            title="🧠 Source Extraction",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            auto_params,
            auto_method,
            pn.Row(manual_corr, manual_pnr),
            title="🎯 Parameter Selection",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            pn.Row(n_procs, max_files),
            use_cuda,
            title="⚡ Performance",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            batch_mode,
            batch_pattern,
            title="📦 Batch",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        pn.Card(
            pn.Row(save_hdf5, save_mat, save_figs),
            title="💾 Export",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["glass-card"],
        ),
        scroll=True,
        width=380,
        margin=(5, 10),
        css_classes=["glass-card"],
    )

    return sidebar, build_config, force_checkbox


def create_corr_pnr_inspector(state: DashboardState, on_apply_thresholds: Callable[[float, float], None]):
    """Interactive correlation/PNR threshold selector; Apply triggers Stage B re-extraction."""
    corr_slider = pn.widgets.FloatSlider(
        name="min_corr threshold",
        start=0.1,
        end=0.99,
        step=0.01,
        value=0.8,
        format="0.00",
        width=350,
    )
    pnr_slider = pn.widgets.FloatSlider(
        name="min_pnr threshold",
        start=1.0,
        end=25.0,
        step=0.25,
        value=8.0,
        format="0.00",
        width=350,
    )
    neuron_count = pn.pane.HTML(
        '<span class="metric-value">—</span><br><span class="metric-label">estimated neurons</span>',
        width=220,
    )

    corr_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=300)
    pnr_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=300)
    scatter_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=340)

    empty_state_html = """
    <div style="text-align:center; padding: 30px 20px; color: var(--text-secondary);">
        <div style="font-size: 2.5em; margin-bottom: 10px;">🎯</div>
        <h3>Threshold Inspector</h3>
        <p>Run Stage A (Preprocess) to inspect Correlation (Cn) & PNR maps and extract neurons.</p>
    </div>
    """
    plot_container = pn.Column(pn.pane.HTML(empty_state_html), sizing_mode="stretch_width")

    def update_plots(_event=None):
        if state.cn_filter is None or state.pnr is None:
            return
        cn = state.cn_filter
        pn_img = state.pnr
        mc = float(corr_slider.value)
        mp = float(pnr_slider.value)

        mask = (cn >= mc) & (pn_img >= mp)
        n_pixels = int(np.sum(mask))
        gsig = state.config.gSig if state.config else 3.0
        neuron_area = max(1.0, np.pi * gsig * gsig)
        est_neurons = max(int(round(n_pixels / neuron_area)), 0)

        color = "var(--color-success)" if 5 <= est_neurons <= 400 else "var(--color-warning)"
        neuron_count.object = (
            f'<span class="metric-value" style="color:{color}">{est_neurons}</span><br>'
            f'<span class="metric-label">estimated neurons ({n_pixels} seed px)</span>'
        )

        if not _BOKEH_OK:
            return

        try:
            p1 = bk_figure(
                title="Correlation Image (Cn)",
                width=380,
                height=280,
                tools="pan,wheel_zoom,box_zoom,reset",
                match_aspect=True,
            )
            p1.image(image=[cn[::-1, :]], x=0, y=0, dw=cn.shape[1], dh=cn.shape[0], palette=Viridis256)
            p1.axis.visible = False
            corr_plot.object = p1

            p2 = bk_figure(
                title="Peak-to-Noise Ratio (PNR)",
                width=380,
                height=280,
                tools="pan,wheel_zoom,box_zoom,reset",
                match_aspect=True,
            )
            p2.image(image=[pn_img[::-1, :]], x=0, y=0, dw=pn_img.shape[1], dh=pn_img.shape[0], palette=Inferno256)
            p2.axis.visible = False
            pnr_plot.object = p2

            flat_cn = cn.ravel()
            flat_pnr = pn_img.ravel()
            if len(flat_cn) > 10000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(flat_cn), 10000, replace=False)
                flat_cn = flat_cn[idx]
                flat_pnr = flat_pnr[idx]

            source = ColumnDataSource(data={"corr": flat_cn, "pnr": flat_pnr})
            p3 = bk_figure(
                title="Correlation vs PNR",
                width=450,
                height=320,
                tools="pan,wheel_zoom,box_zoom,reset",
                x_axis_label="Correlation",
                y_axis_label="PNR",
            )
            p3.scatter("corr", "pnr", source=source, size=2, alpha=0.3, color="#666")
            p3.add_layout(Span(location=mc, dimension="height", line_color=DANGER_COLOR, line_width=2, line_dash="dashed"))
            p3.add_layout(Span(location=mp, dimension="width", line_color=ACCENT_COLOR, line_width=2, line_dash="dashed"))

            above = (flat_cn >= mc) & (flat_pnr >= mp)
            if np.any(above):
                src_above = ColumnDataSource(data={"corr": flat_cn[above], "pnr": flat_pnr[above]})
                p3.scatter("corr", "pnr", source=src_above, size=3, alpha=0.6, color=SUCCESS_COLOR)

            scatter_plot.object = p3
            plot_container.objects = [pn.Row(corr_plot, pnr_plot), scatter_plot]
        except Exception as e:
            logger.warning("Inspector plot update failed: %s", e)

    corr_slider.param.watch(update_plots, "value_throttled")
    pnr_slider.param.watch(update_plots, "value_throttled")

    apply_btn = pn.widgets.Button(
        name="✅ Apply & Extract Neurons (Stage B only)",
        button_type="success",
        width=280,
    )
    auto_btn = pn.widgets.Button(
        name="🔍 Heuristic Estimate",
        button_type="primary",
        width=180,
    )

    def _on_apply(_event):
        on_apply_thresholds(float(corr_slider.value), float(pnr_slider.value))

    def _on_auto(_event):
        if state.cn_filter is not None and state.pnr is not None:
            h_corr, h_pnr = estimate_params_heuristic(state.cn_filter, state.pnr)
            corr_slider.value = h_corr
            pnr_slider.value = h_pnr
            update_plots()
            _notify("info", f"Heuristic estimate: min_corr={h_corr:.3f}, min_pnr={h_pnr:.2f}")

    apply_btn.on_click(_on_apply)
    auto_btn.on_click(_on_auto)

    panel = pn.Column(
        pn.pane.HTML('<p class="dashboard-title">🎯 Threshold Inspector</p>'),
        pn.Row(corr_slider, pnr_slider, neuron_count),
        plot_container,
        pn.Row(auto_btn, apply_btn),
        margin=10,
        css_classes=["glass-card"],
    )

    return panel, update_plots, corr_slider, pnr_slider


def create_neuron_browser(state: DashboardState):
    """Interactive neuron browser: spatial footprint + C & S traces."""
    neuron_selector = pn.widgets.IntSlider(
        name="Neuron Index", start=0, end=1, step=1, value=0, width=400
    )
    status_text = pn.pane.HTML('<span class="metric-label">Run pipeline first to see neurons</span>')
    spatial_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=340)
    trace_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=230)
    deconv_plot = pn.pane.Bokeh(sizing_mode="stretch_width", height=190)
    quality_info = pn.pane.HTML("")

    empty_state_html = """
    <div style="text-align:center; padding: 30px 20px; color: var(--text-secondary);">
        <div style="font-size: 2.5em; margin-bottom: 10px;">🔬</div>
        <h3>Neuron Browser</h3>
        <p>No neurons extracted yet — run extraction to browse cells.</p>
    </div>
    """
    plot_container = pn.Column(pn.pane.HTML(empty_state_html), sizing_mode="stretch_width")

    def update_neuron_view(_event=None):
        if state.result is None or state.result.cnm is None:
            return
        cnm = state.result.cnm
        idx_good = cnm.estimates.idx_components
        if idx_good is None or len(idx_good) == 0:
            status_text.object = "<b>0 accepted neurons</b>"
            return

        neuron_selector.end = max(1, len(idx_good) - 1)
        sel = min(int(neuron_selector.value), len(idx_good) - 1)
        neuron_id = int(idx_good[sel])

        status_text.object = f"<b>Neuron {sel + 1} / {len(idx_good)}</b> (component ID: {neuron_id})"

        if not _BOKEH_OK:
            return

        try:
            cn = state.cn_filter
            A = cnm.estimates.A
            dims = state.dims
            sfp = np.reshape(A[:, neuron_id].toarray(), dims, order="F")

            p1 = bk_figure(
                title=f"Spatial Footprint — Neuron {sel + 1}",
                width=420,
                height=320,
                tools="pan,wheel_zoom,reset",
                match_aspect=True,
            )
            if cn is not None:
                p1.image(image=[cn[::-1, :]], x=0, y=0, dw=cn.shape[1], dh=cn.shape[0], palette=Viridis256, alpha=0.75)
            sfp_disp = sfp.copy()
            sfp_disp[sfp_disp < sfp.max() * 0.2] = np.nan
            p1.image(image=[sfp_disp[::-1, :]], x=0, y=0, dw=sfp.shape[1], dh=sfp.shape[0], palette=Inferno256, alpha=0.65)
            p1.axis.visible = False
            spatial_plot.object = p1

            raw = cnm.estimates.C[neuron_id, :]
            fr = state.config.frate if state.config else 20.0
            t = np.arange(len(raw)) / max(fr, 1e-3)

            p2 = bk_figure(
                title=f"Raw Trace (C) — Neuron {sel + 1}",
                width=580,
                height=210,
                tools="pan,wheel_zoom,box_zoom,reset",
                x_axis_label="Time (s)",
                y_axis_label="Fluorescence",
            )
            p2.line(t, raw, line_width=0.9, color=ACCENT_COLOR)
            trace_plot.object = p2

            deconv = cnm.estimates.S[neuron_id, :]
            p3 = bk_figure(
                title=f"Deconvolved Trace (S) — Neuron {sel + 1}",
                width=580,
                height=175,
                tools="pan,wheel_zoom,box_zoom,reset",
                x_axis_label="Time (s)",
                y_axis_label="Spikes",
            )
            p3.line(t, deconv, line_width=0.9, color=SUCCESS_COLOR)
            deconv_plot.object = p3

            snr_val = cnm.estimates.SNR_comp[neuron_id] if hasattr(cnm.estimates, "SNR_comp") and cnm.estimates.SNR_comp is not None else "N/A"
            r_val = cnm.estimates.r_values[neuron_id] if hasattr(cnm.estimates, "r_values") and cnm.estimates.r_values is not None else "N/A"
            snr_s = f"{snr_val:.2f}" if isinstance(snr_val, (int, float, np.floating)) else str(snr_val)
            r_s = f"{r_val:.3f}" if isinstance(r_val, (int, float, np.floating)) else str(r_val)

            quality_info.object = (
                f"<div style='padding:8px; background:var(--bg-card-hover); border-radius:6px; display:flex; gap:15px;'>"
                f"<span><b>SNR:</b> {snr_s}</span>"
                f"<span><b>r-value:</b> {r_s}</span>"
                f"<span><b>Peak:</b> {raw.max():.2f}</span>"
                f"</div>"
            )
            plot_container.objects = [pn.Row(spatial_plot, pn.Column(trace_plot, deconv_plot))]
        except Exception as e:
            logger.warning("Neuron browser update error: %s", e)

    neuron_selector.param.watch(update_neuron_view, "value")
    prev_btn = pn.widgets.Button(name="◀ Prev", width=80)
    next_btn = pn.widgets.Button(name="Next ▶", width=80)
    prev_btn.on_click(lambda _e: setattr(neuron_selector, "value", max(0, neuron_selector.value - 1)))
    next_btn.on_click(lambda _e: setattr(neuron_selector, "value", min(neuron_selector.end, neuron_selector.value + 1)))

    panel = pn.Column(
        pn.pane.HTML('<p class="dashboard-title">🔬 Neuron Browser</p>'),
        pn.Row(prev_btn, neuron_selector, next_btn),
        status_text,
        quality_info,
        plot_container,
        margin=10,
        css_classes=["glass-card"],
    )
    return panel, update_neuron_view


def create_grid_heatmap(state: DashboardState):
    """Visualize grid search results (`Result.grid`) as a heatmap."""
    plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width", height=350)
    info_pane = pn.pane.HTML('<span class="metric-label">Run grid search to see results</span>')
    empty_state_html = """
    <div style="text-align:center; padding: 30px 20px; color: var(--text-secondary);">
        <div style="font-size: 2.5em; margin-bottom: 10px;">📊</div>
        <h3>Grid Search Results</h3>
        <p>Select Auto Method = <code>grid_search</code> or <code>both</code> to populate this tab.</p>
    </div>
    """
    plot_container = pn.Column(pn.pane.HTML(empty_state_html), sizing_mode="stretch_width")

    def update_heatmap():
        if not state.grid_results or not _BOKEH_OK:
            return
        results = state.grid_results
        corr_vals = sorted(set(k[0] for k in results.keys()))
        pnr_vals = sorted(set(k[1] for k in results.keys()))

        n_accepted, scores, corrs, pnrs = [], [], [], []
        for mc in corr_vals:
            for mp in pnr_vals:
                r = results.get((mc, mp), {})
                corrs.append(mc)
                pnrs.append(mp)
                n_accepted.append(r.get("n_accepted", 0))
                scores.append(r.get("score", -999.0))

        max_n = max(n_accepted) if n_accepted else 1
        sizes = [max(6, 30 * n / max(max_n, 1)) for n in n_accepted]
        source = ColumnDataSource(
            data={"corr": corrs, "pnr": pnrs, "n_accepted": n_accepted, "score": scores, "size": sizes}
        )

        from bokeh.transform import linear_cmap

        low_s = min(scores) if scores else 0.0
        high_s = max(scores) if scores and max(scores) > low_s else low_s + 1.0
        mapper = linear_cmap("score", palette=Viridis256, low=low_s, high=high_s)

        p = bk_figure(
            title="Grid Search: Accepted Neurons per (min_corr, min_pnr)",
            width=500,
            height=330,
            x_axis_label="min_corr",
            y_axis_label="min_pnr",
            tools="hover,reset",
        )
        p.scatter("corr", "pnr", source=source, size="size", color=mapper, alpha=0.85, line_color="white")
        hover = p.select_one(HoverTool)
        if hover:
            hover.tooltips = [
                ("corr", "@corr{0.000}"),
                ("pnr", "@pnr{0.00}"),
                ("neurons", "@n_accepted"),
                ("score", "@score{0.00}"),
            ]
        plot_pane.object = p

        best_key = max(results.keys(), key=lambda k: results[k].get("score", -999.0))
        best = results[best_key]
        info_pane.object = (
            f"<div style='padding:10px; background:var(--bg-card-hover); border-left:4px solid var(--color-success);'>"
            f"<b>Best Parameters:</b> min_corr={best_key[0]}, min_pnr={best_key[1]} → "
            f"{best['n_accepted']} neurons (score={best['score']:.2f})</div>"
        )
        plot_container.objects = [plot_pane, info_pane]

    panel = pn.Column(
        pn.pane.HTML('<p class="dashboard-title">📊 Grid Search Results</p>'),
        plot_container,
        margin=10,
        css_classes=["glass-card"],
    )
    return panel, update_heatmap


def create_pipeline_stepper(state: DashboardState):
    stages_config = [
        ("Preprocessing (Downsample & Background)", "Preprocess", "📉"),
        ("Motion Correction", "Motion / Mmap", "🎬"),
        ("Summary Images", "Cn & PNR", "🖼️"),
        ("Grid Search Parameter Optimization", "Grid Search", "🔍"),
        ("CNMF-E Source Extraction", "CNMF-E", "🧠"),
        ("Component Evaluation", "Evaluation", "⚖️"),
        ("Exporting Results", "Export", "📤"),
    ]
    stepper_pane = pn.pane.HTML("")

    def render_stepper():
        html = '<div class="stepper-container">'
        for key_name, short_label, icon in stages_config:
            match_name = key_name
            if key_name == "Motion Correction" and "Memory Mapping (no motion correction)" in state.stages:
                match_name = "Memory Mapping (no motion correction)"
            status = state.stages.get(match_name, "pending")
            t_spent = state.stage_times.get(match_name, 0.0)
            time_html = f'<div class="step-time">{t_spent:.1f}s</div>' if status == "done" else ""
            html += (
                f'<div class="step-item {status}">'
                f'<div class="step-circle">{icon}</div>'
                f'<div class="step-label">{short_label}</div>'
                f"{time_html}</div>"
            )
        html += "</div>"
        stepper_pane.object = html

    return stepper_pane, render_stepper


def create_log_viewer(state: DashboardState):
    log_pane = pn.pane.HTML(
        '<div class="log-viewer" id="pipeline-log-viewer">Waiting for pipeline to start...</div>',
        sizing_mode="stretch_width",
    )

    def render_logs():
        if not state.log_messages:
            return
        html = '<div class="log-viewer" id="pipeline-log-viewer">'
        for msg in state.log_messages[-100:]:
            html += f'<div class="log-entry">{msg}</div>'
        html += "</div>"
        log_pane.object = html

    def add_log(msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        state.log_messages.append(f"[{ts}] {msg}")
        _ui_execute(render_logs)

    return log_pane, add_log


def create_pipeline_control(
    state: DashboardState,
    build_config_fn: Callable[[], PipelineConfig],
    force_checkbox,
    update_inspector_fn: Callable[[], None],
    update_neuron_fn: Callable[[], None],
    update_grid_fn: Callable[[], None],
    corr_slider,
    pnr_slider,
):
    run_btn = pn.widgets.Button(name="🚀 Run Pipeline", button_type="success", width=180, height=45)
    prep_only_btn = pn.widgets.Button(name="⚡ Stage A Only (Preprocess)", button_type="primary", width=210, height=45)
    stop_btn = pn.widgets.Button(name="⏹ Stop", button_type="danger", width=100, height=45, disabled=True)
    progress_bar = pn.widgets.Progress(name="Progress", active=False, value=0, max=100, width=320)

    stepper_pane, update_stepper = create_pipeline_stepper(state)
    update_stepper()
    log_pane, add_log = create_log_viewer(state)
    timing_display = pn.pane.HTML("")

    def progress_callback(event_type: str, stage: str, value: float, total: int):
        def _upd():
            if event_type == "stage_start":
                state.stages[stage] = "running"
                state.current_stage = stage
                state.stage_times[stage] = 0.0
                update_stepper()
                progress_bar.active = True
            elif event_type == "stage_end":
                state.stages[stage] = "done"
                state.stage_times[stage] = float(value)
                update_stepper()
            elif event_type == "progress" and total > 0:
                progress_bar.value = min(100, int(100 * value / total))

        _ui_execute(_upd)
        if event_type == "stage_start":
            add_log(f"▶ Starting: {stage}")
        elif event_type == "stage_end":
            add_log(f"✓ Finished: {stage} ({value:.1f}s)")

    def _finish_with_result(result: PipelineResult):
        state.result = result
        state.cn_filter = result.cn_filter
        state.pnr = result.pnr
        state.dims = result.dims
        if result.grid is not None:
            state.grid_results = result.grid

        def _sync_ui():
            if result.cn_filter is not None:
                pnr_max = float(np.nanmax(result.pnr)) if result.pnr is not None else 25.0
                pnr_slider.end = max(25.0, float(np.ceil(max(pnr_max, float(result.min_pnr_used)))))
                corr_slider.value = float(np.clip(result.min_corr_used, corr_slider.start, corr_slider.end))
                pnr_slider.value = float(np.clip(result.min_pnr_used, pnr_slider.start, pnr_slider.end))
                update_inspector_fn()
            update_neuron_fn()
            if result.grid is not None:
                update_grid_fn()
            timing_display.object = (
                f"<div style='padding:10px; background:var(--color-success); border-radius:8px; color:white;'>"
                f"<b>✅ Complete: {result.n_neurons} accepted neurons</b> "
                f"(min_corr={result.min_corr_used:.3f}, min_pnr={result.min_pnr_used:.2f})</div>"
            )
            progress_bar.active = False
            progress_bar.value = 100

        _ui_execute(_sync_ui)
        _notify("success", f"Extraction complete: {result.n_neurons} neurons found!")

    def _stage_a_needs_refresh(config: PipelineConfig, force: bool, manual_thresholds: Optional[tuple]) -> bool:
        if state.session is None or force or manual_thresholds is None:
            return True
        if state.session.input_dir != str(os.path.abspath(config.path_to_analyze)):
            return True
        for stg in ("prepare", "motion", "summary"):
            if state.session.config.stage_params(stg) != config.stage_params(stg):
                return True
        return False

    def _worker_run(config: PipelineConfig, force: bool, extract_after_prep: bool, manual_thresholds: Optional[tuple] = None):
        try:
            state.pipeline_running = True
            state.cancel_event.clear()
            _ui_execute(lambda: (setattr(run_btn, "disabled", True), setattr(prep_only_btn, "disabled", True), setattr(stop_btn, "disabled", False)))

            if _stage_a_needs_refresh(config, force, manual_thresholds):
                add_log("🚀 Running Stage A (Preprocess)...")
                session = preprocess(
                    input_dir=config.path_to_analyze,
                    config=config,
                    force=force,
                    progress=progress_callback,
                    cancel_event=state.cancel_event,
                )
                state.session = session
                state.cn_filter = session.cn_filter
                state.pnr = session.pnr
                state.dims = session.dims

                h_c, h_p = estimate_params_heuristic(session.cn_filter, session.pnr)

                def _upd_stage_a():
                    pnr_max = float(np.nanmax(session.pnr)) if session.pnr is not None else 25.0
                    pnr_Val = float(config.min_pnr if config.min_pnr is not None else h_p)
                    pnr_slider.end = max(25.0, float(np.ceil(max(pnr_max, pnr_Val))))
                    corr_Val = float(config.min_corr if config.min_corr is not None else h_c)
                    corr_slider.value = float(np.clip(corr_Val, corr_slider.start, corr_slider.end))
                    pnr_slider.value = float(np.clip(pnr_Val, pnr_slider.start, pnr_slider.end))
                    update_inspector_fn()

                _ui_execute(_upd_stage_a)
                add_log(f"✅ Stage A ready (dims={session.dims}, skipped={session.skipped_stages}).")

            if extract_after_prep and state.session is not None:
                mc_override = manual_thresholds[0] if manual_thresholds else None
                mp_override = manual_thresholds[1] if manual_thresholds else None
                add_log("🧠 Running Stage B (CNMF-E Extraction) on cached memmap...")
                state.session.config = config
                res = extract(
                    state.session,
                    min_corr=mc_override,
                    min_pnr=mp_override,
                    force=force,
                    progress=progress_callback,
                    cancel_event=state.cancel_event,
                )
                _finish_with_result(res)
            else:
                _ui_execute(lambda: (setattr(progress_bar, "active", False), setattr(progress_bar, "value", 100)))
                _notify("info", "Stage A complete! Adjust thresholds and click Apply & Extract.")

        except CancelledError:
            add_log("⏹ Pipeline stopped by user.")
            _notify("warning", "Pipeline stopped by user.")
        except Exception as e:
            logger.error("Dashboard pipeline error: %s", e, exc_info=True)
            add_log(f"❌ Error: {e}")
            _notify("error", f"Pipeline failed: {str(e)[:100]}")
        finally:
            state.pipeline_running = False
            _ui_execute(lambda: (setattr(run_btn, "disabled", False), setattr(prep_only_btn, "disabled", False), setattr(stop_btn, "disabled", True)))

    def _start_thread(extract_after_prep: bool, manual_thresholds: Optional[tuple] = None):
        if state.pipeline_running:
            _notify("warning", "Pipeline is already running.")
            return
        cfg = build_config_fn()
        force = bool(force_checkbox.value)
        t = threading.Thread(
            target=_worker_run,
            args=(cfg, force, extract_after_prep, manual_thresholds),
            daemon=True,
        )
        state.pipeline_thread = t
        t.start()

    def on_apply_thresholds(mc: float, mp: float):
        """Called by 'Apply & Extract Neurons' button — re-runs ONLY Stage B if Stage A is already done."""
        _start_thread(extract_after_prep=True, manual_thresholds=(mc, mp))

    run_btn.on_click(lambda _e: _start_thread(extract_after_prep=True))
    prep_only_btn.on_click(lambda _e: _start_thread(extract_after_prep=False))
    stop_btn.on_click(lambda _e: state.cancel_event.set())

    panel = pn.Column(
        pn.pane.HTML('<p class="dashboard-title">▶️ Pipeline Control</p>'),
        pn.Row(run_btn, prep_only_btn, stop_btn, progress_bar, align="center"),
        stepper_pane,
        timing_display,
        pn.layout.Divider(),
        log_pane,
        margin=10,
        css_classes=["glass-card"],
    )
    return panel, on_apply_thresholds


def create_batch_panel(state: DashboardState, build_config_fn: Callable[[], PipelineConfig]):
    scan_btn = pn.widgets.Button(name="🔍 Scan for Sessions", button_type="primary", width=200)
    run_batch_btn = pn.widgets.Button(name="🚀 Run All Sessions", button_type="success", width=200)
    session_list = pn.pane.HTML("<p>Click Scan to discover sessions (AVI or TIFF).</p>")
    batch_progress = pn.pane.HTML("")

    def scan_sessions(_event):
        cfg = build_config_fn()
        try:
            sessions = discover_batch_sessions(cfg)
            html = f"<b>Found {len(sessions)} session(s):</b><ul>"
            for s in sessions:
                html += f"<li><code>{s}</code></li>"
            html += "</ul>"
            session_list.object = html
        except Exception as e:
            session_list.object = f"<div style='color:var(--color-error);'>Error: {e}</div>"

    def run_all(_event):
        cfg = build_config_fn()
        cfg.batch_mode = True

        def _batch_worker():
            try:
                results = run_batch(cfg)
                html = "<b>Batch Results:</b><ul>"
                for i, r in enumerate(results):
                    html += f"<li>[{i + 1}] {r.n_neurons} neurons</li>"
                html += "</ul>"
                _ui_execute(lambda: setattr(batch_progress, "object", html))
            except Exception as e:
                _ui_execute(lambda: setattr(batch_progress, "object", f"Error: {e}"))

        threading.Thread(target=_batch_worker, daemon=True).start()

    scan_btn.on_click(scan_sessions)
    run_batch_btn.on_click(run_all)

    return pn.Column(
        pn.pane.HTML('<p class="dashboard-title">📦 Batch Processing</p>'),
        pn.Row(scan_btn, run_batch_btn),
        pn.layout.Divider(),
        session_list,
        batch_progress,
        margin=10,
        css_classes=["glass-card"],
    )


def create_dashboard():
    """Create and return the complete Panel interactive dashboard."""
    if not _PANEL_OK:
        raise ImportError("Panel is required for the interactive dashboard. Install with: pip install panel")

    state = DashboardState()
    sidebar, build_config, force_checkbox = create_config_panel(state)

    # Forward reference for apply callback
    apply_holder: Dict[str, Optional[Callable[[float, float], None]]] = {"fn": None}

    def _delegate_apply(mc: float, mp: float):
        if apply_holder["fn"] is not None:
            apply_holder["fn"](mc, mp)

    inspector, update_inspector, corr_slider, pnr_slider = create_corr_pnr_inspector(state, _delegate_apply)
    neuron_browser, update_neurons = create_neuron_browser(state)
    grid_panel, update_grid = create_grid_heatmap(state)
    pipeline_control, on_apply_thresholds = create_pipeline_control(
        state,
        build_config,
        force_checkbox,
        update_inspector,
        update_neurons,
        update_grid,
        corr_slider,
        pnr_slider,
    )
    apply_holder["fn"] = on_apply_thresholds
    batch_panel = create_batch_panel(state, build_config)

    tabs = pn.Tabs(
        ("🎯 Thresholds", pn.Column(pipeline_control, inspector)),
        ("🔬 Neurons", neuron_browser),
        ("📊 Grid Search", grid_panel),
        ("📦 Batch", batch_panel),
        dynamic=True,
    )

    header = pn.Row(
        pn.pane.HTML(
            "<div style='display:flex;align-items:center;gap:12px;'>"
            "<span style='font-size:2em;'>🧠</span>"
            "<div><h2 style='margin:0;color:var(--accent-primary);'>Decovo v2 — CaImAn CNMF-E Pipeline</h2>"
            "<span style='color:var(--text-secondary);'>1-Photon Miniscope (AVI & Multi-Page TIFF) — Resumable Two-Stage Pipeline</span>"
            "</div></div>"
        ),
        sizing_mode="stretch_width",
        margin=(10, 20),
    )

    return pn.template.BootstrapTemplate(
        title="Decovo v2 — CaImAn Interactive Pipeline",
        header_background=ACCENT_COLOR,
        sidebar=[sidebar],
        main=[header, tabs],
        sidebar_width=380,
    )
