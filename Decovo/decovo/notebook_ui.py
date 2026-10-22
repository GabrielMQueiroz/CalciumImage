# -*- coding: utf-8 -*-
"""Interactive ipywidgets UI for Google Colab and local Jupyter notebooks.

Provides:
- `inspect_input_banner(folder_or_file, user_frame_rate, spatial_downsample)`
- `threshold_picker(session, on_extract=None)`
- `neuron_browser(result)`
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from .env import get_available_ram_gb
from .io import InputSpec, discover_inputs
from .params_auto import estimate_params_heuristic
from .pipeline import PipelineResult, Session, extract


def inspect_input_banner(
    folder_or_file: str | Path,
    user_frame_rate: float = 20.0,
    spatial_downsample: float = 0.6,
    pattern: Optional[str] = None,
) -> InputSpec:
    """Inspect input recording, verify metadata frame rate against user selection, and print a clear summary."""
    spec = discover_inputs(folder_or_file, pattern=pattern)
    h, w = spec.frame_shape
    ds_h = max(1, int(round(h * spatial_downsample)))
    ds_w = max(1, int(round(w * spatial_downsample)))
    est_movie_gb = (spec.n_frames * ds_h * ds_w * 4) / (1024 ** 3)
    avail_gb = get_available_ram_gb()

    meta_fps = spec.fps_from_metadata
    effective_fps = user_frame_rate if user_frame_rate > 0 else (meta_fps or 20.0)
    duration_sec = spec.n_frames / max(effective_fps, 1e-3)

    print("=" * 68)
    print(f"✅ Detected Input Format : {spec.kind.upper()} ({len(spec.files)} file(s))")
    print(f"   Frames & Dimensions   : {spec.n_frames:,} frames @ {h}×{w} px ({spec.dtype})")
    print(f"   Downsampled Shape     : {ds_h}×{ds_w} px (spatial_downsample={spatial_downsample})")
    if meta_fps is not None:
        match_str = (
            "✅ Matches your selection"
            if abs(meta_fps - user_frame_rate) <= 0.5
            else f"⚠️ Differs from your form value ({user_frame_rate} Hz) — please verify!"
        )
        print(f"   Metadata Frame Rate   : {meta_fps:.2f} Hz ({match_str})")
    else:
        print(f"   Metadata Frame Rate   : Not stored in file header (using your selection: {user_frame_rate:.2f} Hz)")
    print(f"   Recording Duration    : {duration_sec / 60:.1f} min ({duration_sec:.1f} s at {effective_fps:.1f} Hz)")
    print(f"   Downsampled Movie RAM : ~{est_movie_gb:.2f} GB float32 (Available RAM: ~{avail_gb:.1f} GB)")
    if est_movie_gb > avail_gb * 0.75:
        print(
            "   ⚠️ WARNING: Movie size is large relative to available RAM. "
            "Consider lowering spatial_downsample (e.g. 0.5) if CNMF-E runs out of memory."
        )
    print("=" * 68)
    return spec


def threshold_picker(
    session: Session,
    on_extract: Optional[Callable[[PipelineResult], None]] = None,
):
    """Interactive ipywidgets correlation (Cn) & PNR threshold selector for Google Colab / Jupyter.

    Displays Cn, PNR, and the live seed mask overlay + estimated neuron count.
    Clicking 'Extract with these thresholds' runs only Stage B (`extract`) on the cached memmap.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    h_corr, h_pnr = estimate_params_heuristic(session.cn_filter, session.pnr)
    init_corr = float(session.config.min_corr if session.config.min_corr is not None else h_corr)
    init_pnr = float(session.config.min_pnr if session.config.min_pnr is not None else h_pnr)

    pnr_max_data = float(np.nanmax(session.pnr)) if session.pnr is not None and session.pnr.size > 0 else 30.0
    pnr_min_bound = min(1.0, float(np.floor(init_pnr)))
    pnr_max_bound = max(30.0, float(np.ceil(max(pnr_max_data, init_pnr + 2.0))))

    corr_slider = widgets.FloatSlider(
        value=float(np.clip(init_corr, 0.10, 0.99)),
        min=0.10,
        max=0.99,
        step=0.01,
        description="min_corr:",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="380px"),
    )
    pnr_slider = widgets.FloatSlider(
        value=float(np.clip(init_pnr, pnr_min_bound, pnr_max_bound)),
        min=pnr_min_bound,
        max=pnr_max_bound,
        step=0.25,
        description="min_pnr:",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="380px"),
    )

    auto_btn = widgets.Button(
        description="Auto (Heuristic)",
        button_style="info",
        icon="magic",
        layout=widgets.Layout(width="160px"),
    )
    extract_btn = widgets.Button(
        description="Extract with these",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="190px"),
    )
    status_html = widgets.HTML(value="Adjust sliders to preview seed mask, then click <b>Extract with these</b>.")
    plot_out = widgets.Output()

    state = {"result": None}

    def _estimate_seed_count(mc: float, mp: float) -> Tuple[np.ndarray, int]:
        mask = (session.cn_filter >= mc) & (session.pnr >= mp)
        n_pixels = int(np.sum(mask))
        gsig = float(session.config.gSig)
        neuron_area = max(1.0, np.pi * gsig * gsig)
        return mask, max(0, int(round(n_pixels / neuron_area)))

    def _redraw(*_args):
        mc = float(corr_slider.value)
        mp = float(pnr_slider.value)
        mask, est_n = _estimate_seed_count(mc, mp)
        with plot_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
            im0 = axes[0].imshow(session.cn_filter, cmap="viridis")
            axes[0].set_title(f"Correlation (Cn) — thr ≥ {mc:.2f}")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            im1 = axes[1].imshow(session.pnr, cmap="inferno")
            axes[1].set_title(f"PNR — thr ≥ {mp:.2f}")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            axes[2].imshow(session.cn_filter, cmap="gray", alpha=0.85)
            overlay = np.ma.masked_where(~mask, mask.astype(np.float32))
            axes[2].imshow(overlay, cmap="spring", alpha=0.75, vmin=0, vmax=1)
            axes[2].set_title(f"Seed Mask (~{est_n} est. neurons, {int(mask.sum())} px)")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

    def _on_auto(_btn):
        corr_slider.value = h_corr
        pnr_slider.value = h_pnr
        status_html.value = f"Reset to heuristic thresholds: <b>min_corr={h_corr:.3f}, min_pnr={h_pnr:.2f}</b>"
        _redraw()

    def _on_extract(_btn):
        mc = float(corr_slider.value)
        mp = float(pnr_slider.value)
        extract_btn.disabled = True
        status_html.value = f"⏳ Running Stage B (CNMF-E + OASIS) with <b>min_corr={mc:.3f}, min_pnr={mp:.2f}</b>..."
        try:
            res = extract(session, min_corr=mc, min_pnr=mp)
            state["result"] = res
            status_html.value = (
                f"✅ Extraction complete! Accepted <b>{res.n_neurons}</b> neurons "
                f"(min_corr={res.min_corr_used:.3f}, min_pnr={res.min_pnr_used:.2f})."
            )
            if on_extract is not None:
                on_extract(res)
        except Exception as e:
            status_html.value = f"<span style='color:red;'>❌ Extraction failed: {e}</span>"
            raise
        finally:
            extract_btn.disabled = False

    corr_slider.observe(_redraw, names="value")
    pnr_slider.observe(_redraw, names="value")
    auto_btn.on_click(_on_auto)
    extract_btn.on_click(_on_extract)

    _redraw()

    ui = widgets.VBox(
        [
            widgets.HBox([corr_slider, pnr_slider]),
            widgets.HBox([auto_btn, extract_btn, status_html]),
            plot_out,
        ]
    )
    display(ui)
    return ui, state


def neuron_browser(result: PipelineResult):
    """Interactive ipywidgets neuron browser for Google Colab / Jupyter.

    Shows spatial footprint on Cn, raw fluorescence trace C, deconvolved spike trace S, and SNR / r-value badges.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    if result is None or result.cnm is None:
        print("No extraction result available yet. Run extraction first.")
        return None

    cnm = result.cnm
    idx_good = (
        np.asarray(cnm.estimates.idx_components, dtype=int)
        if getattr(cnm.estimates, "idx_components", None) is not None
        else np.array([], dtype=int)
    )
    if len(idx_good) == 0:
        print("0 accepted neurons in this result. Try lowering min_corr or min_pnr.")
        return None

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(idx_good) - 1,
        step=1,
        description="Neuron:",
        continuous_update=False,
        layout=widgets.Layout(width="420px"),
    )
    prev_btn = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="90px"))
    next_btn = widgets.Button(description="Next ▶", layout=widgets.Layout(width="90px"))
    badge_html = widgets.HTML()
    plot_out = widgets.Output()

    A = cnm.estimates.A
    A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    dims = (
        result.dims
        or (result.cn_filter.shape if result.cn_filter is not None else None)
        or getattr(cnm.estimates, "dims", None)
    )
    fr = result.config.frate if result.config else 20.0

    def _render(*_args):
        sel = int(slider.value)
        comp_id = int(idx_good[sel])
        snr_arr = getattr(cnm.estimates, "SNR_comp", None)
        rval_arr = getattr(cnm.estimates, "r_values", None)
        snr_str = f"{snr_arr[comp_id]:.2f}" if (snr_arr is not None and len(snr_arr) > comp_id) else "N/A"
        rval_str = f"{rval_arr[comp_id]:.3f}" if (rval_arr is not None and len(rval_arr) > comp_id) else "N/A"

        badge_html.value = (
            f"<b>Neuron {sel + 1} / {len(idx_good)}</b> (Component #{comp_id}) &nbsp;|&nbsp; "
            f"<b>SNR:</b> {snr_str} &nbsp;|&nbsp; <b>r-value:</b> {rval_str}"
        )

        sfp = np.reshape(A_csc[:, comp_id].toarray(), dims, order="F")
        raw = cnm.estimates.C[comp_id, :]
        deconv = cnm.estimates.S[comp_id, :] if cnm.estimates.S is not None else np.zeros_like(raw)
        t = np.arange(len(raw)) / max(fr, 1e-3)

        with plot_out:
            clear_output(wait=True)
            fig = plt.figure(figsize=(14, 4.2))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2])

            ax_fov = fig.add_subplot(gs[:, 0])
            if result.cn_filter is not None:
                ax_fov.imshow(result.cn_filter, cmap="viridis", alpha=0.8)
            sfp_masked = np.ma.masked_where(sfp < sfp.max() * 0.2, sfp)
            ax_fov.imshow(sfp_masked, cmap="hot", alpha=0.8)
            ax_fov.set_title(f"Spatial Footprint (Neuron {sel + 1})")
            ax_fov.axis("off")

            ax_raw = fig.add_subplot(gs[0, 1])
            ax_raw.plot(t, raw, color="#0284c7", linewidth=0.9)
            ax_raw.set_title("Denoised Calcium Trace (C)")
            ax_raw.set_ylabel("Fluorescence")

            ax_spk = fig.add_subplot(gs[1, 1], sharex=ax_raw)
            ax_spk.plot(t, deconv, color="#10b981", linewidth=0.9)
            ax_spk.set_title("Deconvolved Activity (OASIS S)")
            ax_spk.set_xlabel("Time (s)")
            ax_spk.set_ylabel("Spikes (a.u.)")

            plt.tight_layout()
            plt.show()

    def _prev(_b):
        slider.value = max(0, slider.value - 1)

    def _next(_b):
        slider.value = min(len(idx_good) - 1, slider.value + 1)

    slider.observe(_render, names="value")
    prev_btn.on_click(_prev)
    next_btn.on_click(_next)

    _render()
    ui = widgets.VBox([widgets.HBox([prev_btn, slider, next_btn, badge_html]), plot_out])
    display(ui)
    return ui
