# -*- coding: utf-8 -*-
"""High-Performance Native Desktop GUI for the Decovo CaImAn CNMF-E Pipeline.

Mirrors the visual design language, responsiveness, and architecture of
CleanMiniscope's Cell Curator GUI:
- 100% native Tkinter + Matplotlib (TkAgg). Zero web server / Bokeh overhead.
- Ultra-fast <5ms neuron navigation using persistent Matplotlib LineCollections.
- 3-Step Guided Workflow:
    1. Setup & Preprocessing (Stage A): File discovery, auto metadata FPS, motion correction.
    2. Interactive Cn/PNR Seed Picker (Stage B): Live real-time candidate seed preview scatter.
    3. Full Neuron Curator: FOV with contours, zoomed ROI, raw + denoised + deconvolved spike
       traces, population peak histogram, and instant keyboard shortcuts.
- Direct one-click handoff to CleanMiniscope Cell Curator GUI.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
import scipy.sparse as sp
from scipy import ndimage

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import PipelineConfig, load_config, save_config
from .export import export_results, save_hdf5_for_curator, save_mat_for_curator
from .io import InputSpec, discover_inputs
from .pipeline import PipelineResult, Session, extract, preprocess

try:
    from caiman.source_extraction.cnmf import cnmf
    from caiman.utils.visualization import get_contours as cm_get_contours
    CAIMAN_AVAILABLE = True
except Exception:
    cnmf = None
    cm_get_contours = None
    CAIMAN_AVAILABLE = False

logger = logging.getLogger("decovo.gui")


def fallback_get_contours(A, dims: Tuple[int, int], thr: float = 0.9) -> List[Dict[str, Any]]:
    """Fast fallback contour extraction when CaImAn is not installed or mocked."""
    if sp.issparse(A):
        A = A.toarray()
    n_components = A.shape[1]
    contours = []
    d1, d2 = dims

    for i in range(n_components):
        img = A[:, i].reshape((d1, d2), order="F")
        max_val = np.max(img)
        if max_val <= 0:
            contours.append({"coordinates": np.empty((0, 2)), "neuron_id": i})
            continue
        mask = (img >= (1.0 - thr) * max_val).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            # Pick largest contour
            largest = max(cnts, key=cv2.contourArea)
            pts = largest.squeeze(axis=1).astype(float)
            if pts.ndim == 1 and len(pts) == 2:
                pts = pts.reshape(1, 2)
            contours.append({"coordinates": pts, "neuron_id": i})
        else:
            contours.append({"coordinates": np.empty((0, 2)), "neuron_id": i})
    return contours


def get_contours(A, dims: Tuple[int, int], thr: float = 0.9) -> List[Dict[str, Any]]:
    if cm_get_contours is not None:
        try:
            return cm_get_contours(A, dims, thr=thr, swap_dim=False)
        except Exception:
            pass
    return fallback_get_contours(A, dims, thr=thr)


def fix_dims(A, dims: Tuple[int, int]) -> Tuple[int, int]:
    pixel_count = A.shape[0]
    d1, d2 = dims
    if pixel_count != d1 * d2:
        aspect = d2 / max(d1, 1)
        d1 = int(np.round(np.sqrt(pixel_count / aspect)))
        d2 = int(np.round(pixel_count / max(d1, 1)))
    return (d1, d2)


class QueueHandler(logging.Handler):
    """Thread-safe logging handler routing messages to a GUI queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


class DecovoApp(tk.Tk):
    """Native Tkinter desktop application for Decovo CNMF-E Deconvolution & Curation."""

    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()

        self.title("🧠 Decovo v2 — CNMF-E Deconvolution & Curation")
        self.geometry("1520x960")
        self.minsize(1240, 820)

        # Pipeline state
        self.current_input_path: Optional[Path] = None
        self.current_output_dir: Optional[Path] = None
        self.input_spec: Optional[InputSpec] = None
        self.session: Optional[Session] = None
        self.pipeline_result: Optional[PipelineResult] = None
        self.current_hdf5_path: Optional[str] = None

        # Curator state
        self.cnm = None
        self.n_cells: int = 0
        self.current_idx: int = 0
        self.accepted_mask: np.ndarray = np.array([], dtype=bool)
        self.snr_values: np.ndarray = np.array([], dtype=float)
        self.r_values: np.ndarray = np.array([], dtype=float)
        self.peak_counts: np.ndarray = np.array([], dtype=int)
        self.peak_indices: List[np.ndarray] = []
        self.dims: Tuple[int, int] = (240, 376)
        self.fov_img: Optional[np.ndarray] = None
        self.contours: List[Dict[str, Any]] = []
        self.contour_segments: List[np.ndarray] = []
        self.centroids: np.ndarray = np.zeros((0, 2))
        self.passing_indices: List[int] = []

        # Threading & Cancellation
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.log_queue: queue.Queue = queue.Queue()

        # Seed picker preview cache
        self._seed_coords: np.ndarray = np.zeros((0, 2))
        self._seed_timer: Optional[str] = None

        # Configure Theme
        self._setup_theme()

        # Build UI layout
        self._build_header_toolbar()
        self._build_notebook()
        self._build_status_bar()

        # Attach logging handler
        self._setup_logging()

        # Bind Global Keyboard Shortcuts
        self.bind("<Left>", lambda e: self.on_prev_cell())
        self.bind("<Right>", lambda e: self.on_next_cell())
        self.bind("<space>", lambda e: self.on_toggle_accepted())
        self.bind("a", lambda e: self.on_set_accepted(True))
        self.bind("A", lambda e: self.on_set_accepted(True))
        self.bind("r", lambda e: self.on_set_accepted(False))
        self.bind("R", lambda e: self.on_set_accepted(False))

        # Check queue periodically
        self.after(100, self._process_log_queue)

        # Load initial file if provided
        if initial_path:
            p = Path(initial_path)
            if p.suffix.lower() in (".hdf5", ".h5"):
                self.load_hdf5(str(p))
            else:
                self.set_input_path(p)

    # -------------------------------------------------------------------------
    # THEME & STYLING
    # -------------------------------------------------------------------------
    def _setup_theme(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.BG_MAIN = "#1e1e1e"
        self.BG_CARD = "#252526"
        self.BG_ELEVATED = "#2d2d2d"
        self.FG_TEXT = "#e0e0e0"
        self.FG_MUTED = "#8e8e8e"
        self.ACCENT_BLUE = "#3b82f6"
        self.ACCENT_GREEN = "#10b981"
        self.ACCENT_RED = "#ef4444"
        self.ACCENT_YELLOW = "#f59e0b"

        self.configure(bg=self.BG_MAIN)

        self.style.configure(".", background=self.BG_MAIN, foreground=self.FG_TEXT, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=self.BG_MAIN)
        self.style.configure("Card.TFrame", background=self.BG_CARD)
        self.style.configure("TLabel", background=self.BG_MAIN, foreground=self.FG_TEXT)
        self.style.configure("Card.TLabel", background=self.BG_CARD, foreground=self.FG_TEXT)
        self.style.configure("Header.TLabel", background=self.BG_MAIN, foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        self.style.configure("Badge.TLabel", background=self.BG_ELEVATED, foreground="#60a5fa", font=("Segoe UI", 8, "bold"), padding=3)

        self.style.configure("TLabelframe", background=self.BG_MAIN, foreground="#ffffff")
        self.style.configure("TLabelframe.Label", background=self.BG_MAIN, foreground="#ffffff", font=("Segoe UI", 9, "bold"))

        self.style.configure("TButton", background=self.BG_ELEVATED, foreground="#ffffff", borderwidth=1, padding=5)
        self.style.map("TButton", background=[("active", self.ACCENT_BLUE), ("disabled", "#333333")])

        self.style.configure("Accent.TButton", background=self.ACCENT_BLUE, foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.map("Accent.TButton", background=[("active", "#2563eb"), ("disabled", "#333333")])

        self.style.configure("Success.TButton", background=self.ACCENT_GREEN, foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.map("Success.TButton", background=[("active", "#059669"), ("disabled", "#333333")])

        self.style.configure("Danger.TButton", background=self.ACCENT_RED, foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.map("Danger.TButton", background=[("active", "#dc2626"), ("disabled", "#333333")])

        self.style.configure("TNotebook", background=self.BG_MAIN, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.BG_CARD, foreground=self.FG_TEXT, padding=[16, 7], font=("Segoe UI", 9, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", self.BG_ELEVATED)], foreground=[("selected", "#60a5fa")])

        self.style.configure("TEntry", fieldbackground=self.BG_ELEVATED, foreground="#ffffff")
        self.style.configure("TSpinbox", fieldbackground=self.BG_ELEVATED, foreground="#ffffff")

    def _setup_logging(self):
        gui_handler = QueueHandler(self.log_queue)
        gui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(gui_handler)
        logging.getLogger("decovo").setLevel(logging.INFO)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                if hasattr(self, "txt_log"):
                    self.txt_log.configure(state="normal")
                    self.txt_log.insert(tk.END, msg + "\n")
                    self.txt_log.see(tk.END)
                    self.txt_log.configure(state="disabled")
            except Exception:
                break
        self.after(100, self._process_log_queue)

    # -------------------------------------------------------------------------
    # UI: HEADER TOOLBAR
    # -------------------------------------------------------------------------
    def _build_header_toolbar(self):
        toolbar = ttk.Frame(self, padding=8, style="Card.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        # Title / Brand
        lbl_brand = ttk.Label(toolbar, text="🧠 Decovo v2", style="Header.TLabel", font=("Segoe UI", 12, "bold"))
        lbl_brand.pack(side=tk.LEFT, padx=(4, 12))

        # Actions on Left
        btn_open_rec = ttk.Button(toolbar, text="📂 Select Recording (TIFF / AVI)...", command=self.browse_recording)
        btn_open_rec.pack(side=tk.LEFT, padx=4)

        btn_open_hdf5 = ttk.Button(toolbar, text="📂 Open HDF5...", command=self.browse_hdf5)
        btn_open_hdf5.pack(side=tk.LEFT, padx=4)

        self.lbl_active_file = ttk.Label(toolbar, text="No recording loaded", font=("Segoe UI", 9, "italic"), foreground=self.FG_MUTED)
        self.lbl_active_file.pack(side=tk.LEFT, padx=10)

        # Actions on Right (Handoff & Save)
        btn_open_curator = ttk.Button(
            toolbar,
            text="🔗 Open in Cell Curator",
            style="Accent.TButton",
            command=self.open_in_cell_curator,
        )
        btn_open_curator.pack(side=tk.RIGHT, padx=4)

        btn_save = ttk.Button(toolbar, text="💾 Save Results", style="Success.TButton", command=self.save_current_results)
        btn_save.pack(side=tk.RIGHT, padx=4)

    # -------------------------------------------------------------------------
    # UI: NOTEBOOK (3-STEP TABS)
    # -------------------------------------------------------------------------
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.tab_setup = ttk.Frame(self.notebook, padding=10)
        self.tab_seeds = ttk.Frame(self.notebook, padding=10)
        self.tab_curator = ttk.Frame(self.notebook, padding=6)

        self.notebook.add(self.tab_setup, text=" 1. Setup & Preprocess ")
        self.notebook.add(self.tab_seeds, text=" 2. Seed & Threshold Picker ")
        self.notebook.add(self.tab_curator, text=" 3. Neuron Curator ")

        self._build_step1_setup()
        self._build_step2_seeds()
        self._build_step3_curator()

    # -------------------------------------------------------------------------
    # TAB 1: SETUP & PREPROCESSING (STAGE A)
    # -------------------------------------------------------------------------
    def _build_step1_setup(self):
        paned = ttk.PanedWindow(self.tab_setup, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned, padding=8)
        right_frame = ttk.Frame(paned, padding=8)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=1)

        # Input & Parameters Box
        grp_params = ttk.LabelFrame(left_frame, text="⚙️ Session Configuration & Parameters", padding=10)
        grp_params.pack(fill=tk.BOTH, expand=True)

        grid = ttk.Frame(grp_params)
        grid.pack(fill=tk.X, pady=4)

        # Path input
        ttk.Label(grid, text="Input Path:").grid(row=0, column=0, sticky="w", pady=4)
        self.ent_input_path = ttk.Entry(grid, width=42)
        self.ent_input_path.grid(row=0, column=1, sticky="ew", pady=4, padx=4)
        ttk.Button(grid, text="Browse...", command=self.browse_recording).grid(row=0, column=2, padx=2)

        # Results directory
        ttk.Label(grid, text="Results Folder:").grid(row=1, column=0, sticky="w", pady=4)
        self.ent_output_dir = ttk.Entry(grid, width=42)
        self.ent_output_dir.grid(row=1, column=1, sticky="ew", pady=4, padx=4)
        ttk.Button(grid, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=2)

        # Frame rate & Metadata badge
        ttk.Label(grid, text="Frame Rate (Hz):").grid(row=2, column=0, sticky="w", pady=4)
        fps_frame = ttk.Frame(grid)
        fps_frame.grid(row=2, column=1, sticky="w", pady=4, padx=4)
        self.spn_fps = ttk.Spinbox(fps_frame, from_=1.0, to=100.0, increment=1.0, width=8)
        self.spn_fps.set("20.0")
        self.spn_fps.pack(side=tk.LEFT)
        self.lbl_fps_meta = ttk.Label(fps_frame, text="", style="Badge.TLabel")
        self.lbl_fps_meta.pack(side=tk.LEFT, padx=6)

        # Spatial Downsample
        ttk.Label(grid, text="Spatial Downsample:").grid(row=3, column=0, sticky="w", pady=4)
        self.spn_ds = ttk.Spinbox(grid, from_=0.1, to=1.0, increment=0.05, width=8)
        self.spn_ds.set("0.60")
        self.spn_ds.grid(row=3, column=1, sticky="w", pady=4, padx=4)

        # gSig & gSiz
        ttk.Label(grid, text="Neuron Radius gSig (px):").grid(row=4, column=0, sticky="w", pady=4)
        gsig_frame = ttk.Frame(grid)
        gsig_frame.grid(row=4, column=1, sticky="w", pady=4, padx=4)
        self.spn_gsig = ttk.Spinbox(gsig_frame, from_=1.0, to=20.0, increment=0.5, width=8, command=self._on_gsig_changed)
        self.spn_gsig.set("3.0")
        self.spn_gsig.bind("<KeyRelease>", lambda e: self._on_gsig_changed())
        self.spn_gsig.pack(side=tk.LEFT)
        ttk.Label(gsig_frame, text="  gSiz:").pack(side=tk.LEFT)
        self.lbl_gsiz = ttk.Label(gsig_frame, text="13 px", font=("Segoe UI", 9, "bold"))
        self.lbl_gsiz.pack(side=tk.LEFT, padx=4)

        # Patch size & stride
        ttk.Label(grid, text="Patch rf / Stride (px):").grid(row=5, column=0, sticky="w", pady=4)
        patch_frame = ttk.Frame(grid)
        patch_frame.grid(row=5, column=1, sticky="w", pady=4, padx=4)
        self.spn_rf = ttk.Spinbox(patch_frame, from_=10, to=120, increment=5, width=6)
        self.spn_rf.set("40")
        self.spn_rf.pack(side=tk.LEFT)
        ttk.Label(patch_frame, text=" / ").pack(side=tk.LEFT)
        self.spn_stride = ttk.Spinbox(patch_frame, from_=5, to=60, increment=5, width=6)
        self.spn_stride.set("20")
        self.spn_stride.pack(side=tk.LEFT)

        # Merge threshold & Quality
        ttk.Label(grid, text="Merge Thr / Min SNR:").grid(row=6, column=0, sticky="w", pady=4)
        merge_frame = ttk.Frame(grid)
        merge_frame.grid(row=6, column=1, sticky="w", pady=4, padx=4)
        self.spn_merge = ttk.Spinbox(merge_frame, from_=0.3, to=0.99, increment=0.05, width=6)
        self.spn_merge.set("0.70")
        self.spn_merge.pack(side=tk.LEFT)
        ttk.Label(merge_frame, text=" / ").pack(side=tk.LEFT)
        self.spn_snr_init = ttk.Spinbox(merge_frame, from_=1.0, to=20.0, increment=0.5, width=6)
        self.spn_snr_init.set("3.0")
        self.spn_snr_init.pack(side=tk.LEFT)

        # Motion correction max shifts
        ttk.Label(grid, text="Max Rigid Shifts (px):").grid(row=7, column=0, sticky="w", pady=4)
        self.spn_shifts = ttk.Spinbox(grid, from_=1, to=30, increment=1, width=8)
        self.spn_shifts.set("4")
        self.spn_shifts.grid(row=7, column=1, sticky="w", pady=4, padx=4)

        grid.columnconfigure(1, weight=1)

        # Action Buttons
        btn_box = ttk.Frame(grp_params, padding=6)
        btn_box.pack(fill=tk.X, pady=(12, 4))

        self.btn_run_stage_a = ttk.Button(
            btn_box,
            text="▶ Run Preprocessing & Motion Correction (Stage A)",
            style="Accent.TButton",
            command=self.run_stage_a,
        )
        self.btn_run_stage_a.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.btn_stop = ttk.Button(btn_box, text="⏹ Stop", style="Danger.TButton", command=self.cancel_task, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=4)

        # Right Frame: Execution Logs & Progress
        grp_log = ttk.LabelFrame(right_frame, text="📋 Real-time Pipeline Log", padding=8)
        grp_log.pack(fill=tk.BOTH, expand=True)

        self.progress_bar = ttk.Progressbar(grp_log, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        self.txt_log = tk.Text(
            grp_log,
            wrap=tk.WORD,
            bg="#181818",
            fg="#cccccc",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            state="disabled",
            borderwidth=0,
        )
        scroll = ttk.Scrollbar(grp_log, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_gsig_changed(self):
        try:
            val = float(self.spn_gsig.get())
            gsiz = int(np.round(4 * val + 1))
            self.lbl_gsiz.config(text=f"{gsiz} px")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # TAB 2: INTERACTIVE SEED & THRESHOLD PICKER (STAGE B)
    # -------------------------------------------------------------------------
    def _build_step2_seeds(self):
        top_bar = ttk.Frame(self.tab_seeds, padding=6, style="Card.TFrame")
        top_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        # Controls Bar
        ttk.Label(top_bar, text="Min Correlation (min_corr):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.slider_corr = ttk.Scale(top_bar, from_=0.60, to=0.98, value=0.85, command=self._on_seed_slider_moved)
        self.slider_corr.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.lbl_corr_val = ttk.Label(top_bar, text="0.85", width=5, font=("Consolas", 9, "bold"))
        self.lbl_corr_val.pack(side=tk.LEFT, padx=2)

        ttk.Label(top_bar, text="Min PNR (min_pnr):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(12, 2))
        self.slider_pnr = ttk.Scale(top_bar, from_=4.0, to=25.0, value=10.0, command=self._on_seed_slider_moved)
        self.slider_pnr.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.lbl_pnr_val = ttk.Label(top_bar, text="10.0", width=5, font=("Consolas", 9, "bold"))
        self.lbl_pnr_val.pack(side=tk.LEFT, padx=2)

        self.lbl_seed_count = ttk.Label(top_bar, text="Seeds: 0", style="Badge.TLabel")
        self.lbl_seed_count.pack(side=tk.LEFT, padx=8)

        # Image view selector
        self.var_seed_view = tk.StringVar(value="Cn")
        rb_cn = ttk.Radiobutton(top_bar, text="Correlation (Cn)", variable=self.var_seed_view, value="Cn", command=self._draw_seed_plot)
        rb_pnr = ttk.Radiobutton(top_bar, text="PNR", variable=self.var_seed_view, value="PNR", command=self._draw_seed_plot)
        rb_cn.pack(side=tk.LEFT, padx=4)
        rb_pnr.pack(side=tk.LEFT, padx=4)

        self.btn_run_stage_b = ttk.Button(
            top_bar,
            text="▶ Run Extraction (Stage B)",
            style="Accent.TButton",
            command=self.run_stage_b,
        )
        self.btn_run_stage_b.pack(side=tk.RIGHT, padx=6)

        # Matplotlib Seed Plot
        self.fig_seeds = Figure(figsize=(10, 6), dpi=100, facecolor=self.BG_MAIN)
        self.ax_seeds = self.fig_seeds.add_subplot(111)
        self.ax_seeds.set_facecolor(self.BG_MAIN)
        self.ax_seeds.axis("off")

        self.canvas_seeds = FigureCanvasTkAgg(self.fig_seeds, master=self.tab_seeds)
        self.toolbar_seeds = NavigationToolbar2Tk(self.canvas_seeds, self.tab_seeds)
        self.toolbar_seeds.config(background=self.BG_CARD)
        self.canvas_seeds.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_seed_slider_moved(self, _=None):
        corr = float(self.slider_corr.get())
        pnr = float(self.slider_pnr.get())
        self.lbl_corr_val.config(text=f"{corr:.2f}")
        self.lbl_pnr_val.config(text=f"{pnr:.1f}")

        # Debounce seed recomputation to 40ms
        if self._seed_timer is not None:
            self.after_cancel(self._seed_timer)
        self._seed_timer = self.after(40, self._update_seed_preview)

    def _update_seed_preview(self):
        self._seed_timer = None
        if self.session is None or self.session.cn is None or self.session.pnr is None:
            return

        corr = float(self.slider_corr.get())
        pnr = float(self.slider_pnr.get())
        Cn = self.session.cn
        PNR = self.session.pnr

        # Fast local maxima detection in Cn & PNR space
        mask = (Cn >= corr) & (PNR >= pnr)
        gsig = float(self.spn_gsig.get())
        footprint_rad = max(3, int(np.round(gsig)))
        footprint = np.ones((footprint_rad * 2 + 1, footprint_rad * 2 + 1), dtype=bool)

        local_max = (PNR == ndimage.maximum_filter(PNR, footprint=footprint)) & mask
        coords = np.column_stack(np.where(local_max))
        self._seed_coords = coords
        self.lbl_seed_count.config(text=f"Seeds: {len(coords)}")

        self._draw_seed_plot()

    def _draw_seed_plot(self):
        self.ax_seeds.clear()
        if self.session is None or self.session.cn is None:
            self.ax_seeds.text(0.5, 0.5, "Run Stage A to generate Cn and PNR summary images",
                               ha="center", va="center", color=self.FG_MUTED, transform=self.ax_seeds.transAxes)
            self.ax_seeds.axis("off")
            self.canvas_seeds.draw_idle()
            return

        img = self.session.cn if self.var_seed_view.get() == "Cn" else self.session.pnr
        cmap = "inferno" if self.var_seed_view.get() == "Cn" else "viridis"

        self.ax_seeds.imshow(img, cmap=cmap, aspect="equal", origin="upper")
        if len(self._seed_coords) > 0:
            # coords are (row, col) = (y, x)
            self.ax_seeds.scatter(self._seed_coords[:, 1], self._seed_coords[:, 0],
                                  s=18, c="#00ffff", marker="o", alpha=0.85, edgecolors="#ffffff", linewidths=0.5)

        title = f"{self.var_seed_view.get()} Summary Image with Candidate Seeds ({len(self._seed_coords)} seeds)"
        self.ax_seeds.set_title(title, color=self.FG_TEXT, fontsize=10, fontweight="bold")
        self.ax_seeds.axis("off")
        self.fig_seeds.tight_layout()
        self.canvas_seeds.draw_idle()

    # -------------------------------------------------------------------------
    # TAB 3: NEURON CURATOR & TRACE INSPECTOR (CELL CURATOR STYLE)
    # -------------------------------------------------------------------------
    def _build_step3_curator(self):
        # 1. Top Quantitative Filter Bar
        filter_bar = ttk.Frame(self.tab_curator, padding=6, style="Card.TFrame")
        filter_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        ttk.Label(filter_bar, text="Min Peaks (≥):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.spn_filter_peaks = ttk.Spinbox(filter_bar, from_=0, to=1000, increment=5, width=6)
        self.spn_filter_peaks.set("10")
        self.spn_filter_peaks.pack(side=tk.LEFT, padx=(0, 8))
        self.spn_filter_peaks.bind("<KeyRelease>", self._on_filter_changed)

        ttk.Label(filter_bar, text="Min SNR (≥):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.spn_filter_snr = ttk.Spinbox(filter_bar, from_=0.0, to=50.0, increment=0.5, width=5)
        self.spn_filter_snr.set("2.0")
        self.spn_filter_snr.pack(side=tk.LEFT, padx=(0, 8))
        self.spn_filter_snr.bind("<KeyRelease>", self._on_filter_changed)

        ttk.Label(filter_bar, text="Min Spatial Corr (≥):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.spn_filter_rval = ttk.Spinbox(filter_bar, from_=0.0, to=1.0, increment=0.05, width=5)
        self.spn_filter_rval.set("0.85")
        self.spn_filter_rval.pack(side=tk.LEFT, padx=(0, 10))
        self.spn_filter_rval.bind("<KeyRelease>", self._on_filter_changed)

        self.var_only_passing = tk.BooleanVar(value=False)
        chk_pass = ttk.Checkbutton(filter_bar, text="Only Display Passing Criteria", variable=self.var_only_passing, command=self._on_filter_changed)
        chk_pass.pack(side=tk.LEFT, padx=6)

        # Population Stats Badges
        self.lbl_stats_total = ttk.Label(filter_bar, text="Total: 0", style="Badge.TLabel")
        self.lbl_stats_total.pack(side=tk.RIGHT, padx=4)
        self.lbl_stats_accepted = ttk.Label(filter_bar, text="Accepted: 0", style="Badge.TLabel", foreground="#4ade80")
        self.lbl_stats_accepted.pack(side=tk.RIGHT, padx=4)
        self.lbl_stats_discarded = ttk.Label(filter_bar, text="Discarded: 0", style="Badge.TLabel", foreground="#f87171")
        self.lbl_stats_discarded.pack(side=tk.RIGHT, padx=4)

        # 2. Main Paned Window (Left FOV, Right Details)
        paned = ttk.PanedWindow(self.tab_curator, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(paned, style="Card.TFrame")
        right_pane = ttk.Frame(paned, style="Card.TFrame")
        paned.add(left_pane, weight=5)
        paned.add(right_pane, weight=4)

        # Left: Interactive FOV Canvas
        self.fig_fov = Figure(figsize=(7, 7), facecolor=self.BG_MAIN)
        self.ax_fov = self.fig_fov.add_subplot(111)
        self.ax_fov.set_facecolor(self.BG_MAIN)
        self.ax_fov.axis("off")

        self.canvas_fov = FigureCanvasTkAgg(self.fig_fov, master=left_pane)
        self.toolbar_fov = NavigationToolbar2Tk(self.canvas_fov, left_pane)
        self.toolbar_fov.config(background=self.BG_CARD)
        self.canvas_fov.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_fov.mpl_connect("button_press_event", self._on_fov_click)

        # Right: Detail Plots (ROI Zoom + Population Histogram + Traces)
        self.fig_detail = Figure(figsize=(6, 7), facecolor=self.BG_MAIN)
        gs = self.fig_detail.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 2.0], hspace=0.35, wspace=0.25)

        self.ax_roi = self.fig_detail.add_subplot(gs[0, 0])
        self.ax_roi.set_facecolor(self.BG_MAIN)
        self.ax_roi.axis("off")

        self.ax_metrics = self.fig_detail.add_subplot(gs[0, 1])
        self.ax_metrics.set_facecolor(self.BG_MAIN)
        self.ax_metrics.axis("off")

        self.ax_hist = self.fig_detail.add_subplot(gs[1, :])
        self.ax_hist.set_facecolor(self.BG_CARD)

        self.ax_trace = self.fig_detail.add_subplot(gs[2, :])
        self.ax_trace.set_facecolor(self.BG_CARD)

        self.canvas_detail = FigureCanvasTkAgg(self.fig_detail, master=right_pane)
        self.toolbar_detail = NavigationToolbar2Tk(self.canvas_detail, right_pane)
        self.toolbar_detail.config(background=self.BG_CARD)
        self.canvas_detail.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3. Bottom Navigation Bar
        nav_bar = ttk.Frame(self.tab_curator, padding=6, style="Card.TFrame")
        nav_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        btn_prev = ttk.Button(nav_bar, text="◀ Previous (Left)", command=self.on_prev_cell)
        btn_prev.pack(side=tk.LEFT, padx=4)

        btn_next = ttk.Button(nav_bar, text="▶ Next (Right)", command=self.on_next_cell)
        btn_next.pack(side=tk.LEFT, padx=4)

        self.lbl_cell_counter = ttk.Label(nav_bar, text="Neuron 0 / 0", font=("Segoe UI", 10, "bold"))
        self.lbl_cell_counter.pack(side=tk.LEFT, padx=12)

        self.btn_accept = ttk.Button(nav_bar, text="✔ Keep (Space)", style="Success.TButton", command=lambda: self.on_set_accepted(True))
        self.btn_accept.pack(side=tk.LEFT, padx=4)

        self.btn_discard = ttk.Button(nav_bar, text="✖ Discard (X/R)", style="Danger.TButton", command=lambda: self.on_set_accepted(False))
        self.btn_discard.pack(side=tk.LEFT, padx=4)

        btn_accept_all = ttk.Button(nav_bar, text="Keep All Passing", command=self.on_accept_all_passing)
        btn_accept_all.pack(side=tk.LEFT, padx=10)

    # -------------------------------------------------------------------------
    # UI: STATUS BAR
    # -------------------------------------------------------------------------
    def _build_status_bar(self):
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=4, font=("Segoe UI", 8))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, text: str):
        self.status_bar.config(text=text)
        self.update_idletasks()

    # -------------------------------------------------------------------------
    # FILE BROWSING & DATA LOADING
    # -------------------------------------------------------------------------
    def browse_recording(self):
        chosen = filedialog.askopenfilename(
            title="Select Multi-Page TIFF or AVI Recording",
            filetypes=[("TIFF / AVI Videos", "*.tif *.tiff *.avi"), ("All Files", "*.*")],
        )
        if chosen:
            self.set_input_path(Path(chosen))

    def browse_output_dir(self):
        chosen = filedialog.askdirectory(title="Select Results Output Directory")
        if chosen:
            self.ent_output_dir.delete(0, tk.END)
            self.ent_output_dir.insert(0, chosen)
            self.current_output_dir = Path(chosen)

    def browse_hdf5(self):
        chosen = filedialog.askopenfilename(
            title="Open Existing CNMF-E HDF5 Results",
            filetypes=[("HDF5 Files", "*.hdf5 *.h5"), ("All Files", "*.*")],
        )
        if chosen:
            self.load_hdf5(chosen)

    def set_input_path(self, path: Path):
        self.current_input_path = path
        self.ent_input_path.delete(0, tk.END)
        self.ent_input_path.insert(0, str(path))

        # Output folder default
        out_dir = (path.parent if path.is_file() else path) / "results_decovo"
        self.current_output_dir = out_dir
        self.ent_output_dir.delete(0, tk.END)
        self.ent_output_dir.insert(0, str(out_dir))

        self.lbl_active_file.config(text=f"{path.name} (Discovering inputs...)")
        self.set_status(f"Inspecting input {path.name}...")

        # Discover metadata & FPS
        try:
            spec = discover_inputs(path)
            self.input_spec = spec
            if spec.fps_from_metadata is not None:
                self.spn_fps.delete(0, tk.END)
                self.spn_fps.insert(0, f"{spec.fps_from_metadata:.2f}")
                self.lbl_fps_meta.config(text=f"✅ {spec.fps_from_metadata:.1f} Hz (Metadata)")
            else:
                self.lbl_fps_meta.config(text="ℹ️ Manual FPS")

            self.lbl_active_file.config(text=f"{spec.kind.upper()}: {path.name} ({spec.n_frames_raw or '?'} frames)")
            self.set_status(f"Ready: {spec.kind.upper()} ({spec.n_frames_raw or '?'} frames, {spec.dims_raw or '?'})")
        except Exception as e:
            logger.warning("Could not auto-inspect input: %s", e)
            self.lbl_active_file.config(text=str(path.name))

    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION (STAGE A & STAGE B)
    # -------------------------------------------------------------------------
    def run_stage_a(self):
        if not self.current_input_path:
            messagebox.showwarning("No Input", "Please select an input recording (.tif or .avi folder).")
            return

        cfg = self._gather_config_from_ui()
        self.cancel_event.clear()
        self.btn_run_stage_a.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_bar.start(10)
        self.set_status("Running Stage A (Preprocessing & Motion Correction)...")

        def _worker():
            try:
                sess = preprocess(
                    self.current_input_path,
                    config=cfg,
                    cancel_event=self.cancel_event,
                )
                self.after(0, lambda: self._on_stage_a_complete(sess))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self._on_pipeline_error(err_msg))

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()

    def _on_stage_a_complete(self, sess: Session):
        self.session = sess
        self.progress_bar.stop()
        self.btn_run_stage_a.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.set_status("Stage A Complete! Cn and PNR summary images generated.")

        # Switch to Tab 2 (Seeds) & update preview
        self.notebook.select(self.tab_seeds)
        self._update_seed_preview()

    def run_stage_b(self):
        if self.session is None:
            messagebox.showwarning("Stage A Required", "Please run Stage A first.")
            return

        min_corr = float(self.slider_corr.get())
        min_pnr = float(self.slider_pnr.get())

        self.cancel_event.clear()
        self.btn_run_stage_b.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_bar.start(10)
        self.set_status(f"Running Stage B Extraction (min_corr={min_corr:.2f}, min_pnr={min_pnr:.1f})...")

        def _worker():
            try:
                res = extract(
                    self.session,
                    min_corr=min_corr,
                    min_pnr=min_pnr,
                    cancel_event=self.cancel_event,
                )
                self.after(0, lambda: self._on_stage_b_complete(res))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self._on_pipeline_error(err_msg))

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()

    def _on_stage_b_complete(self, res: PipelineResult):
        self.pipeline_result = res
        self.progress_bar.stop()
        self.btn_run_stage_b.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.current_hdf5_path = str(res.hdf5_path)
        self.set_status(f"Extraction Complete! Extracted {res.n_neurons} neurons ({res.n_accepted} accepted).")

        # Load into Step 3 Curator
        self._load_from_cnm(res.cnm, fov_background=res.session.cn)
        self.notebook.select(self.tab_curator)

    def cancel_task(self):
        self.cancel_event.set()
        self.set_status("Aborting task... please wait for workers to shut down.")
        logger.info("User requested cancellation.")

    def _on_pipeline_error(self, err_msg: str):
        self.progress_bar.stop()
        self.btn_run_stage_a.config(state=tk.NORMAL)
        self.btn_run_stage_b.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.set_status(f"Error: {err_msg}")
        messagebox.showerror("Pipeline Error", f"An error occurred during execution:\n\n{err_msg}")

    def _gather_config_from_ui(self) -> PipelineConfig:
        out_dir = self.ent_output_dir.get().strip() or str(self.current_output_dir)
        fps = float(self.spn_fps.get())
        ds = float(self.spn_ds.get())
        gsig = float(self.spn_gsig.get())
        gsiz = int(np.round(4 * gsig + 1))
        rf = int(self.spn_rf.get())
        stride = int(self.spn_stride.get())
        merge = float(self.spn_merge.get())
        snr = float(self.spn_snr_init.get())
        shifts = int(self.spn_shifts.get())

        return load_config(
            path_to_analyze=str(self.current_input_path) if self.current_input_path else None,
            results_folder=out_dir,
            frame_rate=fps,
            spatial_downsample=ds,
            gSig=gsig,
            gSiz=gsiz,
            rf=rf,
            stride=stride,
            merge_thr=merge,
            min_SNR=snr,
            max_shifts=[shifts, shifts],
        )

    # -------------------------------------------------------------------------
    # CURATOR DATA MODEL & RENDERING
    # -------------------------------------------------------------------------
    def load_hdf5(self, filepath: str):
        if not CAIMAN_AVAILABLE:
            messagebox.showerror("CaImAn Missing", "CaImAn package not found in current environment.")
            return

        self.set_status(f"Loading {os.path.basename(filepath)}...")
        try:
            loaded_cnm = cnmf.load_CNMF(filepath)
            self.current_hdf5_path = filepath
            fov_bg = None
            if hasattr(loaded_cnm.estimates, "Cn") and loaded_cnm.estimates.Cn is not None:
                fov_bg = loaded_cnm.estimates.Cn
            self._load_from_cnm(loaded_cnm, fov_background=fov_bg)
            self.lbl_active_file.config(text=f"{os.path.basename(filepath)} ({self.n_cells} cells)")
            self.notebook.select(self.tab_curator)
            self.set_status(f"Loaded {os.path.basename(filepath)}: {self.n_cells} neurons.")
        except Exception as e:
            messagebox.showerror("Load Failed", f"Could not load HDF5:\n{e}")

    def _load_from_cnm(self, cnm_obj, fov_background: Optional[np.ndarray] = None):
        self.cnm = cnm_obj
        est = cnm_obj.estimates

        self.n_cells = est.C.shape[0]
        raw_dims = tuple(est.dims) if hasattr(est, "dims") and est.dims is not None else (240, 376)
        self.dims = fix_dims(est.A, raw_dims)

        if not sp.issparse(est.A):
            est.A = sp.csc_matrix(est.A)

        # Background image
        if fov_background is not None and tuple(np.shape(fov_background)) == self.dims:
            self.fov_img = np.array(fov_background, dtype=float)
        elif hasattr(est, "Cn") and est.Cn is not None and tuple(np.shape(est.Cn)) == self.dims:
            self.fov_img = np.array(est.Cn, dtype=float)
        else:
            self.fov_img = np.array(est.A.mean(axis=1)).reshape(self.dims, order="F")

        # Contours & Centroids
        self.contours = get_contours(est.A, self.dims, thr=0.9)
        self.contour_segments = []
        self.centroids = []

        for c in self.contours:
            coords = c.get("coordinates", None)
            center = c.get("center", None)
            if center is not None and not np.any(np.isnan(center)):
                # CaImAn center is usually (y, x) -> store as (x, y)
                cx, cy = float(center[1]), float(center[0])
            elif coords is not None and len(coords) > 2 and not np.any(np.isnan(coords)):
                mean_pt = np.mean(coords, axis=0)
                cx, cy = float(mean_pt[0]), float(mean_pt[1])
            else:
                cx, cy = float(self.dims[1] // 2), float(self.dims[0] // 2)

            if coords is not None and len(coords) > 2 and not np.any(np.isnan(coords)):
                seg = np.vstack([coords, coords[0:1]])
                self.contour_segments.append(seg)
            else:
                self.contour_segments.append(np.empty((0, 2)))

            self.centroids.append([cx, cy])

        self.centroids = np.array(self.centroids, dtype=float)

        # Accepted Mask
        self.accepted_mask = np.ones(self.n_cells, dtype=bool)
        if hasattr(est, "idx_components") and est.idx_components is not None:
            self.accepted_mask = np.zeros(self.n_cells, dtype=bool)
            self.accepted_mask[est.idx_components] = True

        # Precompute Vectors (SNR, rval, peaks)
        C = est.C
        med = np.median(C, axis=1, keepdims=True)
        mad = np.median(np.abs(C - med), axis=1, keepdims=True)
        mad[mad == 0] = 1e-6
        thresh = med + 3.0 * mad
        is_active = (C > thresh).astype(np.int8)
        diffs = np.diff(is_active, axis=1, prepend=0)
        self.peak_counts = np.sum(diffs == 1, axis=1)
        self.peak_indices = [np.where(diffs[i] == 1)[0] for i in range(self.n_cells)]

        if hasattr(est, "SNR_comp") and est.SNR_comp is not None and len(est.SNR_comp) == self.n_cells:
            self.snr_values = np.nan_to_num(np.array(est.SNR_comp, dtype=float), nan=0.0)
        else:
            self.snr_values = np.zeros(self.n_cells)

        if hasattr(est, "r_values") and est.r_values is not None and len(est.r_values) == self.n_cells:
            self.r_values = np.nan_to_num(np.array(est.r_values, dtype=float), nan=0.0)
        else:
            self.r_values = np.zeros(self.n_cells)

        self.current_idx = 0
        self._init_fov_plot()
        self._init_detail_plot()
        self._on_filter_changed()

    def _init_fov_plot(self):
        self.ax_fov.clear()
        if self.fov_img is not None:
            self.ax_fov.imshow(self.fov_img, cmap="gray", aspect="equal", origin="upper")

        self.lc_accepted = LineCollection([], colors="#00FF44", linewidths=0.9)
        self.lc_discarded = LineCollection([], colors="#FF2222", linewidths=0.6)
        self.lc_dimmed = LineCollection([], colors="#555555", linewidths=0.4, alpha=0.25)

        self.ax_fov.add_collection(self.lc_accepted)
        self.ax_fov.add_collection(self.lc_discarded)
        self.ax_fov.add_collection(self.lc_dimmed)

        self.highlight_contour, = self.ax_fov.plot([], [], color="#FFFF00", linewidth=2.5)
        self.highlight_centroid = self.ax_fov.scatter([], [], s=60, facecolors="none", edgecolors="#FFFF00", linewidths=2)

        self.ax_fov.set_axis_off()
        self.fig_fov.tight_layout()
        self.canvas_fov.draw_idle()

    def _init_detail_plot(self):
        self.ax_roi.clear()
        self.ax_metrics.clear()
        self.ax_hist.clear()
        self.ax_trace.clear()

        dummy = np.zeros((40, 40))
        self.im_roi = self.ax_roi.imshow(dummy, cmap="viridis", origin="upper")
        self.ax_roi.axis("off")
        self.ax_metrics.axis("off")

        self.ax_hist.set_title("Population Activity Peaks", fontsize=9, fontweight="bold", color=self.FG_TEXT)
        self.ax_hist.tick_params(colors=self.FG_MUTED, labelsize=8)

        self.line_raw, = self.ax_trace.plot([], [], color="#bbbbbb", linewidth=0.8, alpha=0.8, label="Raw (C+YrA)")
        self.line_trace, = self.ax_trace.plot([], [], color="#0077cc", linewidth=1.4, label="Denoised C")
        self.scatter_peaks = self.ax_trace.scatter([], [], color="#ef4444", s=18, marker="v", label="Peaks", zorder=5)

        self.ax_trace.set_xlabel("Frames", fontsize=8, color=self.FG_MUTED)
        self.ax_trace.set_ylabel("Intensity", fontsize=8, color=self.FG_MUTED)
        self.ax_trace.tick_params(colors=self.FG_MUTED, labelsize=8)
        self.ax_trace.grid(True, linestyle=":", alpha=0.3, color="#666666")
        self.ax_trace.legend(fontsize=8, loc="upper right")

        try:
            self.fig_detail.subplots_adjust(top=0.96, bottom=0.08, left=0.10, right=0.96, hspace=0.38, wspace=0.25)
        except Exception:
            pass
        self.canvas_detail.draw_idle()

    # -------------------------------------------------------------------------
    # FILTERING & FAST UPDATES
    # -------------------------------------------------------------------------
    def _on_filter_changed(self, _=None):
        if self.n_cells == 0:
            return

        try:
            min_peaks = int(self.spn_filter_peaks.get())
            min_snr = float(self.spn_filter_snr.get())
            min_rval = float(self.spn_filter_rval.get())
        except Exception:
            min_peaks, min_snr, min_rval = 10, 2.0, 0.85

        pass_peaks = self.peak_counts >= min_peaks
        pass_snr = self.snr_values >= min_snr
        pass_rval = self.r_values >= min_rval
        passing_mask = pass_peaks & pass_snr & pass_rval
        self.passing_indices = list(np.where(passing_mask)[0])

        # Update badges
        n_acc = np.sum(self.accepted_mask)
        n_disc = self.n_cells - n_acc
        self.lbl_stats_total.config(text=f"Total: {self.n_cells}")
        self.lbl_stats_accepted.config(text=f"Accepted: {n_acc}")
        self.lbl_stats_discarded.config(text=f"Discarded: {n_disc}")

        self._update_all_fast(rebuild_collections=True)

    def _update_all_fast(self, rebuild_collections: bool = False):
        if self.n_cells == 0:
            return

        if rebuild_collections:
            acc_segs, disc_segs, dim_segs = [], [], []
            only_pass = self.var_only_passing.get()
            pass_set = set(self.passing_indices)

            for i, seg in enumerate(self.contour_segments):
                if len(seg) == 0:
                    continue
                if only_pass and i not in pass_set:
                    dim_segs.append(seg)
                elif self.accepted_mask[i]:
                    acc_segs.append(seg)
                else:
                    disc_segs.append(seg)

            self.lc_accepted.set_paths(acc_segs)
            self.lc_discarded.set_paths(disc_segs)
            self.lc_dimmed.set_paths(dim_segs)

        # Highlight current neuron
        idx = self.current_idx
        cur_seg = self.contour_segments[idx] if idx < len(self.contour_segments) else np.empty((0, 2))
        if len(cur_seg) > 0:
            self.highlight_contour.set_data(cur_seg[:, 0], cur_seg[:, 1])
        else:
            self.highlight_contour.set_data([], [])

        if idx < len(self.centroids):
            self.highlight_centroid.set_offsets([self.centroids[idx]])
        else:
            self.highlight_centroid.set_offsets(np.empty((0, 2)))

        self.canvas_fov.draw_idle()

        # Update Detail plots
        self._update_detail_plots()

        # Update counter label
        is_acc = "KEEP" if self.accepted_mask[idx] else "DISCARD"
        self.lbl_cell_counter.config(text=f"Neuron {idx + 1} / {self.n_cells} [{is_acc}]")

    def _update_detail_plots(self):
        idx = self.current_idx
        est = self.cnm.estimates

        # 1. ROI Crop
        if idx < est.A.shape[1]:
            spatial = est.A[:, idx].toarray().reshape(self.dims, order="F")
            cx_raw = self.centroids[idx][0] if idx < len(self.centroids) else self.dims[1] // 2
            cy_raw = self.centroids[idx][1] if idx < len(self.centroids) else self.dims[0] // 2
            cx = int(cx_raw) if not np.isnan(cx_raw) else self.dims[1] // 2
            cy = int(cy_raw) if not np.isnan(cy_raw) else self.dims[0] // 2
            rad = 20
            y1, y2 = max(0, cy - rad), min(self.dims[0], cy + rad)
            x1, x2 = max(0, cx - rad), min(self.dims[1], cx + rad)
            crop = spatial[y1:y2, x1:x2]
            if crop.size > 0:
                self.im_roi.set_data(crop)
                self.im_roi.set_clim(vmin=0, vmax=max(np.max(crop), 1e-4))
                self.ax_roi.set_xlim(0, crop.shape[1])
                self.ax_roi.set_ylim(crop.shape[0], 0)

        # 2. Metrics Card
        self.ax_metrics.clear()
        self.ax_metrics.axis("off")
        snr_v = self.snr_values[idx] if idx < len(self.snr_values) else 0.0
        r_v = self.r_values[idx] if idx < len(self.r_values) else 0.0
        p_v = self.peak_counts[idx] if idx < len(self.peak_counts) else 0
        status_txt = "ACCEPTED" if self.accepted_mask[idx] else "DISCARDED"
        color = "#10b981" if self.accepted_mask[idx] else "#ef4444"

        txt = (
            f"ID: #{idx + 1}\n"
            f"Status: {status_txt}\n"
            f"SNR: {snr_v:.2f}\n"
            f"Spatial Corr (r): {r_v:.2f}\n"
            f"Activity Peaks: {p_v}\n"
            f"Centroid: ({self.centroids[idx][0]:.1f}, {self.centroids[idx][1]:.1f})"
        )
        self.ax_metrics.text(0.05, 0.95, txt, va="top", ha="left", color=color, fontsize=9, fontweight="bold")

        # 3. Histogram
        self.ax_hist.clear()
        self.ax_hist.hist(self.peak_counts, bins=25, color="#3b82f6", alpha=0.7, edgecolor="#1e1e1e")
        try:
            min_p = int(self.spn_filter_peaks.get())
            self.ax_hist.axvline(min_p, color="#ef4444", linestyle="--", linewidth=1.5, label="Min Peaks Cutoff")
        except Exception:
            pass
        self.ax_hist.axvline(p_v, color="#f59e0b", linewidth=2.0, label="Current Cell")
        self.ax_hist.set_title("Population Activity Peaks", fontsize=8, fontweight="bold", color=self.FG_TEXT)
        self.ax_hist.tick_params(colors=self.FG_MUTED, labelsize=7)

        # 4. Traces
        c_trace = est.C[idx]
        if hasattr(est, "YrA") and est.YrA is not None and len(est.YrA) > idx:
            raw_trace = c_trace + est.YrA[idx]
        else:
            raw_trace = c_trace

        frames = np.arange(len(c_trace))
        self.line_raw.set_data(frames, raw_trace)
        self.line_trace.set_data(frames, c_trace)

        # Peaks
        p_idx = self.peak_indices[idx] if idx < len(self.peak_indices) else []
        if len(p_idx) > 0:
            self.scatter_peaks.set_offsets(np.column_stack([p_idx, c_trace[p_idx]]))
        else:
            self.scatter_peaks.set_offsets(np.empty((0, 2)))

        self.ax_trace.set_xlim(0, len(c_trace))
        y_min = min(np.min(raw_trace), np.min(c_trace))
        y_max = max(np.max(raw_trace), np.max(c_trace))
        pad = max((y_max - y_min) * 0.1, 1e-3)
        self.ax_trace.set_ylim(y_min - pad, y_max + pad)

        self.canvas_detail.draw_idle()

    # -------------------------------------------------------------------------
    # NAVIGATION & ACTIONS
    # -------------------------------------------------------------------------
    def on_prev_cell(self):
        if self.n_cells == 0:
            return
        if self.var_only_passing.get() and len(self.passing_indices) > 0:
            cur_pos = self.passing_indices.index(self.current_idx) if self.current_idx in self.passing_indices else 0
            new_pos = (cur_pos - 1) % len(self.passing_indices)
            self.current_idx = self.passing_indices[new_pos]
        else:
            self.current_idx = (self.current_idx - 1) % self.n_cells
        self._update_all_fast(rebuild_collections=False)

    def on_next_cell(self):
        if self.n_cells == 0:
            return
        if self.var_only_passing.get() and len(self.passing_indices) > 0:
            cur_pos = self.passing_indices.index(self.current_idx) if self.current_idx in self.passing_indices else 0
            new_pos = (cur_pos + 1) % len(self.passing_indices)
            self.current_idx = self.passing_indices[new_pos]
        else:
            self.current_idx = (self.current_idx + 1) % self.n_cells
        self._update_all_fast(rebuild_collections=False)

    def on_toggle_accepted(self):
        if self.n_cells == 0:
            return
        self.accepted_mask[self.current_idx] = not self.accepted_mask[self.current_idx]
        self._on_filter_changed()

    def on_set_accepted(self, status: bool):
        if self.n_cells == 0:
            return
        self.accepted_mask[self.current_idx] = status
        self._on_filter_changed()

    def on_accept_all_passing(self):
        if self.n_cells == 0:
            return
        for i in range(self.n_cells):
            self.accepted_mask[i] = i in self.passing_indices
        self._on_filter_changed()

    def _on_fov_click(self, event):
        if event.inaxes != self.ax_fov or event.xdata is None or event.ydata is None or self.n_cells == 0:
            return
        click_pt = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.centroids - click_pt, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] < 20.0:  # click within 20px radius
            self.current_idx = nearest
            self._update_all_fast(rebuild_collections=False)

    # -------------------------------------------------------------------------
    # SAVING & CELL CURATOR HANDOFF
    # -------------------------------------------------------------------------
    def save_current_results(self):
        if self.cnm is None:
            messagebox.showwarning("Nothing to Save", "No active CNMF-E result loaded.")
            return

        est = self.cnm.estimates
        good_idx = np.where(self.accepted_mask)[0]
        bad_idx = np.where(~self.accepted_mask)[0]
        est.idx_components = good_idx
        est.idx_components_bad = bad_idx

        out_dir = self.current_output_dir or Path(self.ent_output_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = out_dir / "analysis_results.hdf5"
        mat_path = out_dir / "ms.mat"

        # Attach Cn and dims
        if self.fov_img is not None:
            est.Cn = self.fov_img
        est.dims = list(self.dims)

        save_hdf5_for_curator(self.cnm, hdf5_path, cn=self.fov_img, dims=self.dims)
        save_mat_for_curator(self.cnm, mat_path)

        self.current_hdf5_path = str(hdf5_path)
        self.set_status(f"Saved: {hdf5_path} ({len(good_idx)} accepted neurons)")
        messagebox.showinfo("Saved", f"Results successfully saved:\n• {hdf5_path}\n• {mat_path}")

    def open_in_cell_curator(self):
        if not self.current_hdf5_path or not os.path.isfile(self.current_hdf5_path):
            if self.cnm is not None:
                self.save_current_results()
            else:
                messagebox.showwarning("No HDF5", "Please run extraction or open an HDF5 result first.")
                return

        # Find CleanMiniscope cell_curator_gui.py
        candidate_paths = [
            Path(__file__).resolve().parents[3] / "CleanMiniscope" / "cell_curator_gui.py",
            Path("D:/Master_Gabriel/Pesquisa/CleanMiniscope/cell_curator_gui.py"),
            Path(os.environ.get("CLEANMINISCOPE_PATH", "")) / "cell_curator_gui.py",
        ]

        curator_script = None
        for p in candidate_paths:
            if p.is_file():
                curator_script = str(p.resolve())
                break

        if not curator_script:
            messagebox.showerror(
                "Not Found",
                "Could not locate CleanMiniscope/cell_curator_gui.py.\n"
                "Please verify CleanMiniscope is present at D:\\Master_Gabriel\\Pesquisa\\CleanMiniscope",
            )
            return

        self.set_status(f"Opening {os.path.basename(self.current_hdf5_path)} in Cell Curator...")
        subprocess.Popen([sys.executable, curator_script, self.current_hdf5_path])


def main():
    """Launch the Decovo native desktop application."""
    initial = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    app = DecovoApp(initial_path=initial)
    app.mainloop()


if __name__ == "__main__":
    main()
