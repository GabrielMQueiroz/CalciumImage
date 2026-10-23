# -*- coding: utf-8 -*-
"""Unified configuration & YAML preset management for Decovo."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .env import auto_n_processes


def _default_preset_path() -> Path:
    """Locate `miniscope_1p_default.yaml` in repo `presets/` or package `decovo/presets/`."""
    pkg_dir = Path(__file__).resolve().parent
    repo_preset = pkg_dir.parent / "presets" / "miniscope_1p_default.yaml"
    if repo_preset.exists():
        return repo_preset
    pkg_preset = pkg_dir / "presets" / "miniscope_1p_default.yaml"
    if pkg_preset.exists():
        return pkg_preset
    raise FileNotFoundError("Could not locate miniscope_1p_default.yaml preset.")


@dataclass
class PipelineConfig:
    """All parameters for the Decovo CaImAn CNMF-E pipeline.

    Defaults match `presets/miniscope_1p_default.yaml` and user-validated 1p parameters:
    - spatial_downsample = 0.6
    - gSig = 3.0, gSiz = 4 * gSig + 1 = 13
    - rf = 40, stride = 20, merge_thr = 0.7
    - min_SNR = 3.0, rval_thr = 0.95
    - max_shifts = (4, 4)
    """

    # --- Input / Output ---
    path_to_analyze: str = ""
    results_folder: str = "results_decovo"
    file_pattern: Optional[str] = None
    max_files: Optional[int] = None

    # --- Acquisition ---
    frame_rate: float = 20.0
    decay_time: float = 0.4

    # --- Preprocessing ---
    spatial_downsample: float = 0.6
    temporal_downsample: float = 1.0
    remove_background: bool = True
    bg_filter_size: int = 4
    bg_scale: float = 0.6
    chunk_frames: int = 1000

    # --- Motion Correction ---
    motion_correct: bool = True
    pw_rigid: bool = False
    gSig_filt: Tuple[int, int] = (3, 3)
    max_shifts: Tuple[int, int] = (4, 4)
    strides: Tuple[int, int] = (48, 48)
    overlaps: Tuple[int, int] = (24, 24)
    max_deviation_rigid: int = 3
    border_nan: str = "copy"

    # --- CNMF-E Source Extraction ---
    gSig: float = 3.0
    gSiz: Optional[int] = None  # If None, computed as int(round(4 * gSig + 1)) -> 13 for gSig=3
    merge_thr: float = 0.7
    p: int = 1
    tsub: int = 2
    ssub: int = 1
    rf: int = 40
    stride: int = 20
    nb: int = 0
    ssub_B: int = 2
    ring_size_factor: float = 1.4
    method_deconvolution: str = "oasis"

    # --- Threshold Selection ---
    # mode: 'auto_heuristic' | 'auto_grid' | 'auto_both' | 'manual'
    threshold_mode: str = "auto_heuristic"
    min_corr: Optional[float] = None
    min_pnr: Optional[float] = None
    grid_n_frames: int = 2000
    grid_corr_steps: int = 3
    grid_pnr_steps: int = 3

    # --- Component Evaluation ---
    min_SNR: float = 3.0
    rval_thr: float = 0.95
    use_cnn: bool = False

    # --- Performance ---
    n_processes: Optional[int] = None
    use_cuda: bool = False
    memory_fact: float = 0.8

    # --- Export & Cleanup ---
    save_hdf5: bool = True
    save_mat: bool = True
    save_figures: bool = True
    save_motion_corrected: bool = False
    cleanup_mmap: bool = False
    convert_mmap_to_npy: bool = False

    # --- Batch ---
    batch_mode: bool = False
    batch_pattern: str = "Animal*/pre,Animal*/post/d*"
    register_batch: bool = False
    register_max_dist: float = 15.0

    def __init__(
        self,
        path_to_analyze: str = "",
        results_folder: str = "results_decovo",
        file_pattern: Optional[str] = None,
        max_files: Optional[int] = None,
        frame_rate: float = 20.0,
        decay_time: float = 0.4,
        spatial_downsample: float = 0.6,
        temporal_downsample: float = 1.0,
        remove_background: bool = True,
        bg_filter_size: int = 4,
        bg_scale: float = 0.6,
        chunk_frames: int = 1000,
        motion_correct: bool = True,
        pw_rigid: bool = False,
        gSig_filt: Tuple[int, int] = (3, 3),
        max_shifts: Tuple[int, int] = (4, 4),
        strides: Tuple[int, int] = (48, 48),
        overlaps: Tuple[int, int] = (24, 24),
        max_deviation_rigid: int = 3,
        border_nan: str = "copy",
        gSig: float = 3.0,
        gSiz: Optional[int] = None,
        merge_thr: float = 0.7,
        p: int = 1,
        tsub: int = 2,
        ssub: int = 1,
        rf: int = 40,
        stride: int = 20,
        nb: int = 0,
        ssub_B: int = 2,
        ring_size_factor: float = 1.4,
        method_deconvolution: str = "oasis",
        threshold_mode: str = "auto_heuristic",
        min_corr: Optional[float] = None,
        min_pnr: Optional[float] = None,
        grid_n_frames: int = 2000,
        grid_corr_steps: int = 3,
        grid_pnr_steps: int = 3,
        min_SNR: float = 3.0,
        rval_thr: float = 0.95,
        use_cnn: bool = False,
        n_processes: Optional[int] = None,
        use_cuda: bool = False,
        memory_fact: float = 0.8,
        save_hdf5: bool = True,
        save_mat: bool = True,
        save_figures: bool = True,
        save_motion_corrected: bool = False,
        cleanup_mmap: bool = False,
        convert_mmap_to_npy: bool = False,
        batch_mode: bool = False,
        batch_pattern: str = "Animal*/pre,Animal*/post/d*",
        register_batch: bool = False,
        register_max_dist: float = 15.0,
        # Legacy aliases
        stride_cnmf: Optional[int] = None,
        gnb: Optional[int] = None,
        r_values_min: Optional[float] = None,
        auto_params: Optional[bool] = None,
        auto_method: Optional[str] = None,
        cuda_batch_size: int = 500,
    ):
        self.path_to_analyze = path_to_analyze
        self.results_folder = results_folder
        self.file_pattern = file_pattern
        self.max_files = max_files
        self.frame_rate = float(frame_rate)
        self.decay_time = float(decay_time)
        self.spatial_downsample = float(spatial_downsample)
        self.temporal_downsample = float(temporal_downsample)
        self.remove_background = bool(remove_background)
        self.bg_filter_size = int(bg_filter_size)
        self.bg_scale = float(bg_scale)
        self.chunk_frames = int(chunk_frames)
        self.motion_correct = bool(motion_correct)
        self.pw_rigid = bool(pw_rigid)
        self.gSig_filt = (int(gSig_filt[0]), int(gSig_filt[1]))
        self.max_shifts = (int(max_shifts[0]), int(max_shifts[1]))
        self.strides = (int(strides[0]), int(strides[1]))
        self.overlaps = (int(overlaps[0]), int(overlaps[1]))
        self.max_deviation_rigid = int(max_deviation_rigid)
        self.border_nan = str(border_nan)
        self.gSig = float(gSig)
        if isinstance(gSiz, (tuple, list)):
            self.gSiz = int(gSiz[0])
        elif gSiz is not None:
            self.gSiz = int(gSiz)
        else:
            self.gSiz = None
        self.merge_thr = float(merge_thr)
        self.p = int(p)
        self.tsub = int(tsub)
        self.ssub = int(ssub)
        self.rf = int(rf)
        self.stride = int(stride_cnmf if stride_cnmf is not None else stride)
        self.nb = int(gnb if gnb is not None else nb)
        self.ssub_B = int(ssub_B)
        self.ring_size_factor = float(ring_size_factor)
        self.method_deconvolution = str(method_deconvolution)
        self.threshold_mode = str(threshold_mode)
        self.min_corr = float(min_corr) if min_corr is not None else None
        self.min_pnr = float(min_pnr) if min_pnr is not None else None
        self.grid_n_frames = int(grid_n_frames)
        self.grid_corr_steps = int(grid_corr_steps)
        self.grid_pnr_steps = int(grid_pnr_steps)
        self.min_SNR = float(min_SNR)
        self.rval_thr = float(r_values_min if r_values_min is not None else rval_thr)
        self.use_cnn = bool(use_cnn)
        self.n_processes = int(n_processes) if n_processes is not None else auto_n_processes()
        self.use_cuda = bool(use_cuda)
        self.memory_fact = float(memory_fact)
        self.save_hdf5 = bool(save_hdf5)
        self.save_mat = bool(save_mat)
        self.save_figures = bool(save_figures)
        self.save_motion_corrected = bool(save_motion_corrected)
        self.cleanup_mmap = bool(cleanup_mmap)
        self.convert_mmap_to_npy = bool(convert_mmap_to_npy)
        self.batch_mode = bool(batch_mode)
        self.batch_pattern = str(batch_pattern)
        self.register_batch = bool(register_batch)
        self.register_max_dist = float(register_max_dist)
        self.cuda_batch_size = int(cuda_batch_size)

        if auto_params is not None or auto_method is not None:
            if auto_params is False:
                self.threshold_mode = "manual"
            elif auto_method:
                self.auto_method = auto_method

    @property
    def stride_cnmf(self) -> int:
        return self.stride

    @stride_cnmf.setter
    def stride_cnmf(self, val: int) -> None:
        self.stride = int(val)

    @property
    def gnb(self) -> int:
        return self.nb

    @gnb.setter
    def gnb(self, val: int) -> None:
        self.nb = int(val)

    @property
    def r_values_min(self) -> float:
        return self.rval_thr

    @r_values_min.setter
    def r_values_min(self, val: float) -> None:
        self.rval_thr = float(val)

    @property
    def auto_params(self) -> bool:
        return self.threshold_mode != "manual"

    @auto_params.setter
    def auto_params(self, val: bool) -> None:
        if not val:
            self.threshold_mode = "manual"
        elif self.threshold_mode == "manual":
            self.threshold_mode = "auto_heuristic"

    @property
    def auto_method(self) -> str:
        mapping = {
            "auto_heuristic": "heuristic",
            "auto_grid": "grid_search",
            "auto_both": "both",
            "manual": "heuristic",
        }
        return mapping.get(self.threshold_mode, "heuristic")

    @auto_method.setter
    def auto_method(self, val: str) -> None:
        mapping = {
            "heuristic": "auto_heuristic",
            "grid_search": "auto_grid",
            "both": "auto_both",
            "auto_heuristic": "auto_heuristic",
            "auto_grid": "auto_grid",
            "auto_both": "auto_both",
            "manual": "manual",
        }
        self.threshold_mode = mapping.get(val, "auto_heuristic")

    @property
    def gSiz_px(self) -> int:
        """Resolved scalar gSiz in pixels: explicit gSiz if set, else 4 * gSig + 1."""
        if self.gSiz is not None:
            return int(self.gSiz)
        return int(round(4 * float(self.gSig) + 1))

    @property
    def gSiz_tuple(self) -> Tuple[int, int]:
        d = self.gSiz_px
        return (d, d)

    @property
    def gSig_tuple(self) -> Tuple[float, float]:
        g = float(self.gSig)
        return (g, g)

    @property
    def frate(self) -> float:
        return float(self.frame_rate) * float(self.temporal_downsample)

    @property
    def path_to_results(self) -> str:
        if not self.path_to_analyze:
            return self.results_folder
        if os.path.isabs(self.results_folder):
            return self.results_folder
        return os.path.join(self.path_to_analyze, self.results_folder)

    # --- Stage parameter dictionaries for deterministic cache hashing ---

    def stage_params(self, stage: str) -> Dict[str, Any]:
        """Return the subset of parameters that affect `stage`."""
        if stage == "prepare":
            return {
                "spatial_downsample": float(self.spatial_downsample),
                "temporal_downsample": float(self.temporal_downsample),
                "remove_background": bool(self.remove_background),
                "bg_filter_size": int(self.bg_filter_size),
                "bg_scale": float(self.bg_scale),
                "chunk_frames": int(self.chunk_frames),
                "max_files": self.max_files,
                "file_pattern": self.file_pattern,
            }
        if stage == "motion":
            return {
                "motion_correct": bool(self.motion_correct),
                "pw_rigid": bool(self.pw_rigid),
                "gSig_filt": list(self.gSig_filt),
                "max_shifts": list(self.max_shifts),
                "strides": list(self.strides),
                "overlaps": list(self.overlaps),
                "max_deviation_rigid": int(self.max_deviation_rigid),
                "border_nan": str(self.border_nan),
                "decay_time": float(self.decay_time),
            }
        if stage == "summary":
            return {
                "gSig": float(self.gSig),
            }
        if stage == "grid":
            return {
                "frame_rate": float(self.frame_rate),
                "decay_time": float(self.decay_time),
                "gSig": float(self.gSig),
                "gSiz": int(self.gSiz_px),
                "merge_thr": float(self.merge_thr),
                "tsub": int(self.tsub),
                "ssub": int(self.ssub),
                "rf": int(self.rf),
                "stride": int(self.stride),
                "nb": int(self.nb),
                "ssub_B": int(self.ssub_B),
                "ring_size_factor": float(self.ring_size_factor),
                "method_deconvolution": str(self.method_deconvolution),
                "min_SNR": float(self.min_SNR),
                "rval_thr": float(self.rval_thr),
                "grid_n_frames": int(self.grid_n_frames),
                "grid_corr_steps": int(self.grid_corr_steps),
                "grid_pnr_steps": int(self.grid_pnr_steps),
            }
        if stage == "extract":
            return {
                "frame_rate": float(self.frame_rate),
                "decay_time": float(self.decay_time),
                "gSig": float(self.gSig),
                "gSiz": int(self.gSiz_px),
                "merge_thr": float(self.merge_thr),
                "p": int(self.p),
                "tsub": int(self.tsub),
                "ssub": int(self.ssub),
                "rf": int(self.rf),
                "stride": int(self.stride),
                "nb": int(self.nb),
                "ssub_B": int(self.ssub_B),
                "ring_size_factor": float(self.ring_size_factor),
                "method_deconvolution": str(self.method_deconvolution),
                "min_SNR": float(self.min_SNR),
                "rval_thr": float(self.rval_thr),
                "use_cnn": bool(self.use_cnn),
                "min_corr": float(self.min_corr) if self.min_corr is not None else None,
                "min_pnr": float(self.min_pnr) if self.min_pnr is not None else None,
            }
        if stage == "export":
            return {
                "save_hdf5": bool(self.save_hdf5),
                "save_mat": bool(self.save_mat),
                "save_figures": bool(self.save_figures),
            }
        raise ValueError(f"Unknown stage: {stage!r}")

    # --- YAML serialization ---

    def to_nested_dict(self) -> Dict[str, Any]:
        """Convert to nested preset dictionary matching `miniscope_1p_default.yaml`."""
        return {
            "acquisition": {
                "frame_rate": float(self.frame_rate),
                "decay_time": float(self.decay_time),
            },
            "preprocess": {
                "spatial_downsample": float(self.spatial_downsample),
                "temporal_downsample": float(self.temporal_downsample),
                "remove_background": bool(self.remove_background),
                "bg_filter_size": int(self.bg_filter_size),
                "bg_scale": float(self.bg_scale),
                "chunk_frames": int(self.chunk_frames),
            },
            "motion": {
                "enabled": bool(self.motion_correct),
                "pw_rigid": bool(self.pw_rigid),
                "gSig_filt": list(self.gSig_filt),
                "max_shifts": list(self.max_shifts),
                "strides": list(self.strides),
                "overlaps": list(self.overlaps),
                "max_deviation_rigid": int(self.max_deviation_rigid),
                "border_nan": str(self.border_nan),
            },
            "cnmfe": {
                "gSig": float(self.gSig),
                "gSiz": int(self.gSiz_px),
                "merge_thr": float(self.merge_thr),
                "p": int(self.p),
                "tsub": int(self.tsub),
                "ssub": int(self.ssub),
                "rf": int(self.rf),
                "stride": int(self.stride),
                "nb": int(self.nb),
                "ssub_B": int(self.ssub_B),
                "ring_size_factor": float(self.ring_size_factor),
                "method_deconvolution": str(self.method_deconvolution),
            },
            "thresholds": {
                "mode": str(self.threshold_mode),
                "min_corr": float(self.min_corr) if self.min_corr is not None else None,
                "min_pnr": float(self.min_pnr) if self.min_pnr is not None else None,
                "grid_n_frames": int(self.grid_n_frames),
                "grid_corr_steps": int(self.grid_corr_steps),
                "grid_pnr_steps": int(self.grid_pnr_steps),
            },
            "evaluation": {
                "min_SNR": float(self.min_SNR),
                "rval_thr": float(self.rval_thr),
                "use_cnn": bool(self.use_cnn),
            },
            "export": {
                "hdf5": bool(self.save_hdf5),
                "mat": bool(self.save_mat),
                "figures": bool(self.save_figures),
                "save_motion_corrected": bool(self.save_motion_corrected),
                "cleanup_mmap": bool(self.cleanup_mmap),
                "convert_mmap_to_npy": bool(self.convert_mmap_to_npy),
            },
        }

    def to_yaml_string(self) -> str:
        return yaml.safe_dump(self.to_nested_dict(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        """Write resolved configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_nested_dict(), f, sort_keys=False)
        return target

    def copy(self, **overrides: Any) -> "PipelineConfig":
        """Return a copy of this config with optional field overrides."""
        c = copy.deepcopy(self)
        _apply_overrides(c, overrides)
        return c


def _apply_nested_dict(cfg: PipelineConfig, data: Dict[str, Any]) -> None:
    """Apply nested YAML dictionary onto `cfg`."""
    acq = data.get("acquisition", {}) or {}
    if "frame_rate" in acq:
        cfg.frame_rate = float(acq["frame_rate"])
    if "decay_time" in acq:
        cfg.decay_time = float(acq["decay_time"])

    prep = data.get("preprocess", {}) or {}
    if "spatial_downsample" in prep:
        cfg.spatial_downsample = float(prep["spatial_downsample"])
    if "temporal_downsample" in prep:
        cfg.temporal_downsample = float(prep["temporal_downsample"])
    if "remove_background" in prep:
        cfg.remove_background = bool(prep["remove_background"])
    if "bg_filter_size" in prep:
        cfg.bg_filter_size = int(prep["bg_filter_size"])
    if "bg_scale" in prep:
        cfg.bg_scale = float(prep["bg_scale"])
    if "chunk_frames" in prep:
        cfg.chunk_frames = int(prep["chunk_frames"])

    mot = data.get("motion", {}) or {}
    if "enabled" in mot:
        cfg.motion_correct = bool(mot["enabled"])
    if "pw_rigid" in mot:
        cfg.pw_rigid = bool(mot["pw_rigid"])
    if "gSig_filt" in mot:
        cfg.gSig_filt = (int(mot["gSig_filt"][0]), int(mot["gSig_filt"][1]))
    if "max_shifts" in mot:
        cfg.max_shifts = (int(mot["max_shifts"][0]), int(mot["max_shifts"][1]))
    if "strides" in mot:
        cfg.strides = (int(mot["strides"][0]), int(mot["strides"][1]))
    if "overlaps" in mot:
        cfg.overlaps = (int(mot["overlaps"][0]), int(mot["overlaps"][1]))
    if "max_deviation_rigid" in mot:
        cfg.max_deviation_rigid = int(mot["max_deviation_rigid"])
    if "border_nan" in mot:
        cfg.border_nan = str(mot["border_nan"])

    cnmfe = data.get("cnmfe", {}) or {}
    gsig_changed = False
    if "gSig" in cnmfe:
        cfg.gSig = float(cnmfe["gSig"])
        gsig_changed = True
    if "gSiz" in cnmfe and cnmfe["gSiz"] is not None:
        raw_gsiz = cnmfe["gSiz"]
        cfg.gSiz = int(raw_gsiz[0]) if isinstance(raw_gsiz, (list, tuple)) else int(raw_gsiz)
    elif gsig_changed:
        cfg.gSiz = None  # compute as 4*gSig + 1
    if "merge_thr" in cnmfe:
        cfg.merge_thr = float(cnmfe["merge_thr"])
    if "p" in cnmfe:
        cfg.p = int(cnmfe["p"])
    if "tsub" in cnmfe:
        cfg.tsub = int(cnmfe["tsub"])
    if "ssub" in cnmfe:
        cfg.ssub = int(cnmfe["ssub"])
    if "rf" in cnmfe:
        cfg.rf = int(cnmfe["rf"])
    if "stride" in cnmfe:
        cfg.stride = int(cnmfe["stride"])
    if "nb" in cnmfe:
        cfg.nb = int(cnmfe["nb"])
    if "ssub_B" in cnmfe:
        cfg.ssub_B = int(cnmfe["ssub_B"])
    if "ring_size_factor" in cnmfe:
        cfg.ring_size_factor = float(cnmfe["ring_size_factor"])
    if "method_deconvolution" in cnmfe:
        cfg.method_deconvolution = str(cnmfe["method_deconvolution"])

    thr = data.get("thresholds", {}) or {}
    if "mode" in thr:
        cfg.threshold_mode = str(thr["mode"])
    if "min_corr" in thr:
        cfg.min_corr = float(thr["min_corr"]) if thr["min_corr"] is not None else None
    if "min_pnr" in thr:
        cfg.min_pnr = float(thr["min_pnr"]) if thr["min_pnr"] is not None else None
    if "grid_n_frames" in thr:
        cfg.grid_n_frames = int(thr["grid_n_frames"])
    if "grid_corr_steps" in thr:
        cfg.grid_corr_steps = int(thr["grid_corr_steps"])
    if "grid_pnr_steps" in thr:
        cfg.grid_pnr_steps = int(thr["grid_pnr_steps"])

    ev = data.get("evaluation", {}) or {}
    if "min_SNR" in ev:
        cfg.min_SNR = float(ev["min_SNR"])
    if "rval_thr" in ev:
        cfg.rval_thr = float(ev["rval_thr"])
    if "use_cnn" in ev:
        cfg.use_cnn = bool(ev["use_cnn"])

    exp = data.get("export", {}) or {}
    if "hdf5" in exp:
        cfg.save_hdf5 = bool(exp["hdf5"])
    if "mat" in exp:
        cfg.save_mat = bool(exp["mat"])
    if "figures" in exp:
        cfg.save_figures = bool(exp["figures"])
    if "save_motion_corrected" in exp:
        cfg.save_motion_corrected = bool(exp["save_motion_corrected"])
    if "cleanup_mmap" in exp:
        cfg.cleanup_mmap = bool(exp["cleanup_mmap"])
    if "convert_mmap_to_npy" in exp:
        cfg.convert_mmap_to_npy = bool(exp["convert_mmap_to_npy"])

    # Also support flat top-level keys if present in YAML
    flat_ignore = {"acquisition", "preprocess", "motion", "cnmfe", "thresholds", "evaluation", "export"}
    flat_overrides = {k: v for k, v in data.items() if k not in flat_ignore}
    if flat_overrides:
        _apply_overrides(cfg, flat_overrides)


def _apply_overrides(cfg: PipelineConfig, overrides: Dict[str, Any]) -> None:
    """Apply flat keyword overrides onto `cfg`."""
    alias_map = {
        "stride_cnmf": "stride",
        "gnb": "nb",
        "r_values_min": "rval_thr",
        "motion_enabled": "motion_correct",
    }
    gsig_overridden = "gSig" in overrides
    gsiz_overridden = "gSiz" in overrides

    for k, v in overrides.items():
        target_key = alias_map.get(k, k)
        if hasattr(cfg, target_key):
            setattr(cfg, target_key, v)
        else:
            raise TypeError(f"Unknown PipelineConfig parameter: {k!r}")

    if gsig_overridden and not gsiz_overridden:
        # When gSig is overridden without an explicit gSiz override, recompute 4*gSig + 1
        cfg.gSiz = int(round(4 * float(cfg.gSig) + 1))
    elif isinstance(cfg.gSiz, (tuple, list)):
        cfg.gSiz = int(cfg.gSiz[0])


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> PipelineConfig:
    """Load `PipelineConfig` from a YAML preset (default: `miniscope_1p_default.yaml`) with overrides."""
    preset_path = Path(path) if path is not None else _default_preset_path()
    with open(preset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = PipelineConfig()
    _apply_nested_dict(cfg, data)
    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg


def save_config(cfg: PipelineConfig, path: str | Path) -> Path:
    """Save `PipelineConfig` to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return p

