# -*- coding: utf-8 -*-
"""Input discovery and memory-bounded chunk streaming for AVI chunks and multi-page TIFF stacks."""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np
import tifffile

logger = logging.getLogger("decovo")


@dataclass
class InputSpec:
    """Metadata describing discovered session recording input(s)."""

    kind: Literal["avi", "tiff"]
    files: List[str]
    n_frames: int
    frame_shape: Tuple[int, int]
    dtype: np.dtype
    fps_from_metadata: Optional[float] = None

    @property
    def estimated_raw_bytes(self) -> int:
        h, w = self.frame_shape
        return int(self.n_frames) * int(h) * int(w) * int(np.dtype(self.dtype).itemsize)

    @property
    def estimated_float32_gb(self) -> float:
        h, w = self.frame_shape
        return (int(self.n_frames) * int(h) * int(w) * 4) / (1024 ** 3)


def _natural_sort_key(fname: str) -> Tuple[int, str]:
    name = os.path.basename(fname)
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else 0, name.lower())


def _extract_tiff_fps(tif: tifffile.TiffFile) -> Optional[float]:
    """Attempt to extract acquisition frame rate (Hz) from ImageJ, OME-XML, or TIFF page tags."""
    # 1. ImageJ metadata
    ij = getattr(tif, "imagej_metadata", None)
    if isinstance(ij, dict):
        if "fps" in ij and ij["fps"]:
            try:
                fps = float(ij["fps"])
                if fps > 0:
                    return round(fps, 4)
            except (ValueError, TypeError):
                pass
        if "finterval" in ij and ij["finterval"]:
            try:
                fint = float(ij["finterval"])
                if fint > 0:
                    return round(1.0 / fint, 4)
            except (ValueError, TypeError):
                pass
        info_str = ij.get("Info") or ij.get("info")
        if isinstance(info_str, str):
            m = re.search(r"(?:FrameRate|fps|frame_rate)\s*[=:]\s*([0-9.]+)", info_str, re.IGNORECASE)
            if m:
                try:
                    fps = float(m.group(1))
                    if fps > 0:
                        return round(fps, 4)
                except ValueError:
                    pass

    # 2. OME-XML metadata
    ome = getattr(tif, "ome_metadata", None)
    if isinstance(ome, str) and "TimeIncrement" in ome:
        try:
            root = ET.fromstring(ome)
            for elem in root.iter():
                if elem.tag.endswith("Pixels") and "TimeIncrement" in elem.attrib:
                    inc = float(elem.attrib["TimeIncrement"])
                    unit = elem.attrib.get("TimeIncrementUnit", "s").strip().lower()
                    if inc > 0:
                        if unit in ("ms", "millisecond", "milliseconds"):
                            inc *= 1e-3
                        elif unit in ("us", "µs", "microsecond", "microseconds"):
                            inc *= 1e-6
                        return round(1.0 / inc, 4)
        except Exception:
            pass

    # 3. Page ImageDescription tag
    try:
        if len(tif.pages) > 0:
            desc_tag = tif.pages[0].tags.get("ImageDescription")
            if desc_tag and isinstance(desc_tag.value, str):
                text = desc_tag.value
                m_fps = re.search(r"(?:fps|frame_rate|FrameRate)\s*[=:]\s*([0-9.]+)", text)
                if m_fps:
                    val = float(m_fps.group(1))
                    if val > 0:
                        return round(val, 4)
                m_int = re.search(r"finterval\s*[=:]\s*([0-9.]+)", text)
                if m_int:
                    val = float(m_int.group(1))
                    if val > 0:
                        return round(1.0 / val, 4)
    except Exception:
        pass

    return None


def _normalize_frame_shape_and_count(raw_shape: Tuple[int, ...]) -> Tuple[int, Tuple[int, int]]:
    """Given a TIFF series/page shape, squeeze singleton channel/Z dimensions (and RGB trailing axis)
    and return (n_frames, (H, W)).
    """
    shape = tuple(int(d) for d in raw_shape)
    # Detect trailing RGB/RGBA channel axis: e.g. (H, W, 3) or (T, H, W, 3)
    if len(shape) >= 3 and shape[-1] in (3, 4) and shape[-2] > 4 and shape[-3] > 4:
        shape = shape[:-1]

    if len(shape) == 2:
        return 1, (shape[0], shape[1])
    if len(shape) == 3:
        return shape[0], (shape[1], shape[2])
    # 4D+ hyperstack (e.g., (T, 1, H, W) or (1, T, H, W) or (T, 1, 1, H, W))
    spatial = (shape[-2], shape[-1])
    leading = [d for d in shape[:-2] if d > 1]
    if not leading:
        n_frames = 1
    else:
        n_frames = int(np.prod(leading))
    return n_frames, spatial


def _squeeze_to_3d(arr: np.ndarray, expected_hw: Tuple[int, int]) -> np.ndarray:
    """Convert an array from TIFF read into (N, H, W), converting RGB/RGBA to grayscale if needed."""
    h, w = expected_hw
    # Handle trailing RGB/RGBA channel axis
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4) and arr.shape[-3:-1] == (h, w):
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim == 2 and arr.shape == (h, w):
        return arr[np.newaxis, :, :]
    if arr.ndim == 3 and arr.shape[1:] == (h, w):
        return arr
    # Squeeze singleton axes except the last two spatial axes
    if arr.shape[-2:] == (h, w):
        return arr.reshape((-1, h, w))
    raise ValueError(f"Unexpected frame array shape {arr.shape}; expected spatial dims {expected_hw}")


def _inspect_tiff_files(files: List[str]) -> InputSpec:
    total_frames = 0
    frame_shape: Optional[Tuple[int, int]] = None
    dtype: Optional[np.dtype] = None
    fps: Optional[float] = None

    for fpath in files:
        with tifffile.TiffFile(fpath) as tif:
            if fps is None:
                fps = _extract_tiff_fps(tif)

            if tif.series:
                s0 = tif.series[0]
                n_f, hw = _normalize_frame_shape_and_count(s0.shape)
                dt = np.dtype(s0.dtype)
                # Guard against multi-page stacks where series only lists 1 page
                if n_f == 1 and len(tif.pages) > 1:
                    p0 = tif.pages[0]
                    _, hw = _normalize_frame_shape_and_count(p0.shape)
                    n_f = len(tif.pages)
            else:
                p0 = tif.pages[0]
                _, hw = _normalize_frame_shape_and_count(p0.shape)
                n_f = len(tif.pages)
                dt = np.dtype(p0.dtype)

            if frame_shape is None:
                frame_shape = hw
                dtype = dt
            elif hw != frame_shape:
                raise ValueError(
                    f"Inconsistent TIFF frame dimensions: {fpath} has {hw}, expected {frame_shape}"
                )
            total_frames += n_f

    assert frame_shape is not None and dtype is not None
    return InputSpec(
        kind="tiff",
        files=[str(Path(f).resolve()) for f in files],
        n_frames=total_frames,
        frame_shape=frame_shape,
        dtype=dtype,
        fps_from_metadata=fps,
    )


def _inspect_avi_files(files: List[str]) -> InputSpec:
    total_frames = 0
    frame_shape: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None

    for fpath in files:
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open AVI file: {fpath}")
        try:
            n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if n_f <= 0 or w <= 0 or h <= 0:
                # Fallback: count frames by reading
                n_f = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_shape is None:
                        h, w = frame.shape[:2]
                    n_f += 1
            if frame_shape is None:
                frame_shape = (h, w)
            if fps is None and file_fps > 0:
                fps = round(file_fps, 4)
            total_frames += n_f
        finally:
            cap.release()

    if frame_shape is None or total_frames == 0:
        raise ValueError(f"AVI files in {files[0]} contain 0 readable frames.")

    return InputSpec(
        kind="avi",
        files=[str(Path(f).resolve()) for f in files],
        n_frames=total_frames,
        frame_shape=frame_shape,
        dtype=np.dtype(np.uint8),
        fps_from_metadata=fps,
    )


def discover_inputs(
    folder_or_file: str | Path,
    pattern: Optional[str] = None,
    max_files: Optional[int] = None,
) -> InputSpec:
    """Auto-detect input movies in `folder_or_file`: either `msCam*.avi`/`ms*.avi` OR `*.tif`/`*.tiff`.

    Raises a plain-language error if both or neither are found.
    """
    target = Path(folder_or_file)
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")

    if target.is_file():
        ext = target.suffix.lower()
        if ext in (".tif", ".tiff"):
            return _inspect_tiff_files([str(target)])
        if ext == ".avi":
            return _inspect_avi_files([str(target)])
        raise ValueError(f"Unsupported input file extension {ext!r}. Expected .tif, .tiff, or .avi.")

    all_entries = [
        f for f in os.listdir(target)
        if (target / f).is_file() and not f.startswith("prep_") and f != "motion_corrected.avi"
    ]

    if pattern:
        import fnmatch
        matched = [f for f in all_entries if fnmatch.fnmatch(f, pattern)]
        avi_files = [f for f in matched if f.lower().endswith(".avi")]
        tif_files = [f for f in matched if f.lower().endswith((".tif", ".tiff"))]
        # If the user left default pattern="msCam*.avi" on a TIFF folder, fallback to auto-detection
        if not avi_files and not tif_files and pattern == "msCam*.avi":
            pattern = None
        else:
            if avi_files and tif_files:
                raise ValueError(
                    f"Both AVI ({len(avi_files)}) and TIFF ({len(tif_files)}) files matched pattern "
                    f"{pattern!r} in {target}. Please keep only one format per session folder or set a specific file_pattern."
                )
            if avi_files:
                avi_sorted = sorted(avi_files, key=_natural_sort_key)
                if max_files is not None:
                    avi_sorted = avi_sorted[:max_files]
                return _inspect_avi_files([str(target / f) for f in avi_sorted])
            if tif_files:
                tif_sorted = sorted(tif_files, key=_natural_sort_key)
                if max_files is not None:
                    tif_sorted = tif_sorted[:max_files]
                return _inspect_tiff_files([str(target / f) for f in tif_sorted])

    # Auto-detect AVI chunks (msCam*.avi, ms*.avi, or any .avi) and TIFF stacks (*.tif, *.tiff)
    avi_candidates = [
        f for f in all_entries
        if f.lower().endswith(".avi") and (f.startswith("msCam") or f.startswith("ms"))
    ]
    if not avi_candidates:
        avi_candidates = [f for f in all_entries if f.lower().endswith(".avi")]

    tif_candidates = [
        f for f in all_entries
        if f.lower().endswith((".tif", ".tiff"))
    ]

    if avi_candidates and tif_candidates:
        raise ValueError(
            f"Ambiguous input folder {target}: found both AVI files ({avi_candidates[:3]}) "
            f"and TIFF files ({tif_candidates[:3]}). Please place only one recording format in the session folder "
            f"or specify `file_pattern` explicitly."
        )

    if not avi_candidates and not tif_candidates:
        raise FileNotFoundError(
            f"No calcium imaging files ( neither msCam*.avi / *.avi nor *.tif / *.tiff ) found in {target}."
        )

    if tif_candidates:
        tif_sorted = sorted(tif_candidates, key=_natural_sort_key)
        if max_files is not None:
            tif_sorted = tif_sorted[:max_files]
        spec = _inspect_tiff_files([str(target / f) for f in tif_sorted])
        logger.info(
            "Discovered TIFF input: %d file(s), %d frames, shape=%s, dtype=%s, metadata_fps=%s",
            len(spec.files), spec.n_frames, spec.frame_shape, spec.dtype, spec.fps_from_metadata
        )
        return spec

    avi_sorted = sorted(avi_candidates, key=_natural_sort_key)
    if max_files is not None:
        avi_sorted = avi_sorted[:max_files]
    spec = _inspect_avi_files([str(target / f) for f in avi_sorted])
    logger.info(
        "Discovered AVI input: %d file(s), %d frames, shape=%s, dtype=%s, metadata_fps=%s",
        len(spec.files), spec.n_frames, spec.frame_shape, spec.dtype, spec.fps_from_metadata
    )
    return spec


def _iter_single_tiff_chunks(fpath: str, expected_hw: Tuple[int, int], chunk_size: int) -> Iterator[np.ndarray]:
    """Stream (n, H, W) float32 blocks from a single TIFF file without loading the full stack."""
    with tifffile.TiffFile(fpath) as tif:
        n_pages = len(tif.pages)
        s0 = tif.series[0] if tif.series else None
        s0_frames = _normalize_frame_shape_and_count(s0.shape)[0] if s0 is not None else n_pages

        # Case 1: ImageJ contiguous hyperstack where all frames are stored in page 0's strip/offset
        if n_pages == 1 and s0_frames > 1:
            try:
                mm = tifffile.memmap(fpath, mode="r")
                mm_3d = _squeeze_to_3d(mm, expected_hw)
                for start in range(0, mm_3d.shape[0], chunk_size):
                    end = min(start + chunk_size, mm_3d.shape[0])
                    yield np.asarray(mm_3d[start:end], dtype=np.float32)
                del mm, mm_3d
                return
            except Exception:
                # Fallback if compressed
                full = _squeeze_to_3d(tif.asarray(), expected_hw)
                for start in range(0, full.shape[0], chunk_size):
                    end = min(start + chunk_size, full.shape[0])
                    yield np.asarray(full[start:end], dtype=np.float32)
                return

        # Case 2: Standard multi-page TIFF or BigTIFF — read page-range by page-range
        for start in range(0, n_pages, chunk_size):
            end = min(start + chunk_size, n_pages)
            block = tif.asarray(key=range(start, end))
            block_3d = _squeeze_to_3d(block, expected_hw)
            yield np.asarray(block_3d, dtype=np.float32)


def _iter_avi_chunks(files: List[str], expected_hw: Tuple[int, int], chunk_size: int) -> Iterator[np.ndarray]:
    """Stream (n, H, W) float32 blocks across one or more AVI files."""
    h, w = expected_hw
    buffer: List[np.ndarray] = []

    for fpath in files:
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open AVI file: {fpath}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                buffer.append(gray.astype(np.float32, copy=False))
                if len(buffer) >= chunk_size:
                    yield np.stack(buffer, axis=0).astype(np.float32, copy=False)
                    buffer.clear()
        finally:
            cap.release()

    if buffer:
        yield np.stack(buffer, axis=0).astype(np.float32, copy=False)
        buffer.clear()


def iter_frames_chunks(spec: InputSpec, chunk_size: int = 1000) -> Iterator[np.ndarray]:
    """Yield `(n, H, W)` float32 blocks of up to `chunk_size` frames without loading the entire movie."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if spec.kind == "tiff":
        carry: Optional[np.ndarray] = None
        for fpath in spec.files:
            for block in _iter_single_tiff_chunks(fpath, spec.frame_shape, chunk_size):
                if carry is not None:
                    block = np.concatenate([carry, block], axis=0)
                    carry = None
                while block.shape[0] >= chunk_size:
                    yield block[:chunk_size]
                    block = block[chunk_size:]
                if block.shape[0] > 0:
                    carry = block
        if carry is not None and carry.shape[0] > 0:
            yield carry
    elif spec.kind == "avi":
        yield from _iter_avi_chunks(spec.files, spec.frame_shape, chunk_size)
    else:
        raise ValueError(f"Unsupported InputSpec kind: {spec.kind!r}")
