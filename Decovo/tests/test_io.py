# -*- coding: utf-8 -*-
"""Tests for `decovo.io`: multi-page TIFF, BigTIFF, ImageJ hyperstack, metadata FPS, and AVI chunks."""

import cv2
import numpy as np
import pytest
import tifffile

from decovo.io import discover_inputs, iter_frames_chunks


def _write_avi(path, frames, fps=20.0):
    n, h, w = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=False)
    for i in range(n):
        writer.write(frames[i])
    writer.release()


def test_discover_and_stream_multipage_uint16_tiff_with_imagej_fps(tmp_path):
    rng = np.random.default_rng(123)
    data = rng.integers(100, 4000, size=(65, 48, 64), dtype=np.uint16)
    tif_path = tmp_path / "recording.tif"
    # ImageJ finterval = 1 / 15.0 s -> 15.0 Hz
    tifffile.imwrite(
        str(tif_path),
        data,
        imagej=True,
        metadata={"axes": "TYX", "finterval": 1.0 / 15.0},
    )

    spec = discover_inputs(tmp_path)
    assert spec.kind == "tiff"
    assert spec.n_frames == 65
    assert spec.frame_shape == (48, 64)
    assert spec.dtype == np.dtype(np.uint16)
    assert spec.fps_from_metadata == pytest.approx(15.0, rel=1e-2)

    chunks = list(iter_frames_chunks(spec, chunk_size=20))
    assert [c.shape for c in chunks] == [(20, 48, 64), (20, 48, 64), (20, 48, 64), (5, 48, 64)]
    assert all(c.dtype == np.float32 for c in chunks)
    reconstructed = np.concatenate(chunks, axis=0)
    np.testing.assert_array_equal(reconstructed, data.astype(np.float32))


def test_bigtiff_and_imagej_hyperstack_singleton_axes(tmp_path):
    rng = np.random.default_rng(456)
    # 4D hyperstack with singleton channel axis: (T, 1, H, W)
    hyper = rng.integers(0, 1000, size=(30, 1, 32, 40), dtype=np.uint16)
    tif_path = tmp_path / "hyper_big.tiff"
    tifffile.imwrite(
        str(tif_path),
        hyper,
        bigtiff=True,
        ome=True,
        metadata={"axes": "TCYX", "TimeIncrement": 0.05, "TimeIncrementUnit": "s"},
    )

    spec = discover_inputs(tif_path)
    assert spec.kind == "tiff"
    assert spec.n_frames == 30
    assert spec.frame_shape == (32, 40)
    assert spec.fps_from_metadata == pytest.approx(20.0, rel=1e-2)

    chunks = list(iter_frames_chunks(spec, chunk_size=12))
    reconstructed = np.concatenate(chunks, axis=0)
    assert reconstructed.shape == (30, 32, 40)
    np.testing.assert_array_equal(reconstructed, hyper[:, 0, :, :].astype(np.float32))


def test_discover_avi_natural_sort_and_stream(tmp_path):
    rng = np.random.default_rng(789)
    # Create msCam1.avi, msCam2.avi, msCam10.avi to verify natural numeric sort
    for idx, n_f in [(1, 15), (2, 15), (10, 10)]:
        frames = rng.integers(20, 200, size=(n_f, 32, 48), dtype=np.uint8)
        _write_avi(tmp_path / f"msCam{idx}.avi", frames, fps=20.0)

    spec = discover_inputs(tmp_path)
    assert spec.kind == "avi"
    assert [f.split("\\")[-1].split("/")[-1] for f in spec.files] == [
        "msCam1.avi",
        "msCam2.avi",
        "msCam10.avi",
    ]
    assert spec.n_frames == 40
    assert spec.frame_shape == (32, 48)

    chunks = list(iter_frames_chunks(spec, chunk_size=18))
    assert [c.shape for c in chunks] == [(18, 32, 48), (18, 32, 48), (4, 32, 48)]


def test_discover_errors_on_empty_or_ambiguous_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_inputs(tmp_path)

    # Ambiguous folder with both AVI and TIFF
    dummy = np.zeros((5, 16, 16), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "rec.tif"), dummy)
    _write_avi(tmp_path / "msCam1.avi", dummy)

    with pytest.raises(ValueError, match="Ambiguous"):
        discover_inputs(tmp_path)


def test_rgb_tiff_and_fractional_temporal_downsample(tmp_path):
    from decovo.stages import _resize_block_cpu

    # 1. RGB multi-page TIFF (T, H, W, 3)
    rgb = np.zeros((25, 40, 56, 3), dtype=np.uint8)
    rgb[:, :, :, 0] = 60
    rgb[:, :, :, 1] = 120
    rgb[:, :, :, 2] = 180
    rgb_path = tmp_path / "rgb_stack.tif"
    tifffile.imwrite(str(rgb_path), rgb, photometric="rgb")

    spec = discover_inputs(rgb_path)
    assert spec.kind == "tiff"
    assert spec.n_frames == 25
    assert spec.frame_shape == (40, 56)

    chunks = list(iter_frames_chunks(spec, chunk_size=10))
    assert [c.shape for c in chunks] == [(10, 40, 56), (10, 40, 56), (5, 40, 56)]
    np.testing.assert_allclose(chunks[0], 120.0)

    # 2. Fractional temporal downsampling (e.g. 0.75) scales frame count accurately
    block = np.ones((100, 40, 56), dtype=np.float32)
    out = _resize_block_cpu(block, spatial_ds=0.5, temporal_ds=0.75)
    assert out.shape == (75, 20, 28)

