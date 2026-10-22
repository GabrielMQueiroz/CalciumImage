# -*- coding: utf-8 -*-
"""Test preprocessing equivalence between multi-page TIFF and lossless AVI inputs."""

import cv2
import numpy as np
import tifffile

from decovo.config import load_config
from decovo.io import discover_inputs
from decovo.stages import ProgressReporter, stage_prepare


def test_tiff_and_avi_produce_equivalent_prep_chunks(tmp_path):
    rng = np.random.default_rng(2026)
    n_frames, h, w = 40, 60, 80
    # Smooth synthetic movie so codec is lossless with raw/FFV1 or PNG/HFYU
    movie = rng.integers(30, 220, size=(n_frames, h, w), dtype=np.uint8)

    tiff_dir = tmp_path / "session_tiff"
    avi_dir = tmp_path / "session_avi"
    tiff_dir.mkdir()
    avi_dir.mkdir()

    # 1. Write as multi-page TIFF
    tifffile.imwrite(str(tiff_dir / "recording.tif"), movie)

    # 2. Write as uncompressed / lossless AVI (try FFV1 or HFYU or raw 0, fallback check)
    avi_path = avi_dir / "msCam1.avi"
    wrote_lossless = False
    for codec in ("FFV1", "HFYU", "Y800"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(avi_path), fourcc, 20.0, (w, h), isColor=False)
        if writer.isOpened():
            for i in range(n_frames):
                writer.write(movie[i])
            writer.release()
            spec_check = discover_inputs(avi_dir)
            if spec_check.n_frames == n_frames:
                wrote_lossless = True
                break
    if not wrote_lossless:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(avi_path), fourcc, 20.0, (w, h), isColor=False)
        for i in range(n_frames):
            writer.write(movie[i])
        writer.release()

    cfg = load_config(spatial_downsample=0.6, remove_background=True, chunk_frames=15)
    spec_tiff = discover_inputs(tiff_dir)
    spec_avi = discover_inputs(avi_dir)

    work_tiff = tiff_dir / "results_decovo"
    work_avi = avi_dir / "results_decovo"

    prep_tiff = stage_prepare(spec_tiff, cfg, work_tiff, ProgressReporter())
    prep_avi = stage_prepare(spec_avi, cfg, work_avi, ProgressReporter())

    assert len(prep_tiff) == len(prep_avi) == 3

    for f_t, f_a in zip(prep_tiff, prep_avi):
        arr_t = tifffile.imread(f_t)
        arr_a = tifffile.imread(f_a)
        assert arr_t.shape == arr_a.shape == (arr_t.shape[0], 36, 48)
        assert arr_t.dtype == np.float32
        assert arr_a.dtype == np.float32
        if wrote_lossless:
            np.testing.assert_allclose(arr_t, arr_a, atol=1e-4)
        else:
            # MJPG compression tolerance
            np.testing.assert_allclose(arr_t, arr_a, atol=5.0)
