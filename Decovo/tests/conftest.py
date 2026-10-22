# -*- coding: utf-8 -*-
"""Synthetic 1-photon movie generator shared by cache and end-to-end tests."""

from typing import Tuple

import numpy as np


def make_synthetic_1p_movie(
    n_frames: int = 300,
    shape: Tuple[int, int] = (96, 96),
    n_neurons: int = 12,
    gSig: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic 1-photon miniscope movie with known Gaussian neurons,
    AR(1) calcium transients, and a smooth background + vignetting + Poisson-Gaussian noise.

    Returns:
        `(movie_uint16, gt_centroids_yx)`
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]

    # Place neurons on a regular grid with slight jitter away from borders
    margin = 16
    grid_side = int(np.ceil(np.sqrt(n_neurons)))
    ys = np.linspace(margin, h - margin, grid_side)
    xs = np.linspace(margin, w - margin, grid_side)
    grid_pts = [(y, x) for y in ys for x in xs][:n_neurons]
    centroids = []
    footprints = []

    for cy, cx in grid_pts:
        cy_j = float(cy + rng.uniform(-2.0, 2.0))
        cx_j = float(cx + rng.uniform(-2.0, 2.0))
        centroids.append((cy_j, cx_j))
        foot = np.exp(-((yy - cy_j) ** 2 + (xx - cx_j) ** 2) / (2.0 * (gSig ** 2)))
        footprints.append(foot.astype(np.float32))

    footprints_arr = np.stack(footprints, axis=0)  # (N, H, W)
    centroids_arr = np.array(centroids, dtype=np.float32)  # (N, 2) in (row, col)

    # AR(1) transients with distinct spike times per neuron
    gamma = 0.92
    traces = np.zeros((n_neurons, n_frames), dtype=np.float32)
    for i in range(n_neurons):
        spikes = np.zeros(n_frames, dtype=np.float32)
        # Space 8 strong calcium events across the recording with per-neuron phase offset
        event_times = np.linspace(15 + (i * 7) % 20, n_frames - 25, 8, dtype=int)
        spikes[event_times] = rng.uniform(1100.0, 1400.0, size=len(event_times)).astype(np.float32)
        c = 0.0
        for t in range(n_frames):
            c = gamma * c + spikes[t]
            traces[i, t] = c

    # Smooth 1-photon background with slight temporal fluctuation
    bg_spatial = 500.0 * np.exp(-((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) / (2.0 * (0.6 * h) ** 2))
    bg_temporal = 1.0 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, n_frames, dtype=np.float32))
    movie = bg_temporal[:, None, None] * bg_spatial[None, :, :]

    # Add neural signal: tensordot (N, T) and (N, H, W) -> (T, H, W)
    movie += np.tensordot(traces.T, footprints_arr, axes=([1], [0]))

    # Add realistic per-pixel noise so non-neuron pixels have low local spatial correlation
    movie += rng.normal(0.0, 25.0, size=movie.shape).astype(np.float32)
    movie = np.clip(movie, 0, 65535).astype(np.uint16)
    return movie, centroids_arr
