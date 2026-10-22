# -*- coding: utf-8 -*-
"""End-to-end test on synthetic 1-photon movie: centroid recovery, Curator HDF5, and ms.mat orientation."""

import h5py
import numpy as np
import tifffile
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from scipy.io import loadmat
from scipy.spatial.distance import cdist

from decovo.config import load_config
from decovo.pipeline import run
from tests.conftest import make_synthetic_1p_movie


def test_end_to_end_synthetic_1p_recovery_and_exports(tmp_path):
    n_gt = 12
    n_frames = 350
    shape = (96, 96)
    movie, gt_centroids = make_synthetic_1p_movie(
        n_frames=n_frames,
        shape=shape,
        n_neurons=n_gt,
        gSig=3.0,
        seed=99,
    )
    tif_path = tmp_path / "synthetic_1p.tif"
    tifffile.imwrite(
        str(tif_path),
        movie,
        imagej=True,
        metadata={"axes": "TYX", "fps": 20.0},
    )

    cfg = load_config(
        path_to_analyze=str(tmp_path),
        spatial_downsample=1.0,
        frame_rate=20.0,
        gSig=3.0,
        min_SNR=2.5,
        rval_thr=0.75,
        n_processes=1,
        save_figures=True,
    )

    result = run(tmp_path, config=cfg)
    assert result.n_neurons > 0

    # 1. Verify ground-truth centroid recovery (>= 70% within 3 px)
    A = result.cnm.estimates.A[:, result.idx_good].toarray()
    dims = result.dims
    est_centroids = []
    for k in range(A.shape[1]):
        fp = np.reshape(A[:, k], dims, order="F")
        total = fp.sum()
        if total > 0:
            yy, xx = np.mgrid[0 : dims[0], 0 : dims[1]]
            cy = float((yy * fp).sum() / total)
            cx = float((xx * fp).sum() / total)
            est_centroids.append((cy, cx))
    est_centroids_arr = np.array(est_centroids, dtype=np.float32)

    dists = cdist(gt_centroids, est_centroids_arr)
    min_dists = dists.min(axis=1)
    matched_fraction = float(np.mean(min_dists <= 3.0))
    assert matched_fraction >= 0.70, f"Expected >= 70% centroid recovery within 3 px, got {matched_fraction:.2%}"

    # 2. Verify Cell Curator HDF5 compatibility (loads via cnmf.load_CNMF with Cn and dims)
    work_dir = result.session.work_dir
    hdf5_path = work_dir / "analysis_results.hdf5"
    assert hdf5_path.exists()

    loaded_cnm = cnmf_module.load_CNMF(str(hdf5_path))
    assert hasattr(loaded_cnm.estimates, "Cn") and loaded_cnm.estimates.Cn is not None
    assert tuple(loaded_cnm.estimates.Cn.shape) == dims
    assert hasattr(loaded_cnm.estimates, "dims") and tuple(loaded_cnm.estimates.dims) == dims
    assert loaded_cnm.estimates.idx_components is not None
    assert len(loaded_cnm.estimates.idx_components) == result.n_neurons

    with h5py.File(str(hdf5_path), "r") as h5f:
        assert "Cn" in h5f["estimates"]
        assert "dims" in h5f["estimates"]

    # 3. Verify ms.mat keys and that BOTH RawTraces and DeconvTraces are time × neurons (T × N)
    ms_mat_path = work_dir / "ms.mat"
    assert ms_mat_path.exists()
    mat_data = loadmat(str(ms_mat_path), simplify_cells=True)["ms"]

    n_total_comp = result.cnm.estimates.C.shape[0]
    raw_tr = mat_data["RawTraces"]
    deconv_tr = mat_data["DeconvTraces"]
    if n_total_comp == 1:
        assert raw_tr.shape[0] == n_frames
        assert deconv_tr.shape[0] == n_frames
    else:
        assert raw_tr.shape == (n_frames, n_total_comp)
        assert deconv_tr.shape == (n_frames, n_total_comp)

    assert mat_data["height"] == dims[0]
    assert mat_data["width"] == dims[1]
    assert mat_data["frameRate"] == 20.0
    assert mat_data["inputType"] == "tiff"
    assert "gSiz: 13" in mat_data["configYaml"]

    # 4. Verify config_used.yaml exists and round-trips
    cfg_used_path = work_dir / "config_used.yaml"
    assert cfg_used_path.exists()
    cfg_reloaded = load_config(cfg_used_path)
    assert cfg_reloaded.min_corr == result.min_corr_used
    assert cfg_reloaded.min_pnr == result.min_pnr_used


def test_multisession_batch_registration_with_inhomogeneous_and_empty_sessions(tmp_path):
    from types import SimpleNamespace
    import scipy.sparse as sp
    from decovo.pipeline import PipelineResult, _register_multisession_batch

    dims = (48, 48)

    def _make_mock_result(n_total: int, n_good: int) -> PipelineResult:
        A = sp.lil_matrix((dims[0] * dims[1], n_total), dtype=np.float32)
        for i in range(n_total):
            cy = 8 + (i * 5) % 32
            cx = 8 + (i * 7) % 32
            A[cy + cx * dims[0], i] = 1.0
        idx_good = np.arange(n_good, dtype=int)
        est = SimpleNamespace(
            A=A.tocsc(),
            C=np.ones((n_total, 60), dtype=np.float32),
            F_dff=np.ones((n_total, 60), dtype=np.float32) * 0.5,
            idx_components=idx_good,
        )
        cnm = SimpleNamespace(estimates=est)
        cn_orig = np.linspace(0.2, 0.9, dims[0] * dims[1], dtype=np.float32).reshape(dims)
        return PipelineResult(
            cnm=cnm,
            cn_filter=cn_orig,
            dims=dims,
            idx_good=idx_good,
            n_neurons=n_good,
        )

    s1 = str(tmp_path / "Animal1" / "pre")
    s2 = str(tmp_path / "Animal1" / "empty_session")
    s3 = str(tmp_path / "Animal1" / "post")

    r1 = _make_mock_result(n_total=10, n_good=6)
    r2 = _make_mock_result(n_total=4, n_good=0)  # 0 accepted neurons!
    r3 = _make_mock_result(n_total=9, n_good=5)  # Different accepted count (5 vs 6)

    cn1_before = r1.cn_filter.copy()
    cfg = load_config(path_to_analyze=str(tmp_path), register_batch=True, save_mat=True, save_hdf5=True)
    _register_multisession_batch(cfg, [s1, s2, s3], [r1, r2, r3])

    # Ensure cn_filter was not mutated in-place by CaImAn register_multisession
    np.testing.assert_array_equal(r1.cn_filter, cn1_before)

    mat_path = tmp_path / "batch_registration.mat"
    h5_path = tmp_path / "batch_registration.hdf5"
    assert mat_path.exists()
    assert h5_path.exists()

    mat_reg = loadmat(str(mat_path), simplify_cells=True)["registration"]
    assert list(mat_reg["sessions"]) == ["Animal1/pre", "Animal1/post"]
    assert len(mat_reg["matchings"][0]) == 6
    assert len(mat_reg["matchings"][1]) == 5

    with h5py.File(str(h5_path), "r") as f:
        assert "assignments" in f
        assert "matchings" in f
        assert f["matchings/session_0"].shape == (6,)
        assert f["matchings/session_1"].shape == (5,)

