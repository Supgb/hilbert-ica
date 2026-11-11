from __future__ import annotations

import numpy as np

from hilbert_ica.ebm import complex_ica_ebm
from hilbert_ica.ica import gift_complex_pca, preprocess_complex_data, run_complex_ica
from hilbert_ica.selection import (
    energymax_phase_correct,
    phase_denoise,
    prepare_reference_vector,
    reconstruct_bold,
    select_components_for_subject,
)


def test_preprocess_and_gift_pca_whitening() -> None:
    rng = np.random.default_rng(10)
    data = rng.normal(size=(30, 200)) + 1j * rng.normal(size=(30, 200))  # (T, V)

    pro_tp = preprocess_complex_data(data, centering="timepoint")
    assert pro_tp.shape == (200, 30)
    np.testing.assert_allclose(pro_tp.mean(axis=0), 0.0, atol=1e-12)

    pro_vx = preprocess_complex_data(data, centering="voxel")
    np.testing.assert_allclose(pro_vx.mean(axis=1), 0.0, atol=1e-12)

    pca_scores = gift_complex_pca(pro_tp, n_components=8)
    assert pca_scores.shape == (8, 200)
    sample_cov = (pca_scores @ pca_scores.conj().T) / (200 - 1)
    np.testing.assert_allclose(sample_cov, np.eye(8), atol=1e-10)


def test_ebm_recovers_planted_noncircular_sources() -> None:
    rng = np.random.default_rng(11)
    n_samples = 1200
    s1 = rng.laplace(scale=1.0, size=n_samples) + 0.2j * rng.uniform(-1, 1, size=n_samples)
    s2 = rng.uniform(-1.5, 1.5, size=n_samples) + 0.3j * rng.laplace(scale=0.8, size=n_samples)
    sources_true = np.vstack([s1, s2])
    sources_true -= sources_true.mean(axis=1, keepdims=True)
    sources_true /= np.sqrt(np.mean(np.abs(sources_true) ** 2, axis=1, keepdims=True))

    mixing = np.array([[1.0 + 0.2j, 0.5 - 0.3j], [-0.4 + 0.5j, 1.1 + 0.1j]])
    mixtures = mixing @ sources_true

    _w, est = complex_ica_ebm(mixtures, random_state=0, max_iter_north=300, max_iter_refine=150)
    est_norm = est / np.sqrt(np.mean(np.abs(est) ** 2, axis=1, keepdims=True))
    cross = np.abs(est_norm @ sources_true.conj().T) / n_samples

    # Permutation matrix up to phase/scale: each true source matched with |r| > 0.95
    assert float(cross.max(axis=0).min()) > 0.95


def test_energymax_phase_correct_and_phase_denoise() -> None:
    rng = np.random.default_rng(12)
    n_time, n_vox = 120, 300
    ref = np.zeros(n_vox)
    ref[20:80] = 4.0

    # Planted source mostly on real axis, then rotated by an arbitrary complex phase + negative sign
    tc_base = rng.standard_t(df=4, size=n_time) + 0.05j * rng.normal(size=n_time)
    sm_base = np.zeros(n_vox, dtype=np.complex128)
    sm_base[20:80] = 2.0 * np.exp(1j * 0.15)  # inside [-pi/4, pi/4] and >= 0.5
    sm_base[100:130] = 2.0 * np.exp(1j * 1.3)  # outside phase cone -> must be zeroed
    sm_base[150:180] = 0.25 * np.exp(1j * 0.05)  # below 0.5 mag cutoff -> must be zeroed

    rot = np.exp(1j * 2.1)  # flips sign + rotates phase
    tc_rot = tc_base * rot
    sm_rot = sm_base * np.conj(rot)

    _theta, c_sm, c_tc = energymax_phase_correct(tc_rot, sm_rot, ref)
    # Product c_tc[:, None] * c_sm[None, :] equals tc_rot[:, None] * sm_rot[None, :]
    np.testing.assert_allclose(
        np.outer(c_tc, c_sm),
        np.outer(tc_rot, sm_rot),
        atol=1e-10,
    )
    # Sign aligned positively with reference
    assert float(np.corrcoef(c_sm.real, ref)[0, 1]) > 0.8

    denoised = phase_denoise(c_sm, phase_cone=np.pi / 4, mag_threshold=0.5)
    assert np.all(np.abs(denoised[20:80]) >= 0.5)
    assert np.all(denoised[100:130] == 0.0)
    assert np.all(denoised[150:180] == 0.0)


def test_select_components_and_reconstruct_bold() -> None:
    rng = np.random.default_rng(13)
    n_time, n_vox = 50, 400
    mask = np.ones((10, 10, 4), dtype=bool)

    t1 = np.zeros((10, 10, 4))
    t1[:3, :3, :2] = 5.0
    t2 = np.zeros((10, 10, 4))
    t2[6:9, 6:9, 2:] = 4.5

    ref_dmn = prepare_reference_vector(t1, mask, ref_threshold=2.5)
    ref_aud = prepare_reference_vector(t2, mask, ref_threshold=2.5)

    # Plant 2 clear spatial networks + random background in complex data
    a1 = rng.laplace(size=n_time) + 0.05j * rng.normal(size=n_time)
    a2 = rng.laplace(size=n_time) + 0.05j * rng.normal(size=n_time)
    s1 = ref_dmn * np.exp(1j * 0.05)
    s2 = ref_aud * np.exp(-1j * 0.08)
    noise = 0.05 * (rng.normal(size=(n_time, n_vox)) + 1j * rng.normal(size=(n_time, n_vox)))
    data = np.outer(a1, s1) + np.outer(a2, s2) + noise

    grid = run_complex_ica(
        data,
        model_orders=(4,),
        num_runs=2,
        random_state=3,
        max_iter_sea=40,
        max_iter_north=150,
        max_iter_refine=80,
    )
    selected = select_components_for_subject(
        grid,
        {"default_mode": ref_dmn, "auditory": ref_aud},
    )

    assert selected.network_names == ("default_mode", "auditory")
    assert selected.selected_maps.shape == (2, n_vox)
    assert selected.selected_time_courses.shape == (n_time, 2)
    assert float(selected.correlations[0]) > 0.75
    assert float(selected.correlations[1]) > 0.75
    assert int(selected.selected_components[0]) != int(selected.selected_components[1])

    bold = reconstruct_bold(
        selected.selected_maps,
        selected.selected_time_courses,
        selected_components=selected.selected_components,
        selected_runs=selected.selected_runs,
        selected_orders=selected.selected_orders,
    )
    assert bold.shape == (n_vox, n_time)
    assert bold.dtype == np.complex64

    # Verify deduplication if two templates map to the exact same (run, order, component)
    dup_maps = np.vstack([selected.selected_maps[0:1], selected.selected_maps[0:1]])
    dup_tc = np.column_stack([selected.selected_time_courses[:, 0], selected.selected_time_courses[:, 0]])
    bold_dedup = reconstruct_bold(
        dup_maps,
        dup_tc,
        selected_components=np.array([2, 2], dtype=np.int32),
        selected_runs=np.array([0, 0], dtype=np.int32),
        selected_orders=np.array([4, 4], dtype=np.int32),
        deduplicate=True,
    )
    single = (dup_maps[0:1].T @ dup_tc[:, 0:1].T).astype(np.complex64)
    np.testing.assert_allclose(bold_dedup, single, atol=1e-6)
