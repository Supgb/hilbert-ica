from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from hilbert_ica.hilbert import (
    brain_mask_mean_rule,
    construct_complex_bold,
    hilbert_phase,
    mask_from_inbrain_indices,
)


def test_brain_mask_mean_rule_matches_matlab_definition() -> None:
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(4, 5, 3, 24)) + 6.0
    volume[0, 0, 0, :] = 0.0  # background voxel
    volume[1, 2, 1, :15] = 0.0  # below grand mean for > half frames -> excluded

    mask = brain_mask_mean_rule(volume)
    flat = np.abs(volume.reshape(-1, 24)).T
    thr = flat.mean()
    expected = ((flat > thr).mean(axis=0) >= 0.5).reshape(4, 5, 3)

    np.testing.assert_array_equal(mask, expected)
    assert not mask[0, 0, 0]
    assert not mask[1, 2, 1]


def test_mask_from_inbrain_indices_fortran_one_based() -> None:
    shape = (3, 4, 2)
    vol = np.arange(1, 25).reshape(shape, order="F")
    ind_one_based = np.array([1, 5, 24])
    mask = mask_from_inbrain_indices(shape, ind_one_based, one_based=True)
    assert mask.shape == shape
    assert int(mask.sum()) == 3
    np.testing.assert_array_equal(np.sort(vol[mask]), np.array([1, 5, 24]))


def test_construct_complex_bold_preserves_signed_magnitude_and_hilbert_phase() -> None:
    rng = np.random.default_rng(1)
    mag = rng.normal(loc=300.0, scale=15.0, size=(80, 12))
    mag[5, 0] = -3.5  # verify signed mag * exp(1j * angle(hilbert(mag))) matches MATLAB

    z = construct_complex_bold(mag)
    expected_phase = np.angle(hilbert(mag, axis=0))
    expected_z = mag * np.exp(1j * expected_phase)

    assert z.shape == (80, 12)
    np.testing.assert_allclose(hilbert_phase(mag), expected_phase, atol=1e-12)
    np.testing.assert_allclose(z, expected_z, atol=1e-12)
    np.testing.assert_allclose(np.abs(z), np.abs(mag), atol=1e-12)


def test_construct_complex_bold_max_frames_and_n_drop() -> None:
    rng = np.random.default_rng(2)
    mag = rng.normal(size=(50, 6)) + 10.0
    z = construct_complex_bold(mag, max_frames=40, n_drop=5)
    assert z.shape == (30, 6)

    with pytest.raises(ValueError, match="removes all frames"):
        construct_complex_bold(mag, max_frames=10, n_drop=5)
