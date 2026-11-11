"""In-brain voxel masking and Hilbert-transform complex fMRI construction.

Standard fMRI datasets only provide real-valued magnitude BOLD images. Following
Yu's workflow, this module extracts an in-brain mask and turns each voxel's 1D
time series into a complex-valued signal by pairing its original BOLD amplitude
with the instantaneous phase of its discrete-time Hilbert analytic signal:

- ``brain_mask_mean_rule``        : mirrors ``Complex_ICA.m``
- ``mask_from_inbrain_indices``   : converts MATLAB ``inbrain_ind.mat`` indices to a 3D mask
- ``hilbert_phase``               : mirrors ``phase_fmri_Hilbert.m`` / ``hilbert_transform.m``
- ``construct_complex_bold``      : mirrors ``complex_fmri_HilbertPhase.m``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def brain_mask_mean_rule(volume: NDArray) -> NDArray[np.bool_]:
    """Derive a 3D brain mask by keeping voxels above the grand mean for >= 50% of frames.

    In ``Complex_ICA.m``, background air voxels have near-zero intensity while
    brain tissue stays well above the whole-volume average. Specifically:
    1. Compute the global mean of ``|volume|`` across all voxels and time frames.
    2. Keep any voxel whose ``|volume|`` exceeds that global mean in at least
       half of the time frames.
    """
    data = np.asarray(volume)
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD volume, got shape {data.shape}")
    n_time = data.shape[-1]
    if n_time < 1:
        raise ValueError("4D volume has zero time frames")

    # Flatten spatial axes to (n_time, n_total_voxels) to match MATLAB's layout
    flat = np.abs(data.reshape(-1, n_time)).astype(np.float64).T
    threshold = float(flat.mean())
    frac_above = (flat > threshold).mean(axis=0)
    mask = (frac_above >= 0.5).reshape(data.shape[:3])
    if not np.any(mask):
        raise ValueError("mean-rule brain mask is empty")
    return mask


def mask_from_inbrain_indices(
    shape_3d: tuple[int, int, int],
    inbrain_ind: NDArray[np.integer],
    *,
    one_based: bool = True,
) -> NDArray[np.bool_]:
    """Build a 3D boolean mask from MATLAB linear voxel indices (``inbrain_ind``).

    MATLAB flattens 3D arrays in column-major (Fortran) order and uses 1-based
    indexing. Subtracting 1 and reshaping with ``order="F"`` maps those indices
    onto the exact same 3D voxel coordinates in NumPy.
    """
    if len(shape_3d) != 3:
        raise ValueError(f"expected 3D shape, got {shape_3d}")
    indices = np.asarray(inbrain_ind, dtype=np.int64).ravel()
    if one_based:
        indices = indices - 1
    n_total = int(np.prod(shape_3d))
    if np.any(indices < 0) or np.any(indices >= n_total):
        raise ValueError(f"inbrain_ind out of bounds for volume of size {n_total}")
    flat_f = np.zeros(n_total, dtype=bool)
    flat_f[indices] = True
    return flat_f.reshape(shape_3d, order="F")


def trim_timepoints(
    mag_t_by_v: NDArray[np.floating],
    *,
    max_frames: int | None = None,
    n_drop: int = 0,
) -> NDArray[np.float64]:
    """Optionally cap leading frames (``max_frames``) or trim both ends (``n_drop``)."""
    arr = np.asarray(mag_t_by_v, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (n_time, n_voxels), got shape {arr.shape}")
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError(f"max_frames must be >= 2, got {max_frames}")
        arr = arr[:max_frames]
    if n_drop < 0:
        raise ValueError(f"n_drop must be >= 0, got {n_drop}")
    if n_drop > 0:
        if 2 * n_drop >= arr.shape[0]:
            raise ValueError(
                f"n_drop={n_drop} removes all frames from length {arr.shape[0]}"
            )
        arr = arr[n_drop : arr.shape[0] - n_drop]
    return arr


def hilbert_phase(mag_t_by_v: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute instantaneous phase ``angle(hilbert(mag))`` along the time axis (axis 0).

    ``scipy.signal.hilbert`` uses the exact same Marple (1999) FFT method as
    MATLAB's ``hilbert`` / ``hilbert_transform.m``: double positive frequencies,
    zero negative frequencies, inverse FFT, and take ``atan2(Im, Re)``.
    """
    arr = np.asarray(mag_t_by_v, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (n_time, n_voxels), got shape {arr.shape}")
    analytic = hilbert(arr, axis=0)
    return np.angle(analytic)


def construct_complex_bold(
    mag_t_by_v: NDArray[np.floating],
    *,
    max_frames: int | None = None,
    n_drop: int = 0,
) -> NDArray[np.complex128]:
    """Construct complex-valued fMRI ``z = mag * exp(1j * angle(hilbert(mag)))``.

    Parameters
    ----------
    mag_t_by_v :
        In-brain magnitude BOLD matrix of shape ``(n_time, n_voxels)``.
    max_frames :
        Optional cap ``1..max_frames`` applied *before* the Hilbert transform
        (matching ``mag_nii.img(:,:,:,1:146)`` in ``phase_fmri_Hilbert.m``).
    n_drop :
        Optional number of edge frames discarded from both ends *after* complex
        construction to suppress Hilbert wrap-around boundary effects.

    Returns
    -------
    complex_bold :
        Complex-valued fMRI matrix of shape ``(n_time_used, n_voxels)``.
    """
    arr = np.asarray(mag_t_by_v, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (n_time, n_voxels), got shape {arr.shape}")
    if max_frames is not None:
        arr = trim_timepoints(arr, max_frames=max_frames, n_drop=0)

    # Compute voxel-wise instantaneous phase and combine with raw BOLD values
    phase = hilbert_phase(arr)
    complex_bold = arr * np.exp(1j * phase)

    if n_drop > 0:
        if 2 * n_drop >= complex_bold.shape[0]:
            raise ValueError(
                f"n_drop={n_drop} removes all frames from length {complex_bold.shape[0]}"
            )
        complex_bold = complex_bold[n_drop : complex_bold.shape[0] - n_drop]
    return complex_bold
