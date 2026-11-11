"""Spatial demeaning, GIFT-equivalent Complex PCA, and Complex ICA-EBM driver.

This module reproduces ``ICA_complex.m`` together with the GIFT toolbox preprocessing
functions ``icatb_preproc_data.m`` and ``icatb_calculate_pca.m``:

1. Centering (``preprocess_complex_data``): Transpose ``data`` ``(T, V)`` without
   conjugation to ``(V, T)`` and subtract the spatial mean per timepoint.
2. Complex PCA whitening (``gift_complex_pca``): Reduce the ``T`` timepoints to
   ``K`` uncorrelated spatial components of shape ``(K, V)``.
3. Complex ICA-EBM (``run_complex_ica``): Separate ``(K, V)`` into independent
   spatial sources ``Sm = W @ data_pca`` and back-project complex time courses
   ``Tc = data @ pinv(Se)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from hilbert_ica.ebm import complex_ica_ebm

CenteringMode = Literal["timepoint", "voxel"]
_EPS = 1e-12


@dataclass(frozen=True)
class IcaRunResult:
    """Output of a single Complex ICA-EBM run at a specific PCA model order."""

    order: int
    run: int
    spatial_maps: NDArray[np.complex128]  # (K, n_voxels), corresponds to MATLAB Sm{r, m}
    time_courses: NDArray[np.complex128]  # (n_time, K), corresponds to MATLAB Tc{r, m}
    weights: NDArray[np.complex128]       # (K, K) demixing matrix W


def preprocess_complex_data(
    data_t_by_v: NDArray,
    *,
    centering: CenteringMode = "timepoint",
) -> NDArray[np.complex128]:
    """Transpose ``data`` ``(T, V) -> (V, T)`` and remove its mean prior to PCA.

    Parameters
    ----------
    data_t_by_v :
        Complex fMRI data of shape ``(n_time, n_voxels)``.
    centering :
        - ``"timepoint"`` (default, matching ``ICA_complex.m``): subtracts the
          mean across all in-brain voxels at each frame, removing global frame-wise
          baseline offsets (``icatb_preproc_data(data.', 'remove mean per timepoint')``).
        - ``"voxel"`` (matching ``Complex_ICA.m``): subtracts each voxel's own
          temporal mean across frames.
    """
    data = np.asarray(data_t_by_v, dtype=np.complex128)
    if data.ndim != 2:
        raise ValueError(f"expected 2D (n_time, n_voxels), got shape {data.shape}")

    # Non-conjugate transpose matches MATLAB's `data.'`
    vox_by_time = data.T
    if centering == "timepoint":
        return vox_by_time - vox_by_time.mean(axis=0, keepdims=True)
    if centering == "voxel":
        return vox_by_time - vox_by_time.mean(axis=1, keepdims=True)
    raise ValueError(f"unknown centering mode {centering!r}")


def gift_complex_pca(
    data_vox_by_time: NDArray[np.complex128],
    n_components: int,
) -> NDArray[np.complex128]:
    """Reduce ``(n_voxels, n_time)`` complex data to ``(K, n_voxels)`` whitened PCs.

    Matches GIFT's ``icatb_calculate_pca(data_pro, K, 'remove_mean', 0)`` followed
    by ``data_pca.'``:
    - GIFT forms the ``(T, T)`` Hermitian covariance ``C = X^H X / (V - 1)``,
      extracts the top ``K`` eigenvectors ``V_K`` and eigenvalues ``Lambda_K``,
      and computes whitened spatial scores ``X V_K Lambda_K^{-1/2}``.
    - Using an economy SVD ``X = U diag(S) V^H``, ``X V_K`` equals ``U_K diag(S_K)``
      and ``Lambda_K^{1/2} = S_K / sqrt(V - 1)``, so the whitened spatial scores
      simplify directly to ``sqrt(V - 1) * U_K`` without explicitly forming ``C``.
    """
    x = np.asarray(data_vox_by_time, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"expected 2D (n_voxels, n_time), got shape {x.shape}")
    n_voxels, n_time = x.shape
    max_rank = min(n_voxels - 1, n_time)
    if n_components < 1 or n_components > max_rank:
        raise ValueError(
            f"n_components={n_components} invalid for shape {x.shape} (max {max_rank})"
        )

    u, singular_values, _vh = np.linalg.svd(x, full_matrices=False)
    valid = singular_values[:n_components] > _EPS
    scores = np.zeros((n_voxels, n_components), dtype=np.complex128)
    scores[:, valid] = u[:, :n_components][:, valid] * np.sqrt(float(n_voxels - 1))
    return scores.T  # (K, n_voxels), matching `data_pca.'` input to complex_ICA_EBM


def run_complex_ica(
    data_t_by_v: NDArray,
    *,
    model_orders: Sequence[int] = (40,),
    num_runs: int = 1,
    centering: CenteringMode = "timepoint",
    random_state: int | np.random.Generator | None = 0,
    max_iter_sea: int = 100,
    max_iter_north: int = 1000,
    max_iter_refine: int = 500,
    tolerance: float = 1e-4,
) -> list[list[IcaRunResult]]:
    """Run spatial Complex ICA-EBM across ``num_runs`` and ``model_orders``.

    Because EBM uses random initialization and stochastic sample weights in its
    search stage, Yu's paper optionally runs multiple random restarts (e.g.
    ``num_runs=4``) across multiple PCA orders (e.g. ``model_orders=(40, 50, 60)``)
    and lets template matching pick the most reproducible component. By default,
    a single ``K=40`` run is executed for speed.

    Returns
    -------
    grid :
        2D list ``grid[run_idx][order_idx]`` of :class:`IcaRunResult`.
    """
    data = np.asarray(data_t_by_v, dtype=np.complex128)
    if data.ndim != 2:
        raise ValueError(f"expected 2D (n_time, n_voxels), got shape {data.shape}")
    orders = [int(k) for k in model_orders]
    if not orders:
        raise ValueError("model_orders must be non-empty")
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")

    base_rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # Demean and compute PCA once per unique model order (PCA is deterministic)
    data_pro = preprocess_complex_data(data, centering=centering)
    pca_by_order = {order: gift_complex_pca(data_pro, order) for order in set(orders)}

    grid: list[list[IcaRunResult]] = []
    for run_idx in range(num_runs):
        row: list[IcaRunResult] = []
        for order in orders:
            child_seed = int(base_rng.integers(0, 2**31 - 1))
            pca_k = pca_by_order[order]  # (K, n_voxels)
            weights, se = complex_ica_ebm(
                pca_k,
                random_state=child_seed,
                max_iter_sea=max_iter_sea,
                max_iter_north=max_iter_north,
                max_iter_refine=max_iter_refine,
                tolerance=tolerance,
            )
            # Reconstruct spatial maps Sm = W @ data_pca.' and back-project
            # time courses Tc = data @ pinv(Se) exactly as in ICA_complex.m
            spatial_maps = weights @ pca_k
            time_courses = data @ np.linalg.pinv(se)
            row.append(
                IcaRunResult(
                    order=order,
                    run=run_idx,
                    spatial_maps=spatial_maps,
                    time_courses=time_courses,
                    weights=weights,
                )
            )
        grid.append(row)
    return grid
