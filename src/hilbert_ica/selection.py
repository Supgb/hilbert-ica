"""Smith-2009 template loading, phase de-ambiguity, denoising, and RSN selection.

Complex ICA has inherent phase and permutation ambiguities:
1. Any component pair ``(a_k, s_k)`` can be multiplied by ``(e^{-j theta}, e^{+j theta})``
   without changing their outer product ``a_k s_k``.
2. Components are returned in arbitrary (negentropy) order rather than anatomical order.

Following Yu's ``energymax_T1.m``, ``phase_denoising.m``, and
``selection_component_complex.m``, this module resolves both ambiguities:
- Rotates each component's phase ``theta`` so its time-course variance is maximally
  concentrated along the real axis, then flips sign ``+/-1`` so ``Re(s_k)`` correlates
  positively with the target Smith (2009) resting-state network template.
- Zeroes noisy voxels whose spatial phase lies outside ``[-pi/4, +pi/4]`` or whose
  spatial magnitude is below ``0.5``.
- Selects the component whose denoised magnitude map correlates most strongly with
  each thresholded Smith template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.image import resample_to_img
from numpy.typing import NDArray

from hilbert_ica.ica import IcaRunResult

_EPS = 1e-12

# All 10 canonical resting-state networks in Smith et al. (2009), PNAS.
SMITH_RSN_NAMES: tuple[str, ...] = (
    "visual_medial",
    "visual_occipital",
    "visual_lateral",
    "default_mode",
    "cerebellum",
    "sensorimotor",
    "auditory",
    "executive_control",
    "frontoparietal_right",
    "frontoparietal_left",
)

# Default 3 networks analyzed in Yu's Nature Computational Science paper (Spatial_ref.mat).
DEFAULT_NETWORKS: tuple[str, ...] = (
    "default_mode",
    "auditory",
    "executive_control",
)

DEFAULT_REF_THRESHOLD = 2.5
DEFAULT_PHASE_CONE = float(np.pi / 4.0)
DEFAULT_MAG_THRESHOLD = 0.5


@dataclass(frozen=True)
class SelectedNetworksResult:
    """Selected complex spatial maps, time courses, and provenance per target RSN."""

    network_names: tuple[str, ...]
    selected_maps: NDArray[np.complex64]        # (n_networks, n_voxels), cone + mag denoised
    selected_raw_maps: NDArray[np.complex64]    # (n_networks, n_voxels), phase/sign-corrected only
    selected_time_courses: NDArray[np.complex64]  # (n_time, n_networks), phase/sign-corrected
    correlations: NDArray[np.float32]           # (n_networks,), |corr(|selected_maps|, ref)|
    selected_components: NDArray[np.int32]      # (n_networks,), 0-based component index
    selected_runs: NDArray[np.int32]            # (n_networks,), 0-based run index
    selected_orders: NDArray[np.int32]          # (n_networks,), winning PCA model order


def fetch_smith_atlas_path(data_dir: Path | str | None = None) -> str:
    """Download or locate the 10-network Smith (2009) 4D NIfTI file.

    Called once in the parent process before launching parallel workers so
    multiple processes never download or unpack the atlas concurrently.
    """
    atlas = fetch_atlas_smith_2009(
        data_dir=None if data_dir is None else str(data_dir),
        dimension=10,
        resting=True,
        verbose=0,
    )
    return str(atlas.maps)


def load_smith_templates(
    reference_img: Nifti1Image,
    *,
    networks: Sequence[str] = DEFAULT_NETWORKS,
    atlas_maps_path: str | Path | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Resample the requested Smith-2009 RSN templates onto a subject's 3D voxel grid."""
    unknown = [name for name in networks if name not in SMITH_RSN_NAMES]
    if unknown:
        raise ValueError(
            f"unknown Smith network(s) {unknown}; valid choices: {SMITH_RSN_NAMES}"
        )
    maps_path = (
        str(atlas_maps_path)
        if atlas_maps_path is not None
        else fetch_smith_atlas_path(data_dir=data_dir)
    )
    resampled = resample_to_img(
        maps_path,
        reference_img,
        interpolation="continuous",
        copy_header=True,
        force_resample=True,
    )
    data = np.asanyarray(resampled.dataobj, dtype=np.float64)
    if data.ndim != 4 or data.shape[3] != len(SMITH_RSN_NAMES):
        raise ValueError(f"expected 10 Smith RSN volumes, got shape {data.shape}")
    index_by_name = {name: idx for idx, name in enumerate(SMITH_RSN_NAMES)}
    return {name: data[..., index_by_name[name]] for name in networks}


def prepare_reference_vector(
    template_volume: NDArray[np.floating],
    mask: NDArray[np.bool_],
    *,
    ref_threshold: float = DEFAULT_REF_THRESHOLD,
) -> NDArray[np.float64]:
    """Extract in-brain template values and zero out voxels below ``ref_threshold``.

    Matches ``run_ucla_author_source.m``:
    ``reference = Sr(network, :); reference(reference < 2.5) = 0;``
    """
    ref = np.asarray(template_volume, dtype=np.float64)[np.asarray(mask, dtype=bool)].copy()
    ref[ref < ref_threshold] = 0.0
    return ref


def _pearson_corr(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Compute Pearson correlation between two 1D real vectors (0.0 if constant)."""
    x0 = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y0 = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom < _EPS:
        return 0.0
    return float(np.dot(x0, y0) / denom)


def energymax_phase_correct(
    time_course: NDArray[np.complexfloating],
    spatial_map: NDArray[np.complexfloating],
    reference: NDArray[np.floating],
) -> tuple[float, NDArray[np.complex128], NDArray[np.complex128]]:
    """Resolve complex phase and sign ambiguity for one ICA component (``energymax_T1.m``).

    Steps:
    1. Treat the centered complex time course ``a(t)`` as a 2D point cloud
       ``[Re(a), Im(a)]`` and find its principal axis angle ``theta`` via SVD.
       Rotating ``a`` by ``-theta`` maximizes real-axis time-course energy,
       while rotating ``s`` by ``+theta`` preserves the product ``a(t) * s(v)``.
    2. Because ``theta`` and ``theta + pi`` span the same line, check the Pearson
       correlation between ``Re(cS)`` and the spatial ``reference``. If negative,
       negate both ``cA`` and ``cS`` so in-network BOLD voxels sit near phase ``0``
       rather than ``+/-pi``.
    """
    tc = np.asarray(time_course, dtype=np.complex128).reshape(-1)
    sm = np.asarray(spatial_map, dtype=np.complex128).reshape(-1)

    # 2D PCA on demeaned [Re(tc), Im(tc)] to locate maximum-variance axis
    cloud = np.column_stack((tc.real, tc.imag))
    cloud_centered = cloud - cloud.mean(axis=0, keepdims=True)
    if cloud_centered.shape[0] < 2 or float(np.linalg.norm(cloud_centered)) < _EPS:
        theta = 0.0
    else:
        _u, _s, vh = np.linalg.svd(cloud_centered, full_matrices=False)
        axis = vh[0]
        theta = float(np.arctan2(axis[1], axis[0]))

    rot = np.exp(1j * theta)
    c_tc = tc * np.conj(rot)
    c_sm = sm * rot

    # Disambiguate +/-1 sign against the anatomical reference template
    if _pearson_corr(c_sm.real, reference) <= 0.0:
        c_tc = -c_tc
        c_sm = -c_sm
    return theta, c_sm, c_tc


def phase_denoise(
    spatial_map: NDArray[np.complexfloating],
    *,
    phase_cone: float = DEFAULT_PHASE_CONE,
    mag_threshold: float = DEFAULT_MAG_THRESHOLD,
) -> NDArray[np.complex128]:
    """Suppress out-of-phase and low-amplitude background voxels on a corrected map.

    Matches ``phase_denoising.m`` followed by ``pS1(abs(pS1) < 0.5) = 0`` in
    ``selection_component_complex.m``:
    - Phase cone: retain only voxels with ``|angle(s)| <= phase_cone`` (default ``pi/4``),
      isolating voxels phase-locked to the primary network time course.
    - Magnitude floor: retain only voxels with ``|s| >= mag_threshold`` (default ``0.5``).
    """
    sm = np.asarray(spatial_map, dtype=np.complex128)
    keep = (np.abs(np.angle(sm)) <= phase_cone) & (np.abs(sm) >= mag_threshold)
    return np.where(keep, sm, 0.0 + 0.0j)


def select_components_for_subject(
    ica_grid: list[list[IcaRunResult]],
    templates_in_mask: dict[str, NDArray[np.float64]],
    *,
    phase_cone: float = DEFAULT_PHASE_CONE,
    mag_threshold: float = DEFAULT_MAG_THRESHOLD,
) -> SelectedNetworksResult:
    """Select the highest-correlating complex ICA component for each target RSN.

    Mirrors ``selection_component_complex.m``:
    For each target Smith RSN, iterate over every ``(run, order)`` ICA solution
    and every candidate component ``0..K-1``:
      1. Phase- and sign-correct the component against this network's template.
      2. Apply phase-cone and magnitude denoising to get ``pS1``.
      3. Score the candidate by ``|corr(|pS1|, reference)|``.
    The winning component across all runs and orders is stored for that network.
    """
    if not ica_grid or not ica_grid[0]:
        raise ValueError("ica_grid must be non-empty")
    if not templates_in_mask:
        raise ValueError("templates_in_mask must be non-empty")

    first = ica_grid[0][0]
    n_voxels = first.spatial_maps.shape[1]
    n_time = first.time_courses.shape[0]
    network_names = tuple(templates_in_mask.keys())
    n_networks = len(network_names)

    sel_denoised = np.zeros((n_networks, n_voxels), dtype=np.complex64)
    sel_raw = np.zeros((n_networks, n_voxels), dtype=np.complex64)
    sel_tc = np.zeros((n_time, n_networks), dtype=np.complex64)
    sel_corr = np.zeros(n_networks, dtype=np.float32)
    sel_comp = np.zeros(n_networks, dtype=np.int32)
    sel_run = np.zeros(n_networks, dtype=np.int32)
    sel_order = np.zeros(n_networks, dtype=np.int32)

    for net_idx, (net_name, ref_vec) in enumerate(templates_in_mask.items()):
        if ref_vec.shape != (n_voxels,):
            raise ValueError(
                f"reference {net_name} has shape {ref_vec.shape}, expected ({n_voxels},)"
            )
        best_corr = -1.0
        best_denoised: NDArray[np.complex128] | None = None
        best_raw: NDArray[np.complex128] | None = None
        best_tc: NDArray[np.complex128] | None = None
        best_meta = (0, 0, first.order)

        for row in ica_grid:
            for cell in row:
                n_comp = cell.spatial_maps.shape[0]
                for comp_idx in range(n_comp):
                    _theta, c_sm, c_tc = energymax_phase_correct(
                        cell.time_courses[:, comp_idx],
                        cell.spatial_maps[comp_idx],
                        ref_vec,
                    )
                    denoised = phase_denoise(
                        c_sm,
                        phase_cone=phase_cone,
                        mag_threshold=mag_threshold,
                    )
                    corr = abs(_pearson_corr(np.abs(denoised), ref_vec))
                    if corr > best_corr:
                        best_corr = corr
                        best_denoised = denoised
                        best_raw = c_sm
                        best_tc = c_tc
                        best_meta = (comp_idx, cell.run, cell.order)

        assert best_denoised is not None and best_raw is not None and best_tc is not None
        sel_denoised[net_idx] = best_denoised.astype(np.complex64)
        sel_raw[net_idx] = best_raw.astype(np.complex64)
        sel_tc[:, net_idx] = best_tc.astype(np.complex64)
        sel_corr[net_idx] = np.float32(best_corr)
        sel_comp[net_idx] = np.int32(best_meta[0])
        sel_run[net_idx] = np.int32(best_meta[1])
        sel_order[net_idx] = np.int32(best_meta[2])

    return SelectedNetworksResult(
        network_names=network_names,
        selected_maps=sel_denoised,
        selected_raw_maps=sel_raw,
        selected_time_courses=sel_tc,
        correlations=sel_corr,
        selected_components=sel_comp,
        selected_runs=sel_run,
        selected_orders=sel_order,
    )


def reconstruct_bold(
    maps: NDArray[np.complexfloating],
    time_courses: NDArray[np.complexfloating],
    *,
    selected_components: NDArray[np.integer] | None = None,
    selected_runs: NDArray[np.integer] | None = None,
    selected_orders: NDArray[np.integer] | None = None,
    deduplicate: bool = True,
) -> NDArray[np.complex64]:
    """Reconstruct ``(n_voxels, n_time)`` complex BOLD on the fly from ``(S, A)``.

    Because ``selected_maps`` ``(M, V)`` and ``selected_time_courses`` ``(T, M)``
    are low-rank factors, storing the full ``(V, T)`` array on disk would waste
    ~50x space. Calling ``reconstruct_bold`` forms ``S^T @ A^T`` in memory when
    needed for voxel- or ROI-level time-series models.

    When ``deduplicate=True`` and ``selected_components`` is supplied, any
    component accidentally chosen by more than one network template is included
    only once in the sum.
    """
    s = np.asarray(maps, dtype=np.complex64)
    a = np.asarray(time_courses, dtype=np.complex64)
    if s.ndim != 2 or a.ndim != 2:
        raise ValueError("maps and time_courses must be 2D")
    if s.shape[0] != a.shape[1]:
        raise ValueError(
            f"maps has {s.shape[0]} networks but time_courses has {a.shape[1]}"
        )
    if deduplicate and selected_components is not None:
        comp = np.asarray(selected_components).ravel()
        runs = (
            np.asarray(selected_runs).ravel()
            if selected_runs is not None
            else np.zeros_like(comp)
        )
        orders = (
            np.asarray(selected_orders).ravel()
            if selected_orders is not None
            else np.zeros_like(comp)
        )
        keep_indices: list[int] = []
        seen: set[tuple[int, int, int]] = set()
        for idx in range(s.shape[0]):
            key = (int(runs[idx]), int(orders[idx]), int(comp[idx]))
            if key not in seen:
                seen.add(key)
                keep_indices.append(idx)
        s = s[keep_indices]
        a = a[:, keep_indices]
    return (s.T @ a.T).astype(np.complex64)
