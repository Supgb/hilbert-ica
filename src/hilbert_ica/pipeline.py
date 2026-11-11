"""Single-subject and parallel cohort orchestration for Hilbert Complex ICA.

Coordinates the end-to-end subject pipeline (masking -> Hilbert complex BOLD ->
PCA + EBM -> Smith-2009 RSN selection -> atomic ``.npz`` save) and distributes
subjects across multiple CPU worker processes while keeping BLAS single-threaded
inside each worker to avoid CPU oversubscription.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed
from nibabel.nifti1 import Nifti1Image
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from hilbert_ica.hilbert import brain_mask_mean_rule, construct_complex_bold
from hilbert_ica.ica import CenteringMode, run_complex_ica
from hilbert_ica.io import (
    SubjectInput,
    load_bold_nifti,
    load_mask_nifti,
    nifti_stem,
    subject_npz_is_complete,
    subject_npz_path,
    write_subject_npz,
)
from hilbert_ica.selection import (
    DEFAULT_MAG_THRESHOLD,
    DEFAULT_NETWORKS,
    DEFAULT_PHASE_CONE,
    DEFAULT_REF_THRESHOLD,
    fetch_smith_atlas_path,
    load_smith_templates,
    prepare_reference_vector,
    select_components_for_subject,
)


def pin_blas_threads(n_threads: int = 1) -> None:
    """Set OpenMP / MKL / OpenBLAS / NumExpr thread environment variables.

    When running ``N`` subject processes in parallel via ``joblib``, pinning
    each worker's linear algebra library to 1 thread prevents ``N x cores``
    thread contention during SVD and matrix multiplications.
    """
    value = str(int(n_threads))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def process_subject(
    bold_path: Path | str,
    out_dir: Path | str,
    *,
    subject: str | None = None,
    mask_path: Path | str | None = None,
    model_orders: Sequence[int] = (40,),
    num_runs: int = 1,
    centering: CenteringMode = "timepoint",
    networks: Sequence[str] = DEFAULT_NETWORKS,
    ref_threshold: float = DEFAULT_REF_THRESHOLD,
    phase_cone: float = DEFAULT_PHASE_CONE,
    mag_threshold: float = DEFAULT_MAG_THRESHOLD,
    max_frames: int | None = None,
    n_drop: int = 0,
    random_state: int = 0,
    skip_existing: bool = True,
    atlas_maps_path: str | Path | None = None,
    templates_override: dict[str, np.ndarray] | None = None,
    max_iter_sea: int = 100,
    max_iter_north: int = 1000,
    max_iter_refine: int = 500,
) -> dict[str, Any]:
    """Execute the full Hilbert Complex ICA + Smith selection workflow for one subject.

    Writes ``<out_dir>/<subject>.npz`` containing the selected RSN spatial maps,
    time courses, mask, affine, and selection metadata, and returns a JSON-ready
    status summary dictionary.
    """
    bold_file = Path(bold_path)
    subject_id = subject if subject is not None else nifti_stem(bold_file)
    out_file = subject_npz_path(out_dir, subject_id)

    # Resume-safe check: skip immediately if a valid .npz already exists
    if skip_existing and subject_npz_is_complete(out_file):
        return {
            "subject": subject_id,
            "status": "skipped",
            "output": str(out_file),
        }

    t0 = time.perf_counter()
    pin_blas_threads(1)

    # 1. Load 4D BOLD NIfTI and derive (or load) the 3D in-brain mask
    volume, affine, tr, _img = load_bold_nifti(bold_file)
    mask = (
        load_mask_nifti(mask_path, volume.shape[:3])
        if mask_path is not None
        else brain_mask_mean_rule(volume)
    )
    mag_t_by_v = volume[mask].T  # (n_time, n_voxels)
    del volume

    # 2. Construct complex-valued fMRI z = mag * exp(1j * hilbert_phase(mag))
    complex_bold = construct_complex_bold(
        mag_t_by_v,
        max_frames=max_frames,
        n_drop=n_drop,
    )
    del mag_t_by_v

    # 3. Resample Smith-2009 RSN templates onto this subject's 3D grid and threshold
    if templates_override is not None:
        raw_templates = {name: templates_override[name] for name in networks}
    else:
        ref_img = Nifti1Image(mask.astype(np.uint8), affine)
        raw_templates = load_smith_templates(
            ref_img,
            networks=networks,
            atlas_maps_path=atlas_maps_path,
        )
    templates_in_mask = {
        name: prepare_reference_vector(vol, mask, ref_threshold=ref_threshold)
        for name, vol in raw_templates.items()
    }

    # 4. Run Complex PCA + EBM and select the best component per RSN
    with threadpool_limits(limits=1):
        ica_grid = run_complex_ica(
            complex_bold,
            model_orders=model_orders,
            num_runs=num_runs,
            centering=centering,
            random_state=random_state,
            max_iter_sea=max_iter_sea,
            max_iter_north=max_iter_north,
            max_iter_refine=max_iter_refine,
        )
        selected = select_components_for_subject(
            ica_grid,
            templates_in_mask,
            phase_cone=phase_cone,
            mag_threshold=mag_threshold,
        )

    # 5. Save compact factorized outputs atomically to <out_dir>/<subject>.npz
    elapsed = time.perf_counter() - t0
    write_subject_npz(
        out_file,
        {
            "selected_maps": selected.selected_maps,
            "selected_raw_maps": selected.selected_raw_maps,
            "selected_time_courses": selected.selected_time_courses,
            "network_names": np.asarray(selected.network_names),
            "correlations": selected.correlations,
            "selected_components": selected.selected_components,
            "selected_runs": selected.selected_runs,
            "selected_orders": selected.selected_orders,
            "mask": mask,
            "affine": affine,
            "tr": np.float64(tr),
        },
    )

    return {
        "subject": subject_id,
        "status": "ok",
        "output": str(out_file),
        "bold_path": str(bold_file),
        "n_voxels": int(mask.sum()),
        "n_time": int(complex_bold.shape[0]),
        "tr": float(tr),
        "orders": [int(k) for k in model_orders],
        "num_runs": int(num_runs),
        "correlations": {
            name: float(corr)
            for name, corr in zip(selected.network_names, selected.correlations)
        },
        "selected_components": {
            name: int(comp)
            for name, comp in zip(selected.network_names, selected.selected_components)
        },
        "seconds": round(elapsed, 3),
    }


def _worker_entry(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry wrapper enforcing single-threaded BLAS inside each worker."""
    pin_blas_threads(1)
    with threadpool_limits(limits=1):
        return process_subject(**kwargs)


def process_cohort(
    inputs: Sequence[SubjectInput],
    out_dir: Path | str,
    *,
    n_jobs: int | None = None,
    mask_path: Path | str | None = None,
    model_orders: Sequence[int] = (40,),
    num_runs: int = 1,
    centering: CenteringMode = "timepoint",
    networks: Sequence[str] = DEFAULT_NETWORKS,
    ref_threshold: float = DEFAULT_REF_THRESHOLD,
    phase_cone: float = DEFAULT_PHASE_CONE,
    mag_threshold: float = DEFAULT_MAG_THRESHOLD,
    max_frames: int | None = None,
    n_drop: int = 0,
    random_state: int = 0,
    skip_existing: bool = True,
    templates_override: dict[str, np.ndarray] | None = None,
    atlas_data_dir: Path | str | None = None,
    max_iter_sea: int = 100,
    max_iter_north: int = 1000,
    max_iter_refine: int = 500,
) -> list[dict[str, Any]]:
    """Process a cohort of subjects in parallel across ``n_jobs`` worker processes.

    Pre-fetches the Smith-2009 template NIfTI once in the main process before
    spawning workers so parallel workers only perform local file reads.
    """
    if not inputs:
        return []
    pin_blas_threads(1)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Ensure Smith-2009 atlas is cached locally before spawning worker processes
    atlas_maps_path: str | None = None
    if templates_override is None:
        need_any = any(
            not (skip_existing and subject_npz_is_complete(subject_npz_path(out_path, item.subject)))
            for item in inputs
        )
        if need_any:
            atlas_maps_path = fetch_smith_atlas_path(data_dir=atlas_data_dir)

    jobs = [
        {
            "bold_path": item.bold_path,
            "out_dir": out_path,
            "subject": item.subject,
            "mask_path": mask_path,
            "model_orders": tuple(int(k) for k in model_orders),
            "num_runs": int(num_runs),
            "centering": centering,
            "networks": tuple(networks),
            "ref_threshold": float(ref_threshold),
            "phase_cone": float(phase_cone),
            "mag_threshold": float(mag_threshold),
            "max_frames": max_frames,
            "n_drop": int(n_drop),
            "random_state": int(random_state),
            "skip_existing": bool(skip_existing),
            "atlas_maps_path": atlas_maps_path,
            "templates_override": templates_override,
            "max_iter_sea": int(max_iter_sea),
            "max_iter_north": int(max_iter_north),
            "max_iter_refine": int(max_iter_refine),
        }
        for item in inputs
    ]

    workers = (os.cpu_count() or 1) if n_jobs is None else int(n_jobs)
    if workers == 1 or len(jobs) == 1:
        return [
            _worker_entry(job)
            for job in tqdm(jobs, desc="hilbert-ica", unit="subject")
        ]

    iterator = Parallel(n_jobs=workers, backend="loky", return_as="generator")(
        delayed(_worker_entry)(job) for job in jobs
    )
    return list(tqdm(iterator, total=len(jobs), desc="hilbert-ica", unit="subject"))
