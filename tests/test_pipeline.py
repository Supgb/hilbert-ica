from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image

from hilbert_ica import discover_inputs, load_subject_npz, process_cohort, reconstruct_bold
from hilbert_ica.main import main as cli_main


def _write_synthetic_bold(path: Path, seed: int, shape: tuple[int, int, int, int] = (6, 6, 5, 28)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nx, ny, nz, nt = shape
    vol = rng.normal(loc=200.0, scale=8.0, size=shape).astype(np.float32)
    # Add dead corners so mean-rule brain mask excludes them
    vol[0, 0, :, :] = 0.0

    t_axis = np.linspace(0, 4 * np.pi, nt, dtype=np.float32)
    dmn_tpl = np.zeros((nx, ny, nz), dtype=np.float64)
    dmn_tpl[1:3, 1:3, 1:3] = 5.0
    aud_tpl = np.zeros((nx, ny, nz), dtype=np.float64)
    aud_tpl[3:5, 3:5, 2:4] = 5.0
    ecn_tpl = np.zeros((nx, ny, nz), dtype=np.float64)
    ecn_tpl[1:3, 3:5, 3:5] = 5.0

    vol += (dmn_tpl[..., None] * (6.0 * np.sin(t_axis))[None, None, None, :]).astype(np.float32)
    vol += (aud_tpl[..., None] * (6.0 * np.cos(1.7 * t_axis))[None, None, None, :]).astype(np.float32)
    vol += (ecn_tpl[..., None] * (6.0 * np.sin(2.3 * t_axis + 0.4))[None, None, None, :]).astype(np.float32)

    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    img = Nifti1Image(vol, affine)
    img.header.set_zooms((3.0, 3.0, 3.0, 2.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib_save(img, str(path))
    return {
        "default_mode": dmn_tpl,
        "auditory": aud_tpl,
        "executive_control": ecn_tpl,
    }


def test_discover_bids_and_manifests(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    b1 = bids / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"
    b2 = bids / "sub-02" / "ses-1" / "func" / "sub-02_ses-1_task-rest_bold.nii.gz"
    _write_synthetic_bold(b1, seed=1)
    _write_synthetic_bold(b2, seed=2)

    found = discover_inputs(bids_dir=bids)
    assert [item.subject for item in found] == ["sub-01", "sub-02"]

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "subjects": [
                    {"subject": "sub-01", "bold": str(b1)},
                    {"subject": "sub-02", "bold": str(b2)},
                ]
            }
        ),
        encoding="utf-8",
    )
    from_manifest = discover_inputs(file_list=manifest, limit=1)
    assert len(from_manifest) == 1
    assert from_manifest[0].subject == "sub-01"


def test_parallel_cohort_and_resume_skip_existing(tmp_path: Path) -> None:
    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    templates = _write_synthetic_bold(in_dir / "sub-001.nii.gz", seed=10)
    _write_synthetic_bold(in_dir / "sub-002.nii.gz", seed=11)

    inputs = discover_inputs(data_dir=in_dir)
    assert [i.subject for i in inputs] == ["sub-001", "sub-002"]

    results = process_cohort(
        inputs,
        out_dir,
        n_jobs=2,
        model_orders=(5,),
        num_runs=1,
        templates_override=templates,
        max_iter_sea=20,
        max_iter_north=60,
        max_iter_refine=30,
    )
    assert [r["status"] for r in results] == ["ok", "ok"]

    payload = load_subject_npz(out_dir / "sub-001.npz")
    expected_keys = {
        "selected_maps",
        "selected_raw_maps",
        "selected_time_courses",
        "network_names",
        "correlations",
        "selected_components",
        "selected_runs",
        "selected_orders",
        "mask",
        "affine",
        "tr",
    }
    assert expected_keys.issubset(set(payload.keys()))
    assert "bold" not in payload  # compact factorized storage only

    n_vox = int(payload["mask"].sum())
    assert payload["selected_maps"].shape == (3, n_vox)
    assert payload["selected_maps"].dtype == np.complex64
    assert payload["selected_time_courses"].shape == (28, 3)
    assert list(payload["network_names"]) == [
        "default_mode",
        "auditory",
        "executive_control",
    ]

    bold = reconstruct_bold(
        payload["selected_maps"],
        payload["selected_time_courses"],
        selected_components=payload["selected_components"],
    )
    assert bold.shape == (n_vox, 28)

    # Re-running with skip_existing=True skips both subjects without touching ICA
    rerun = process_cohort(
        inputs,
        out_dir,
        n_jobs=2,
        model_orders=(5,),
        templates_override=templates,
        skip_existing=True,
    )
    assert [r["status"] for r in rerun] == ["skipped", "skipped"]


def test_cli_help_exits_cleanly() -> None:
    try:
        cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
