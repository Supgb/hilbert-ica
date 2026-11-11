"""Command-line interface for parallel Hilbert-phase Complex ICA.

Parses input discovery options (explicit NIfTIs, flat directory, manifest, or
BIDS tree) and algorithm hyperparameters, runs ``process_cohort``, and writes
a cohort-level ``summary.json`` alongside the per-subject ``<subject>.npz`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hilbert_ica.io import discover_inputs
from hilbert_ica.pipeline import pin_blas_threads, process_cohort
from hilbert_ica.selection import (
    DEFAULT_MAG_THRESHOLD,
    DEFAULT_NETWORKS,
    DEFAULT_PHASE_CONE,
    DEFAULT_REF_THRESHOLD,
    SMITH_RSN_NAMES,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``hilbert-ica`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hilbert-ica",
        description=(
            "Run Yu's Hilbert-phase Complex ICA (EBM) workflow in parallel and "
            "save Smith-2009 selected RSN spatial maps and time courses per subject."
        ),
    )

    # Mutually exclusive ways to specify subject NIfTI inputs
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--files",
        nargs="+",
        help="One or more 4D BOLD NIfTI files (.nii or .nii.gz)",
    )
    source.add_argument(
        "--data-dir",
        help="Directory containing flat 4D BOLD NIfTI files",
    )
    source.add_argument(
        "--file-list",
        help="Text file (one NIfTI path per line) or JSON manifest",
    )
    source.add_argument(
        "--bids-dir",
        help="BIDS root directory searched via sub-*/**/func/<bids-pattern>",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for per-subject <subject>.npz files and summary.json",
    )
    parser.add_argument(
        "--bids-pattern",
        default="*_bold.nii*",
        help="Filename glob inside func/ when using --bids-dir (default: *_bold.nii*)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subset of subject IDs to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N discovered subjects",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional explicit 3D brain mask NIfTI (default: Yu mean-rule mask per subject)",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[40],
        help="Complex PCA/ICA model orders (default: 40; paper uses: 40 50 60)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent EBM runs per model order (default: 1; paper uses: 4)",
    )
    parser.add_argument(
        "--centering",
        choices=("timepoint", "voxel"),
        default="timepoint",
        help="Pre-PCA centering: 'timepoint' (ICA_complex.m default) or 'voxel'",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=list(DEFAULT_NETWORKS),
        help=(
            "Smith-2009 RSN names to select, or 'all' for all 10 "
            f"(default: {' '.join(DEFAULT_NETWORKS)})"
        ),
    )
    parser.add_argument(
        "--ref-threshold",
        type=float,
        default=DEFAULT_REF_THRESHOLD,
        help=f"Smith template lower cutoff (default: {DEFAULT_REF_THRESHOLD})",
    )
    parser.add_argument(
        "--phase-cone",
        type=float,
        default=DEFAULT_PHASE_CONE,
        help="Phase denoising half-width in radians (default: pi/4)",
    )
    parser.add_argument(
        "--mag-threshold",
        type=float,
        default=DEFAULT_MAG_THRESHOLD,
        help=f"Source magnitude denoising cutoff (default: {DEFAULT_MAG_THRESHOLD})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on leading timepoints (e.g. 146 in UCLA script)",
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=0,
        help="Optional edge frames trimmed from each end after Hilbert transform (default: 0)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel worker processes (default: CPU core count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Complex ICA-EBM initialization (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute subjects even if <out>/<subject>.npz already exists",
    )
    return parser


def _resolve_networks(raw_networks: Sequence[str]) -> tuple[str, ...]:
    """Expand ``--networks all`` to all 10 Smith-2009 RSN names."""
    if len(raw_networks) == 1 and raw_networks[0].lower() == "all":
        return SMITH_RSN_NAMES
    return tuple(raw_networks)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags, run cohort processing, and write ``<out>/summary.json``."""
    pin_blas_threads(1)
    parser = build_parser()
    args = parser.parse_args(argv)

    networks = _resolve_networks(args.networks)
    inputs = discover_inputs(
        files=args.files,
        data_dir=args.data_dir,
        file_list=args.file_list,
        bids_dir=args.bids_dir,
        bids_pattern=args.bids_pattern,
        subjects=args.subjects,
        limit=args.limit,
    )
    if not inputs:
        raise SystemExit("No matching NIfTI inputs found.")

    out_dir = Path(args.out)
    results = process_cohort(
        inputs,
        out_dir,
        n_jobs=args.n_jobs,
        mask_path=args.mask,
        model_orders=args.orders,
        num_runs=args.num_runs,
        centering=args.centering,
        networks=networks,
        ref_threshold=args.ref_threshold,
        phase_cone=args.phase_cone,
        mag_threshold=args.mag_threshold,
        max_frames=args.max_frames,
        n_drop=args.n_drop,
        random_state=args.seed,
        skip_existing=not args.overwrite,
    )

    n_ok = sum(1 for row in results if row.get("status") == "ok")
    n_skip = sum(1 for row in results if row.get("status") == "skipped")
    summary = {
        "n_subjects": len(results),
        "n_processed": n_ok,
        "n_skipped": n_skip,
        "params": {
            "orders": list(args.orders),
            "num_runs": args.num_runs,
            "centering": args.centering,
            "networks": list(networks),
            "ref_threshold": args.ref_threshold,
            "phase_cone": args.phase_cone,
            "mag_threshold": args.mag_threshold,
            "max_frames": args.max_frames,
            "n_drop": args.n_drop,
            "seed": args.seed,
        },
        "subjects": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_subjects": len(results),
                "n_processed": n_ok,
                "n_skipped": n_skip,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
