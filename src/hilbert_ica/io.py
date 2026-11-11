"""Dataset-agnostic NIfTI / BIDS discovery and atomic per-subject ``.npz`` I/O.

This module isolates all filesystem interactions:
- Discovering input 4D BOLD NIfTI files from explicit file paths, a flat directory,
  a newline-separated or JSON manifest, or a BIDS dataset tree.
- Loading 4D BOLD NIfTIs and optional 3D brain-mask NIfTIs.
- Writing and reading per-subject ``.npz`` archives atomically so interrupted or
  parallel runs never leave half-written files behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

NIFTI_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")


@dataclass(frozen=True)
class SubjectInput:
    """Resolved subject identifier and path to its 4D BOLD NIfTI file."""

    subject: str
    bold_path: Path


def nifti_stem(path: Path | str) -> str:
    """Strip ``.nii.gz`` or ``.nii`` from a filename to get a clean base name."""
    name = Path(path).name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def sidecar_json_path(nifti_path: Path | str) -> Path:
    """Return the companion BIDS JSON sidecar path (``<stem>.json``) for a NIfTI file."""
    return Path(nifti_path).with_name(nifti_stem(nifti_path) + ".json")


def _tr_from_sidecar(sidecar: Path) -> float | None:
    """Read ``RepetitionTime`` (in seconds) from a BIDS JSON sidecar if available."""
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    val = payload.get("RepetitionTime")
    if val is None:
        return None
    tr = float(val)
    return tr if tr > 0 else None


def load_bold_nifti(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64], float, Nifti1Image]:
    """Load a 4D BOLD NIfTI image, its 4x4 affine matrix, and repetition time (TR).

    TR is read from the BIDS sidecar JSON if present, otherwise from NIfTI header
    ``pixdim[4]`` (falling back to ``2.0`` s if unset, since the FFT Hilbert
    transform itself is scale-invariant in sampling rate).
    """
    nifti_path = Path(path)
    if not nifti_path.exists():
        raise FileNotFoundError(nifti_path)
    img = nib_load(str(nifti_path))
    if not isinstance(img, Nifti1Image):
        raise TypeError(f"expected Nifti1Image for {nifti_path}, got {type(img)}")
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"expected 4D BOLD NIfTI at {nifti_path}, got shape {data.shape}")
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) >= 4 else 0.0
    sidecar_tr = _tr_from_sidecar(sidecar_json_path(nifti_path))
    if sidecar_tr is not None:
        tr = sidecar_tr
    elif tr <= 0:
        tr = 2.0
    return data, np.asarray(img.affine, dtype=np.float64), tr, img


def load_mask_nifti(path: Path | str, expected_shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Load an explicit 3D brain-mask NIfTI and verify it matches ``expected_shape``."""
    img = nib_load(str(path))
    mask = np.asanyarray(img.dataobj) > 0
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask shape {mask.shape} at {path} does not match BOLD shape {expected_shape}"
        )
    if not np.any(mask):
        raise ValueError(f"mask at {path} is empty")
    return mask


def _resolve_relative(raw: str, base_dir: Path) -> Path:
    """Resolve relative paths in a manifest against the manifest's parent directory."""
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _subject_from_path(path: Path) -> str:
    """Infer a subject ID from a ``sub-*`` parent folder if present, else the file stem."""
    stem = nifti_stem(path)
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return stem


def _parse_file_list(file_list_path: Path) -> list[SubjectInput]:
    """Parse either a plain-text NIfTI list (one path per line) or a JSON manifest."""
    text = file_list_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    base = file_list_path.parent
    if text.startswith("[") or text.startswith("{"):
        payload = json.loads(text)
        return _parse_json_manifest(payload, base)
    items: list[SubjectInput] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        bold = _resolve_relative(stripped, base)
        items.append(SubjectInput(subject=_subject_from_path(bold), bold_path=bold))
    return items


def _parse_json_manifest(payload: Any, base: Path) -> list[SubjectInput]:
    """Extract ``SubjectInput`` items from list-style or object-style JSON manifests."""
    if isinstance(payload, list):
        items: list[SubjectInput] = []
        for entry in payload:
            if isinstance(entry, str):
                bold = _resolve_relative(entry, base)
                items.append(SubjectInput(subject=_subject_from_path(bold), bold_path=bold))
            elif isinstance(entry, dict):
                bold_raw = entry.get("bold") or entry.get("local_bold") or entry.get("path")
                if not bold_raw:
                    raise ValueError(f"manifest entry missing bold/path key: {entry}")
                bold = _resolve_relative(str(bold_raw), base)
                subj = str(entry.get("subject") or entry.get("name") or _subject_from_path(bold))
                items.append(SubjectInput(subject=subj, bold_path=bold))
            else:
                raise ValueError(f"unsupported manifest list element: {type(entry)}")
        return items
    if isinstance(payload, dict):
        if "files" in payload:
            return _parse_json_manifest(payload["files"], base)
        if "subjects" in payload:
            return _parse_json_manifest(payload["subjects"], base)
    raise ValueError("JSON file list must be a list or an object with 'files' or 'subjects'")


def _deduplicate_subject_names(items: list[SubjectInput]) -> list[SubjectInput]:
    """Ensure every ``SubjectInput`` has a unique output name so files never collide."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.subject] = counts.get(item.subject, 0) + 1
    if all(count == 1 for count in counts.values()):
        return items
    resolved: list[SubjectInput] = []
    seen: dict[str, int] = {}
    for item in items:
        if counts[item.subject] == 1:
            resolved.append(item)
        else:
            stem = nifti_stem(item.bold_path)
            if stem != item.subject:
                name = stem
            else:
                idx = seen.get(item.subject, 0) + 1
                seen[item.subject] = idx
                name = f"{item.subject}_{idx}"
            resolved.append(SubjectInput(subject=name, bold_path=item.bold_path))
    return resolved


def discover_bids_bold(
    bids_dir: Path | str,
    *,
    bids_pattern: str = "*_bold.nii*",
    subjects: Sequence[str] | None = None,
) -> list[SubjectInput]:
    """Discover functional BOLD NIfTIs inside a BIDS tree (``sub-*/**/func/<pattern>``).

    If a subject has a single matching functional run, the output is named
    ``sub-<id>``; if a subject has multiple sessions/runs, each run is named by
    its full BIDS filename stem so runs do not overwrite one another.
    """
    root = Path(bids_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    wanted = set(subjects) if subjects else None
    found: list[SubjectInput] = []
    sub_dirs = sorted(p for p in root.glob("sub-*") if p.is_dir())
    for sub_dir in sub_dirs:
        sub_name = sub_dir.name
        if wanted is not None and sub_name not in wanted:
            continue
        matches = sorted(
            p
            for p in sub_dir.glob(f"**/func/{bids_pattern}")
            if p.is_file() and p.name.endswith(NIFTI_SUFFIXES)
        )
        if not matches:
            continue
        if len(matches) == 1:
            found.append(SubjectInput(subject=sub_name, bold_path=matches[0]))
        else:
            for match in matches:
                found.append(SubjectInput(subject=nifti_stem(match), bold_path=match))
    return _deduplicate_subject_names(found)


def discover_inputs(
    *,
    files: Sequence[Path | str] | None = None,
    data_dir: Path | str | None = None,
    file_list: Path | str | None = None,
    bids_dir: Path | str | None = None,
    bids_pattern: str = "*_bold.nii*",
    subjects: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[SubjectInput]:
    """Resolve input subjects from exactly one source (files, dir, manifest, or BIDS)."""
    sources_set = sum(
        x is not None and (not isinstance(x, Sequence) or len(x) > 0)
        for x in (files, data_dir, file_list, bids_dir)
    )
    if sources_set == 0:
        raise ValueError(
            "provide one input source: files, data_dir, file_list, or bids_dir"
        )
    if sources_set > 1:
        raise ValueError(
            "provide only one input source among files, data_dir, file_list, bids_dir"
        )

    if files:
        resolved = [
            SubjectInput(subject=nifti_stem(Path(p)), bold_path=Path(p))
            for p in files
        ]
    elif file_list is not None:
        resolved = _parse_file_list(Path(file_list))
    elif data_dir is not None:
        directory = Path(data_dir)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        niftis: dict[Path, Path] = {}
        for suffix in NIFTI_SUFFIXES:
            for p in directory.glob(f"*{suffix}"):
                if p.is_file():
                    niftis[p.resolve()] = p
        ordered = sorted(niftis.values(), key=lambda p: p.name)
        resolved = [SubjectInput(subject=nifti_stem(p), bold_path=p) for p in ordered]
    else:
        assert bids_dir is not None
        resolved = discover_bids_bold(
            bids_dir, bids_pattern=bids_pattern, subjects=subjects
        )

    resolved = _deduplicate_subject_names(resolved)
    if subjects and bids_dir is None:
        wanted = set(subjects)
        resolved = [
            item
            for item in resolved
            if item.subject in wanted or nifti_stem(item.bold_path) in wanted
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        resolved = resolved[:limit]
    return resolved


def subject_npz_path(out_dir: Path | str, subject: str) -> Path:
    """Return the canonical output path ``<out_dir>/<subject>.npz``."""
    return Path(out_dir) / f"{subject}.npz"


def subject_npz_is_complete(path: Path | str) -> bool:
    """Return True if ``path`` exists and contains all required output arrays."""
    npz_file = Path(path)
    if not npz_file.is_file():
        return False
    try:
        with np.load(npz_file, allow_pickle=False) as archive:
            required = {"selected_maps", "selected_time_courses", "mask", "affine"}
            return required.issubset(set(archive.files))
    except Exception:
        return False


def write_subject_npz(out_path: Path | str, arrays: Mapping[str, Any]) -> Path:
    """Write a compressed ``.npz`` archive atomically via temporary file + rename.

    Writing to a hidden sibling temporary file and calling ``os.replace``
    guarantees that an interrupted process never leaves a corrupted ``<subject>.npz``
    that would fool ``skip_existing`` on a subsequent run.
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}_",
        suffix=".tmp.npz",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_subject_npz(path: Path | str) -> dict[str, NDArray]:
    """Load all arrays from a per-subject ``.npz`` archive into a plain dictionary."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}
