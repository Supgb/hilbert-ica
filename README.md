# `hilbert-ica`

**English** | [简体中文](README_zh.md)

Minimal, self-contained Python implementation of Yu's **Hilbert-phase Complex ICA** workflow for magnitude-only fMRI, supporting parallel cohort execution and Smith (2009) resting-state network (RSN) component selection.

## References

1. M.-C. Yu *et al.*, *"Uncovering abnormal patterns of schizophrenia from fMRI regional source phase"* (`phase_fmri_Hilbert.m`, `complex_fmri_HilbertPhase.m`, `ICA_complex.m`, `energymax_T1.m`, `phase_denoising.m`, `selection_component_complex.m`).
2. X.-L. Li and T. Adali, *"Complex independent component analysis by entropy bound minimization,"* IEEE Trans. Circuits Syst. I, vol. 57, no. 7, pp. 1417-1430, 2010 (`complex_ICA_EBM.m`).
3. S. M. Smith *et al.*, *"Correspondence of the brain's functional architecture during activation and rest,"* PNAS, 2009.

---

## Pipeline Overview

For each 4D magnitude BOLD NIfTI volume (executed in parallel across subjects with BLAS pinned to 1 thread per worker):

1. **In-brain Masking (`hilbert.py`)**: Keeps voxels whose magnitude exceeds the grand mean for at least half of the time frames (`Complex_ICA.m`), or uses an explicit 3D mask via `--mask`.
2. **Hilbert Complex Construction (`hilbert.py`)**: Computes instantaneous phase via FFT analytic signal along time and constructs complex fMRI data:
   \[
   z(t, v) = x(t, v) \exp\!\bigl(i \arg(\mathcal{H}\{x\}(t, v))\bigr)
   \]
3. **Spatial Demeaning & Complex PCA (`ica.py`)**: Subtracts the spatial mean per timepoint (`icatb_preproc_data(..., 'remove mean per timepoint')`) and reduces dimensionality to model order \(K\) (default `40`; configurable e.g. `40 50 60`).
4. **Complex ICA-EBM & Back-Projection (`ebm.py`, `ica.py`)**: Separates complex spatial sources via Entropy Bound Minimization (`SEA` initialization + stochastic `North` stage + elliptical refinement) and estimates complex time courses via pseudo-inverse projection \(T_c = Z \, S_e^{+}\).
5. **Phase De-Ambiguity, Denoising & Smith-2009 Selection (`selection.py`)**:
   - Rotates each component's phase to maximize real-part time-course energy (`energymax_T1`) and aligns sign positively with the Smith template (`ref_threshold=2.5`).
   - Applies phase-cone denoising (\(|\theta| \le \pi/4\)) and magnitude thresholding (\(|s| < 0.5 \to 0\)).
   - Selects the winning component (and winning run/order when `--num-runs > 1` or multiple `--orders` are used) maximizing \(|\operatorname{corr}(|s_{\text{denoised}}|, \text{ref})|\) for each target RSN (default: `default_mode`, `auditory`, `executive_control`; pass `--networks all` for all 10 Smith RSNs).

---

## Installation

```bash
pip install -e .
```

## CLI Usage

Run directly via `python main.py` from the repository root or the installed `hilbert-ica` entry point:

```bash
# Flat directory of NIfTI files (fast default: K=40, 1 run, 3 paper RSNs)
python main.py --data-dir /path/to/niftis --out /path/to/out --n-jobs 8

# BIDS dataset tree
python main.py --bids-dir /path/to/bids --bids-pattern "*_desc-preproc_bold.nii.gz" \
    --out /path/to/out --n-jobs 8

# Full paper multi-order x multi-run configuration across all 10 Smith RSNs
python main.py --files sub-001.nii sub-002.nii --out /path/to/out \
    --orders 40 50 60 --num-runs 4 --networks all --max-frames 146
```

Re-running on the same `--out` directory automatically skips completed `<subject>.npz` files unless `--overwrite` is passed.

---

## Output Format (`<out>/<subject>.npz`)

Each subject is written atomically to a standalone compressed `.npz` archive containing only the compact factorized maps and time courses:

| Key | Shape & Dtype | Description |
|---|---|---|
| `selected_maps` | `(M, V)` `complex64` | Phase-corrected, denoised (`\|angle\| <= pi/4`, `\|s\| >= 0.5`) spatial maps |
| `selected_raw_maps` | `(M, V)` `complex64` | Phase- and sign-corrected spatial maps before cone/magnitude zeroing |
| `selected_time_courses` | `(T, M)` `complex64` | Phase- and sign-corrected complex time courses |
| `network_names` | `(M,)` `str` | Selected Smith-2009 network names |
| `correlations` | `(M,)` `float32` | Winning `\|corr(\|selected_maps[m]\|, ref_m)\|` |
| `selected_components` | `(M,)` `int32` | Winning ICA component index (plus `selected_runs`, `selected_orders`) |
| `mask` | `(nx, ny, nz)` `bool` | 3D in-brain voxel mask (`volume[mask]` maps to axis `V`) |
| `affine` / `tr` | `(4, 4)` / scalar `float64` | Spatial affine matrix and repetition time (seconds) |

### Python API & On-the-Fly Complex BOLD Reconstruction

```python
import numpy as np
from hilbert_ica import load_subject_npz, reconstruct_bold

sub = load_subject_npz("outputs/sub-001.npz")
maps = sub["selected_maps"]               # (M, V) complex64: magnitude = np.abs(maps), phase = np.angle(maps)
tc = sub["selected_time_courses"]         # (T, M) complex64

# Optional: reconstruct (V, T) complex-valued fMRI time series on the fly
bold_vt = reconstruct_bold(
    maps,
    tc,
    selected_components=sub["selected_components"],
    selected_runs=sub["selected_runs"],
    selected_orders=sub["selected_orders"],
)
```
