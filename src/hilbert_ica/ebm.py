"""Complex Independent Component Analysis by Entropy Bound Minimization (EBM).

This module implements Li & Adali's Complex ICA-EBM algorithm in NumPy, matching
the official MLSP MATLAB implementation (``complex_ICA_EBM.m``) and using the
official piecewise-polynomial lookup tables stored in ``ebm_nf_tables.npz``.

Why EBM is used for complex fMRI:
- Standard real-valued ICA assumes real signals, while many complex ICA methods
  assume circularly symmetric (proper) complex sources whose real and imaginary
  parts have identical variance and zero correlation.
- Complex fMRI sources can be non-circular (improper). EBM handles arbitrary
  sub-Gaussian, super-Gaussian, skewed, and non-circular complex distributions
  by approximating each source's differential entropy with tight upper bounds
  computed from smooth measuring functions.

Algorithm overview (3 optimization stages):
1. Pre-whitening (``_prewhiten``): Center each mixture row and apply a Hermitian
   whitening matrix ``P`` so the mixtures have identity sample covariance.
2. Orthogonal initialization (``_sea_initialize``): Run up to ``max_iter_sea``
   kurtosis-like fixed-point iterations with symmetric orthogonalization to get
   a reliable starting demixing matrix.
3. Non-orthogonal Newton/gradient stages (``_north_stage``, run twice):
   - Stage 3a (stochastic search, real-bivariate bounds only, ``mu0 = 1/5``)
     escapes local minima using random sample weights.
   - Stage 3b (deterministic refinement, adding circular/elliptical bounds,
     ``mu0 = 1/25``) polishes the demixing vectors and sorts components by
     descending estimated negentropy.

Reference
---------
Xi-Lin Li and Tulay Adali, "Complex independent component analysis by entropy
bound minimization," IEEE Trans. Circuits Syst. I, vol. 57, no. 7,
pp. 1417-1430, 2010.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_TABLE_PATH = Path(__file__).resolve().parent / "ebm_nf_tables.npz"
_EPS = 1e-12


@dataclass(frozen=True)
class _SplineTable:
    """Piecewise-polynomial lookup table mapping E[G(x)] -> negentropy bound.

    Each table comes from ``nf_table.mat`` or ``complex_nf_table.mat`` and holds
    both the cubic spline coefficients for the negentropy bound itself and the
    spline coefficients for its first derivative (slope).
    """

    min_val: float
    max_val: float
    breaks: NDArray[np.float64]
    coefs: NDArray[np.float64]
    slope_breaks: NDArray[np.float64]
    slope_coefs: NDArray[np.float64]


def _as_mixtures(x: NDArray) -> NDArray[np.complex128]:
    """Validate that input mixtures form a 2D complex128 matrix ``(K, T)``."""
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    return arr


def _ppval(breaks: NDArray[np.float64], coefs: NDArray[np.float64], value: float) -> float:
    """Evaluate a 1D piecewise polynomial at ``value`` using Horner's method.

    Mirrors ``simplified_ppval`` in ``complex_ICA_EBM.m``: locate the interval
    ``[breaks[i], breaks[i+1])`` containing ``value`` (clamped to the first or
    last piece at the boundaries) and evaluate the polynomial in local
    coordinates ``value - breaks[i]``.
    """
    n_pieces = int(coefs.shape[0])
    index = int(np.searchsorted(breaks, value, side="right") - 1)
    index = min(max(index, 0), n_pieces - 1)
    local = value - float(breaks[index])
    result = float(coefs[index, 0])
    for coeff in coefs[index, 1:]:
        result = local * result + float(coeff)
    return result


def _negentropy_bound(table: _SplineTable, expectation: float, *, clamp_only: bool) -> float:
    """Look up the negentropy upper bound for sample expectation ``E[G(x)]``.

    When ``expectation`` falls outside the tabulated range ``[min_val, max_val]``:
    - ``clamp_only=True`` (used by quartic functions ``G1`` and ``CG1``) clamps
      the input to the nearest endpoint.
    - ``clamp_only=False`` linearly extrapolates outward using the endpoint
      derivative magnitude so the bound continues to grow smoothly.
    """
    if clamp_only:
        clipped = min(max(expectation, table.min_val), table.max_val)
        return _ppval(table.breaks, table.coefs, clipped)
    if expectation < table.min_val:
        extra = _ppval(table.slope_breaks, table.slope_coefs, table.min_val) * (
            expectation - table.min_val
        )
        return _ppval(table.breaks, table.coefs, table.min_val) + abs(extra)
    if expectation > table.max_val:
        extra = _ppval(table.slope_breaks, table.slope_coefs, table.max_val) * (
            expectation - table.max_val
        )
        return _ppval(table.breaks, table.coefs, table.max_val) + abs(extra)
    return _ppval(table.breaks, table.coefs, expectation)


def _slope_at(table: _SplineTable, expectation: float, *, clamp: bool) -> float:
    """Evaluate the derivative of the negentropy bound with respect to ``E[G(x)]``."""
    value = (
        min(max(expectation, table.min_val), table.max_val) if clamp else expectation
    )
    return _ppval(table.slope_breaks, table.slope_coefs, value)


def _load_tables() -> dict[str, _SplineTable]:
    """Load the 6 active real and complex negentropy spline tables from disk."""
    data = np.load(_TABLE_PATH)
    tables: dict[str, _SplineTable] = {}
    for key in ("real_nf1", "real_nf3", "real_nf5", "real_nf7", "complex_nf1", "complex_nf3"):
        tables[key] = _SplineTable(
            min_val=float(data[f"{key}_min"]),
            max_val=float(data[f"{key}_max"]),
            breaks=np.asarray(data[f"{key}_breaks"], dtype=np.float64),
            coefs=np.asarray(data[f"{key}_coefs"], dtype=np.float64),
            slope_breaks=np.asarray(data[f"{key}_slope_breaks"], dtype=np.float64),
            slope_coefs=np.asarray(data[f"{key}_slope_coefs"], dtype=np.float64),
        )
    return tables


_TABLES = _load_tables()
_G1 = _TABLES["real_nf1"]      # Real measuring function 1: x^4
_G3 = _TABLES["real_nf3"]      # Real measuring function 3: |x| / (1 + |x|)
_G5 = _TABLES["real_nf5"]      # Real measuring function 5: x |x| / (10 + |x|)
_G7 = _TABLES["real_nf7"]      # Real measuring function 7: x / (1 + x^2)
_CG1 = _TABLES["complex_nf1"]  # Elliptical measuring function 1: r^4
_CG3 = _TABLES["complex_nf3"]  # Elliptical measuring function 3: r / (1 + r)


def _inv_sqrtm_hermitian(gram: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Compute the inverse matrix square root ``B^{-1/2}`` of a Hermitian matrix ``B``."""
    hermitian = 0.5 * (gram + gram.conj().T)
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    eigvals = np.maximum(np.real(eigvals), _EPS)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.conj().T


def _symmetric_orthogonalize(weights: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Project square matrix ``W`` onto the unitary group via ``(W W^H)^{-1/2} W``."""
    return _inv_sqrtm_hermitian(weights @ weights.conj().T) @ weights


def _prewhiten(
    mixtures: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Remove row means and whiten ``mixtures`` ``(K, T)`` to identity covariance.

    Mirrors ``pre_processing`` in ``complex_ICA_EBM.m``: returns both the whitened
    data ``Xc = P @ (X - mean(X))`` and the whitening matrix ``P = R^{-1/2}``.
    """
    n_sources, n_samples = mixtures.shape
    centered = mixtures - mixtures.mean(axis=1, keepdims=True)
    cov = (centered @ centered.conj().T) / n_samples
    whitener = _inv_sqrtm_hermitian(cov)
    return whitener @ centered, whitener


def _real_expectations(values: NDArray[np.float64]) -> dict[int, float]:
    """Compute sample expectations ``E[G_k(x)]`` for the 4 real measuring functions."""
    n_samples = values.size
    abs_values = np.abs(values)
    return {
        1: float(np.dot(values**2, values**2) / n_samples),
        3: float(np.mean(abs_values / (1.0 + abs_values))),
        5: float(np.mean(values * abs_values / (10.0 + abs_values))),
        7: float(np.mean(values / (1.0 + values * values))),
    }


def _standardize_uv(
    source: NDArray[np.complex128],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
    float,
    float,
]:
    """Whiten the 2D real/imaginary parts of a complex source into uncorrelated ``(u, v)``.

    Given complex samples ``z = z_r + j z_i`` with variances ``(sigma_r^2, sigma_i^2)``,
    real-imaginary covariance ``rho``, and determinant ``Delta1 = sigma_r^2 sigma_i^2 - rho^2``:
    - ``u = z_r / sigma_r`` has unit variance.
    - ``v = (sigma_r z_i - rho z_r / sigma_r) / sqrt(Delta1)`` has unit variance and
      is uncorrelated with ``u``, allowing 2D entropy to be bounded via 1D marginals.
    """
    real = np.real(source)
    imag = np.imag(source)
    n_samples = source.size
    sigma_r2 = float(np.dot(real, real) / n_samples)
    sigma_i2 = float(np.dot(imag, imag) / n_samples)
    rho = float(np.dot(real, imag) / n_samples)
    sigma_r2 = max(sigma_r2, _EPS)
    sigma_i2 = max(sigma_i2, _EPS)
    sigma_r = float(np.sqrt(sigma_r2))
    delta1 = max(sigma_r2 * sigma_i2 - rho * rho, _EPS)
    sqrt_delta = float(np.sqrt(delta1))
    u_values = real / sigma_r
    v_values = sigma_r * imag / sqrt_delta - rho * real / (sigma_r * sqrt_delta)
    return u_values, v_values, sigma_r, sigma_r2, sigma_i2, rho, delta1


def _best_pair(bound_u: dict[int, float], bound_v: dict[int, float]) -> tuple[int, int, float]:
    """Find the measuring-function pair ``(p, q)`` giving the tightest (largest) negentropy bound."""
    best = -np.inf
    best_p = 1
    best_q = 1
    for p_idx, bound_p in bound_u.items():
        for q_idx, bound_q in bound_v.items():
            total = bound_p + bound_q
            if total > best:
                best = total
                best_p = p_idx
                best_q = q_idx
    return best_p, best_q, float(best)


def _null_space_vector(
    weights: NDArray[np.complex128],
    row: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Construct a vector orthogonal to all rows of ``weights`` except ``row``.

    In non-orthogonal ICA, the ``-log |det(W)|`` penalty gradient requires a
    unit-direction vector ``h`` in the null space of the other ``K - 1`` rows
    (computed via Gram-Schmidt projection of a random Gaussian probe vector).
    """
    n_sources = weights.shape[0]
    probe = rng.normal(size=n_sources) + 1j * rng.normal(size=n_sources)
    if n_sources == 1:
        return probe
    others = np.concatenate([weights[:row], weights[row + 1 :]], axis=0)
    gram = others @ others.conj().T
    projected = others.conj().T @ np.linalg.solve(gram, others @ probe)
    return probe - projected


def _g_score(kind: int, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the first derivative ``g_k(x) = G_k'(x)`` of real measuring function ``kind``."""
    abs_values = np.abs(values)
    if kind == 1:
        return 4.0 * values**3
    if kind == 3:
        return np.sign(values) / (1.0 + abs_values) ** 2
    if kind == 5:
        return abs_values * (20.0 + abs_values) / (10.0 + abs_values) ** 2
    if kind == 7:
        return (1.0 - values * values) / (1.0 + values * values) ** 2
    raise ValueError(f"unsupported measuring function {kind}")


def _slope_for(kind: int, expectation: float) -> float:
    """Return spline derivative ``d(NE_bound)/d(E[G_k])`` for real measuring function ``kind``."""
    if kind == 1:
        return _slope_at(_G1, expectation, clamp=False)
    if kind == 3:
        return _slope_at(_G3, expectation, clamp=True)
    if kind == 5:
        return _slope_at(_G5, expectation, clamp=True)
    if kind == 7:
        return _slope_at(_G7, expectation, clamp=True)
    raise ValueError(f"unsupported measuring function {kind}")


def _bound_vector(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict[int, float]]:
    """Evaluate all 4 real negentropy bounds and expectations on standardized 1D samples."""
    expectations = _real_expectations(values)
    bounds = np.zeros(9, dtype=np.float64)
    bounds[1] = _negentropy_bound(_G1, expectations[1], clamp_only=True)
    bounds[3] = _negentropy_bound(_G3, expectations[3], clamp_only=False)
    bounds[5] = _negentropy_bound(_G5, expectations[5], clamp_only=False)
    bounds[7] = _negentropy_bound(_G7, expectations[7], clamp_only=False)
    return bounds, expectations


def _sea_initialize(
    whitened: NDArray[np.complex128],
    rng: np.random.Generator,
    *,
    max_iter: int,
    tolerance: float,
    max_cost_increase: int,
) -> NDArray[np.complex128]:
    """Stage 1: Symmetric kurtosis-based fixed-point initialization (SEA).

    Provides a fast unitary initial guess ``W`` that accounts for non-circular
    pseudo-covariance ``C = E[x x^T]`` while tracking the EBM entropy-bound cost
    to retain the lowest-cost iterate.
    """
    n_sources, n_samples = whitened.shape
    weights = rng.normal(size=(n_sources, n_sources)) + 1j * rng.normal(
        size=(n_sources, n_sources)
    )
    weights = _symmetric_orthogonalize(weights)
    last_weights = weights.copy()
    best_weights = weights.copy()
    pseudo_cov = (whitened @ whitened.T) / n_samples
    min_cost = np.inf
    cost_increase = 0

    for _ in range(max_iter):
        cost = 0.0
        for row in range(n_sources):
            vector = weights[row].copy()
            source = vector @ whitened
            # Fixed-point kurtosis update accounting for non-circular pseudo-covariance
            updated = (
                whitened.conj() @ (source * source * source.conj())
            ) / n_samples - 2.0 * vector - (vector @ pseudo_cov @ vector) * np.conj(
                pseudo_cov @ vector
            )
            weights[row] = updated

            # Accumulate EBM entropy upper-bound objective for early-stopping tracking
            u_values, v_values, _, _, _, _, delta1 = _standardize_uv(source)
            cost += 0.5 * np.log(delta1)
            cost -= float(np.max(_bound_vector(u_values)[0]))
            cost -= float(np.max(_bound_vector(v_values)[0]))

        if cost < min_cost:
            min_cost = cost
            cost_increase = 0
            best_weights = last_weights.copy()
        else:
            cost_increase += 1

        weights = _symmetric_orthogonalize(weights)
        if cost_increase > max_cost_increase:
            break

        # Stop early once all rows change by less than `tolerance`
        overlap = np.abs(np.diag(weights @ last_weights.conj().T))
        if 1.0 - float(np.min(overlap)) < tolerance:
            break
        last_weights = weights.copy()

    return best_weights


def _north_stage(
    whitened: NDArray[np.complex128],
    weights0: NDArray[np.complex128],
    rng: np.random.Generator,
    *,
    max_iter: int,
    mu0: float,
    max_cost_increase: int,
    use_elliptical: bool,
    stochastic_search: bool,
) -> NDArray[np.complex128]:
    """Stage 2 & 3: Non-orthogonal EBM gradient descent on the unit-norm sphere.

    Minimizes ``J(W) = -log |det(W)| + sum_n H_bound(w_n^H Xc)`` row by row:
    - When ``use_elliptical=False, stochastic_search=True`` (Stage 2), uses random
      per-sample weights and step-size halving down to ``1/20`` to escape local minima.
    - When ``use_elliptical=True, stochastic_search=False`` (Stage 3), adds the
      complex elliptical bounds (``CG1``, ``CG3``) and deterministic weights with
      step-size halving down to ``1/200`` for high-precision convergence.
    """
    n_sources, n_samples = whitened.shape
    pseudo_cov = (whitened @ whitened.T) / n_samples
    mu = mu0
    weights = weights0.copy()
    best_weights = weights.copy()
    last_weights = weights.copy()
    min_cost = np.inf
    cost_increase = 0
    max_negentropy = np.zeros(n_sources)

    for _ in range(max_iter):
        gram_det = float(np.abs(np.linalg.det(weights.conj().T @ weights)).real)
        cost = -np.log(max(gram_det, _EPS))
        negentropy = np.zeros(n_sources)

        for row in range(n_sources):
            complement = _null_space_vector(weights, row, rng)
            w_col = weights[row].conj()
            source = w_col.conj() @ whitened
            u_values, v_values, sigma_r, sigma_r2, sigma_i2, rho, delta1 = _standardize_uv(
                source
            )
            real = np.real(source)
            imag = np.imag(source)

            # Evaluate factorized (u, v) real negentropy bounds
            bound_u, exp_u = _bound_vector(u_values)
            bound_v, exp_v = _bound_vector(v_values)
            p0, q0, bound_real = _best_pair(
                {idx: float(bound_u[idx]) for idx in range(1, 9)},
                {idx: float(bound_v[idx]) for idx in range(1, 9)},
            )

            # Optionally evaluate radial elliptical negentropy bounds (refinement stage)
            max_elliptical = -np.inf
            r_values = None
            rr_values = None
            elliptical_kind = 1
            if use_elliptical:
                delta2 = sigma_i2 * real**2 + sigma_r2 * imag**2 - 2.0 * rho * real * imag
                rr_values = np.maximum(delta2 / delta1, 0.0)
                r_values = np.sqrt(rr_values)
                exp_r4 = float(np.mean(rr_values**2))
                exp_r_frac = float(np.mean(r_values / (1.0 + r_values)))
                bound_r = {
                    1: _negentropy_bound(_CG1, exp_r4, clamp_only=True),
                    3: _negentropy_bound(_CG3, exp_r_frac, clamp_only=False),
                }
                elliptical_kind = 1 if bound_r[1] >= bound_r[3] else 3
                max_elliptical = bound_r[elliptical_kind]

            cost += 0.5 * np.log(delta1)
            negentropy[row] = -0.5 * np.log(delta1)
            grad_delta1 = (0.5 * (sigma_i2 - sigma_r2) + 1j * rho) * (
                pseudo_cov @ np.conj(w_col)
            )

            # Choose whichever family (bivariate u/v vs radial elliptical) gives the tighter bound
            if (not use_elliptical) or bound_real > max_elliptical:
                cost -= bound_real
                negentropy[row] += bound_real
                sample_weights = (
                    rng.random(n_samples) if stochastic_search else np.ones(n_samples)
                )
                weight_sum = float(np.sum(sample_weights))
                grad = 0.5 * grad_delta1 / delta1

                # Gradient contribution from standardized real component u
                if p0 in {1, 3, 5, 7}:
                    g_u = _g_score(p0, u_values)
                    vegu = _slope_for(p0, exp_u[p0])
                    grad = grad - vegu * (whitened @ (sample_weights * g_u)) / (
                        weight_sum * 2.0 * sigma_r
                    )
                    grad = grad - vegu * float(
                        np.dot(sample_weights, -g_u * real)
                    ) / (weight_sum * 4.0 * sigma_r**3) * (
                        pseudo_cov @ np.conj(w_col)
                    )

                sample_weights = (
                    rng.random(n_samples) if stochastic_search else np.ones(n_samples)
                )
                weight_sum = float(np.sum(sample_weights))

                # Gradient contribution from decorrelated imaginary component v
                if q0 in {1, 3, 5, 7}:
                    g_v = _g_score(q0, v_values)
                    vegv = _slope_for(q0, exp_v[q0])
                    scale = 2.0 * sigma_r * np.sqrt(delta1)
                    grad = grad + vegv * (whitened @ (sample_weights * g_v)) / (
                        weight_sum * scale
                    ) * (rho + 1j * sigma_r2)
                    energy = float(np.dot(sample_weights, g_v * v_values))
                    grad = (
                        grad
                        + vegv * energy / (weight_sum * 2.0 * delta1) * grad_delta1
                    )
                    corr_term = imag + (rho / sigma_r2 + 2j) * real
                    corr_scale = np.dot(sample_weights, g_v * corr_term)
                    grad = grad - vegv * corr_scale / (
                        weight_sum * 2.0 * scale
                    ) * (pseudo_cov @ np.conj(w_col))
            else:
                assert r_values is not None and rr_values is not None
                cost -= max_elliptical
                negentropy[row] += max_elliptical
                sample_weights = (
                    rng.random(n_samples) if stochastic_search else np.ones(n_samples)
                )
                weight_sum = float(np.sum(sample_weights))
                part1 = sigma_i2 * real - 1j * sigma_r2 * imag + 1j * rho * source
                part2 = 0.5 * (imag**2 - real**2) + 1j * real * imag
                if elliptical_kind == 1:
                    vegr = _slope_at(_CG1, float(np.mean(rr_values**2)), clamp=False)
                    kernel = 4.0 * sample_weights * rr_values
                else:
                    vegr = _slope_at(
                        _CG3,
                        float(np.mean(r_values / (1.0 + r_values))),
                        clamp=True,
                    )
                    inv_r = 1.0 / (r_values + np.finfo(np.float64).eps)
                    kernel = sample_weights * inv_r / (1.0 + r_values) ** 2
                grad = 0.5 * (
                    1.0 / delta1
                    + vegr * float(np.dot(kernel, delta2)) / (weight_sum * delta1**2)
                ) * grad_delta1
                grad = grad - 0.5 / delta1 * vegr * (
                    whitened @ (kernel * part1)
                ) / weight_sum
                grad = grad - 0.5 / delta1 * vegr * (
                    np.dot(kernel, part2) / weight_sum
                ) * (pseudo_cov @ np.conj(w_col))

            # Add -log|det(W)| gradient term, project onto tangent plane of unit sphere, and step
            inner = np.vdot(w_col, complement)
            if abs(inner) > _EPS:
                grad = grad - complement / inner
            grad = grad - float(np.real(np.vdot(w_col, grad))) * w_col
            norm = np.linalg.norm(grad)
            if norm > _EPS:
                grad = grad / norm
            updated = w_col - mu * grad
            updated_norm = np.linalg.norm(updated)
            if updated_norm > _EPS:
                updated = updated / updated_norm
            weights[row] = updated.conj()

        # Track lowest-cost iterate and anneal step size `mu` when cost stalls
        if cost < min_cost:
            min_cost = cost
            best_weights = last_weights.copy()
            max_negentropy = negentropy.copy()
            cost_increase = 0
        else:
            cost_increase += 1

        if cost_increase > max_cost_increase:
            mu_floor = 1.0 / 20.0 if stochastic_search else 1.0 / 200.0
            if mu > mu_floor:
                mu *= 0.5
                cost_increase = 0
                weights = best_weights.copy()
                last_weights = weights.copy()
                continue
            break
        last_weights = weights.copy()

    # Order separated components from highest to lowest estimated negentropy
    order = np.argsort(max_negentropy)[::-1]
    return best_weights[order]


def complex_ica_ebm(
    mixtures: NDArray,
    *,
    random_state: int | np.random.Generator | None = 0,
    max_iter_sea: int = 100,
    max_iter_north: int = 1000,
    max_iter_refine: int = 500,
    tolerance: float = 1e-4,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Estimate Complex ICA-EBM demixing matrix ``W`` and separated sources ``Shat``.

    Parameters
    ----------
    mixtures :
        2D complex array of shape ``(K, n_samples)`` (in spatial fMRI ICA, ``K``
        is the PCA model order and ``n_samples`` is the number of in-brain voxels).
    random_state :
        Integer seed or NumPy ``Generator`` controlling random initialization and
        stochastic-stage weights.

    Returns
    -------
    weights :
        Non-unitary demixing matrix ``W`` of shape ``(K, K)`` (composed with pre-whitener).
    sources :
        Estimated complex sources ``Shat = W @ (mixtures - mean(mixtures, axis=1))``
        of shape ``(K, n_samples)``, matching ``complex_ICA_EBM.m``.
    """
    data = _as_mixtures(mixtures)
    n_sources, n_samples = data.shape
    if n_samples < n_sources:
        raise ValueError("need at least as many samples as sources")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # Pre-whiten mixtures across samples (voxels)
    centered = data - data.mean(axis=1, keepdims=True)
    whitened, whitener = _prewhiten(centered)

    # Stage 1: Orthogonal kurtosis initialization
    weights = _sea_initialize(
        whitened,
        rng,
        max_iter=max_iter_sea,
        tolerance=tolerance,
        max_cost_increase=10,
    )
    # Stage 2: Stochastic non-orthogonal search (real-bivariate bounds)
    weights = _north_stage(
        whitened,
        weights,
        rng,
        max_iter=max_iter_north,
        mu0=1.0 / 5.0,
        max_cost_increase=5,
        use_elliptical=False,
        stochastic_search=True,
    )
    # Stage 3: Deterministic refinement (real-bivariate + elliptical bounds)
    weights = _north_stage(
        whitened,
        weights,
        rng,
        max_iter=max_iter_refine,
        mu0=1.0 / 25.0,
        max_cost_increase=3,
        use_elliptical=True,
        stochastic_search=False,
    )

    demixing = weights @ whitener
    sources = demixing @ centered
    return demixing, sources
