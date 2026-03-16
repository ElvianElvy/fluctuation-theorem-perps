"""Detailed Fluctuation Theorem (DFT) test.

The correct symmetry test for dissipation sigma = W - dF:

    ln[ P(sigma) / P(-sigma) ] = beta * sigma

Tested on the FULL dissipation distribution (all cycles).
We do NOT split by funding direction — that would require
time-reversed protocols of the SAME process, which positive/negative
funding is not.

Uses KDE for robust density estimation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

from data.schemas import CrooksResult


def dft_test(
    dissipation: np.ndarray,
    beta: float,
    n_grid: int = 200,
    kde_bw: str = "silverman",
) -> CrooksResult:
    """Test the Detailed Fluctuation Theorem via KDE.

    Computes P(sigma) and P(-sigma) from the same distribution,
    then tests: ln[P(sigma)/P(-sigma)] = beta*sigma (slope = beta).
    """
    n = len(dissipation)
    if n < 50:
        return CrooksResult(n_forward=n, n_reverse=n)

    try:
        kde = gaussian_kde(dissipation, bw_method=kde_bw)
    except (np.linalg.LinAlgError, ValueError):
        return CrooksResult(n_forward=n, n_reverse=n)

    extent = np.percentile(np.abs(dissipation), 98)
    if extent < 1e-15:
        return CrooksResult(n_forward=n, n_reverse=n)

    # Evaluate on positive sigma only — compare P(sigma) vs P(-sigma)
    x_pos = np.linspace(extent * 0.01, extent, n_grid)
    p_pos = kde(x_pos)
    p_neg = kde(-x_pos)

    min_density = max(np.max(p_pos) * 1e-4, 1e-15)
    valid = (p_pos > min_density) & (p_neg > min_density)
    if np.sum(valid) < 10:
        return CrooksResult(n_forward=n, n_reverse=n)

    x_valid = x_pos[valid]
    log_ratio = np.log(p_pos[valid]) - np.log(p_neg[valid])

    slope, intercept, r_value, _, _ = stats.linregress(x_valid, log_ratio)

    return CrooksResult(
        work_bins=x_valid,
        log_ratio=log_ratio,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value ** 2),
        beta_est=float(slope),
        delta_f_est=float(-intercept / slope) if abs(slope) > 1e-10 else 0.0,
        n_forward=n, n_reverse=n,
    )


def dft_significance(cr: CrooksResult, beta_true: float, n_bootstrap: int = 5000) -> dict:
    """Bootstrap test: is DFT slope consistent with beta?"""
    if len(cr.work_bins) < 10:
        return {"slope": cr.slope, "beta": beta_true, "p_value": 1.0, "significant": False}

    rng = np.random.default_rng(42)
    n = len(cr.work_bins)
    boot_slopes = [stats.linregress(cr.work_bins[idx := rng.choice(n, n, True)],
                                     cr.log_ratio[idx]).slope
                   for _ in range(n_bootstrap)]
    boot_slopes = np.array(boot_slopes)
    centered = boot_slopes - beta_true
    p_val = float(2 * min(np.mean(centered <= 0), np.mean(centered >= 0)))

    return {
        "slope": cr.slope, "beta": beta_true,
        "slope_ci": [float(np.percentile(boot_slopes, 2.5)),
                     float(np.percentile(boot_slopes, 97.5))],
        "p_value": p_val, "significant": p_val < 0.05,
        "slope_error_pct": abs(cr.slope - beta_true) / max(abs(beta_true), 1e-10) * 100,
    }


# Aliases for backward compatibility
crooks_test = dft_test
crooks_significance = dft_significance
