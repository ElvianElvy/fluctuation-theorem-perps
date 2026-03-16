"""Entropy production — the arrow of time in funding markets.

    σ(t) = β(t) · (W(t) − ΔF(t))

Second law:  ⟨σ⟩ ≥ 0
High σ  →  market far from equilibrium  →  potential arb window.

Includes:
- Rolling σ(t) computation
- Second law significance test
- Relaxation time analysis (τ = 1/κ vs τ_funding)
- Winsorized variants for robustness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from theory.market_temperature import estimate_temperature
from theory.thermodynamic_quantities import free_energy


def compute_entropy_production_series(
    cycles_df: pd.DataFrame,
    window_cycles: int = 90,
    global_beta: float | None = None,
    full_basis: np.ndarray | None = None,
    basis_dt: float = 1.0,
) -> pd.DataFrame:
    """Rolling σ(t) from funding cycles.

    Args:
        cycles_df: DataFrame with 'work', 'delta_free_energy', 'basis_mean', 'start_time'
        window_cycles: Rolling window for temperature estimation
        global_beta: If provided, use this fixed β instead of estimating
        full_basis: Full instantaneous basis at basis_dt resolution
        basis_dt: Time step of full_basis in hours
    """
    work = cycles_df["work"].values.astype(float)
    delta_f = (cycles_df["delta_free_energy"].values.astype(float)
               if "delta_free_energy" in cycles_df.columns
               else np.zeros(len(work)))
    basis = cycles_df["basis_mean"].values.astype(float)
    times = pd.to_datetime(cycles_df["start_time"])

    spc = int(8.0 / basis_dt) if full_basis is not None else 0

    rows, cum = [], 0.0
    for i in range(window_cycles, len(work)):
        if global_beta is not None:
            beta_i = global_beta
            temp_i = 1.0 / beta_i if beta_i > 0 else 1e6
            kappa_i = 0.0
            is_eq = True
        elif full_basis is not None:
            end_idx = min((i + window_cycles) * spc, len(full_basis))
            start_idx = max(end_idx - window_cycles * spc, 0)
            win = full_basis[start_idx:end_idx]
            te = estimate_temperature(win, dt=basis_dt)
            beta_i, temp_i, kappa_i, is_eq = te.beta, te.temperature, te.kappa, te.is_equilibrium
        else:
            win = basis[i - window_cycles:i]
            te = estimate_temperature(win, dt=8.0)
            beta_i, temp_i, kappa_i, is_eq = te.beta, te.temperature, te.kappa, te.is_equilibrium

        # Per-cycle dissipation with overflow protection
        dissipation_i = work[i] - delta_f[i]
        sigma = np.clip(beta_i * dissipation_i, -1e6, 1e6)
        cum += sigma

        rows.append(dict(
            timestamp=times.iloc[i], sigma=sigma,
            dissipation=dissipation_i,
            beta=beta_i, temperature=temp_i, kappa=kappa_i,
            work=work[i], delta_f=delta_f[i],
            basis_mean=basis[i],
            cumulative_sigma=cum, is_equilibrium=is_eq,
        ))
    return pd.DataFrame(rows)


def second_law_test(sigma_df: pd.DataFrame) -> dict:
    """Test ⟨σ⟩ ≥ 0 with full significance analysis.

    Returns:
        mean_sigma, std_sigma, t_statistic, p_value (one-sided),
        fraction_negative, n_cycles, second_law_holds,
        ci_lower, ci_upper (95% CI for ⟨σ⟩)
    """
    s = sigma_df["sigma"].values
    n = len(s)
    if n == 0:
        return dict(mean_sigma=0, std_sigma=0, fraction_negative=0,
                    t_statistic=0, p_value_positive=1, n_cycles=0,
                    second_law_holds=False, ci_lower=0, ci_upper=0,
                    mean_sigma_significant=False)

    mu = float(np.mean(s))
    std = float(np.std(s, ddof=1))
    se = std / np.sqrt(n)
    t_stat = mu / se if se > 0 else 0.0

    # One-sided p-value: H₀: ⟨σ⟩ ≤ 0, H₁: ⟨σ⟩ > 0
    p_positive = float(1 - sp_stats.t.cdf(t_stat, df=n - 1))

    # Two-sided p-value: H₀: ⟨σ⟩ = 0
    p_two_sided = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1)))

    # 95% CI for ⟨σ⟩
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    ci_lo = mu - t_crit * se
    ci_hi = mu + t_crit * se

    return dict(
        mean_sigma=mu,
        std_sigma=std,
        fraction_negative=float(np.mean(s < 0)),
        t_statistic=t_stat,
        p_value_positive=p_positive,
        p_value_two_sided=p_two_sided,
        n_cycles=n,
        second_law_holds=mu > 0 and p_positive < 0.05,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        mean_sigma_significant=p_two_sided < 0.05,
    )


def relaxation_time_analysis(kappa: float, funding_period_hours: float = 8.0) -> dict:
    """Compute relaxation time and compare to funding period.

    τ_relax = 1/κ  (time for basis to mean-revert by factor e)
    τ_funding = 8h  (funding settlement period)

    JE requires τ_relax < τ_funding (system equilibrates between cycles).
    """
    tau_relax = 1.0 / kappa if kappa > 0 else float("inf")
    ratio = tau_relax / funding_period_hours
    equilibrates = tau_relax < funding_period_hours

    return dict(
        kappa=kappa,
        tau_relax_hours=tau_relax,
        tau_funding_hours=funding_period_hours,
        ratio=ratio,
        equilibrates=equilibrates,
        verdict=("FAST (JE valid)" if equilibrates
                 else "SLOW (JE assumption violated)"),
    )


def winsorize_dissipation(
    dissipation: np.ndarray, percentile: float = 1.0,
) -> np.ndarray:
    """Winsorize extreme dissipation values.

    Clips at the given percentile on both tails. Report results
    both with and without winsorization for the paper.
    """
    lo = np.percentile(dissipation, percentile)
    hi = np.percentile(dissipation, 100 - percentile)
    return np.clip(dissipation, lo, hi)
