"""Jarzynski Equality estimators — numerically stable.

    ⟨exp(−β W)⟩ = exp(−β ΔF)

Equivalently, for dissipation σ = W − ΔF:
    ⟨exp(−β σ)⟩ = 1

Three estimators, increasing in sophistication:
1. **Naive** — direct exponential average (high variance)
2. **Cumulant-2** — Gaussian approx: ΔF ≈ ⟨W⟩ − β/2·Var(W)
3. **BAR** — Bennett Acceptance Ratio (optimal, lowest variance)

All use numerically stable exp/sigmoid implementations to prevent overflow.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from config.settings import settings
from data.schemas import JarzynskiResult


# ── Numerical Stability Helpers ───────────────────────────

_CLIP = 500.0  # exp(500) ≈ 1.4e217, safe for float64


def _safe_exp(x, clip: float = _CLIP) -> np.ndarray:
    """exp() with argument clipping to prevent overflow."""
    return np.exp(np.clip(np.asarray(x, dtype=float), -clip, clip))


def _safe_fermi(x) -> np.ndarray:
    """Numerically stable Fermi function: f(x) = 1/(1+exp(x)).

    Uses the identity: for x > 0, f(x) = exp(-x)/(1+exp(-x))
    to avoid overflow in exp(x) for large positive x.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # For x >= 0: use exp(-x)/(1+exp(-x))
    ex_neg = np.exp(-x[pos])
    result[pos] = ex_neg / (1.0 + ex_neg)
    # For x < 0: use 1/(1+exp(x)) directly
    ex_pos = np.exp(x[neg])
    result[neg] = 1.0 / (1.0 + ex_pos)
    return result


# ── Estimators ────────────────────────────────────────────

def jarzynski_naive(dissipation: np.ndarray, beta: float) -> JarzynskiResult:
    """Direct exponential average on dissipation.

    Tests: ⟨exp(−β·σ)⟩ = 1  where σ = W − ΔF.
    """
    exp_vals = _safe_exp(-beta * dissipation)
    je_mean = float(np.mean(exp_vals))

    # Bootstrap CI
    rng = np.random.default_rng(settings.random_seed)
    boots = np.empty(settings.bootstrap_n)
    for b in range(settings.bootstrap_n):
        idx = rng.choice(len(dissipation), len(dissipation), replace=True)
        boots[b] = float(np.mean(_safe_exp(-beta * dissipation[idx])))
    ci = np.percentile(boots, [2.5, 97.5])

    # p-value: is je_mean significantly different from 1.0?
    boot_centered = boots - 1.0
    p_value = float(2 * min(np.mean(boot_centered <= 0), np.mean(boot_centered >= 0)))

    return JarzynskiResult(
        estimator="naive",
        lhs=je_mean, rhs=1.0,
        ratio=je_mean,
        beta=beta,
        delta_f_je=float(-np.log(max(je_mean, 1e-300)) / beta) if beta > 0 else 0.0,
        delta_f_direct=0.0,
        n_samples=len(dissipation),
        ci_lower=float(ci[0]), ci_upper=float(ci[1]),
        p_value=p_value,
    )


def jarzynski_cumulant2(dissipation: np.ndarray, beta: float) -> JarzynskiResult:
    """Second-order cumulant expansion.

    For dissipation σ: ⟨exp(−βσ)⟩ ≈ exp(−β⟨σ⟩ + β²Var(σ)/2)
    If JE holds, this should equal 1, so: ⟨σ⟩ ≈ β·Var(σ)/2.
    """
    mu = float(np.mean(dissipation))
    var = float(np.var(dissipation))
    # Cumulant prediction: ⟨exp(-βσ)⟩ ≈ exp(-β·μ + β²·var/2)
    cumulant_pred = float(np.exp(np.clip(-beta * mu + beta**2 * var / 2, -_CLIP, _CLIP)))

    # Actual exponential average
    je_mean = float(np.mean(_safe_exp(-beta * dissipation)))

    rng = np.random.default_rng(settings.random_seed)
    boots = np.empty(settings.bootstrap_n)
    for b in range(settings.bootstrap_n):
        idx = rng.choice(len(dissipation), len(dissipation), replace=True)
        s = dissipation[idx]
        boots[b] = float(np.exp(np.clip(-beta * np.mean(s) + beta**2 * np.var(s) / 2, -_CLIP, _CLIP)))
    ci = np.percentile(boots, [2.5, 97.5])

    return JarzynskiResult(
        estimator="cumulant2",
        lhs=je_mean, rhs=1.0,
        ratio=cumulant_pred,
        beta=beta,
        delta_f_je=mu - beta / 2 * var,  # ΔF from cumulant
        delta_f_direct=0.0,
        n_samples=len(dissipation),
        ci_lower=float(ci[0]), ci_upper=float(ci[1]),
    )


def jarzynski_bar(
    diss_fwd: np.ndarray, diss_rev: np.ndarray, beta: float,
) -> JarzynskiResult:
    """Bennett Acceptance Ratio — optimal estimator.

    Uses numerically stable Fermi functions throughout.
    Solves for C such that:
        ⟨f(β·σ_F − C)⟩_F = ⟨f(C − β·σ_R)⟩_R
    where f is the Fermi function.
    """
    nf, nr = len(diss_fwd), len(diss_rev)
    if nf < 20 or nr < 20:
        return JarzynskiResult(
            estimator="bar", lhs=0, rhs=1, ratio=0, beta=beta,
            delta_f_je=0, delta_f_direct=0, n_samples=nf + nr)

    M = np.log(nf / nr)

    def bar_obj(C):
        lhs = np.mean(_safe_fermi(beta * diss_fwd - C + M))
        rhs = np.mean(_safe_fermi(C - beta * diss_rev - M))
        return float(lhs - rhs)

    # Initial guess from cumulant on forward dissipation
    C0 = beta * float(np.mean(diss_fwd)) - beta**2 / 2 * float(np.var(diss_fwd))
    lo, hi = C0 - 50, C0 + 50

    try:
        C_opt = float(optimize.brentq(bar_obj, lo, hi, maxiter=200))
    except ValueError:
        C_opt = C0

    je_mean = float(np.mean(_safe_exp(-beta * diss_fwd)))

    # Bootstrap
    rng = np.random.default_rng(settings.random_seed)
    n_boot = min(settings.bootstrap_n, 2000)
    boots = []
    for _ in range(n_boot):
        sf = rng.choice(diss_fwd, nf, replace=True)
        sr = rng.choice(diss_rev, nr, replace=True)
        try:
            def obj(C):
                return float(np.mean(_safe_fermi(beta * sf - C + M))
                             - np.mean(_safe_fermi(C - beta * sr - M)))
            boots.append(float(optimize.brentq(obj, lo, hi, maxiter=100)))
        except (ValueError, RuntimeError):
            boots.append(C_opt)
    ci = np.percentile(boots, [2.5, 97.5]) if boots else [C_opt, C_opt]

    return JarzynskiResult(
        estimator="bar",
        lhs=je_mean, rhs=1.0,
        ratio=je_mean,
        beta=beta,
        delta_f_je=C_opt / beta if beta > 0 else 0.0,
        delta_f_direct=0.0,
        n_samples=nf + nr,
        ci_lower=float(ci[0]), ci_upper=float(ci[1]),
    )


# ── Convenience ───────────────────────────────────────────

def run_all_estimators(
    cycles_df, beta: float, delta_f_direct: float = 0.0,
) -> list[JarzynskiResult]:
    """Run all JE estimators on dissipation σ = W − ΔF."""
    w = cycles_df["work"].values.astype(float)
    df = cycles_df["delta_free_energy"].values.astype(float)
    diss = w - df  # dissipation per cycle

    fwd_mask = cycles_df["funding_rate"].values > 0
    diss_fwd = diss[fwd_mask]
    diss_rev = diss[~fwd_mask]

    results = []
    for fn in [jarzynski_naive, jarzynski_cumulant2]:
        r = fn(diss, beta)
        r.delta_f_direct = delta_f_direct
        results.append(r)

    if len(diss_fwd) > 20 and len(diss_rev) > 20:
        r = jarzynski_bar(diss_fwd, diss_rev, beta)
        r.delta_f_direct = delta_f_direct
        results.append(r)

    return results
