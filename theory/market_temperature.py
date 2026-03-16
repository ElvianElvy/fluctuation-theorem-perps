"""Market temperature estimation: T = kappa * Var(b), beta = 1/T."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class TemperatureEstimate:
    beta: float
    temperature: float
    kappa: float
    basis_mean: float
    basis_std: float
    basis_variance: float
    n_samples: int
    normality_pvalue: float
    is_equilibrium: bool


def estimate_temperature(basis: np.ndarray, dt: float = 1.0, significance: float = 0.05) -> TemperatureEstimate:
    b = basis[~np.isnan(basis)]
    if len(b) < 10:
        return TemperatureEstimate(beta=1.0, temperature=1.0, kappa=0.01,
            basis_mean=0, basis_std=0, basis_variance=0, n_samples=len(b),
            normality_pvalue=0, is_equilibrium=False)
    y, x = b[1:], b[:-1]
    d = np.sum(x * x)
    phi = np.clip(np.sum(x * y) / d, 1e-6, 1.0 - 1e-6) if d > 0 else 0.5
    kappa = max(-np.log(phi) / dt, 1e-6)
    var_b = float(np.var(b))
    temperature = kappa * var_b
    beta = 1.0 / temperature if temperature > 0 else 1e6
    if len(b) > 5000:
        _, p = stats.kstest((b - np.mean(b)) / np.std(b), "norm")
    elif len(b) >= 3:
        _, p = stats.shapiro(b)
    else:
        p = 0.0
    return TemperatureEstimate(beta=beta, temperature=temperature, kappa=kappa,
        basis_mean=float(np.mean(b)), basis_std=float(np.std(b)),
        basis_variance=var_b, n_samples=len(b),
        normality_pvalue=float(p), is_equilibrium=p > significance)


def rolling_temperature(basis: np.ndarray, window: int, dt: float = 1.0) -> list[TemperatureEstimate]:
    return [estimate_temperature(basis[i - window:i], dt) for i in range(window, len(basis))]
