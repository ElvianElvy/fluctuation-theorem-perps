"""OU synthetic simulator — exact discretization, correct Jarzynski work."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SyntheticParams:
    kappa: float = 0.05
    sigma_noise: float = 5.0
    funding_rate_mean: float = 0.0001
    funding_rate_std: float = 0.0002
    dt_hours: float = 1.0
    funding_period_hours: float = 8.0
    n_cycles: int = 5000
    spot_price: float = 50_000.0
    seed: int = 42
    equilibrium_reset: bool = False

    @property
    def temperature(self) -> float:
        return self.sigma_noise ** 2 / 2.0

    @property
    def beta(self) -> float:
        return 1.0 / self.temperature

    @property
    def eq_basis_variance(self) -> float:
        return self.temperature / self.kappa

    @property
    def eq_basis_std(self) -> float:
        return np.sqrt(self.eq_basis_variance)

    @property
    def free_energy(self) -> float:
        T = self.temperature
        return -T / 2.0 * np.log(2.0 * np.pi * T / self.kappa)


class SyntheticMarketSimulator:
    def __init__(self, params: SyntheticParams | None = None):
        self.p = params or SyntheticParams()
        self.rng = np.random.default_rng(self.p.seed)

    def simulate(self) -> tuple[pd.DataFrame, np.ndarray]:
        spc = int(self.p.funding_period_hours / self.p.dt_hours)
        total = spc * self.p.n_cycles
        dt = self.p.dt_hours
        kappa = self.p.kappa

        funding_rates = self.rng.normal(self.p.funding_rate_mean, self.p.funding_rate_std, self.p.n_cycles)

        phi = np.exp(-kappa * dt)
        exact_noise_std = self.p.sigma_noise * np.sqrt((1.0 - phi**2) / (2.0 * kappa))

        b = np.zeros(total + 1)
        b[0] = self.rng.normal(0, self.p.eq_basis_std)
        eps = self.rng.standard_normal(total)

        for i in range(total):
            if self.p.equilibrium_reset and i > 0 and i % spc == 0:
                b[i] = self.rng.normal(0, self.p.eq_basis_std)
            c = i // spc
            F = funding_rates[min(c, len(funding_rates) - 1)]
            b[i + 1] = phi * b[i] - (F / kappa) * (1.0 - phi) + exact_noise_std * eps[i]

        base = pd.Timestamp("2024-01-01", tz="UTC")
        rows = []
        for c in range(self.p.n_cycles):
            seg = b[c * spc:(c + 1) * spc + 1]
            fr = funding_rates[c]
            ph = self.p.funding_period_hours
            e0 = 0.5 * kappa * seg[0] ** 2
            e1 = 0.5 * kappa * seg[-1] ** 2
            w = -fr * seg[0]  # W = -F * b_start
            d_f = -fr ** 2 / (2.0 * kappa)
            rows.append(dict(
                symbol="SYNTHETIC",
                start_time=base + pd.Timedelta(hours=c * ph),
                end_time=base + pd.Timedelta(hours=(c + 1) * ph),
                funding_rate=fr, basis_start=seg[0], basis_end=seg[-1],
                basis_mean=float(np.mean(seg)), basis_std=float(np.std(seg)),
                mark_price_start=self.p.spot_price + seg[0],
                mark_price_end=self.p.spot_price + seg[-1],
                work=w, energy_start=e0, energy_end=e1,
                heat=(e1 - e0) - w, delta_free_energy=d_f,
            ))
        return pd.DataFrame(rows), b.copy()

    def analytical(self) -> dict:
        return dict(kappa=self.p.kappa, temperature=self.p.temperature,
                    beta=self.p.beta, free_energy=self.p.free_energy,
                    eq_basis_std=self.p.eq_basis_std,
                    eq_basis_variance=self.p.eq_basis_variance, jarzynski_ratio=1.0)


STRONG_DRIVING_PARAMS = SyntheticParams(
    kappa=0.5, sigma_noise=1.0, funding_rate_mean=0.1, funding_rate_std=0.2,
    n_cycles=10_000, seed=42, equilibrium_reset=True)

REALISTIC_MARKET_PARAMS = SyntheticParams(
    kappa=0.05, sigma_noise=5.0, funding_rate_mean=0.0001, funding_rate_std=0.0003,
    n_cycles=5_000, seed=42)
