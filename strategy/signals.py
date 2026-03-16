"""4 Novel Physics-Derived Trading Signals.

Each signal comes directly from our thermodynamic framework:
1. Relaxation Ratio    — basis mean-reversion speed vs funding period
2. Temperature Z-score — market stability regime from β(t)
3. Entropy Production  — dissipation rate (arb edge strength)
4. JE Health           — how thermodynamic the market is right now

All computed from rolling windows on funding cycles + hourly basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from theory.thermodynamic_quantities import estimate_kappa
from theory.market_temperature import estimate_temperature
from theory.jarzynski import _safe_exp


@dataclass
class SignalSnapshot:
    """All 4 physics signals at a single point in time."""
    timestamp: pd.Timestamp
    relaxation_ratio: float     # τ_relax / τ_funding; < 1 = safe
    temperature_zscore: float   # > 0 = cold/stable; < 0 = hot/volatile
    entropy_rate: float         # > 0 = dissipative (arb edge); < 0 = NESS
    je_health: float            # close to 0 = thermodynamic; large = broken
    raw_beta: float
    raw_kappa: float
    raw_sigma: float


class ThermodynamicSignals:
    """Compute physics signals from market data.

    All signals use ONLY data from the past (no look-ahead).
    """

    def __init__(
        self,
        basis_window_hours: int = 720,   # 30 days of hourly data
        entropy_window_cycles: int = 90,  # 90 funding cycles = 30 days
        je_window_cycles: int = 180,      # 180 cycles = 60 days
        beta_train_cycles: int = 360,     # 360 cycles = 120 days for z-score
        funding_period_hours: float = 8.0,
    ):
        self.basis_window = basis_window_hours
        self.entropy_window = entropy_window_cycles
        self.je_window = je_window_cycles
        self.beta_train = beta_train_cycles
        self.funding_period = funding_period_hours

    def relaxation_ratio(self, basis_hourly: np.ndarray, dt: float = 1.0) -> float:
        """Signal 1: r = τ_relax / τ_funding.

        r < 1  → basis mean-reverts within one funding cycle → SAFE to arb
        r > 1  → basis persists across cycles → DANGEROUS
        r ≈ 1  → marginal
        """
        if len(basis_hourly) < 50:
            return 2.0  # default: unsafe
        kappa = estimate_kappa(basis_hourly, dt=dt)
        tau_relax = 1.0 / max(kappa, 1e-6)
        return tau_relax / self.funding_period

    def temperature_zscore(
        self, betas_history: list[float], current_beta: float,
    ) -> float:
        """Signal 2: Z-score of current β vs historical distribution.

        Positive → colder than average → tighter basis → STABLE → arb works
        Negative → hotter than average → wider basis → VOLATILE → risky
        Zero     → average conditions
        """
        if len(betas_history) < 10:
            return 0.0
        arr = np.array(betas_history)
        mu, sigma = np.mean(arr), np.std(arr)
        if sigma < 1e-10:
            return 0.0
        return float((current_beta - mu) / sigma)

    def entropy_rate(
        self, work: np.ndarray, delta_f: np.ndarray, beta: float,
    ) -> float:
        """Signal 3: Rolling mean entropy production σ = β(W − ΔF).

        σ > 0  → market is dissipative → funding mechanism working → arb edge
        σ ≈ 0  → near equilibrium → weak signal
        σ < 0  → second law violated → NESS → don't trade
        """
        diss = work - delta_f
        sigma = beta * diss
        # Clip extreme values for stability
        sigma = np.clip(sigma, -1e4, 1e4)
        return float(np.mean(sigma))

    def je_health(
        self, work: np.ndarray, delta_f: np.ndarray, beta: float,
    ) -> float:
        """Signal 4: JE deviation |⟨exp(−βσ)⟩ − 1|.

        ≈ 0    → market behaves thermodynamically → safe to trade
        large  → JE assumption violated → anomalous regime
        """
        diss = work - delta_f
        exp_vals = _safe_exp(-beta * diss)
        je_val = float(np.mean(exp_vals))
        return abs(je_val - 1.0)

    def compute_signals_series(
        self, cycles_df: pd.DataFrame, full_basis: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute all 4 signals at every cycle in the dataframe.

        Uses ONLY past data at each point (causal / no look-ahead).
        """
        n = len(cycles_df)
        work = cycles_df["work"].values.astype(float)
        d_f = cycles_df["delta_free_energy"].values.astype(float)
        times = pd.to_datetime(cycles_df["start_time"])
        funding_rates = cycles_df["funding_rate"].values.astype(float)
        basis_means = cycles_df["basis_mean"].values.astype(float)

        # Hourly basis samples per cycle (approx 8)
        spc = int(self.funding_period) if full_basis is not None else 0

        rows = []
        beta_history = []

        min_start = max(self.beta_train, self.je_window, self.entropy_window)

        for i in range(min_start, n):
            # --- Temperature estimation from hourly basis ---
            if full_basis is not None and spc > 0:
                end_idx = min((i + 1) * spc, len(full_basis))
                start_idx = max(end_idx - self.basis_window, 0)
                basis_win = full_basis[start_idx:end_idx]
            else:
                basis_win = basis_means[max(0, i - self.entropy_window):i]

            te = estimate_temperature(basis_win, dt=1.0 if full_basis is not None else 8.0)
            beta_i = te.beta
            kappa_i = te.kappa

            # Signal 1: Relaxation ratio
            if full_basis is not None and spc > 0:
                rr = self.relaxation_ratio(basis_win, dt=1.0)
            else:
                rr = (1.0 / max(kappa_i, 1e-6)) / self.funding_period

            # Signal 2: Temperature z-score
            beta_history.append(beta_i)
            train_betas = beta_history[-self.beta_train:]
            tz = self.temperature_zscore(train_betas[:-1], beta_i)

            # Signal 3: Entropy rate (rolling window)
            w_win = work[max(0, i - self.entropy_window):i]
            df_win = d_f[max(0, i - self.entropy_window):i]
            er = self.entropy_rate(w_win, df_win, beta_i)

            # Signal 4: JE health (longer window)
            w_je = work[max(0, i - self.je_window):i]
            df_je = d_f[max(0, i - self.je_window):i]
            jh = self.je_health(w_je, df_je, beta_i)

            rows.append(dict(
                timestamp=times.iloc[i],
                cycle_index=i,
                funding_rate=funding_rates[i],
                basis_mean=basis_means[i],
                relaxation_ratio=rr,
                temperature_zscore=tz,
                entropy_rate=er,
                je_health=jh,
                raw_beta=beta_i,
                raw_kappa=kappa_i,
            ))

        return pd.DataFrame(rows)
