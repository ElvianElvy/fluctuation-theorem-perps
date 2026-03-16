"""Core thermodynamic quantities — uses percentage basis for scale-invariance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from data.storage.db import Database


def estimate_kappa(basis: np.ndarray, dt: float = 1.0) -> float:
    b = basis[~np.isnan(basis)]
    if len(b) < 10: return 0.01
    y, x = b[1:], b[:-1]
    d = np.sum(x * x)
    if d == 0: return 0.01
    phi = np.clip(np.sum(x * y) / d, 1e-6, 1.0 - 1e-6)
    return max(-np.log(phi) / dt, 1e-6)


def energy(basis, kappa: float):
    return 0.5 * kappa * np.asarray(basis) ** 2


def work(funding_rate: float, basis_start: float) -> float:
    """W = -F * b_start (Jarzynski sudden-quench work)."""
    return -funding_rate * basis_start


def delta_free_energy_cycle(funding_rate: float, kappa: float) -> float:
    """dF = -F^2 / (2*kappa)."""
    return -funding_rate ** 2 / (2.0 * kappa)


def heat(e_start: float, e_end: float, w: float) -> float:
    return (e_end - e_start) - w


def free_energy(kappa: float, temperature: float) -> float:
    if temperature <= 0 or kappa <= 0: return 0.0
    return -temperature / 2.0 * np.log(2.0 * np.pi * temperature / kappa)


def entropy_production(beta: float, w: float, delta_f: float) -> float:
    return beta * (w - delta_f)


def build_funding_cycles(db: Database, symbol: str) -> pd.DataFrame:
    """Build funding cycles using percentage basis for scale-invariance."""
    df = db.get_basis_at_funding_times(symbol)
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    df = df.sort_values("funding_time").reset_index(drop=True)

    basis_full = db.get_basis(symbol)
    if basis_full.empty:
        return pd.DataFrame()
    kappa = estimate_kappa(basis_full["basis_pct"].values.astype(float), dt=1.0)

    cycles = []
    for i in range(len(df) - 1):
        row, nxt = df.iloc[i], df.iloc[i + 1]
        t0, t1 = pd.Timestamp(row["funding_time"]), pd.Timestamp(nxt["funding_time"])
        intra = db.get_basis_between(symbol, t0, t1)
        if not intra.empty:
            bvals = intra["basis_pct"].values.astype(float)
        else:
            bvals = np.array([float(row["basis_pct"]) if pd.notna(row.get("basis_pct")) else 0.0])
        b_mean, b_std = float(np.mean(bvals)), float(np.std(bvals))
        b0 = float(row["basis_pct"]) if pd.notna(row.get("basis_pct")) else 0.0
        b1 = float(nxt["basis_pct"]) if pd.notna(nxt.get("basis_pct")) else 0.0
        fr = float(row["funding_rate"])
        e0, e1 = float(energy(b0, kappa)), float(energy(b1, kappa))
        w = work(fr, b0)
        d_f = delta_free_energy_cycle(fr, kappa)
        q = heat(e0, e1, w)
        cycles.append(dict(
            symbol=symbol, start_time=t0, end_time=t1, funding_rate=fr,
            basis_start=b0, basis_end=b1, basis_mean=b_mean, basis_std=b_std,
            mark_price_start=float(row.get("mark_price", 0) or 0),
            mark_price_end=float(nxt.get("mark_price", 0) or 0),
            work=w, energy_start=e0, energy_end=e1, heat=q, delta_free_energy=d_f,
        ))
    return pd.DataFrame(cycles)
