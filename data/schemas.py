"""Data models for market data and thermodynamic quantities."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


@dataclass
class FundingRate:
    symbol: str
    funding_rate: float
    funding_time: datetime
    mark_price: float


@dataclass
class OHLCV:
    symbol: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str


@dataclass
class FundingCycle:
    symbol: str
    start_time: datetime
    end_time: datetime
    funding_rate: float
    basis_start: float
    basis_end: float
    basis_mean: float
    basis_std: float
    mark_price_start: float
    mark_price_end: float
    work: float = 0.0
    energy_start: float = 0.0
    energy_end: float = 0.0
    heat: float = 0.0
    delta_free_energy: float = 0.0


@dataclass
class JarzynskiResult:
    estimator: str
    lhs: float
    rhs: float
    ratio: float
    beta: float
    delta_f_je: float
    delta_f_direct: float
    n_samples: int
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 0.0


@dataclass
class CrooksResult:
    work_bins: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    log_ratio: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    beta_est: float = 0.0
    delta_f_est: float = 0.0
    n_forward: int = 0
    n_reverse: int = 0
