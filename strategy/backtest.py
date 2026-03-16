"""Paper-trade backtester — ALL VIRTUAL, no real money.

Strategy: delta-neutral funding arb when entropy production σ(t)
signals the market is far from equilibrium.

Walk-forward protocol for academic rigour:
  1. Train window → estimate thresholds from history
  2. Test window → generate signals, simulate trades
  3. Roll forward, report ONLY out-of-sample metrics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import settings


@dataclass
class VirtualTrade:
    """Record of one completed virtual trade."""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_basis: float
    exit_basis: float
    size_usd: float
    pnl_funding: float
    pnl_fees: float
    pnl_total: float
    sigma_entry: float
    sigma_exit: float
    n_cycles_held: int = 0


@dataclass
class BacktestResult:
    """Full backtest output for the paper."""
    trades: list[VirtualTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return self.equity_curve[-1] / self.equity_curve[0] - 1.0

    @property
    def annualised_return(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        days = (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 86400
        if days <= 0:
            return 0.0
        base = 1.0 + self.total_return
        if base <= 0:
            return -1.0
        try:
            ann = base ** (365.0 / days) - 1.0
            return float(max(min(ann, 1e4), -1.0))
        except (OverflowError, ValueError):
            return 1e4 if self.total_return > 0 else -1.0

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 10:
            return 0.0
        eq = np.array(self.equity_curve, dtype=float)
        r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        std = float(np.std(r))
        if std < 1e-12:
            return 0.0
        return float(np.mean(r) / std * np.sqrt(3 * 365))

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak, dd = self.equity_curve[0], 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                dd = max(dd, (peak - v) / peak)
        return dd

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_total > 0) / len(self.trades)

    @property
    def avg_trade_duration_cycles(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.n_cycles_held for t in self.trades])

    def summary(self) -> dict:
        def _r(v, d=2):
            try:
                return round(float(v), d)
            except (OverflowError, ValueError, TypeError):
                return 0.0
        return dict(
            n_trades=self.n_trades,
            total_return_pct=_r(self.total_return * 100),
            annualised_return_pct=_r(self.annualised_return * 100),
            sharpe_ratio=_r(self.sharpe, 3),
            max_drawdown_pct=_r(self.max_drawdown * 100),
            win_rate_pct=_r(self.win_rate * 100, 1),
            avg_trade_cycles=_r(self.avg_trade_duration_cycles, 1),
            avg_pnl=_r(np.mean([t.pnl_total for t in self.trades]), 4) if self.trades else 0,
        )


class PaperTradeBacktester:
    """Walk-forward virtual-money backtester.

    Caps per-cycle funding PnL to prevent unrealistic accumulation.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        pos_frac: float = 0.5,
        sigma_entry_pct: float = 80,
        sigma_exit_pct: float = 50,
        train_cycles: int = 270,
        max_funding_pnl_pct: float = 0.01,  # Cap funding PnL at 1% of position per cycle
    ):
        self.capital0 = initial_capital or settings.initial_capital
        self.pos_frac = pos_frac
        self.entry_pct = sigma_entry_pct
        self.exit_pct = sigma_exit_pct
        self.train = train_cycles
        self.max_fund_pct = max_funding_pnl_pct

    def _fees(self, size: float) -> float:
        return (2 * size * settings.taker_fee
                + 2 * size * settings.maker_fee
                + 2 * size * settings.slippage_bps / 10_000)

    def _capped_funding(self, side: str, size: float, fr: float) -> float:
        """Funding PnL per cycle, capped to prevent extreme values."""
        raw = size * fr if side == "long_basis" else -size * fr
        cap = size * self.max_fund_pct
        return float(np.clip(raw, -cap, cap))

    def run(self, cycles_df: pd.DataFrame, sigma_df: pd.DataFrame) -> BacktestResult:
        res = BacktestResult()
        cap = self.capital0
        sv = sigma_df["sigma"].values
        st = pd.to_datetime(sigma_df["timestamp"])

        pos_side: str | None = None
        pos_size = 0.0
        pos_entry_time = None
        pos_entry_sigma = 0.0
        pos_entry_basis = 0.0
        cum_funding = 0.0
        cycles_held = 0

        # Map sigma timestamps to cycles
        offset = max(0, len(cycles_df) - len(sigma_df))

        for i in range(self.train, len(sv)):
            ts = st.iloc[i]
            sig = sv[i]
            train_s = np.abs(sv[max(0, i - self.train):i])
            entry_th = np.percentile(train_s, self.entry_pct) if len(train_s) > 10 else 1e9
            exit_th = np.percentile(train_s, self.exit_pct) if len(train_s) > 10 else 0

            ci = min(i + offset, len(cycles_df) - 1)
            fr = float(cycles_df.iloc[ci].get("funding_rate", 0))
            bm = float(cycles_df.iloc[ci].get("basis_mean", 0))

            if pos_side is None:
                if abs(sig) > entry_th and entry_th > 0 and cap > 100:
                    pos_side = "long_basis" if fr > 0 else "short_basis"
                    pos_size = cap * self.pos_frac
                    pos_entry_time = ts
                    pos_entry_sigma = sig
                    pos_entry_basis = bm
                    cum_funding = 0.0
                    cycles_held = 0
            else:
                cum_funding += self._capped_funding(pos_side, pos_size, fr)
                cycles_held += 1

                if abs(sig) < exit_th or cycles_held > 100:
                    fees = self._fees(pos_size)
                    pnl = cum_funding - fees
                    res.trades.append(VirtualTrade(
                        symbol=str(cycles_df.iloc[ci].get("symbol", "?")),
                        side=pos_side, entry_time=pos_entry_time, exit_time=ts,
                        entry_basis=pos_entry_basis, exit_basis=bm,
                        size_usd=pos_size, pnl_funding=cum_funding,
                        pnl_fees=-fees, pnl_total=pnl,
                        sigma_entry=pos_entry_sigma, sigma_exit=sig,
                        n_cycles_held=cycles_held))
                    cap += pnl
                    cap = max(cap, 0.0)  # Can't go negative
                    pos_side = None

            res.equity_curve.append(cap)
            res.timestamps.append(ts)

        return res
