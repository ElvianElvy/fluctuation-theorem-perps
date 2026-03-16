"""V2 Backtester — Corrected Realistic PnL Model.

CRITICAL FIXES from previous version:
1. Signed funding: short basis RECEIVES F when F>0, PAYS when F<0
2. Basis risk: mark-to-market from basis_pct changes
3. Realistic drawdowns from basis volatility

PnL model for SHORT BASIS (long spot + short perp):
  funding_pnl = notional × F           (signed — you pay when F < 0)
  basis_pnl   = -notional × Δbasis/100 (profit when basis narrows)
  total_pnl   = funding_pnl + basis_pnl - fees

ALL VIRTUAL MONEY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

from config.settings import settings
from strategy.signals import ThermodynamicSignals
from strategy.regime import (
    ThermodynamicRegimeClassifier, RegimeState, RegimeClassification,
)


@dataclass
class FeeModel:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    slippage_bps: float = 1.0

    @property
    def round_trip(self) -> float:
        avg = (self.maker_fee + self.taker_fee) / 2
        return 2 * avg + 2 * self.slippage_bps / 10_000

    def rebalance_cost(self, delta_notional: float) -> float:
        return abs(delta_notional) * self.round_trip


@dataclass
class V2Trade:
    strategy: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime | None = None
    notional: float = 0.0
    cum_funding_pnl: float = 0.0
    cum_basis_pnl: float = 0.0
    fees_paid: float = 0.0
    pnl_total: float = 0.0
    cycles_held: int = 0
    regime_at_entry: str = ""
    exit_reason: str = ""


@dataclass
class StrategyResult:
    name: str
    trades: list[V2Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)
    position_sizes: list[float] = field(default_factory=list)

    def metrics(self) -> dict:
        eq = np.array(self.equity_curve, dtype=float)
        if len(eq) < 10:
            return dict(name=self.name, sharpe=0, sortino=0, calmar=0,
                        total_return_pct=0, annual_return_pct=0,
                        max_dd_pct=0, win_rate_pct=0, n_trades=0,
                        avg_pnl=0, avg_hold_cycles=0, profit_factor=0)
        r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        r = r[np.isfinite(r)]
        if len(r) == 0: r = np.array([0.0])

        mu, sigma = float(np.mean(r)), float(np.std(r))
        ann = np.sqrt(3 * 365)
        sharpe = mu / sigma * ann if sigma > 1e-12 else 0.0
        down = r[r < 0]
        down_std = float(np.std(down)) if len(down) > 1 else sigma
        sortino = mu / down_std * ann if down_std > 1e-12 else 0.0

        peak, dd = eq[0], 0.0
        for v in eq:
            peak = max(peak, v)
            if peak > 0: dd = max(dd, (peak - v) / peak)

        days = len(eq) * 8.0 / 24.0
        total_ret = eq[-1] / eq[0] - 1.0 if eq[0] > 0 else 0.0
        ann_ret = (1 + total_ret) ** (365 / max(days, 1)) - 1 if total_ret > -1 else -1
        calmar = ann_ret / dd if dd > 0.01 else 0.0

        wins = [t for t in self.trades if t.pnl_total > 0]
        losses = [t for t in self.trades if t.pnl_total <= 0]
        gp = sum(t.pnl_total for t in wins)
        gl = abs(sum(t.pnl_total for t in losses))
        pf = gp / gl if gl > 0 else 999.0

        def _r(v, d=3): return round(float(v), d)
        return dict(
            name=self.name,
            sharpe=_r(sharpe), sortino=_r(sortino), calmar=_r(calmar, 2),
            total_return_pct=_r(total_ret * 100, 2),
            annual_return_pct=_r(ann_ret * 100, 2),
            max_dd_pct=_r(dd * 100, 2),
            win_rate_pct=_r(len(wins) / max(len(self.trades), 1) * 100, 1),
            n_trades=len(self.trades),
            avg_pnl=_r(np.mean([t.pnl_total for t in self.trades]), 4) if self.trades else 0,
            avg_hold_cycles=_r(np.mean([t.cycles_held for t in self.trades]), 1) if self.trades else 0,
            profit_factor=_r(pf, 2),
        )


def _cycle_pnl(notional: float, fr: float, basis_start: float,
               basis_end: float, side: str) -> tuple[float, float]:
    """Compute REALISTIC per-cycle PnL for delta-neutral position.

    Args:
        side: "short_basis" (long spot + short perp) or
              "long_basis" (short spot + long perp)

    Returns: (funding_pnl, basis_pnl)
    """
    if side == "short_basis":
        # Short perp: receive F when F>0, pay when F<0
        funding_pnl = notional * fr
        # Short basis profits when basis narrows (perp drops vs spot)
        basis_pnl = -notional * (basis_end - basis_start) / 100.0
    else:  # long_basis
        # Long perp: pay F when F>0, receive when F<0
        funding_pnl = -notional * fr
        # Long basis profits when basis widens
        basis_pnl = notional * (basis_end - basis_start) / 100.0

    # Cap extreme single-cycle PnL (defensive)
    funding_pnl = np.clip(funding_pnl, -notional * 0.01, notional * 0.01)
    basis_pnl = np.clip(basis_pnl, -notional * 0.05, notional * 0.05)

    return float(funding_pnl), float(basis_pnl)


# ── V2 Physics Strategy ──────────────────────────────────

class PhysicsStrategy:
    """Always-on funding harvest with physics-based position sizing.

    Default: short basis (collect positive funding).
    Regime controls exposure: EQUIL 100%, WARM 60%, NESS 25%.
    """

    def __init__(self, base_fraction: float = 0.50, rebal_threshold: float = 0.20):
        self.base_frac = base_fraction
        self.rebal_thresh = rebal_threshold
        self.fee = FeeModel()

    def run(
        self, cycles_df: pd.DataFrame, signals_df: pd.DataFrame,
        regimes: list[RegimeClassification], initial_capital: float,
    ) -> StrategyResult:
        res = StrategyResult(name="V2 Physics")
        capital = initial_capital
        current_notional = 0.0
        side = "short_basis"  # default: collect positive funding
        cum_fund = 0.0
        cum_basis = 0.0
        cum_fees = 0.0
        trade_start = None
        trade_regime = ""
        cycle_count = 0

        for i in range(len(signals_df)):
            sig = signals_df.iloc[i]
            ci = int(sig["cycle_index"])
            if ci >= len(cycles_df):
                break
            cyc = cycles_df.iloc[ci]
            fr = float(cyc["funding_rate"])
            b_start = float(cyc["basis_start"])
            b_end = float(cyc["basis_end"])
            regime = regimes[i]

            target_notional = capital * self.base_frac * regime.config.position_multiplier
            target_notional = max(target_notional, 0.0)

            # Rebalance check
            if current_notional < 1.0:
                current_notional = target_notional
                cost = self.fee.rebalance_cost(current_notional)
                capital -= cost
                cum_fees += cost
                trade_start = sig["timestamp"]
                trade_regime = regime.state.value
                cycle_count = 0
            else:
                pct_change = abs(target_notional - current_notional) / max(current_notional, 1)
                if pct_change > self.rebal_thresh:
                    delta = abs(target_notional - current_notional)
                    cost = self.fee.rebalance_cost(delta)
                    capital -= cost

                    res.trades.append(V2Trade(
                        strategy="V2 Physics", symbol=str(cyc.get("symbol", "?")),
                        side=side, entry_time=trade_start,
                        exit_time=sig["timestamp"],
                        notional=current_notional,
                        cum_funding_pnl=cum_fund, cum_basis_pnl=cum_basis,
                        fees_paid=cum_fees, pnl_total=cum_fund + cum_basis - cum_fees,
                        cycles_held=cycle_count,
                        regime_at_entry=trade_regime,
                        exit_reason=f"rebal_{regime.state.value}",
                    ))
                    cum_fund = 0.0
                    cum_basis = 0.0
                    cum_fees = cost
                    trade_start = sig["timestamp"]
                    trade_regime = regime.state.value
                    current_notional = target_notional
                    cycle_count = 0

            # REALISTIC PnL: signed funding + basis change
            f_pnl, b_pnl = _cycle_pnl(current_notional, fr, b_start, b_end, side)
            cum_fund += f_pnl
            cum_basis += b_pnl
            capital += f_pnl + b_pnl
            capital = max(capital, 0)
            cycle_count += 1

            res.equity_curve.append(capital)
            res.timestamps.append(sig["timestamp"])
            res.regimes.append(regime.state.value)
            res.position_sizes.append(current_notional)

        # Final segment
        if cycle_count > 0:
            res.trades.append(V2Trade(
                strategy="V2 Physics", symbol="?", side=side,
                entry_time=trade_start,
                exit_time=res.timestamps[-1] if res.timestamps else None,
                notional=current_notional,
                cum_funding_pnl=cum_fund, cum_basis_pnl=cum_basis,
                fees_paid=cum_fees, pnl_total=cum_fund + cum_basis - cum_fees,
                cycles_held=cycle_count, regime_at_entry=trade_regime,
                exit_reason="end_of_data",
            ))
        return res


# ── Benchmarks ────────────────────────────────────────────

class NaiveAlwaysOn:
    """Permanent short basis. Weekly rebalance. Realistic signed funding + basis risk."""
    def run(self, cycles_df, initial_capital, pos_frac=0.5) -> StrategyResult:
        res = StrategyResult(name="Naive Always-On")
        fee = FeeModel()
        capital = initial_capital
        notional = capital * pos_frac
        cost = fee.rebalance_cost(notional)
        capital -= cost
        cum_fund, cum_basis, cum_fees = 0.0, 0.0, cost
        rebal_interval = 21
        last_rebal = 0
        trade_start = pd.to_datetime(cycles_df.iloc[0]["start_time"])
        times = pd.to_datetime(cycles_df["start_time"])

        for i in range(len(cycles_df)):
            fr = float(cycles_df.iloc[i]["funding_rate"])
            b_start = float(cycles_df.iloc[i]["basis_start"])
            b_end = float(cycles_df.iloc[i]["basis_end"])

            f_pnl, b_pnl = _cycle_pnl(notional, fr, b_start, b_end, "short_basis")
            cum_fund += f_pnl
            cum_basis += b_pnl
            capital += f_pnl + b_pnl

            if i - last_rebal >= rebal_interval:
                cost = fee.rebalance_cost(notional)
                capital -= cost
                res.trades.append(V2Trade(
                    strategy="Naive", symbol="?", side="short_basis",
                    entry_time=trade_start, exit_time=times.iloc[i],
                    notional=notional, cum_funding_pnl=cum_fund,
                    cum_basis_pnl=cum_basis,
                    fees_paid=cum_fees + cost,
                    pnl_total=cum_fund + cum_basis - cum_fees - cost,
                    cycles_held=rebal_interval, exit_reason="rebalance",
                ))
                notional = max(capital, 0) * pos_frac
                cum_fund, cum_basis, cum_fees = 0.0, 0.0, 0.0
                last_rebal = i
                trade_start = times.iloc[i]

            capital = max(capital, 0)
            res.equity_curve.append(capital)
            res.timestamps.append(times.iloc[i])
        return res


class ThresholdArb:
    """Enter short basis when |F| > mean+1sigma. Realistic PnL."""
    def run(self, cycles_df, initial_capital, train_cycles=360, pos_frac=0.10) -> StrategyResult:
        res = StrategyResult(name="Threshold Arb")
        fee = FeeModel()
        capital = initial_capital
        position = None
        times = pd.to_datetime(cycles_df["start_time"])

        for i in range(train_cycles, len(cycles_df)):
            fr = float(cycles_df.iloc[i]["funding_rate"])
            b_start = float(cycles_df.iloc[i]["basis_start"])
            b_end = float(cycles_df.iloc[i]["basis_end"])
            ts = times.iloc[i]
            train_fr = cycles_df.iloc[max(0, i-train_cycles):i]["funding_rate"].values.astype(float)
            mu_fr, std_fr = np.mean(train_fr), np.std(train_fr)

            if position is not None:
                f_pnl, b_pnl = _cycle_pnl(position["notional"], fr, b_start, b_end,
                                            position["side"])
                position["cum_fund"] += f_pnl
                position["cum_basis"] += b_pnl
                capital += f_pnl + b_pnl
                position["cycles"] += 1

                if abs(fr) < abs(mu_fr) or position["cycles"] >= 12:
                    fees = fee.rebalance_cost(position["notional"])
                    pnl = position["cum_fund"] + position["cum_basis"] - fees
                    capital -= fees
                    res.trades.append(V2Trade(
                        strategy="Threshold", symbol="?",
                        side=position["side"], entry_time=position["entry_time"],
                        exit_time=ts, notional=position["notional"],
                        cum_funding_pnl=position["cum_fund"],
                        cum_basis_pnl=position["cum_basis"],
                        fees_paid=fees, pnl_total=pnl,
                        cycles_held=position["cycles"],
                        exit_reason="normalized" if abs(fr) < abs(mu_fr) else "max_hold",
                    ))
                    position = None
            else:
                threshold = abs(mu_fr) + std_fr
                if abs(fr) > threshold and capital > 100:
                    side = "short_basis" if fr > 0 else "long_basis"
                    not_sz = capital * pos_frac
                    entry_cost = fee.rebalance_cost(not_sz)
                    capital -= entry_cost
                    position = dict(notional=not_sz, entry_time=ts,
                                     cum_fund=0.0, cum_basis=0.0, cycles=0, side=side)

            capital = max(capital, 0)
            res.equity_curve.append(capital)
            res.timestamps.append(ts)
        return res


class BuyAndHold:
    """Buy and hold perpetual."""
    def run(self, cycles_df, initial_capital) -> StrategyResult:
        res = StrategyResult(name="Buy-Hold")
        prices = cycles_df["mark_price_start"].values.astype(float)
        prices = np.where(prices > 0, prices, np.nan)
        mask = np.isnan(prices)
        if mask.all():
            res.equity_curve = [initial_capital] * len(cycles_df)
            res.timestamps = list(pd.to_datetime(cycles_df["start_time"]))
            return res
        prices[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), prices[~mask])
        base = prices[0]
        times = pd.to_datetime(cycles_df["start_time"])
        for i, p in enumerate(prices):
            res.equity_curve.append(initial_capital * p / base if base > 0 else initial_capital)
            res.timestamps.append(times.iloc[i])
        return res


# ── Walk-Forward Engine ───────────────────────────────────

def run_walk_forward(
    cycles_df: pd.DataFrame,
    full_basis: np.ndarray | None = None,
    initial_capital: float | None = None,
    train_cycles: int = 360,
) -> dict[str, StrategyResult]:
    capital = initial_capital or settings.initial_capital
    n = len(cycles_df)
    if n < train_cycles + 90:
        return {}

    sig_engine = ThermodynamicSignals()
    signals_df = sig_engine.compute_signals_series(cycles_df, full_basis)
    if signals_df.empty:
        return {}

    classifier = ThermodynamicRegimeClassifier()
    regimes = classifier.classify_series(signals_df, train_window=train_cycles)

    physics = PhysicsStrategy()
    v2_result = physics.run(cycles_df, signals_df, regimes, capital)

    min_idx = int(signals_df.iloc[0]["cycle_index"]) if not signals_df.empty else train_cycles
    cycles_sub = cycles_df.iloc[min_idx:].reset_index(drop=True)

    naive_result = NaiveAlwaysOn().run(cycles_sub, capital)
    threshold_result = ThresholdArb().run(cycles_sub, capital,
                                          train_cycles=min(train_cycles, len(cycles_sub) // 2))
    bh_result = BuyAndHold().run(cycles_sub, capital)

    return {
        "V2 Physics": v2_result,
        "Naive Always-On": naive_result,
        "Threshold Arb": threshold_result,
        "Buy-Hold": bh_result,
    }
