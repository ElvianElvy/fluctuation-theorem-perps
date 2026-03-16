"""Strategy comparison — statistical tests and attribution."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from strategy.backtest_v2 import StrategyResult

console = Console()


def compare_strategies(results: dict[str, StrategyResult], symbol: str = "") -> dict:
    console.print(f"\n[bold cyan]Strategy Comparison{' — ' + symbol if symbol else ''}[/bold cyan]")

    tab = Table(title=f"Strategy Metrics{' — ' + symbol if symbol else ''}")
    for c in ["Strategy", "Sharpe", "Sortino", "Return%", "MaxDD%",
              "WinRate%", "Trades", "AvgPnL", "ProfitFactor"]:
        tab.add_column(c)

    all_metrics = {}
    for name, res in results.items():
        m = res.metrics()
        all_metrics[name] = m
        tab.add_row(m["name"], str(m["sharpe"]), str(m["sortino"]),
                     str(m["total_return_pct"]), str(m["max_dd_pct"]),
                     str(m["win_rate_pct"]), str(m["n_trades"]),
                     str(m["avg_pnl"]), str(m["profit_factor"]))
    console.print(tab)

    # Bootstrap significance: V2 Sharpe vs each baseline
    v2 = results.get("V2 Physics")
    if v2 and len(v2.equity_curve) > 20:
        console.print(f"\n[cyan]Significance Tests (V2 vs baselines):[/cyan]")
        v2_eq = np.array(v2.equity_curve, dtype=float)
        v2_r = np.diff(v2_eq) / np.maximum(v2_eq[:-1], 1e-12)
        v2_r = v2_r[np.isfinite(v2_r)]

        for name, res in results.items():
            if name == "V2 Physics" or len(res.equity_curve) < 20:
                continue
            eq = np.array(res.equity_curve, dtype=float)
            bl_r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
            bl_r = bl_r[np.isfinite(bl_r)]
            mn = min(len(v2_r), len(bl_r))
            if mn < 20: continue

            rng = np.random.default_rng(42)
            ann = np.sqrt(3 * 365)
            diffs = []
            for _ in range(5000):
                idx = rng.choice(mn, mn, replace=True)
                sv = np.mean(v2_r[:mn][idx]) / max(np.std(v2_r[:mn][idx]), 1e-12) * ann
                sb = np.mean(bl_r[:mn][idx]) / max(np.std(bl_r[:mn][idx]), 1e-12) * ann
                diffs.append(sv - sb)
            diffs = np.array(diffs)
            ci = np.percentile(diffs, [2.5, 97.5])
            p = float(np.mean(diffs <= 0))
            verdict = "V2 BETTER" if p < 0.05 else ("V2 WORSE" if p > 0.95 else "COMPARABLE")
            console.print(f"  V2 vs {name}: dSharpe={np.mean(diffs):.3f}  "
                          f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  p={p:.4f}  {verdict}")

    # Regime breakdown
    if v2 and v2.regimes:
        console.print(f"\n[cyan]Time in Regime:[/cyan]")
        rc = pd.Series(v2.regimes).value_counts()
        total = len(v2.regimes)
        for r, c in rc.items():
            console.print(f"  {r}: {c} cycles ({c/total*100:.1f}%)")

    if v2 and v2.trades:
        console.print(f"\n[cyan]V2 Regime Attribution:[/cyan]")
        rtab = Table(title="Performance by Regime")
        rtab.add_column("Regime"); rtab.add_column("Trades"); rtab.add_column("WinRate%")
        rtab.add_column("AvgPnL"); rtab.add_column("TotalPnL")
        for regime in ["EQUILIBRIUM", "WARM", "NESS"]:
            rt = [t for t in v2.trades if t.regime_at_entry == regime]
            if not rt:
                rtab.add_row(regime, "0", "-", "-", "-"); continue
            wins = sum(1 for t in rt if t.pnl_total > 0)
            rtab.add_row(regime, str(len(rt)), f"{wins/len(rt)*100:.1f}",
                         f"{np.mean([t.pnl_total for t in rt]):.2f}",
                         f"{sum(t.pnl_total for t in rt):.2f}")
        console.print(rtab)

    # Drawdown comparison
    console.print(f"\n[cyan]Drawdown Analysis:[/cyan]")
    for name, res in results.items():
        eq = np.array(res.equity_curve, dtype=float)
        if len(eq) < 2: continue
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-12)
        console.print(f"  {name}: MaxDD={np.max(dd)*100:.2f}%  "
                      f"AvgDD={np.mean(dd)*100:.2f}%  "
                      f"DD>5%={np.sum(dd>0.05)} cycles")

    return all_metrics


def multi_asset_summary(all_results: dict[str, dict[str, StrategyResult]]):
    console.print("\n")
    console.rule("[bold magenta]CROSS-ASSET STRATEGY SUMMARY[/bold magenta]")

    tab = Table(title="V2 Physics — All Assets")
    for c in ["Asset", "Sharpe", "Return%", "MaxDD%", "WinRate%", "Trades", "ProfitFactor"]:
        tab.add_column(c)

    for symbol, strats in all_results.items():
        v2 = strats.get("V2 Physics")
        if not v2: continue
        m = v2.metrics()
        tab.add_row(symbol, str(m["sharpe"]), str(m["total_return_pct"]),
                     str(m["max_dd_pct"]), str(m["win_rate_pct"]),
                     str(m["n_trades"]), str(m["profit_factor"]))
    console.print(tab)

    # Compare V2 vs Naive across all assets
    console.print("\n[cyan]V2 vs Naive (per-asset):[/cyan]")
    for symbol, strats in all_results.items():
        v2m = strats.get("V2 Physics", StrategyResult("?")).metrics()
        nm = strats.get("Naive Always-On", StrategyResult("?")).metrics()
        dd_improve = nm["max_dd_pct"] - v2m["max_dd_pct"]
        sharpe_diff = v2m["sharpe"] - nm["sharpe"]
        console.print(f"  {symbol}: V2 Sharpe={v2m['sharpe']}  Naive Sharpe={nm['sharpe']}  "
                      f"dSharpe={sharpe_diff:+.3f}  DD improvement={dd_improve:+.2f}%")
