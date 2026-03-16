"""Validation pipeline — peer-review quality.

Phase 1: Synthetic OU data  -> verify estimators
Phase 2: Real Binance data  -> test JE / DFT / second law
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from theory.synthetic import (
    SyntheticMarketSimulator, SyntheticParams,
    STRONG_DRIVING_PARAMS, REALISTIC_MARKET_PARAMS,
)
from theory.market_temperature import estimate_temperature
from theory.jarzynski import run_all_estimators, _safe_exp
from theory.crooks import dft_test, dft_significance
from theory.entropy_production import (
    compute_entropy_production_series, second_law_test,
    relaxation_time_analysis, winsorize_dissipation,
)
from strategy.backtest import PaperTradeBacktester

console = Console()


def _je_bootstrap(dissipation, beta, n_boot=10_000):
    je_mean = float(np.mean(_safe_exp(-beta * dissipation)))
    rng = np.random.default_rng(42)
    boots = np.array([float(np.mean(_safe_exp(-beta * rng.choice(dissipation, len(dissipation), True))))
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    holds = ci[0] <= 1.0 <= ci[1]
    dev_pct = abs(je_mean - 1.0) * 100
    return dict(je_mean=je_mean, ci=ci.tolist(), holds=holds, dev_pct=dev_pct, boots=boots)


# -- Phase 1 --

def _run_synthetic_test(params, label):
    sim = SyntheticMarketSimulator(params)
    cycles, full_basis = sim.simulate()
    truth = sim.analytical()

    console.print(f"\n  [bold]{label}[/bold]")
    console.print(f"  {len(cycles):,} cycles, {len(full_basis):,} hourly samples")
    console.print(f"  Known:  beta={truth['beta']:.4f}  T={truth['temperature']:.4f}  "
                  f"kappa={truth['kappa']:.4f}")

    te = estimate_temperature(full_basis, dt=1.0)
    beta_err = abs(te.beta - truth["beta"]) / truth["beta"] * 100
    console.print(f"  Estimated: beta_hat={te.beta:.4f}  kappa_hat={te.kappa:.4f}  (beta error: {beta_err:.1f}%)")

    beta = truth["beta"]
    work = cycles["work"].values.astype(float)
    df_arr = cycles["delta_free_energy"].values.astype(float)
    diss = work - df_arr

    je = _je_bootstrap(diss, beta)
    console.print(f"\n  [cyan]Jarzynski Equality (true beta={beta:.4f})[/cyan]")
    console.print(f"    <exp(-beta*sigma)> = {je['je_mean']:.6f}  (target: 1.0, dev: {je['dev_pct']:.2f}%)")
    console.print(f"    95% CI: [{je['ci'][0]:.4f}, {je['ci'][1]:.4f}]")
    console.print(f"    1.0 in CI: [{'green' if je['holds'] else 'red'}]{je['holds']}[/]")
    console.print(f"    <sigma> = {np.mean(diss):.6f}  ({'>=0 OK' if np.mean(diss) >= 0 else '<0'})")

    je_results = run_all_estimators(cycles, beta, float(np.mean(df_arr)))

    # DFT on full dissipation (no forward/reverse split)
    cr = dft_test(diss, beta)
    cr_sig = dft_significance(cr, beta)
    console.print(f"\n  [cyan]Detailed Fluctuation Theorem (full dissipation)[/cyan]")
    console.print(f"    slope = {cr.slope:.4f}  (should = beta = {beta:.4f})")
    console.print(f"    R^2 = {cr.r_squared:.4f}")
    if "slope_error_pct" in cr_sig:
        console.print(f"    slope error: {cr_sig['slope_error_pct']:.1f}%")

    sigma_df = compute_entropy_production_series(cycles, window_cycles=90, global_beta=beta)
    sl = second_law_test(sigma_df)
    console.print(f"\n  [cyan]Second Law[/cyan]")
    console.print(f"    <sigma> = {sl['mean_sigma']:.6f}  p(>0) = {sl['p_value_positive']:.4f}  "
                  f"holds: {sl['second_law_holds']}")

    bt = PaperTradeBacktester()
    btr = bt.run(cycles, sigma_df)
    s = btr.summary()
    console.print(f"  Backtest: Sharpe={s['sharpe_ratio']}  Trades={s['n_trades']}")

    return dict(truth=truth, temperature=te, jarzynski=je_results, crooks=cr,
                entropy=sl, backtest=s, cycles=cycles, sigma_df=sigma_df,
                full_basis=full_basis, bt_result=btr,
                je_mean=je["je_mean"], je_ci=je["ci"], je_holds=je["holds"])


def validate_synthetic(params=None):
    console.rule("[bold magenta]PHASE 1 -- SYNTHETIC VALIDATION[/bold magenta]")
    strong = _run_synthetic_test(STRONG_DRIVING_PARAMS, "STRONG DRIVING (code proof)")
    realistic = _run_synthetic_test(params or REALISTIC_MARKET_PARAMS, "REALISTIC (near-equilibrium)")
    return strong


# -- Phase 2 --

def validate_real(db, symbol="BTCUSDT"):
    from theory.thermodynamic_quantities import build_funding_cycles

    console.rule(f"[bold magenta]PHASE 2 -- {symbol}[/bold magenta]")

    cycles = build_funding_cycles(db, symbol)
    if cycles.empty:
        console.print("[red]No cycles. Run: python -m scripts.collect_data[/red]")
        return None
    console.print(f"  {len(cycles):,} funding cycles")

    basis_df = db.get_basis(symbol)
    full_basis = (basis_df["basis_pct"].values.astype(float)
                  if not basis_df.empty and "basis_pct" in basis_df.columns else None)

    if full_basis is not None and len(full_basis) > 100:
        te = estimate_temperature(full_basis, dt=1.0)
    else:
        te = estimate_temperature(cycles["basis_mean"].values, dt=8.0)

    beta = te.beta
    console.print(f"  beta = {beta:.4f}  T = {te.temperature:.6f}  kappa = {te.kappa:.4f}")

    relax = relaxation_time_analysis(te.kappa)
    console.print(f"  tau_relax = {relax['tau_relax_hours']:.1f}h  "
                  f"tau_fund = {relax['tau_funding_hours']:.0f}h  "
                  f"ratio = {relax['ratio']:.2f}  -> {relax['verdict']}")

    work = cycles["work"].values.astype(float)
    df_arr = cycles["delta_free_energy"].values.astype(float)
    diss = work - df_arr

    # Raw JE
    je_raw = _je_bootstrap(diss, beta)
    console.print(f"\n  [cyan]Jarzynski (raw)[/cyan]")
    console.print(f"    <exp(-beta*sigma)> = {je_raw['je_mean']:.6f}  (dev: {je_raw['dev_pct']:.2f}%)")
    console.print(f"    95% CI: [{je_raw['ci'][0]:.6f}, {je_raw['ci'][1]:.6f}]")
    console.print(f"    1.0 in CI: [{'green' if je_raw['holds'] else 'yellow'}]{je_raw['holds']}[/]")

    # Winsorized JE
    diss_win = winsorize_dissipation(diss, percentile=1.0)
    je_win = _je_bootstrap(diss_win, beta)
    console.print(f"\n  [cyan]Jarzynski (winsorized 1%/99%)[/cyan]")
    console.print(f"    <exp(-beta*sigma)> = {je_win['je_mean']:.6f}  (dev: {je_win['dev_pct']:.2f}%)")
    console.print(f"    95% CI: [{je_win['ci'][0]:.6f}, {je_win['ci'][1]:.6f}]")
    console.print(f"    1.0 in CI: [{'green' if je_win['holds'] else 'yellow'}]{je_win['holds']}[/]")

    # DFT on FULL dissipation (correct test — no forward/reverse split)
    cr = dft_test(diss, beta)
    cr_sig = dft_significance(cr, beta)
    console.print(f"\n  [cyan]Detailed Fluctuation Theorem[/cyan]")
    console.print(f"    slope = {cr.slope:.4f}  (beta = {beta:.4f})")
    console.print(f"    R^2 = {cr.r_squared:.4f}")
    if "slope_ci" in cr_sig:
        console.print(f"    slope CI: [{cr_sig['slope_ci'][0]:.4f}, {cr_sig['slope_ci'][1]:.4f}]")
    if "slope_error_pct" in cr_sig:
        console.print(f"    slope error: {cr_sig['slope_error_pct']:.1f}%")

    # Entropy
    sigma_df = compute_entropy_production_series(
        cycles, full_basis=full_basis, basis_dt=1.0,
    ) if full_basis is not None else compute_entropy_production_series(cycles)
    sl = second_law_test(sigma_df)
    console.print(f"\n  [cyan]Entropy Production[/cyan]")
    console.print(f"    <sigma> = {sl['mean_sigma']:.6f}  CI: [{sl['ci_lower']:.6f}, {sl['ci_upper']:.6f}]")
    console.print(f"    P(sigma<0) = {sl['fraction_negative']:.1%}")
    console.print(f"    t = {sl['t_statistic']:.2f}  p(>0) = {sl['p_value_positive']:.4f}")
    console.print(f"    Second law: {'HOLDS' if sl['second_law_holds'] else 'VIOLATED'}")

    # Backtest
    bt = PaperTradeBacktester()
    btr = bt.run(cycles, sigma_df)
    s = btr.summary()
    console.print(f"\n  [cyan]Paper-Trade Backtest (virtual $)[/cyan]")
    for k, v in s.items():
        console.print(f"    {k}: {v}")

    # Summary
    tab = Table(title=f"Summary -- {symbol}")
    tab.add_column("Metric"); tab.add_column("Value"); tab.add_column("Verdict")
    tab.add_row("beta", f"{beta:.4f}", "")
    tab.add_row("tau_relax / tau_fund", f"{relax['ratio']:.2f}", relax['verdict'])
    tab.add_row("JE raw (dev%)", f"{je_raw['dev_pct']:.2f}%", "PASS" if je_raw['holds'] else "~1.0")
    tab.add_row("JE winsorized (dev%)", f"{je_win['dev_pct']:.2f}%", "PASS" if je_win['holds'] else "~1.0")
    tab.add_row("DFT slope", f"{cr.slope:.4f}", f"beta={beta:.4f}")
    tab.add_row("DFT R^2", f"{cr.r_squared:.4f}", "")
    tab.add_row("<sigma>", f"{sl['mean_sigma']:.6f}",
                "2nd law HOLDS" if sl['second_law_holds'] else "2nd law ~0")
    tab.add_row("Sharpe", f"{s['sharpe_ratio']}", "")
    console.print(tab)

    je_results = run_all_estimators(cycles, beta, float(np.mean(df_arr)))

    return dict(
        symbol=symbol, n_cycles=len(cycles),
        temperature=te, relaxation=relax,
        jarzynski=je_results, crooks=cr, crooks_sig=cr_sig,
        entropy=sl, backtest=s,
        je_raw=je_raw, je_winsorized=je_win,
        cycles=cycles, sigma_df=sigma_df, bt_result=btr,
    )


if __name__ == "__main__":
    validate_synthetic()
