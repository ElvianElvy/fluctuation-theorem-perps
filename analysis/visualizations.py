"""Paper-quality figures — peer-review standard. No Unicode glyph issues."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from config.settings import settings
from data.schemas import CrooksResult

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})

C = dict(fwd="#2166ac", rev="#b2182b", theory="#4daf4a", equity="#1b9e77",
         naive="#d95f02", bench="#7570b3", sigma="#e7298a", beta="#66a61e",
         v2="#e6550d", threshold="#3182bd", buyhold="#636363")


def _save(fig, name, out=None):
    d = out or settings.figures_dir
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf"); fig.savefig(d / f"{name}.png")
    plt.close(fig)


# ── FIGS 1-9: Physics figures (unchanged) ─────────────────

def fig1_basis_distribution(basis, kappa, temperature, label="", out=None):
    fig, (ax, ax_qq) = plt.subplots(1, 2, figsize=(10, 4))
    b = basis[~np.isnan(basis)]
    mu, sig = np.mean(b), np.std(b)
    ax.hist(b, bins=80, density=True, alpha=0.6, color=C["fwd"], label="Observed")
    x = np.linspace(mu-4*sig, mu+4*sig, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sig), "k-", lw=2, label=f"N({mu:.3f}, {sig:.3f}^2)")
    ax.set_xlabel("Basis (%)"); ax.set_ylabel("Density")
    ax.set_title(f"Equilibrium Basis Distribution {label}")
    _, p = stats.kstest((b-mu)/sig, "norm")
    ax.text(0.98, 0.95, f"KS p = {p:.4f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.legend()
    stats.probplot(b, dist="norm", plot=ax_qq); ax_qq.set_title("Q-Q Plot")
    fig.tight_layout(); _save(fig, f"fig1_basis{'_'+label if label else ''}", out)


def fig2_work_distributions(dissipation, label="", out=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dissipation, bins=80, density=True, alpha=0.6, color=C["fwd"], label="sigma")
    ax.axvline(0, ls="--", color="k", lw=1, label="sigma=0")
    ax.set_xlabel("Dissipation sigma = W - dF"); ax.set_ylabel("Density")
    ax.set_title(f"Dissipation Distribution {label}"); ax.legend()
    fig.tight_layout(); _save(fig, f"fig2_dissipation{'_'+label if label else ''}", out)


def fig3_dft_plot(cr, beta_true=None, label="", out=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(cr.work_bins) > 0:
        ax.scatter(cr.work_bins, cr.log_ratio, s=18, color=C["fwd"], zorder=3, label="KDE estimate")
        w = np.linspace(cr.work_bins.min(), cr.work_bins.max(), 200)
        ax.plot(w, cr.slope*w + cr.intercept, "k-", lw=2,
                label=f"Fit: slope={cr.slope:.4f}  R^2={cr.r_squared:.4f}")
        if beta_true is not None:
            ax.plot(w, beta_true*w, "--", color=C["theory"], lw=1.5,
                    label=f"Theory: slope=beta={beta_true:.4f}")
    ax.axhline(0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("Dissipation sigma"); ax.set_ylabel("ln[P(sigma) / P(-sigma)]")
    ax.set_title(f"Detailed Fluctuation Theorem {label}"); ax.legend()
    fig.tight_layout(); _save(fig, f"fig3_dft{'_'+label if label else ''}", out)


def fig4_je_convergence(dissipation, beta, label="", out=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(42)
    ns = np.unique(np.geomspace(20, len(dissipation), 60).astype(int))
    meds, lo, hi = [], [], []
    for n in ns:
        vals = [float(np.mean(np.exp(np.clip(-beta*rng.choice(dissipation, n, True), -500, 500))))
                for _ in range(200)]
        meds.append(np.median(vals)); lo.append(np.percentile(vals, 5)); hi.append(np.percentile(vals, 95))
    ax.plot(ns, meds, "o-", ms=3, color=C["naive"], label="Naive exp average")
    ax.fill_between(ns, lo, hi, alpha=0.15, color=C["naive"])
    ax.axhline(1.0, ls="--", color="k", lw=1.2, label="JE target (1.0)")
    ax.set_xscale("log"); ax.set_xlabel("Number of cycles"); ax.set_ylabel("<exp(-beta*sigma)>")
    ax.set_title(f"Jarzynski Convergence {label}"); ax.legend()
    fig.tight_layout(); _save(fig, f"fig4_je_conv{'_'+label if label else ''}", out)


def fig5_entropy_production(sigma_df, cycles_df=None, label="", out=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    t = pd.to_datetime(sigma_df["timestamp"]); sig = sigma_df["sigma"].values
    ax = axes[0]
    ax.plot(t, sig, color=C["sigma"], lw=0.5, alpha=0.8)
    ax.axhline(0, ls="--", color="k", lw=0.8)
    p80 = np.percentile(np.abs(sig), 80)
    ax.fill_between(t, sig.min(), sig.max(), where=np.abs(sig)>p80, alpha=0.1, color=C["sigma"],
                    label="|sigma| > 80th pct")
    ax.set_ylabel("sigma(t)"); ax.set_title(f"Entropy Production {label}"); ax.legend(loc="upper right")
    ax2 = axes[1]
    if "basis_mean" in sigma_df.columns:
        ax2.plot(t, sigma_df["basis_mean"].values, color=C["fwd"], lw=0.5)
    ax2.axhline(0, ls=":", color="gray"); ax2.set_ylabel("Basis (%)"); ax2.set_xlabel("Time")
    fig.tight_layout(); _save(fig, f"fig5_entropy{'_'+label if label else ''}", out)


def fig6_rolling_temperature(sigma_df, label="", out=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = pd.to_datetime(sigma_df["timestamp"])
    ax.plot(t, sigma_df["beta"].values, color=C["beta"], lw=0.8)
    ax.set_ylabel("beta(t)"); ax.set_xlabel("Time")
    ax.set_title(f"Rolling Market Temperature {label}")
    fig.tight_layout(); _save(fig, f"fig6_temp{'_'+label if label else ''}", out)


def fig7_relaxation_times(results, out=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    symbols, taus = [], []
    for sym, rr in results.items():
        if rr and "relaxation" in rr:
            symbols.append(sym); taus.append(rr["relaxation"]["tau_relax_hours"])
    if not symbols: plt.close(fig); return
    x = np.arange(len(symbols))
    colors = [C["fwd"] if t < 8 else C["rev"] for t in taus]
    ax.bar(x, taus, color=colors, alpha=0.7)
    ax.axhline(8.0, ls="--", color="k", lw=1.5, label="Funding period (8h)")
    ax.set_xticks(x); ax.set_xticklabels(symbols)
    ax.set_ylabel("Relaxation time 1/kappa (hours)")
    ax.set_title("Relaxation Time vs Funding Period")
    for i, t in enumerate(taus):
        verdict = "JE valid" if t < 8 else "JE violated"
        ax.text(i, t + 0.3, f"{t:.1f}h\n{verdict}", ha="center", fontsize=9)
    ax.legend(); fig.tight_layout(); _save(fig, "fig7_relaxation", out)


def fig8_backtest_equity(bt_result, label="", out=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    t, eq = bt_result.timestamps, bt_result.equity_curve
    ax.plot(t, eq, color=C["equity"], lw=1.5, label="FT Strategy (virtual)")
    if eq: ax.axhline(eq[0], ls=":", color="gray", label="Initial capital")
    ax.set_xlabel("Time"); ax.set_ylabel("Virtual USDT")
    ax.set_title(f"Paper-Trade Backtest {label}"); ax.legend()
    fig.tight_layout(); _save(fig, f"fig8_bt{'_'+label if label else ''}", out)


def fig9_phase_space(sigma_df, label="", out=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    b, s = sigma_df["beta"].values, sigma_df["sigma"].values
    sc = ax.scatter(b, s, c=np.arange(len(b)), cmap="viridis", s=4, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Time index")
    ax.axhline(0, ls="--", color="gray"); ax.set_xlabel("beta"); ax.set_ylabel("sigma")
    ax.set_title(f"Phase Space {label}"); fig.tight_layout()
    _save(fig, f"fig9_phase{'_'+label if label else ''}", out)


def fig_synthetic_panel(synth_result, out=None):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)
    cycles = synth_result["cycles"]; cr = synth_result["crooks"]
    sigma_df = synth_result["sigma_df"]; truth = synth_result["truth"]
    w = cycles["work"].values; df = cycles["delta_free_energy"].values; diss = w - df

    ax1 = fig.add_subplot(gs[0, 0])
    b = cycles["basis_mean"].values; mu, sig = np.mean(b), np.std(b)
    ax1.hist(b, bins=80, density=True, alpha=0.6, color=C["fwd"])
    x = np.linspace(mu-4*sig, mu+4*sig, 300)
    ax1.plot(x, stats.norm.pdf(x, mu, sig), "k-", lw=2)
    ax1.set_xlabel("Basis"); ax1.set_ylabel("Density"); ax1.set_title("(a) Basis vs Boltzmann")

    ax2 = fig.add_subplot(gs[0, 1])
    if len(cr.work_bins) > 0:
        ax2.scatter(cr.work_bins, cr.log_ratio, s=12, color=C["fwd"])
        wr = np.linspace(cr.work_bins.min(), cr.work_bins.max(), 200)
        ax2.plot(wr, cr.slope*wr+cr.intercept, "k-", lw=2)
    ax2.set_xlabel("sigma"); ax2.set_ylabel("ln[P(s)/P(-s)]"); ax2.set_title(f"(b) DFT R^2={cr.r_squared:.3f}")

    ax3 = fig.add_subplot(gs[1, 0])
    beta = truth["beta"]; rng = np.random.default_rng(42)
    ns = np.unique(np.geomspace(20, len(diss), 40).astype(int))
    meds = [np.median([float(np.mean(np.exp(np.clip(-beta*rng.choice(diss, n, True), -500, 500))))
            for _ in range(100)]) for n in ns]
    ax3.plot(ns, meds, "o-", ms=3, color=C["naive"]); ax3.axhline(1.0, ls="--", color="k")
    ax3.set_xscale("log"); ax3.set_xlabel("Samples"); ax3.set_ylabel("<exp(-beta*sigma)>")
    ax3.set_title("(c) JE Convergence")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(np.arange(len(sigma_df)), sigma_df["sigma"].values, color=C["sigma"], lw=0.5)
    ax4.axhline(0, ls="--", color="k"); ax4.set_xlabel("Cycle"); ax4.set_ylabel("sigma")
    ax4.set_title(f"(d) Entropy <sigma>={sigma_df['sigma'].mean():.4f}")
    _save(fig, "fig_synthetic_panel", out)


# ── NEW: V2 Strategy Figures ──────────────────────────────

def fig10_strategy_equity_comparison(strat_results, label="", out=None):
    """Fig 10: Equity curves — V2 vs all baselines."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"V2 Physics": C["v2"], "Naive Always-On": C["naive"],
              "Threshold Arb": C["threshold"], "Buy-Hold": C["buyhold"]}
    lws = {"V2 Physics": 2.5, "Naive Always-On": 1.5,
           "Threshold Arb": 1.5, "Buy-Hold": 1.0}

    for name, res in strat_results.items():
        if not res.equity_curve:
            continue
        t = res.timestamps
        eq = res.equity_curve
        ax.plot(t, eq, color=colors.get(name, "gray"), lw=lws.get(name, 1),
                label=f"{name} ({res.metrics()['total_return_pct']:.1f}%)", alpha=0.85)

    ax.axhline(10000, ls=":", color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time"); ax.set_ylabel("Virtual USDT")
    ax.set_title(f"Strategy Comparison {label}"); ax.legend(loc="best")
    fig.tight_layout(); _save(fig, f"fig10_strategy_comparison{'_'+label if label else ''}", out)


def fig11_regime_heatmap(strat_results, label="", out=None):
    """Fig 11: Regime classification over time with trade markers."""
    v2 = strat_results.get("V2 Physics")
    if not v2 or not v2.regimes:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 3]})

    t = v2.timestamps
    regime_map = {"EQUILIBRIUM": 2, "WARM": 1, "NESS": 0}
    r_vals = [regime_map.get(r, 1) for r in v2.regimes]

    # Regime strip
    ax1.fill_between(t, 0, 1, where=[r == 2 for r in r_vals],
                     color=C["fwd"], alpha=0.6, label="EQUILIBRIUM", transform=ax1.get_xaxis_transform())
    ax1.fill_between(t, 0, 1, where=[r == 1 for r in r_vals],
                     color=C["naive"], alpha=0.6, label="WARM", transform=ax1.get_xaxis_transform())
    ax1.fill_between(t, 0, 1, where=[r == 0 for r in r_vals],
                     color=C["rev"], alpha=0.6, label="NESS", transform=ax1.get_xaxis_transform())
    ax1.set_yticks([]); ax1.set_title(f"Market Regime {label}"); ax1.legend(loc="upper right", ncol=3)

    # Equity
    ax2.plot(t, v2.equity_curve, color=C["v2"], lw=1.5)
    # Trade markers
    for trade in v2.trades:
        if trade.pnl_total > 0:
            ax2.axvline(trade.entry_time, color=C["theory"], alpha=0.2, lw=0.5)
        else:
            ax2.axvline(trade.entry_time, color=C["rev"], alpha=0.1, lw=0.5)
    ax2.set_ylabel("Virtual USDT"); ax2.set_xlabel("Time")
    fig.tight_layout(); _save(fig, f"fig11_regime{'_'+label if label else ''}", out)


def fig12_signal_distributions(strat_results, signals_data=None, label="", out=None):
    """Fig 12: Distribution of 4 physics signals at trade entry."""
    v2 = strat_results.get("V2 Physics")
    if not v2 or not v2.trades:
        return
    # We can show PnL by regime
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    regimes = ["EQUILIBRIUM", "WARM", "NESS"]
    for ax, regime in zip(axes, regimes):
        rtrades = [t for t in v2.trades if t.regime_at_entry == regime]
        if not rtrades:
            ax.set_title(f"{regime}\nNo trades"); continue
        pnls = [t.pnl_total for t in rtrades]
        ax.hist(pnls, bins=30, alpha=0.6, color=C["fwd"] if regime == "EQUILIBRIUM"
                else C["naive"] if regime == "WARM" else C["rev"])
        ax.axvline(0, ls="--", color="k")
        ax.axvline(np.mean(pnls), ls="-", color="red", lw=2, label=f"mean={np.mean(pnls):.2f}")
        ax.set_title(f"{regime} ({len(rtrades)} trades)")
        ax.set_xlabel("PnL (virtual USDT)"); ax.legend()
    fig.suptitle(f"Trade PnL by Regime {label}"); fig.tight_layout()
    _save(fig, f"fig12_pnl_by_regime{'_'+label if label else ''}", out)


# ── Generate All ──────────────────────────────────────────

def generate_all_figures(synth_result, real_results=None, strategy_results=None):
    settings.ensure_dirs(); out = settings.figures_dir
    con = __import__("rich").console.Console()
    con.print("\n[bold cyan]Generating paper figures...[/bold cyan]")

    cycles = synth_result["cycles"]; cr = synth_result["crooks"]
    sigma_df = synth_result["sigma_df"]; truth = synth_result["truth"]
    w = cycles["work"].values; df_arr = cycles["delta_free_energy"].values; diss = w - df_arr

    fig1_basis_distribution(cycles["basis_mean"].values, 0, 0, "synthetic", out)
    fig2_work_distributions(diss, "synthetic", out)
    fig3_dft_plot(cr, truth["beta"], "synthetic", out)
    fig4_je_convergence(diss, truth["beta"], "synthetic", out)
    fig5_entropy_production(sigma_df, cycles, "synthetic", out)
    fig6_rolling_temperature(sigma_df, "synthetic", out)
    if "bt_result" in synth_result and synth_result["bt_result"]:
        fig8_backtest_equity(synth_result["bt_result"], "synthetic", out)
    fig9_phase_space(sigma_df, "synthetic", out)
    fig_synthetic_panel(synth_result, out)

    if real_results:
        fig7_relaxation_times(real_results, out)
        for sym, rr in real_results.items():
            if not rr: continue
            c2, s2 = rr["cycles"], rr["sigma_df"]
            w2 = c2["work"].values; df2 = c2["delta_free_energy"].values; d2 = w2 - df2
            fig1_basis_distribution(c2["basis_mean"].values, 0, 0, sym, out)
            fig2_work_distributions(d2, sym, out)
            fig3_dft_plot(rr["crooks"], rr["temperature"].beta, sym, out)
            fig4_je_convergence(d2, rr["temperature"].beta, sym, out)
            fig5_entropy_production(s2, c2, sym, out)
            fig6_rolling_temperature(s2, sym, out)
            fig9_phase_space(s2, sym, out)
            if "bt_result" in rr and rr["bt_result"]:
                fig8_backtest_equity(rr["bt_result"], sym, out)

    # V2 Strategy figures
    if strategy_results:
        for sym, strats in strategy_results.items():
            fig10_strategy_equity_comparison(strats, sym, out)
            fig11_regime_heatmap(strats, sym, out)
            fig12_signal_distributions(strats, label=sym, out=out)

    con.print(f"[green]Figures saved to {out}/[/green]")
