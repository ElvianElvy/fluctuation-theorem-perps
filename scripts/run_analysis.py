"""Full analysis pipeline.  python -m scripts.run_analysis

Phase 1: Synthetic validation
Phase 2: Real data — JE, DFT, entropy, relaxation
Phase 3: V2 Strategy — physics-informed trading vs baselines
"""
from rich.console import Console
from config.settings import settings
from analysis.validation import validate_synthetic, validate_real
from analysis.visualizations import generate_all_figures

console = Console()


def main():
    settings.ensure_dirs()
    console.rule("[bold]Fluctuation Theorem Perps — Full Analysis Pipeline[/bold]")

    # Phase 1
    console.print("\n[yellow]Running synthetic validation (virtual money)...[/yellow]")
    synth = validate_synthetic()

    # Phase 2 + 3
    real_results: dict = {}
    strategy_results: dict = {}

    try:
        from data.storage.db import Database
        db = Database()
        s = db.summary()
        if s.get("funding_cycles", 0) > 0:
            for symbol in settings.symbols:
                r = validate_real(db, symbol)
                if r is not None:
                    real_results[symbol] = r

            # Phase 3: V2 Strategy Backtests
            if real_results:
                console.print("\n")
                console.rule("[bold magenta]PHASE 3 -- V2 STRATEGY BACKTEST[/bold magenta]")
                from strategy.backtest_v2 import run_walk_forward
                from analysis.strategy_analysis import compare_strategies, multi_asset_summary

                for symbol, rr in real_results.items():
                    console.print(f"\n[bold cyan]Running V2 for {symbol}...[/bold cyan]")
                    cycles = rr["cycles"]
                    basis_df = db.get_basis(symbol)
                    full_basis = (basis_df["basis_pct"].values.astype(float)
                                  if not basis_df.empty and "basis_pct" in basis_df.columns
                                  else None)
                    strats = run_walk_forward(cycles, full_basis)
                    if strats:
                        strategy_results[symbol] = strats
                        compare_strategies(strats, symbol)

                if strategy_results:
                    multi_asset_summary(strategy_results)
        else:
            console.print("\n[yellow]No real data yet.  Run:[/yellow]")
            console.print("  [bold]python -m scripts.collect_data[/bold]")
        db.close()
    except Exception as e:
        console.print(f"\n[yellow]Skipping real data: {e}[/yellow]")
        import traceback
        traceback.print_exc()

    # Figures
    console.print("\n[yellow]Generating paper figures...[/yellow]")
    generate_all_figures(synth, real_results if real_results else None,
                         strategy_results if strategy_results else None)

    console.rule("[bold green]Pipeline Complete[/bold green]")
    console.print(f"  Figures -> {settings.figures_dir}/")
    console.print(f"  Database -> {settings.db_path}")
    console.print("\n[dim]All trading results use VIRTUAL MONEY only.[/dim]")


if __name__ == "__main__":
    main()
