"""Data collection CLI.  python -m scripts.collect_data"""
import asyncio
import sys
from datetime import datetime, timezone
from rich.console import Console
from config.settings import settings
from data.collectors.binance import BinanceCollector
from data.storage.db import Database
from theory.thermodynamic_quantities import build_funding_cycles

console = Console()


async def main():
    settings.ensure_dirs()
    db = Database()
    collector = BinanceCollector()
    start = datetime.strptime(settings.collection_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(settings.collection_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    console.print(f"[bold]Collecting {len(settings.symbols)} symbols: {', '.join(settings.symbols)}[/bold]")
    console.print(f"  Period: {start.date()} → {end.date()}")

    for symbol in settings.symbols:
        try:
            data = await collector.collect_all(symbol, start, end)
            db.insert_funding_rates(data["funding_rates"])
            db.insert_klines(data["futures_klines"])
            db.insert_klines(data["spot_klines"])
            db.compute_basis(symbol)
            cycles = build_funding_cycles(db, symbol)
            if not cycles.empty:
                db.insert_funding_cycles(cycles)
                console.print(f"[green]✓ {symbol}: {len(cycles):,} funding cycles[/green]")
        except Exception as e:
            console.print(f"[red]✗ {symbol}: {e}[/red]")

    console.rule("[bold green]Collection Complete[/bold green]")
    for k, v in db.summary().items():
        console.print(f"  {k}: {v:,} rows")
    db.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
