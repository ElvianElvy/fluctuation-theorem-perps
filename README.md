# Fluctuation Theorem Perps

Non-equilibrium thermodynamics of cryptocurrency perpetual futures.

> **ALL trading is virtual. No real money.**

## Quick Start (Windows)

```powershell
cd D:\Projects\fluctuation-theorem-perps
uv pip install -e ".[dev]"

# 1. Synthetic validation (instant, no internet)
python -m scripts.run_analysis

# 2. Collect 6 years of Binance data (public API, no keys)
python -m scripts.collect_data

# 3. Full analysis + paper figures
python -m scripts.run_analysis
```

Use **Windows Terminal** (not cmd.exe) for Unicode output.

## Database: DuckDB

Embedded columnar OLAP database. Key feature: ASOF JOIN for
aligning funding timestamps to nearest basis measurement.
Single file: `data/ftp.duckdb`.
