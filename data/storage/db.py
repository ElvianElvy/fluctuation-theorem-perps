"""DuckDB storage manager — columnar OLAP database for time series."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd
from rich.console import Console
from config.settings import settings
from data.schemas import FundingRate, OHLCV

console = Console()

SCHEMA_STMTS = [
    """CREATE TABLE IF NOT EXISTS funding_rates (
        symbol VARCHAR, funding_rate DOUBLE, funding_time TIMESTAMPTZ, mark_price DOUBLE,
        PRIMARY KEY (symbol, funding_time))""",
    """CREATE TABLE IF NOT EXISTS klines (
        symbol VARCHAR, open_time TIMESTAMPTZ, open DOUBLE, high DOUBLE,
        low DOUBLE, close DOUBLE, volume DOUBLE, source VARCHAR,
        PRIMARY KEY (symbol, source, open_time))""",
    """CREATE TABLE IF NOT EXISTS basis (
        symbol VARCHAR, timestamp TIMESTAMPTZ, perp_price DOUBLE,
        spot_price DOUBLE, basis DOUBLE, basis_pct DOUBLE,
        PRIMARY KEY (symbol, timestamp))""",
    """CREATE TABLE IF NOT EXISTS funding_cycles (
        symbol VARCHAR, start_time TIMESTAMPTZ, end_time TIMESTAMPTZ,
        funding_rate DOUBLE, basis_start DOUBLE, basis_end DOUBLE,
        basis_mean DOUBLE, basis_std DOUBLE, mark_price_start DOUBLE,
        mark_price_end DOUBLE, work DOUBLE, energy_start DOUBLE,
        energy_end DOUBLE, heat DOUBLE, delta_free_energy DOUBLE,
        PRIMARY KEY (symbol, start_time))""",
]


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        for stmt in SCHEMA_STMTS:
            self.conn.execute(stmt)
        console.print(f"[dim]DuckDB ready: {self.db_path}[/dim]")

    def close(self):
        self.conn.close()

    def insert_funding_rates(self, rates: list[FundingRate]) -> int:
        if not rates: return 0
        df = pd.DataFrame([vars(r) for r in rates])
        self.conn.execute("INSERT OR REPLACE INTO funding_rates SELECT * FROM df")
        return len(df)

    def insert_klines(self, klines: list[OHLCV]) -> int:
        if not klines: return 0
        df = pd.DataFrame([vars(k) for k in klines])
        self.conn.execute("INSERT OR REPLACE INTO klines SELECT * FROM df")
        return len(df)

    def compute_basis(self, symbol: str) -> int:
        self.conn.execute("""
            INSERT OR REPLACE INTO basis
            SELECT f.symbol, f.open_time, f.close, s.close,
                   f.close - s.close, (f.close - s.close)/s.close*100
            FROM klines f JOIN klines s ON f.symbol=s.symbol AND f.open_time=s.open_time
            WHERE f.source='futures' AND s.source='spot' AND f.symbol=?
        """, [symbol])
        n = self.conn.execute("SELECT COUNT(*) FROM basis WHERE symbol=?", [symbol]).fetchone()[0]
        console.print(f"  [green]✓ {n:,} basis points for {symbol}[/green]")
        return n

    def get_funding_rates(self, symbol: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM funding_rates WHERE symbol=? ORDER BY funding_time", [symbol]).fetchdf()

    def get_basis(self, symbol: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM basis WHERE symbol=? ORDER BY timestamp", [symbol]).fetchdf()

    def get_basis_at_funding_times(self, symbol: str) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT f.symbol, f.funding_time, f.funding_rate, f.mark_price,
                   b.basis, b.basis_pct, b.perp_price, b.spot_price
            FROM funding_rates f ASOF JOIN basis b ON f.symbol=b.symbol AND f.funding_time>=b.timestamp
            WHERE f.symbol=? ORDER BY f.funding_time
        """, [symbol]).fetchdf()

    def get_basis_between(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM basis WHERE symbol=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
            [symbol, start, end]).fetchdf()

    def insert_funding_cycles(self, df: pd.DataFrame) -> int:
        if df.empty: return 0
        self.conn.execute("INSERT OR REPLACE INTO funding_cycles SELECT * FROM df")
        return len(df)

    def get_funding_cycles(self, symbol: str) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT * FROM funding_cycles WHERE symbol=? ORDER BY start_time", [symbol]).fetchdf()

    def summary(self) -> dict:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ["funding_rates", "klines", "basis", "funding_cycles"]}
