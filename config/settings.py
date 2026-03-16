"""Configuration settings."""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    project_root: Path = Field(default=Path("."))
    db_path: Path = Field(default=Path("data/ftp.duckdb"))
    figures_dir: Path = Field(default=Path("output/figures"))
    results_dir: Path = Field(default=Path("output/results"))

    binance_futures_base: str = "https://fapi.binance.com"
    binance_spot_base: str = "https://api.binance.com"
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    collection_start: str = "2020-01-01"
    collection_end: str = "2026-03-16"
    kline_interval: str = "1h"
    request_delay_ms: int = 120

    funding_period_hours: float = 8.0
    equilibrium_window_days: int = 30
    min_equilibrium_samples: int = 90

    bootstrap_n: int = 10_000
    significance_level: float = 0.05
    random_seed: int = 42

    initial_capital: float = 10_000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    slippage_bps: float = 1.0

    def ensure_dirs(self):
        for d in [self.db_path.parent, self.figures_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
