"""Binance USDT-M Futures and Spot data collector.

All public endpoints — **no API key required**.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import httpx
from rich.console import Console
from tqdm import tqdm
from config.settings import settings
from data.schemas import FundingRate, OHLCV

console = Console()


def _safe_float(val, default: float = 0.0) -> float:
    """Parse float from API — handles empty strings, None, junk."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class BinanceCollector:
    def __init__(self):
        self.futures_base = settings.binance_futures_base
        self.spot_base = settings.binance_spot_base
        self.delay = settings.request_delay_ms / 1000.0

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict) -> list:
        for attempt in range(3):
            try:
                resp = await client.get(url, params=params, timeout=30.0)
                if resp.status_code == 429:
                    await asyncio.sleep(int(resp.headers.get("Retry-After", 5)))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.ReadTimeout):
                if attempt == 2:
                    return []
                await asyncio.sleep(2 ** attempt)
        return []

    async def collect_funding_rates(self, symbol: str, start: datetime, end: datetime) -> list[FundingRate]:
        results: list[FundingRate] = []
        cursor, end_ms = _ms(start), _ms(end)
        async with httpx.AsyncClient() as client:
            with tqdm(desc=f"Funding {symbol}", unit="batch") as pbar:
                while cursor < end_ms:
                    data = await self._get(client, f"{self.futures_base}/fapi/v1/fundingRate",
                        {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
                    if not data:
                        break
                    for row in data:
                        results.append(FundingRate(
                            symbol=symbol,
                            funding_rate=_safe_float(row.get("fundingRate"), 0.0),
                            funding_time=_from_ms(row["fundingTime"]),
                            mark_price=_safe_float(row.get("markPrice"), 0.0),
                        ))
                    cursor = data[-1]["fundingTime"] + 1
                    pbar.update(1)
                    pbar.set_postfix(total=len(results))
                    await asyncio.sleep(self.delay)
        console.print(f"  [green]✓ {len(results)} funding rates for {symbol}[/green]")
        return results

    async def collect_klines(self, symbol: str, start: datetime, end: datetime,
                             source: str = "futures") -> list[OHLCV]:
        base = self.futures_base if source == "futures" else self.spot_base
        endpoint = "/fapi/v1/klines" if source == "futures" else "/api/v3/klines"
        results: list[OHLCV] = []
        cursor, end_ms = _ms(start), _ms(end)
        async with httpx.AsyncClient() as client:
            with tqdm(desc=f"Klines {symbol} ({source})", unit="batch") as pbar:
                while cursor < end_ms:
                    data = await self._get(client, f"{base}{endpoint}",
                        {"symbol": symbol, "interval": settings.kline_interval,
                         "startTime": cursor, "endTime": end_ms, "limit": 1500})
                    if not data:
                        break
                    for row in data:
                        results.append(OHLCV(
                            symbol=symbol, open_time=_from_ms(row[0]),
                            open=_safe_float(row[1]), high=_safe_float(row[2]),
                            low=_safe_float(row[3]), close=_safe_float(row[4]),
                            volume=_safe_float(row[5]), source=source,
                        ))
                    cursor = data[-1][0] + 1
                    pbar.update(1)
                    pbar.set_postfix(total=len(results))
                    await asyncio.sleep(self.delay)
        console.print(f"  [green]✓ {len(results)} {source} klines for {symbol}[/green]")
        return results

    async def collect_all(self, symbol: str, start: datetime, end: datetime) -> dict:
        console.print(f"\n[bold cyan]Collecting {symbol}[/bold cyan]  {start.date()} → {end.date()}")
        return {
            "funding_rates": await self.collect_funding_rates(symbol, start, end),
            "futures_klines": await self.collect_klines(symbol, start, end, "futures"),
            "spot_klines": await self.collect_klines(symbol, start, end, "spot"),
        }
