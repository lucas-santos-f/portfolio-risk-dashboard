"""Download and store daily prices.

The snapshot in ``data/daily_prices.csv`` is a wide table: one row per
trading day, one column per ticker, holding the close adjusted for splits
and dividends. Days before a ticker started trading are empty.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "daily_prices.csv"
YEARS_OF_HISTORY = 15

TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "VTI": "US total stock market",
    "VEA": "Developed markets ex-US",
    "VWO": "Emerging markets",
    "AGG": "US aggregate bonds",
    "TLT": "US Treasuries 20+ years",
    "GLD": "Gold",
    "VNQ": "US real estate (REITs)",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
}

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"


def history_start(today: date, years: int = YEARS_OF_HISTORY) -> date:
    return today.replace(year=today.year - years, day=1)


def download_yahoo(tickers: list[str], start: date, today: date) -> pd.DataFrame:
    """Daily closes adjusted for splits and dividends, excluding ``today``.

    Today's row would be an intraday price while the market is open.
    """
    frame = yf.download(tickers, start=start.isoformat(), auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no data for {', '.join(tickers)}")
    close: pd.DataFrame = frame["Close"]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.dropna(axis=1, how="all")
    return close[close.index < pd.Timestamp(today)]


def parse_stooq_csv(text: str, ticker: str) -> pd.Series:
    """Parse Stooq's daily CSV (Date,Open,High,Low,Close,Volume).

    Stooq adjusts for splits but not for dividends: returns built from it
    understate total return by roughly the dividend yield.
    """
    if not text.startswith("Date,"):
        raise ValueError(f"Stooq did not return a CSV for {ticker}")
    frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    return frame["Close"].rename(ticker)


def stooq_symbol(ticker: str) -> str:
    return ticker.lower().replace("-", ".")


def download_stooq(ticker: str, session: requests.Session) -> pd.Series:
    response = session.get(STOOQ_URL.format(symbol=stooq_symbol(ticker)), timeout=30)
    response.raise_for_status()
    return parse_stooq_csv(response.text, ticker)


def save_prices(prices: pd.DataFrame, path: Path = PRICES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date", float_format="%.8g", date_format="%Y-%m-%d")


def load_prices(path: Path = PRICES_FILE) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
