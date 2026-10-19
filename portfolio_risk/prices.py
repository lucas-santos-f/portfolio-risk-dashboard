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
    """Daily closes adjusted for splits and dividends, one column per ticker."""
    frame = yf.download(tickers, start=start.isoformat(), auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no data for {', '.join(tickers)}")

    close = frame["Close"]
    # yfinance has gone back and forth between versions on returning a
    # tz-aware index; joining that with the naive snapshot index raises
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.dropna(axis=1, how="all")
    # today's row is an intraday price while the market is open, skip it
    return close[close.index < pd.Timestamp(today)]


def parse_stooq_csv(text: str, ticker: str) -> pd.Series:
    # Stooq answers 200 with an html "verify you are human" page when it
    # doesn't like the client, so the status code alone means nothing
    if not text.startswith("Date,"):
        raise ValueError(f"Stooq did not return a CSV for {ticker}")
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    # NB: split-adjusted only, no dividends -> understates total return
    return df["Close"].rename(ticker)


def stooq_symbol(ticker):
    return ticker.lower().replace("-", ".")  # BRK-B -> brk.b


def download_stooq(ticker: str, session: requests.Session) -> pd.Series:
    resp = session.get(STOOQ_URL.format(symbol=stooq_symbol(ticker)), timeout=30)
    resp.raise_for_status()
    return parse_stooq_csv(resp.text, ticker)


def save_prices(prices: pd.DataFrame, path: Path = PRICES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # %.8g and not %.4f: NVDA's split-adjusted 2011 closes are ~0.30, four
    # decimals throws away real precision there
    prices.to_csv(path, index_label="date", float_format="%.8g", date_format="%Y-%m-%d")


def load_prices(path=PRICES_FILE):
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
