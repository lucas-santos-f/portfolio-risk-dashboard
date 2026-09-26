"""Refresh data/daily_prices.csv with the last 15 years of daily prices.

Usage:
    python scripts/update_prices.py              # every ticker
    python scripts/update_prices.py NVDA TSLA    # some tickers
    python scripts/update_prices.py --file SPY spy_us_d.csv
        # import a daily CSV saved from stooq.com in a browser

Prices come from Yahoo Finance, adjusted for dividends. A ticker Yahoo
does not return is tried on Stooq, whose prices are NOT adjusted for
dividends (the script says so). A ticker that fails on both keeps its
previous snapshot.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_risk.prices import (
    PRICES_FILE,
    TICKERS,
    download_stooq,
    download_yahoo,
    history_start,
    load_prices,
    parse_stooq_csv,
    save_prices,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tickers", nargs="*", help="tickers to refresh (default: all)")
    parser.add_argument("--file", nargs=2, metavar=("TICKER", "PATH"), help="import a Stooq daily CSV")
    args = parser.parse_args()

    today = date.today()
    start = history_start(today)
    prices = load_prices() if PRICES_FILE.exists() else pd.DataFrame()

    if args.file:
        ticker = args.file[0].upper()
        wanted = [ticker]
        updates = parse_stooq_csv(Path(args.file[1]).read_text(encoding="utf-8"), ticker).to_frame()
    else:
        wanted = [t.upper() for t in args.tickers] or list(TICKERS)
        try:
            updates = download_yahoo(wanted, start, today)
        except (ValueError, KeyError, requests.RequestException) as exc:
            print(f"Yahoo Finance failed ({exc})")
            updates = pd.DataFrame()
        session = requests.Session()
        for ticker in [t for t in wanted if t not in updates.columns]:
            try:
                daily = download_stooq(ticker, session)
            except (ValueError, requests.RequestException) as exc:
                print(f"{ticker}: Stooq failed too ({exc})")
                continue
            print(f"{ticker}: using Stooq prices, NOT adjusted for dividends")
            updates = updates.join(daily, how="outer")

    updates = updates[updates.index >= pd.Timestamp(start)]
    for ticker in updates.columns:
        series = updates[ticker].dropna()
        print(f"{ticker}: {len(series)} days, {series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d}")
    prices = prices.drop(columns=updates.columns, errors="ignore").join(updates, how="outer")
    prices = prices[prices.index >= pd.Timestamp(start)].dropna(how="all")

    ordered = [t for t in TICKERS if t in prices.columns] + [t for t in prices.columns if t not in TICKERS]
    save_prices(prices[ordered].sort_index())

    failed = [t for t in wanted if t not in updates.columns]
    if failed:
        print(f"\nNot updated: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
