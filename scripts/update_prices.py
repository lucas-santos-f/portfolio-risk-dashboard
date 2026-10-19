"""Refresh data/daily_prices.csv (last 15 years).

    python scripts/update_prices.py              # every ticker
    python scripts/update_prices.py NVDA TSLA    # just these
    python scripts/update_prices.py --file SPY spy_us_d.csv   # CSV saved from stooq.com by hand
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_risk.prices import (  # noqa: E402
    PRICES_FILE,
    TICKERS,
    download_stooq,
    download_yahoo,
    history_start,
    load_prices,
    parse_stooq_csv,
    save_prices,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file", nargs=2, metavar=("TICKER", "PATH"))
    args = ap.parse_args()

    today = date.today()
    start = history_start(today)
    prices = load_prices() if PRICES_FILE.exists() else pd.DataFrame()

    if args.file:
        ticker = args.file[0].upper()
        wanted = [ticker]
        updates = parse_stooq_csv(Path(args.file[1]).read_text(encoding="utf-8"), ticker).to_frame()
        print(updates.tail(3))
    else:
        wanted = [t.upper() for t in args.tickers] or list(TICKERS)
        try:
            updates = download_yahoo(wanted, start, today)
        except (ValueError, KeyError, requests.RequestException) as exc:
            print(f"Yahoo Finance failed ({exc})")
            updates = pd.DataFrame()

        # whatever Yahoo didn't return gets one shot on Stooq
        sess = requests.Session()
        for ticker in [t for t in wanted if t not in updates.columns]:
            try:
                daily = download_stooq(ticker, sess)
            except (ValueError, requests.RequestException) as exc:
                print(f"{ticker}: Stooq failed too ({exc})")
                continue
            print(f"{ticker}: using Stooq prices, NOT adjusted for dividends")
            updates = updates.join(daily, how="outer")

    updates = updates[updates.index >= pd.Timestamp(start)]
    for t in updates.columns:
        s = updates[t].dropna()
        print(f"{t}: {len(s)} days, {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d}")

    prices = prices.drop(columns=updates.columns, errors="ignore").join(updates, how="outer")
    prices = prices[prices.index >= pd.Timestamp(start)].dropna(how="all")

    # keep the TICKERS order in the csv, anything extra goes at the end
    cols = [t for t in TICKERS if t in prices.columns] + [t for t in prices.columns if t not in TICKERS]
    save_prices(prices[cols].sort_index())

    failed = [t for t in wanted if t not in updates.columns]
    if failed:
        print(f"\nNot updated: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
