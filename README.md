# portfolio-risk-dashboard

Streamlit app that shows how risky a mix of ETFs/stocks actually is: drawdowns, volatility, beta, VaR, and which holding the risk really comes from. 15 years of daily prices, compared against SPY.

![screenshot](docs/screenshot.png)

Example of what it shows: a 60/40 VTI/AGG portfolio puts ~95% of its risk in the stocks. Bonds are 40% of the money and under 5% of the swings. Put 5% each in NVDA and TSLA and those two end up as ~27% of the risk.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

No API keys. A price snapshot ships in `data/daily_prices.csv` so it works offline.

Refresh the snapshot with `python scripts/update_prices.py` (or pass tickers). Yahoo Finance first, Stooq as fallback.

## Gotchas

- The portfolio is rebalanced back to target **every day**. Monthly/yearly rebalancing would give slightly different numbers.
- Stooq prices are split-adjusted but not dividend-adjusted, so a ticker that fell back to Stooq understates its total return by roughly its yield. The script prints a warning when that happens.
- Stooq sometimes serves a captcha page instead of the CSV. Download it in a browser and run `python scripts/update_prices.py --file SPY spy_us_d.csv`.
- The analysis window is the overlap of the selected tickers (META starts in 2012).

Tests: `pip install -r requirements-dev.txt && pytest`

Not investment advice.
