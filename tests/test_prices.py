from datetime import date

import pandas as pd
import pytest

from portfolio_risk.prices import (
    download_yahoo,
    history_start,
    load_prices,
    parse_stooq_csv,
    save_prices,
    stooq_symbol,
)

STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-30,100,101,99,100.5,1000
2024-01-31,100,102,99,101.0,1000
"""


def test_history_starts_on_the_first_of_the_month_years_ago():
    assert history_start(date(2026, 9, 26), years=15) == date(2011, 9, 1)


def test_parse_stooq_csv_reads_the_close_column():
    close = parse_stooq_csv(STOOQ_CSV, "SPY")
    assert close.name == "SPY"
    assert close.to_list() == [100.5, 101.0]


def test_parse_stooq_rejects_a_non_csv_answer():
    with pytest.raises(ValueError, match="did not return a CSV"):
        parse_stooq_csv("<!DOCTYPE html><html>verify you are human</html>", "SPY")


def test_share_classes_use_stooq_dot_notation():
    assert stooq_symbol("BRK-B") == "brk.b"


def test_yahoo_download_keeps_close_and_drops_today(monkeypatch):
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    frame = pd.concat(
        {"Close": pd.DataFrame({"SPY": [1.0, 2.0, 3.0], "XXX": [None] * 3}, index=index)}, axis=1
    )
    monkeypatch.setattr("portfolio_risk.prices.yf.download", lambda *args, **kwargs: frame)
    close = download_yahoo(["SPY", "XXX"], date(2024, 1, 1), today=date(2024, 2, 1))
    assert list(close.columns) == ["SPY"]
    assert close.index[-1] == pd.Timestamp("2024-01-31")


def test_yahoo_download_with_no_rows_is_an_error(monkeypatch):
    monkeypatch.setattr("portfolio_risk.prices.yf.download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="no data"):
        download_yahoo(["SPY"], date(2024, 1, 1), today=date(2024, 2, 1))


def test_saved_prices_load_back_unchanged(tmp_path):
    prices = pd.DataFrame(
        {"SPY": [1.5, 2.25], "NVDA": [0.30367845, 3.0]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31"]),
    )
    path = tmp_path / "prices.csv"
    save_prices(prices, path)
    pd.testing.assert_frame_equal(load_prices(path), prices, check_names=False, check_freq=False)
