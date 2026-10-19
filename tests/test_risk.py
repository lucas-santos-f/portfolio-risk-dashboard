import numpy as np
import pandas as pd
import pytest

from portfolio_risk.risk import (
    TRADING_DAYS,
    annualized_return,
    beta,
    daily_returns,
    drawdown,
    historical_var,
    max_drawdown,
    monthly_returns,
    normalize_weights,
    portfolio_returns,
    risk_contribution,
    risk_report,
    sharpe_ratio,
)


def days(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# +10%, -50%, +100%: peak after day 1, bottom on day 2, back at the high on day 3
ROUND_TRIP = days([0.10, -0.50, 1.00, 0.05])


def test_weights_rescale():
    w = normalize_weights({"VTI": 60_000, "AGG": 40_000, "GLD": 0})
    assert w.to_dict() == {"VTI": pytest.approx(0.6), "AGG": pytest.approx(0.4)}

    with pytest.raises(ValueError):
        normalize_weights({"VTI": 0})


def test_returns_only_where_all_assets_trade():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [np.nan, 50.0, 55.0, 55.0]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    r = daily_returns(prices)
    assert len(r) == 2
    assert r["B"].to_list() == pytest.approx([0.10, 0.0])


def test_portfolio_return():
    r = pd.DataFrame({"A": [0.10, -0.02], "B": [0.00, 0.04]})
    out = portfolio_returns(r, pd.Series({"A": 0.25, "B": 0.75}))
    assert out.to_list() == pytest.approx([0.025, 0.025])


def test_drawdown():
    assert drawdown(ROUND_TRIP).to_list() == pytest.approx([0.0, -0.5, 0.0, 0.0])
    # falling on day one counts from the starting value, not the first close
    assert drawdown(days([-0.2, 0.1])).to_list() == pytest.approx([-0.2, -0.12])


def test_max_drawdown_dates():
    worst = max_drawdown(ROUND_TRIP)
    d = ROUND_TRIP.index
    assert worst.depth == pytest.approx(-0.5)
    assert (worst.peak, worst.bottom, worst.recovered) == (d[0], d[1], d[2])
    assert worst.days_to_recover == 2


def test_max_drawdown_not_recovered():
    worst = max_drawdown(days([-0.1, -0.1, 0.05]))
    assert worst.recovered is None
    assert worst.days_to_recover is None


def test_cagr():
    daily = 1.21 ** (1 / TRADING_DAYS) - 1
    assert annualized_return(days([daily] * TRADING_DAYS * 2)) == pytest.approx(0.21)


def test_sharpe_uses_rf():
    r = days([0.002, 0.0] * 100)
    assert sharpe_ratio(r) == pytest.approx(0.001 / r.std() * np.sqrt(252))
    assert sharpe_ratio(r, risk_free_rate=0.05) < sharpe_ratio(r)


def test_beta_of_2x_market():
    mkt = days([0.01, -0.02, 0.015, 0.003, -0.007])
    assert beta(2 * mkt, mkt) == pytest.approx(2.0)


def test_risk_shares_sum_to_one():
    rng = np.random.default_rng(7)
    r = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.005], size=(500, 3)), columns=["A", "B", "C"])
    shares = risk_contribution(r, pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}))
    assert shares["risk_share"].sum() == pytest.approx(1.0)
    # B is the volatile one, it should punch above its weight
    assert shares.loc["B", "risk_share"] > shares.loc["B", "weight"]


def test_uncorrelated_split_by_squared_weight():
    r = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01], "B": [0.01, 0.01, -0.01, -0.01]})
    shares = risk_contribution(r, pd.Series({"A": 0.75, "B": 0.25}))
    assert shares.loc["A", "risk_share"] == pytest.approx(0.5625 / 0.625)


def test_monthly_compounding():
    r = pd.Series([0.10, 0.10, -0.5], index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]))
    assert monthly_returns(r).to_list() == pytest.approx([0.21, -0.5])


def test_var_95():
    r = days([-0.05] + [-0.02] * 9 + [0.01] * 90)
    assert historical_var(r) == pytest.approx(0.02)


def test_report():
    mkt = days([0.01, -0.02, 0.015, 0.003, -0.007] * 60)
    rep = risk_report(mkt * 0.5, mkt)
    assert rep.beta == pytest.approx(0.5)
    assert rep.worst_day == pytest.approx(-0.01)
    assert rep.worst_day_date == mkt.index[1]
