import numpy as np
import pandas as pd
import pytest

from portfolio_risk.risk import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
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
    rolling_volatility,
    sharpe_ratio,
)


def days(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# Up 10%, down 50%, back up 100%: peak after day 1, bottom after day 2, recovered on day 3.
ROUND_TRIP = days([0.10, -0.50, 1.00, 0.05])


def test_weights_are_rescaled_to_sum_to_one_and_zeros_dropped():
    weights = normalize_weights({"VTI": 60_000, "AGG": 40_000, "GLD": 0})
    assert weights.to_dict() == {"VTI": pytest.approx(0.6), "AGG": pytest.approx(0.4)}


def test_weights_need_something_positive():
    with pytest.raises(ValueError, match="positive"):
        normalize_weights({"VTI": 0})


def test_daily_returns_use_only_days_where_every_asset_has_a_price():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [np.nan, 50.0, 55.0, 55.0]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    returns = daily_returns(prices)
    assert len(returns) == 2
    assert returns["A"].to_list() == pytest.approx([0.10, 0.10])
    assert returns["B"].to_list() == pytest.approx([0.10, 0.0])


def test_portfolio_return_is_the_weighted_sum_of_asset_returns():
    returns = pd.DataFrame({"A": [0.10, -0.02], "B": [0.00, 0.04]})
    result = portfolio_returns(returns, pd.Series({"A": 0.25, "B": 0.75}))
    assert result.to_list() == pytest.approx([0.025, 0.025])


def test_drawdown_measures_the_fall_from_the_highest_value_so_far():
    assert drawdown(ROUND_TRIP).to_list() == pytest.approx([0.0, -0.5, 0.0, 0.0])


def test_a_fall_on_the_first_day_counts_from_the_starting_value():
    assert drawdown(days([-0.2, 0.1])).to_list() == pytest.approx([-0.2, -0.12])


def test_max_drawdown_reports_peak_bottom_and_recovery():
    worst = max_drawdown(ROUND_TRIP)
    dates = ROUND_TRIP.index
    assert worst.depth == pytest.approx(-0.5)
    assert (worst.peak, worst.bottom, worst.recovered) == (dates[0], dates[1], dates[2])
    assert worst.days_to_recover == 2


def test_a_drawdown_not_yet_recovered_has_no_recovery_date():
    worst = max_drawdown(days([-0.1, -0.1, 0.05]))
    assert worst.peak == worst.bottom - pd.offsets.BDay(1)
    assert worst.recovered is None
    assert worst.days_to_recover is None


def test_annualized_return_compounds_daily_growth_over_a_year():
    daily = (1.21) ** (1 / TRADING_DAYS) - 1
    assert annualized_return(days([daily] * TRADING_DAYS * 2)) == pytest.approx(0.21)


def test_volatility_scales_with_the_square_root_of_time():
    returns = days([0.01, -0.01] * 50)
    assert annualized_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_sharpe_ratio_subtracts_the_risk_free_rate():
    returns = days([0.002, 0.0] * 100)
    no_rate = sharpe_ratio(returns)
    assert no_rate == pytest.approx(0.001 / returns.std() * np.sqrt(252))
    assert sharpe_ratio(returns, risk_free_rate=0.05) < no_rate


def test_beta_of_a_leveraged_copy_of_the_market_is_the_leverage():
    market = days([0.01, -0.02, 0.015, 0.003, -0.007])
    assert beta(2 * market, market) == pytest.approx(2.0)
    assert beta(market, market) == pytest.approx(1.0)


def test_risk_shares_add_up_to_one():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.005], size=(500, 3)), columns=["A", "B", "C"])
    shares = risk_contribution(returns, pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}))
    assert shares["risk_share"].sum() == pytest.approx(1.0)
    assert shares.loc["B", "risk_share"] > shares.loc["B", "weight"]


def test_uncorrelated_equal_volatility_assets_split_risk_by_squared_weight():
    returns = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01], "B": [0.01, 0.01, -0.01, -0.01]})
    shares = risk_contribution(returns, pd.Series({"A": 0.75, "B": 0.25}))
    # w_i^2 / sum(w^2) = 0.5625 / 0.625
    assert shares.loc["A", "risk_share"] == pytest.approx(0.9)


def test_rolling_volatility_starts_once_the_window_is_full():
    returns = days([0.01, -0.01] * 10)
    assert len(rolling_volatility(returns, window=5)) == 16


def test_monthly_returns_compound_the_daily_ones():
    returns = pd.Series([0.10, 0.10, -0.5], index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]))
    assert monthly_returns(returns).to_list() == pytest.approx([0.21, -0.5])


def test_var_95_is_the_loss_beaten_on_only_5_percent_of_days():
    returns = days([-0.05] + [-0.02] * 9 + [0.01] * 90)
    assert historical_var(returns) == pytest.approx(0.02)


def test_risk_report_collects_the_headline_numbers():
    market = days([0.01, -0.02, 0.015, 0.003, -0.007] * 60)
    report = risk_report(market * 0.5, market)
    assert report.beta == pytest.approx(0.5)
    assert report.worst_day == pytest.approx(-0.01)
    assert report.worst_day_date == market.index[1]
    assert report.max_drawdown.depth < 0
    assert report.start == market.index[0]
