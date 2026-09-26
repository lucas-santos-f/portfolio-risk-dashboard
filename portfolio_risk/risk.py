"""Risk measures for a portfolio with fixed weights.

Assumption used throughout: the portfolio is rebalanced back to its target
weights every day, so its daily return is the weighted sum of the assets'
daily returns. Returns are simple (not log) returns, as fractions
(0.01 = 1%), and a year has 252 trading days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VAR_LEVEL = 0.95


def date_of_min(series: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(series.index[int(series.to_numpy().argmin())])


def normalize_weights(weights: dict[str, float]) -> pd.Series:
    """Rescale amounts in any unit (dollars, %) to weights summing to 1, dropping zeros."""
    series = pd.Series(weights, dtype=float)
    series = series[series > 0]
    if series.empty:
        raise ValueError("At least one weight must be positive")
    return series / series.sum()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns over the period where every column has a price."""
    return prices.dropna().pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return (returns[weights.index] * weights).sum(axis=1).rename("portfolio")


def growth(returns: pd.Series) -> pd.Series:
    """Value of 1 invested at the start, after each day."""
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    years = len(returns) / TRADING_DAYS
    return float(growth(returns).iloc[-1] ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Average yearly return above the risk-free rate per unit of volatility.

    Uses the arithmetic mean of daily excess returns, the textbook
    definition, so it is not exactly (CAGR - rate) / volatility.
    """
    daily_rate = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rate
    return float(excess.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def drawdown(returns: pd.Series) -> pd.Series:
    """Loss from the highest value reached so far, at each day (0 at a new high)."""
    wealth = growth(returns)
    return wealth / wealth.cummax().clip(lower=1) - 1


@dataclass
class MaxDrawdown:
    depth: float  # worst peak-to-bottom loss, negative fraction
    peak: pd.Timestamp  # last high before the fall
    bottom: pd.Timestamp
    recovered: pd.Timestamp | None  # first day back at the old high, None if not yet

    @property
    def days_to_recover(self) -> int | None:
        return None if self.recovered is None else (self.recovered - self.peak).days


def max_drawdown(returns: pd.Series) -> MaxDrawdown:
    dd = drawdown(returns)
    bottom = date_of_min(dd)
    before = dd.loc[:bottom]
    at_high = before[before == 0]
    # Falling from day one means the peak is the starting value.
    peak = pd.Timestamp(at_high.index[-1] if len(at_high) else dd.index[0])
    after = dd.loc[bottom:]
    back = after[after == 0]
    return MaxDrawdown(
        depth=float(dd.loc[bottom]),
        peak=peak,
        bottom=bottom,
        recovered=pd.Timestamp(back.index[0]) if len(back) else None,
    )


def beta(returns: pd.Series, market: pd.Series) -> float:
    """Sensitivity to the market: 1.5 means it moved 1.5% for every 1% of the market."""
    aligned = pd.concat([returns, market], axis=1, join="inner").to_numpy()
    cov = np.cov(aligned, rowvar=False)
    return float(cov[0, 1] / cov[1, 1])


def rolling_volatility(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Annualized volatility over the trailing ``window`` days."""
    volatility: pd.Series = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return volatility.dropna()


def risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Share of the portfolio's volatility that comes from each asset.

    contribution_i = w_i * (Cov @ w)_i / portfolio_variance. The shares add
    up to 1. An asset can contribute more (or less) risk than its weight,
    and a negative share means it reduces the portfolio's risk.
    """
    cov = returns[weights.index].cov() * TRADING_DAYS
    w = weights.to_numpy()
    marginal = cov.to_numpy() @ w
    variance = float(w @ marginal)
    share = w * marginal / variance
    return pd.DataFrame(
        {
            "weight": weights,
            "risk_share": share,
            "volatility": np.sqrt(np.diag(cov.to_numpy())),
        },
        index=weights.index,
    )


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def historical_var(returns: pd.Series, level: float = VAR_LEVEL) -> float:
    """Daily loss exceeded on only (1 - level) of days, as a positive fraction."""
    return float(-returns.quantile(1 - level))


@dataclass
class RiskReport:
    start: pd.Timestamp
    end: pd.Timestamp
    annual_return: float
    annual_volatility: float
    sharpe: float
    beta: float
    max_drawdown: MaxDrawdown
    worst_day: float
    worst_day_date: pd.Timestamp
    worst_month: float
    worst_month_date: pd.Timestamp
    var_95: float


def risk_report(returns: pd.Series, market: pd.Series, risk_free_rate: float = 0.0) -> RiskReport:
    months = monthly_returns(returns)
    worst_day = date_of_min(returns)
    worst_month = date_of_min(months)
    return RiskReport(
        start=returns.index[0],
        end=returns.index[-1],
        annual_return=annualized_return(returns),
        annual_volatility=annualized_volatility(returns),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        beta=beta(returns, market),
        max_drawdown=max_drawdown(returns),
        worst_day=float(returns.loc[worst_day]),
        worst_day_date=worst_day,
        worst_month=float(months.loc[worst_month]),
        worst_month_date=worst_month,
        var_95=historical_var(returns),
    )
