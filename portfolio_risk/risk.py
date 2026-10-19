# Everything here assumes the portfolio is rebalanced back to target weights
# every day, so the portfolio return is just the weighted sum of asset returns.
# Simple returns, not log. 0.01 == 1%.

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VAR_LEVEL = 0.95


def date_of_min(series):
    return pd.Timestamp(series.index[int(series.to_numpy().argmin())])


def normalize_weights(weights: dict[str, float]) -> pd.Series:
    # the sidebar takes dollars or % interchangeably, so just rescale
    w = pd.Series(weights, dtype=float)
    w = w[w > 0]
    if w.empty:
        raise ValueError("At least one weight must be positive")
    return w / w.sum()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # dropna first: META only starts in 2012, the window shrinks to fit
    return prices.dropna().pct_change().dropna()


def portfolio_returns(returns, weights):
    return (returns[weights.index] * weights).sum(axis=1).rename("portfolio")


def growth(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    years = len(returns) / TRADING_DAYS
    return float(growth(returns).iloc[-1] ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    # textbook version (mean of daily excess returns), so it won't match
    # (CAGR - rf) / vol exactly. People ask about this, it's expected.
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    return float((returns - daily_rf).mean() / returns.std() * np.sqrt(TRADING_DAYS))


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = growth(returns)
    # clip at 1 so a fall on day one is measured from the money you put in,
    # not from the (already lower) first close
    return wealth / wealth.cummax().clip(lower=1) - 1


@dataclass
class MaxDrawdown:
    depth: float  # negative
    peak: pd.Timestamp
    bottom: pd.Timestamp
    recovered: pd.Timestamp | None  # None = still underwater

    @property
    def days_to_recover(self):
        if self.recovered is None:
            return None
        return (self.recovered - self.peak).days


def max_drawdown(returns: pd.Series) -> MaxDrawdown:
    """Worst peak-to-bottom fall, with the dates and the recovery (if any)."""
    dd = drawdown(returns)
    bottom = date_of_min(dd)

    before = dd.loc[:bottom]
    highs = before[before == 0]
    peak = pd.Timestamp(highs.index[-1] if len(highs) else dd.index[0])

    after = dd.loc[bottom:]
    back = after[after == 0]
    recovered = pd.Timestamp(back.index[0]) if len(back) else None

    return MaxDrawdown(float(dd.loc[bottom]), peak, bottom, recovered)


def beta(returns: pd.Series, market: pd.Series) -> float:
    both = pd.concat([returns, market], axis=1, join="inner").to_numpy()
    cov = np.cov(both, rowvar=False)
    return float(cov[0, 1] / cov[1, 1])


def rolling_volatility(returns, window=TRADING_DAYS):
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol.dropna()


def risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Each asset's share of portfolio variance: w_i * (Cov @ w)_i / (w' Cov w). Sums to 1."""
    cov = returns[weights.index].cov() * TRADING_DAYS
    w = weights.to_numpy()
    marginal = cov.to_numpy() @ w
    variance = float(w @ marginal)

    # a negative share is legit (e.g. TLT next to stocks pre-2022), it means
    # the asset is pulling total risk down
    out = pd.DataFrame(index=weights.index)
    out["weight"] = weights
    out["risk_share"] = w * marginal / variance
    out["volatility"] = np.sqrt(np.diag(cov.to_numpy()))
    return out


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def historical_var(returns: pd.Series, level: float = VAR_LEVEL) -> float:
    # returned as a positive number, 0.02 == "lost more than 2% on 5% of days"
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
    """All the headline numbers the dashboard shows, for one return series."""
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
