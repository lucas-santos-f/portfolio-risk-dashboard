import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio_risk.prices import TICKERS, load_prices
from portfolio_risk.risk import (
    RiskReport,
    daily_returns,
    drawdown,
    growth,
    normalize_weights,
    portfolio_returns,
    risk_contribution,
    risk_report,
    rolling_volatility,
)

# palette is the same as .streamlit/config.toml, keep them in sync by hand
PORTFOLIO, MARKET, WEIGHT = "#2a78d6", "#898781", "#cdccc5"
DIVERGING = [[0.0, "#256abf"], [0.5, "#f0efec"], [1.0, "#c4481d"]]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CHART_CONFIG = {"displayModeBar": False}
BENCHMARK = "SPY"
DEFAULT_WEIGHTS = {"VTI": 60.0, "AGG": 40.0}
NEW_ASSET_WEIGHT = 10.0
START_VALUE = 10_000
DAYS_PER_MONTH = 30.44  # 365.25 / 12

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")


@st.cache_data
def get_prices() -> pd.DataFrame:
    return load_prices()


def base_layout(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        legend=dict(orientation="h", y=1.1, x=0, title=None),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def versus_market(portfolio: pd.Series, market: pd.Series, fmt: str, fill: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=market.index,
        y=market,
        name=f"{BENCHMARK} (S&P 500)",
        line=dict(color=MARKET, width=1.5),
        hovertemplate=f"%{{y:{fmt}}}",
    )
    fig.add_scatter(
        x=portfolio.index,
        y=portfolio,
        name="Your portfolio",
        line=dict(color=PORTFOLIO, width=2),
        fill="tozeroy" if fill else None,
        fillcolor="rgba(42, 120, 214, 0.15)",
        hovertemplate=f"%{{y:{fmt}}}",
    )
    return base_layout(fig, 320)


def contribution_chart(shares: pd.DataFrame) -> go.Figure:
    data = shares.sort_values("risk_share")
    fig = go.Figure()
    fig.add_bar(
        y=data.index,
        x=data["weight"] * 100,
        name="Share of money",
        orientation="h",
        marker=dict(color=WEIGHT),
        text=[f"{v:.0f}%" for v in data["weight"] * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% of the money<extra></extra>",
    )
    fig.add_bar(
        y=data.index,
        x=data["risk_share"] * 100,
        name="Share of risk",
        orientation="h",
        marker=dict(color=PORTFOLIO),
        text=[f"{v:.0f}%" for v in data["risk_share"] * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% of the risk<extra></extra>",
    )
    # leave room for the "27%" labels outside the bars, and for negative shares
    top = max(100.0, float(data[["weight", "risk_share"]].max().max() * 100)) * 1.12
    low = min(0.0, float(data["risk_share"].min() * 100)) * 1.2
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(ticksuffix="%", range=[low, top])
    fig.update_yaxes(color=INK_2, gridcolor=SURFACE)
    base_layout(fig, 60 * len(data) + 90)
    fig.update_layout(hovermode="closest", legend=dict(traceorder="reversed"))
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=13),
            xgap=2,
            ygap=2,
            hovertemplate="%{y} vs %{x}: <b>%{z:.2f}</b><extra></extra>",
            colorbar=dict(thickness=10, outlinewidth=0),
        )
    )
    fig.update_yaxes(autorange="reversed", color=INK_2, showgrid=False)
    fig.update_xaxes(side="top", color=INK_2, showgrid=False)
    base_layout(fig, max(260, 44 * len(corr) + 80))
    fig.update_layout(hovermode="closest")
    return fig


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def gap(value):
    # difference of two percentages -> percentage points, not %
    return f"{value * 100:+.1f} pts vs S&P 500"


def comparison_table(mine: RiskReport, market: RiskReport) -> pd.DataFrame:
    rows = {
        "Growth per year": (pct(mine.annual_return), pct(market.annual_return)),
        "Volatility": (pct(mine.annual_volatility), pct(market.annual_volatility)),
        "Max drawdown": (pct(mine.max_drawdown.depth), pct(market.max_drawdown.depth)),
        "Sharpe ratio": (f"{mine.sharpe:.2f}", f"{market.sharpe:.2f}"),
        "Worst day": (pct(mine.worst_day), pct(market.worst_day)),
        "Worst month": (pct(mine.worst_month), pct(market.worst_month)),
        "1-in-20 daily loss (VaR 95%)": (pct(-mine.var_95, 2), pct(-market.var_95, 2)),
    }
    return pd.DataFrame(rows, index=["Portfolio", BENCHMARK]).T


prices = get_prices()

st.sidebar.header("Your portfolio")
selected = st.sidebar.multiselect(
    "Assets",
    options=list(TICKERS),
    default=list(DEFAULT_WEIGHTS),
    format_func=lambda t: f"{t} · {TICKERS[t]}",
)
st.sidebar.caption("How much is in each one? Dollars or percentages, it's rescaled automatically.")
amounts = {
    t: st.sidebar.number_input(
        t,
        min_value=0.0,
        value=DEFAULT_WEIGHTS.get(t, NEW_ASSET_WEIGHT),
        step=5.0,
        format="%.0f",
        key=f"w_{t}",
    )
    for t in selected
}
risk_free = st.sidebar.number_input(
    "Risk-free rate (% per year)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.25,
    format="%.2f",
    help="What cash or short-term Treasury bills pay. Used only for the Sharpe ratio.",
)
st.sidebar.divider()
st.sidebar.caption(
    f"Daily prices from Yahoo Finance, adjusted for dividends, through {prices.index[-1]:%b %d, %Y}. "
    "Not investment advice."
)

st.title("Portfolio Risk Dashboard")
st.markdown(
    "How much could this portfolio fall, how bumpy is the ride, and where the risk really "
    "comes from. Computed from daily prices, compared with the S&P 500."
)

if sum(amounts.values()) <= 0:
    st.info("Pick at least one asset and put some amount in it.")
    st.stop()

weights = normalize_weights(amounts)
# dict.fromkeys dedupes while keeping order, SPY can be picked as an asset too
returns = daily_returns(prices[list(dict.fromkeys([*weights.index, BENCHMARK]))])
mine = portfolio_returns(returns, weights)
market = returns[BENCHMARK]
report = risk_report(mine, market, risk_free / 100)
market_report = risk_report(market, market, risk_free / 100)
worst = report.max_drawdown

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Growth per year",
    pct(report.annual_return),
    gap(report.annual_return - market_report.annual_return),
    delta_color="off",
    delta_arrow="off",
    help="Compound annual growth rate: the steady yearly return that ends at the same value.",
)
k2.metric(
    "Volatility",
    pct(report.annual_volatility),
    gap(report.annual_volatility - market_report.annual_volatility),
    delta_color="off",
    delta_arrow="off",
    help="How much the value typically swings in a year (annualized standard deviation of daily returns).",
)
k3.metric(
    "Max drawdown",
    pct(worst.depth),
    gap(worst.depth - market_report.max_drawdown.depth),
    delta_color="off",
    delta_arrow="off",
    help="The worst peak-to-bottom loss over the whole period.",
)
k4.metric(
    "Sharpe ratio",
    f"{report.sharpe:.2f}",
    f"{report.sharpe - market_report.sharpe:+.2f} vs S&P 500",
    delta_color="off",
    delta_arrow="off",
    help=f"Return above a {risk_free:.2f}% risk-free rate for each unit of volatility. Higher is better.",
)
k5, k6, k7, k8 = st.columns(4)
k5.metric(
    "Beta vs S&P 500",
    f"{report.beta:.2f}",
    help="How much the portfolio moves when the market moves 1%. Below 1 means calmer than the market.",
)
k6.metric(
    "1-in-20 daily loss",
    pct(-report.var_95, 2),
    help="95% historical Value at Risk: only 5% of days (about one a month) lost more than this.",
)
k7.metric(
    "Worst day",
    pct(report.worst_day),
    f"{report.worst_day_date:%b %d, %Y}",
    delta_color="off",
    delta_arrow="off",
)
k8.metric(
    "Worst month",
    pct(report.worst_month),
    f"{report.worst_month_date:%B %Y}",
    delta_color="off",
    delta_arrow="off",
)

if worst.recovered is not None and worst.days_to_recover is not None:
    months = worst.days_to_recover / DAYS_PER_MONTH
    recovery = f"took {months:.0f} months to get back to its old high (on {worst.recovered:%b %d, %Y})"
else:
    recovery = "has not yet recovered to its old high"
st.markdown(
    f"From {report.start:%B %Y} to {report.end:%B %Y} this portfolio grew "
    f"**{pct(report.annual_return)} a year**. "
    f"At its worst it fell **{pct(-worst.depth)}** from its peak on {worst.peak:%b %d, %Y} "
    f"to the bottom on {worst.bottom:%b %d, %Y}, and {recovery}. "
    f"It moved about **{report.beta:.2f}%** for every 1% move of the S&P 500."
)
if report.start > prices[BENCHMARK].dropna().index[1]:
    st.caption(f"The analysis starts on {report.start:%b %d, %Y}, when every selected asset had a price.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Drawdown")
    st.caption("How far below its previous high the portfolio was on each day. 0% = at a new high.")
    fig = versus_market(drawdown(mine) * 100, drawdown(market) * 100, ".1f", fill=True)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
with right:
    st.subheader("Where the risk comes from")
    st.caption(
        "Each asset's share of the portfolio's volatility, next to its share of the money. "
        "Volatile assets that move together take more than their share."
    )
    shares = risk_contribution(returns, weights)
    st.plotly_chart(contribution_chart(shares), width="stretch", config=CHART_CONFIG)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Volatility over time")
    st.caption("Volatility over the previous 12 months, on each day. Spikes are market crises.")
    fig = versus_market(rolling_volatility(mine) * 100, rolling_volatility(market) * 100, ".1f")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
with right:
    st.subheader("Correlation")
    st.caption(
        "How much two assets move together, from -1 (opposite) to +1 (in lockstep). "
        "Low or negative numbers are what make diversification work."
    )
    corr_assets = list(dict.fromkeys([*weights.index, BENCHMARK]))
    st.plotly_chart(correlation_heatmap(returns[corr_assets].corr()), width="stretch", config=CHART_CONFIG)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader(f"Growth of ${START_VALUE:,}")
    st.caption("Your portfolio against the S&P 500, dividends reinvested.")
    fig = versus_market(growth(mine) * START_VALUE, growth(market) * START_VALUE, "$,.0f")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
with right:
    st.subheader("Side by side")
    st.caption("Same period, same measures.")
    st.dataframe(comparison_table(report, market_report), width="stretch")

st.divider()
st.caption(
    "Weights are assumed to be rebalanced back to target every day. Taxes and trading costs are ignored. "
    "Past risk doesn't guarantee future risk. For education only, not investment advice."
)
