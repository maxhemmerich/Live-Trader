from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from edge_test import BASE_COLUMNS, build_lightweight_features

CSV_PATH = "D:/XBTUSD_1.csv"
FEE_RATE = 0.0026
START_CASH = 100.0
BARS_PER_YEAR = 365 * 24 * 60
MIN_HOLD_BARS = 5


def prepare_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    src = build_lightweight_features(csv_path, quote_currency="USD")
    print(f"[backtest] Loaded {len(src)} rows")
    feature_columns = [c for c in src.columns if c not in BASE_COLUMNS]

    X = src[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (src["close"].shift(-1) > src["close"]).astype(int)
    prices = src["close"].astype(float)

    X = X.iloc[:-1].reset_index(drop=True)
    y = y.iloc[:-1].reset_index(drop=True)
    prices = prices.iloc[:-1].reset_index(drop=True)
    print("[backtest] Features built")
    return X, y, prices, feature_columns


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return (returns.mean() / std) * np.sqrt(BARS_PER_YEAR)


def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / (peaks + 1e-8)
    return float(drawdowns.min())


def main() -> None:
    X, y, prices, feature_columns = prepare_dataset(CSV_PATH)
    split_idx = int(len(X) * 0.8)

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    prices_test = prices.iloc[split_idx:].reset_index(drop=True)
    print("[backtest] Training GBM...")

    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[backtest] GBM trained, running backtest simulation...")

    prob_up = model.predict_proba(X_test)[:, 1]
    prices = prices_test.to_numpy(dtype=float)

    if len(prices) < 2:
        print("[backtest] Not enough test bars to run simulation.")
        return

    price_returns = np.diff(prices) / prices[:-1]
    prob_up = prob_up[:-1]

    prob_min = float(prob_up.min())
    prob_max = float(prob_up.max())
    prob_mean = float(prob_up.mean())
    prob_median = float(np.median(prob_up))
    pct_above = float((prob_up > 0.58).mean()) * 100.0
    pct_below = float((prob_up < 0.42).mean()) * 100.0
    pct_between = float(((prob_up >= 0.42) & (prob_up <= 0.58)).mean()) * 100.0

    target = np.where(prob_up > 0.58, 1.0, np.where(prob_up < 0.42, 0.0, np.nan))
    desired_allocation = pd.Series(target).ffill().fillna(0.5).to_numpy(dtype=float)

    allocation = desired_allocation.copy()
    last_trade_idx = -10**9
    for i in range(1, len(allocation)):
        if allocation[i] != allocation[i - 1]:
            if i - last_trade_idx <= MIN_HOLD_BARS:
                allocation[i] = allocation[i - 1]
            else:
                last_trade_idx = i

    prev_allocation = np.concatenate([[allocation[0]], allocation[:-1]])
    allocation_change = np.abs(allocation - prev_allocation)
    trades = allocation_change > 0
    n_trades = int(trades.sum())

    fees_applied = FEE_RATE * allocation_change
    bar_returns = allocation * price_returns - fees_applied

    portfolio = START_CASH * np.cumprod(1.0 + bar_returns)

    diagnostics = pd.DataFrame(
        {
            "bar": np.arange(1, len(bar_returns) + 1),
            "prob_up": prob_up,
            "allocation": allocation,
            "prev_allocation": prev_allocation,
            "allocation_change": allocation_change,
            "trade": trades,
            "price_return": price_returns,
            "fee_applied": fees_applied,
            "bar_return": bar_returns,
            "portfolio": portfolio,
        }
    )

    debug_bars = min(20, len(bar_returns))
    print("[backtest] First 20 simulation bars:")
    print(
        "bar\tprob_up\tallocation\tprice_return\tfee_applied\tbar_return\tportfolio_value"
    )
    for i in range(debug_bars):
        print(
            f"{i + 1}\t"
            f"{prob_up[i]:.6f}\t"
            f"{allocation[i]:.2f}\t"
            f"{price_returns[i]:.6f}\t"
            f"{fees_applied[i]:.4f}\t"
            f"{bar_returns[i]:.6f}\t"
            f"{portfolio[i]:.6f}"
        )

    min_bar_return = float(diagnostics["bar_return"].min())
    max_bar_return = float(diagnostics["bar_return"].max())
    print(f"[backtest] Min bar_return: {min_bar_return:.6f}")
    print(f"[backtest] Max bar_return: {max_bar_return:.6f}")

    for threshold in (50.0, 1.0):
        threshold_hits = diagnostics[diagnostics["portfolio"] < threshold]
        if threshold_hits.empty:
            print(f"[backtest] Portfolio never drops below ${threshold:.0f}.")
            continue

        event_idx = int(threshold_hits.index[0])
        event_bar = int(diagnostics.loc[event_idx, "bar"])
        print(f"[backtest] First bar where portfolio drops below ${threshold:.0f}: {event_bar}")

        window_start = max(0, event_idx - 5)
        window_end = min(len(diagnostics), event_idx + 6)
        print(
            f"[backtest] Bars {window_start + 1} to {window_end} around ${threshold:.0f} breach "
            "(all columns):"
        )
        print(diagnostics.iloc[window_start:window_end].to_string(index=False))

    final_value = float(portfolio[-1])
    total_return = (final_value - START_CASH) / START_CASH
    sharpe = sharpe_ratio(bar_returns)
    mdd = max_drawdown(portfolio)

    start_idx = len(prices) - len(price_returns) - 1
    bh_return = (prices[-1] - prices[start_idx]) / prices[start_idx]

    avg_allocation = float(allocation.mean())
    pct_long = float((allocation == 1.0).mean()) * 100.0
    pct_short = float((allocation == 0.0).mean()) * 100.0
    pct_hold = float((allocation == 0.5).mean()) * 100.0

    print("[backtest] Results:")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max drawdown: {mdd * 100:.2f}%")
    print(f"Number of trades: {n_trades}")
    print(f"Buy & hold BTC return (same period): {bh_return * 100:.2f}%")
    print(
        f"prob_up distribution -> min: {prob_min:.4f}, max: {prob_max:.4f}, "
        f"mean: {prob_mean:.4f}, median: {prob_median:.4f}"
    )
    print(
        f"prob_up threshold hits -> >0.58: {pct_above:.2f}%, "
        f"<0.42: {pct_below:.2f}%, between [0.42, 0.58]: {pct_between:.2f}%"
    )
    print(f"Average prob_up: {prob_mean:.4f}")
    print(f"Average allocation: {avg_allocation:.4f}")
    print(f"Allocation long (1.0): {pct_long:.2f}%")
    print(f"Allocation short (0.0): {pct_short:.2f}%")
    print(f"Allocation hold (0.5): {pct_hold:.2f}%")


if __name__ == "__main__":
    main()
