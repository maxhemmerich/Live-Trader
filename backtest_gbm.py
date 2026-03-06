from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from edge_test import BASE_COLUMNS, build_lightweight_features

CSV_PATH = "D:/XBTUSD_1.csv"
FEE_RATE = 0.0026
START_CASH = 100.0
BARS_PER_YEAR = 365 * 24 * 60


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

    target = np.where(prob_up > 0.58, 1.0, np.where(prob_up < 0.42, 0.0, np.nan))
    allocation = pd.Series(target).ffill().fillna(0.5).to_numpy(dtype=float)

    trades = np.diff(np.concatenate([[allocation[0]], allocation])) != 0
    n_trades = int(trades.sum())

    bar_returns = allocation * price_returns
    bar_returns[trades] -= FEE_RATE

    portfolio = START_CASH * np.cumprod(1.0 + bar_returns)

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
    print(f"Average allocation: {avg_allocation:.4f}")
    print(f"Allocation long (1.0): {pct_long:.2f}%")
    print(f"Allocation short (0.0): {pct_short:.2f}%")
    print(f"Allocation hold (0.5): {pct_hold:.2f}%")


if __name__ == "__main__":
    main()
