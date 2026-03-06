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
    std = returns.std(ddof=1)
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

    cash = START_CASH
    btc = 0.0
    trades = 0
    wins = 0
    open_trade_value = None

    equity_curve: list[float] = []
    step_returns: list[float] = []
    prev_equity = START_CASH

    total_bars = len(X_test)
    for i in range(total_bars):
        price = float(prices_test.iloc[i])
        prob_up = float(model.predict_proba(X_test.iloc[[i]])[0, 1])

        if (i + 1) % 100_000 == 0:
            print(f"[backtest] Simulated {i + 1}/{total_bars} bars...")

        if prob_up > 0.58:
            target_btc_alloc = 1.0
        elif prob_up < 0.42:
            target_btc_alloc = 0.0
        else:
            target_btc_alloc = None

        if target_btc_alloc is not None:
            equity = cash + (btc * price)
            target_btc_value = equity * target_btc_alloc
            current_btc_value = btc * price
            gap = target_btc_value - current_btc_value

            if abs(gap) > 0.10 * equity:
                if gap > 0 and cash > 0:
                    gross_btc = cash / price
                    fee_btc = gross_btc * FEE_RATE
                    btc += max(0.0, gross_btc - fee_btc)
                    cash = 0.0
                    trades += 1
                    open_trade_value = equity
                elif gap < 0 and btc > 0:
                    gross_cash = btc * price
                    fee_cash = gross_cash * FEE_RATE
                    cash += max(0.0, gross_cash - fee_cash)
                    btc = 0.0
                    trades += 1
                    if open_trade_value is not None and cash > open_trade_value:
                        wins += 1
                    open_trade_value = None

        equity = cash + (btc * price)
        equity_curve.append(equity)
        step_returns.append((equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0)
        prev_equity = equity

    final_value = equity_curve[-1] if equity_curve else START_CASH
    total_return = ((final_value / START_CASH) - 1.0) * 100.0

    n_bars = len(equity_curve)
    annualized = 0.0
    if n_bars > 0 and final_value > 0:
        annualized = ((final_value / START_CASH) ** (BARS_PER_YEAR / n_bars) - 1.0) * 100.0

    sharpe = sharpe_ratio(np.array(step_returns, dtype=float))
    mdd = max_drawdown(np.array(equity_curve, dtype=float)) * 100.0
    win_rate = (wins / trades) * 100.0 if trades > 0 else 0.0

    buy_hold_final = START_CASH * (float(prices_test.iloc[-1]) / float(prices_test.iloc[0])) if len(prices_test) > 1 else START_CASH
    buy_hold_return = ((buy_hold_final / START_CASH) - 1.0) * 100.0

    print("[backtest] Results:")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized:.2f}%")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max drawdown: {mdd:.2f}%")
    print(f"Number of trades: {trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Buy & hold BTC return (same period): {buy_hold_return:.2f}%")


if __name__ == "__main__":
    main()
