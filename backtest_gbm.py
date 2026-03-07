from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from edge_test import BASE_COLUMNS, build_lightweight_features

CSV_PATH = "D:/XBTUSD_1.csv"
FEE_RATE = 0.0026
START_CASH = 100.0
BARS_PER_YEAR = 365 * 24 * 60
MIN_HOLD_BARS = 60


def apply_min_hold_period(
    desired_allocation: np.ndarray, min_hold_bars: int = MIN_HOLD_BARS
) -> np.ndarray:
    allocation = np.empty_like(desired_allocation)
    allocation[0] = desired_allocation[0]
    bars_since_last_trade = min_hold_bars

    for i in range(1, len(desired_allocation)):
        bars_since_last_trade += 1
        previous_allocation = allocation[i - 1]
        target_allocation = desired_allocation[i]

        if target_allocation == previous_allocation:
            allocation[i] = previous_allocation
            continue

        if bars_since_last_trade >= min_hold_bars:
            allocation[i] = target_allocation
            bars_since_last_trade = 0
        else:
            allocation[i] = previous_allocation

    return allocation


def evaluate_backtest(
    prob_up: np.ndarray, price_returns: np.ndarray, min_hold_bars: int = MIN_HOLD_BARS
) -> dict[str, float]:
    target = np.where(prob_up > 0.58, 1.0, np.where(prob_up < 0.42, 0.0, np.nan))
    desired_allocation = pd.Series(target).ffill().fillna(0.5).to_numpy(dtype=float)
    allocation = apply_min_hold_period(desired_allocation, min_hold_bars=min_hold_bars)

    # Allocation is encoded as long=1.0, flat=0.5, short=0.0.
    # Convert this to directional exposure before applying returns.
    position = (allocation * 2.0) - 1.0

    prev_position = np.concatenate([[position[0]], position[:-1]])
    position_change = np.abs(position - prev_position)
    trade_mask = position_change > 0.0
    n_trades = int(trade_mask.sum())

    # A full side change (flat->long or flat->short) costs one fee.
    # A reversal (long->short or short->long) costs two fees.
    fees_applied = FEE_RATE * position_change
    bar_returns = position * price_returns - fees_applied
    portfolio = START_CASH * np.cumprod(1.0 + bar_returns)

    final_value = float(portfolio[-1])
    total_return = (final_value - START_CASH) / START_CASH
    sharpe = sharpe_ratio(bar_returns)
    mdd = max_drawdown(portfolio)

    return {
        "min_hold_bars": float(min_hold_bars),
        "final_value": final_value,
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "n_trades": float(n_trades),
        "avg_allocation": float(allocation.mean()),
        "pct_long": float((allocation == 1.0).mean()) * 100.0,
        "pct_short": float((allocation == 0.0).mean()) * 100.0,
        "pct_hold": float((allocation == 0.5).mean()) * 100.0,
    }


def prepare_dataset(
    csv_path: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    src = build_lightweight_features(csv_path, quote_currency="USD")
    print(f"[backtest] Loaded {len(src)} rows")
    feature_columns = [c for c in src.columns if c not in BASE_COLUMNS]

    X = src[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (src["close"].shift(-1) > src["close"]).astype(int)
    prices = src["close"].astype(float)
    timestamps = src["ts"].astype(np.int64)

    X = X.iloc[:-1].reset_index(drop=True)
    y = y.iloc[:-1].reset_index(drop=True)
    prices = prices.iloc[:-1].reset_index(drop=True)
    timestamps = timestamps.iloc[:-1].reset_index(drop=True)
    print("[backtest] Features built")
    return X, y, prices, timestamps, feature_columns


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


def average_consecutive_run_length(binary_series: np.ndarray) -> float:
    if len(binary_series) == 0:
        return 0.0

    direction_changes = np.where(binary_series[1:] != binary_series[:-1])[0] + 1
    run_boundaries = np.concatenate(([0], direction_changes, [len(binary_series)]))
    run_lengths = np.diff(run_boundaries)
    return float(run_lengths.mean())


def run_timeframe_sweep() -> None:
    print("[timeframe_sweep] Running independent timeframe sweep...")

    raw = pd.read_csv(
        CSV_PATH,
        header=None,
        names=["ts", "open", "high", "low", "close", "vol"],
        usecols=[0, 1, 2, 3, 4, 5],
    )
    if raw.empty:
        print("[timeframe_sweep] No raw rows found, skipping.")
        return

    raw["ts"] = pd.to_numeric(raw["ts"], errors="coerce")
    for col in ["open", "high", "low", "close", "vol"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["ts", "open", "high", "low", "close", "vol"]).copy()
    if raw.empty:
        print("[timeframe_sweep] No valid OHLCV rows after cleaning, skipping.")
        return

    ts_scale = 1000 if int(raw["ts"].median()) < 10**12 else 1
    raw["ts"] = raw["ts"].astype(np.int64) * ts_scale
    raw = raw.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    raw["dt"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
    raw = raw.set_index("dt")

    timeframe_minutes = [5, 15, 30, 60, 240, 1440]
    ranking: list[dict[str, float]] = []

    for tf in timeframe_minutes:
        resampled = (
            raw.resample(f"{tf}min")
            .agg(
                {
                    "ts": "last",
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "vol": "sum",
                }
            )
            .dropna(subset=["ts", "open", "high", "low", "close", "vol"])
            .reset_index(drop=True)
        )

        if len(resampled) < 300:
            print(
                f"TF={tf}min | GBM_acc=nan% | Oracle=$nan | GBM=$nan | "
                "avg_move=nan% | avg_consecutive=nan bars"
            )
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as tmp:
            resampled[["ts", "open", "high", "low", "close", "vol"]].to_csv(
                tmp.name, index=False, header=False
            )

            src = build_lightweight_features(tmp.name, quote_currency="USD")

        feature_columns = [c for c in src.columns if c not in BASE_COLUMNS]
        X = src[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = (src["close"].shift(-1) > src["close"]).astype(int)
        prices = src["close"].astype(float)

        X = X.iloc[:-1].reset_index(drop=True)
        y = y.iloc[:-1].reset_index(drop=True)
        prices = prices.iloc[:-1].reset_index(drop=True)

        if len(X) < 200:
            print(
                f"TF={tf}min | GBM_acc=nan% | Oracle=$nan | GBM=$nan | "
                "avg_move=nan% | avg_consecutive=nan bars"
            )
            continue

        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        prices_test = prices.iloc[split_idx:].reset_index(drop=True)

        if len(X_test) < 3 or len(prices_test) < 3:
            print(
                f"TF={tf}min | GBM_acc=nan% | Oracle=$nan | GBM=$nan | "
                "avg_move=nan% | avg_consecutive=nan bars"
            )
            continue

        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = float((y_pred == y_test.to_numpy(dtype=int)).mean())

        prob_up = model.predict_proba(X_test)[:, 1]
        test_prices_np = prices_test.to_numpy(dtype=float)
        price_returns = np.diff(test_prices_np) / test_prices_np[:-1]
        prob_up = prob_up[:-1]

        oracle_prob_up = np.where(price_returns > 0.0, 1.0, 0.0)
        oracle_result = evaluate_backtest(oracle_prob_up, price_returns, min_hold_bars=0)
        gbm_result = evaluate_backtest(prob_up, price_returns, min_hold_bars=3)
        avg_move = float(np.abs(price_returns).mean()) * 100.0
        avg_consecutive = average_consecutive_run_length(oracle_prob_up.astype(int))

        print(
            f"TF={tf}min | GBM_acc={accuracy * 100:.2f}% | "
            f"Oracle=${oracle_result['final_value']:.2f} | "
            f"GBM=${gbm_result['final_value']:.2f} | "
            f"avg_move={avg_move:.4f}% | "
            f"avg_consecutive={avg_consecutive:.2f} bars"
        )

        ranking.append(
            {
                "timeframe": float(tf),
                "gbm_final_value": gbm_result["final_value"],
            }
        )

    if not ranking:
        print("[timeframe_sweep] No valid timeframe results to rank.")
        return

    print("[timeframe_sweep] Summary ranking by GBM final value:")
    ranking_sorted = sorted(ranking, key=lambda item: item["gbm_final_value"], reverse=True)
    for rank, item in enumerate(ranking_sorted, start=1):
        print(
            f"{rank}. TF={int(item['timeframe'])}min | GBM=${item['gbm_final_value']:.2f}"
        )


def main() -> None:
    X, y, prices, timestamps, feature_columns = prepare_dataset(CSV_PATH)
    split_idx = int(len(X) * 0.8)

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:].reset_index(drop=True)

    train_start_dt = pd.to_datetime(timestamps.iloc[0], unit="ms", utc=True)
    train_end_dt = pd.to_datetime(timestamps.iloc[split_idx - 1], unit="ms", utc=True)
    test_start_dt = pd.to_datetime(timestamps.iloc[split_idx], unit="ms", utc=True)
    test_end_dt = pd.to_datetime(timestamps.iloc[-1], unit="ms", utc=True)

    print(
        "[backtest] Training period: "
        f"{train_start_dt.isoformat()} -> {train_end_dt.isoformat()}"
    )
    print(
        "[backtest] Test period: "
        f"{test_start_dt.isoformat()} -> {test_end_dt.isoformat()}"
    )
    print(
        "[backtest] BTC close (training): "
        f"start={prices_train.iloc[0]:.2f}, end={prices_train.iloc[-1]:.2f}"
    )
    print(
        "[backtest] BTC close (test): "
        f"start={prices_test.iloc[0]:.2f}, end={prices_test.iloc[-1]:.2f}"
    )
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

    oracle_prob_up = np.where(price_returns > 0.0, 1.0, 0.0)
    oracle_result = evaluate_backtest(
        oracle_prob_up, price_returns, min_hold_bars=1
    )
    avg_same_direction_bars = average_consecutive_run_length(oracle_prob_up.astype(int))

    prob_min = float(prob_up.min())
    prob_max = float(prob_up.max())
    prob_mean = float(prob_up.mean())
    prob_median = float(np.median(prob_up))
    pct_above = float((prob_up > 0.58).mean()) * 100.0
    pct_below = float((prob_up < 0.42).mean()) * 100.0
    pct_between = float(((prob_up >= 0.42) & (prob_up <= 0.58)).mean()) * 100.0

    start_idx = len(prices) - len(price_returns) - 1
    bh_return = (prices[-1] - prices[start_idx]) / prices[start_idx]

    print("[backtest] Hold period sweep:")
    hold_periods = (30, 60, 120, 240)
    results: list[dict[str, float]] = []
    for hold_bars in hold_periods:
        result = evaluate_backtest(prob_up, price_returns, min_hold_bars=hold_bars)
        results.append(result)
        print(
            f"min_hold_bars={hold_bars:>3} | "
            f"Final=${result['final_value']:.2f} | "
            f"Return={result['total_return'] * 100:.2f}% | "
            f"Sharpe={result['sharpe']:.4f} | "
            f"MDD={result['max_drawdown'] * 100:.2f}% | "
            f"Trades={int(result['n_trades'])}"
        )

    best_by_final_value = max(results, key=lambda item: item["final_value"])
    print(
        "[backtest] Best hold by final value: "
        f"min_hold_bars={int(best_by_final_value['min_hold_bars'])}, "
        f"Final=${best_by_final_value['final_value']:.2f}, "
        f"Sharpe={best_by_final_value['sharpe']:.4f}"
    )

    base_result = next(
        result for result in results if int(result["min_hold_bars"]) == MIN_HOLD_BARS
    )

    print("[backtest] Results (min_hold_bars=60 baseline):")
    print(f"Final portfolio value: ${base_result['final_value']:.2f}")
    print(f"Total return: {base_result['total_return'] * 100:.2f}%")
    print(f"Sharpe ratio: {base_result['sharpe']:.4f}")
    print(f"Max drawdown: {base_result['max_drawdown'] * 100:.2f}%")
    print(f"Number of trades: {int(base_result['n_trades'])}")
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
    print(f"Average allocation: {base_result['avg_allocation']:.4f}")
    print(f"Allocation long (1.0): {base_result['pct_long']:.2f}%")
    print(f"Allocation short (0.0): {base_result['pct_short']:.2f}%")
    print(f"Allocation hold (0.5): {base_result['pct_hold']:.2f}%")
    print(
        "[backtest] Perfect oracle (actual next-bar direction, min_hold_bars=1) "
        f"final portfolio value: ${oracle_result['final_value']:.2f}, "
        f"trades: {int(oracle_result['n_trades'])}"
    )
    print(
        "[backtest] Average consecutive same-direction bars in test period: "
        f"{avg_same_direction_bars:.2f}"
    )


if __name__ == "__main__":
    main()
    run_timeframe_sweep()
