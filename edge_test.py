"""Quick edge test: do engineered features predict next-bar direction?"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

BASE_COLUMNS = {"ts", "open", "high", "low", "close", "vol"}
MIN_REQUIRED_ROWS = 1000


@dataclass
class EdgeResult:
    csv_path: str
    timeframe_label: str
    samples: int
    train_samples: int
    test_samples: int
    accuracy: float
    top_features: list[tuple[str, float]]


def _csv_has_header_row(csv_path: str) -> bool:
    first_row = pd.read_csv(csv_path, nrows=1, header=None, dtype=str)
    if first_row.empty:
        return False
    first_cell = str(first_row.iat[0, 0]).strip().lower().lstrip("\ufeff")
    return first_cell in {"ts", "timestamp", "time", "datetime", "date"}


def _compute_cci_williams_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    cci_window: int = 20,
    willr_window: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    typical = (high + low + close) / 3.0

    cci = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) >= cci_window:
        cci_windows = np.lib.stride_tricks.sliding_window_view(typical, cci_window)
        sma = cci_windows.mean(axis=1)
        mad = np.abs(cci_windows - sma[:, None]).mean(axis=1)
        cci_values = (typical[cci_window - 1 :] - sma) / ((0.015 * mad) + 1e-8)
        cci[cci_window - 1 :] = np.clip(cci_values / 200.0, -1.0, 1.0)

    willr = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) >= willr_window:
        high_windows = np.lib.stride_tricks.sliding_window_view(high, willr_window)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, willr_window)
        highest_high = high_windows.max(axis=1)
        lowest_low = low_windows.min(axis=1)
        willr_raw = ((highest_high - close[willr_window - 1 :]) / ((highest_high - lowest_low) + 1e-8)) * -100.0
        willr[willr_window - 1 :] = (willr_raw + 100.0) / 100.0

    return cci, willr


def build_lightweight_features(csv_path: str) -> pd.DataFrame:
    skip_header_row = 1 if _csv_has_header_row(csv_path) else 0
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["ts", "open", "high", "low", "close", "vol"],
        skiprows=skip_header_row,
        usecols=[0, 1, 2, 3, 4, 5],
        dtype={
            "ts": np.int64,
            "open": np.float32,
            "high": np.float32,
            "low": np.float32,
            "close": np.float32,
            "vol": np.float32,
        },
    )
    if df.empty:
        return df

    ts_scale = 1000 if int(df["ts"].median()) < 10**12 else 1
    df["ts"] = df["ts"] * ts_scale
    two_years_ms = int(2 * 365 * 24 * 60 * 60 * 1000)
    sample_cutoff_ts = int(df["ts"].max()) - two_years_ms
    df = df[df["ts"] >= sample_cutoff_ts].copy().reset_index(drop=True)

    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    vol = df["vol"].astype(np.float64)

    df["rsi_7_norm"] = RSIIndicator(close=close, window=7).rsi() / 100.0
    df["rsi_14_norm"] = RSIIndicator(close=close, window=14).rsi() / 100.0
    df["rsi_21_norm"] = RSIIndicator(close=close, window=21).rsi() / 100.0

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["stoch_k_norm"] = stoch.stoch() / 100.0
    df["stoch_d_norm"] = stoch.stoch_signal() / 100.0

    cci_20_clipped, willr_14_norm = _compute_cci_williams_np(
        high=high.to_numpy(dtype=np.float64),
        low=low.to_numpy(dtype=np.float64),
        close=close.to_numpy(dtype=np.float64),
        cci_window=20,
        willr_window=14,
    )
    df["cci_20_clipped"] = np.nan_to_num(cci_20_clipped, nan=0.0)
    df["willr_14_norm"] = np.nan_to_num(willr_14_norm, nan=0.0)

    bb20 = BollingerBands(close=close, window=20, window_dev=2)
    bb50 = BollingerBands(close=close, window=50, window_dev=2)
    atr14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    ema9 = EMAIndicator(close=close, window=9).ema_indicator()
    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema50 = EMAIndicator(close=close, window=50).ema_indicator()
    ema200 = EMAIndicator(close=close, window=200).ema_indicator()
    macd_hist = MACD(close=close, window_fast=12, window_slow=26, window_sign=9).macd_diff()
    obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()

    df["bb20_p"] = bb20.bollinger_pband()
    df["atr_14_over_price"] = atr14 / (close + 1e-8)
    df["realized_vol_20_norm"] = close.pct_change().rolling(20).std() / 0.02
    df["ema9"] = (close / (ema9 + 1e-8)) - 1.0
    df["ema20"] = (close / (ema20 + 1e-8)) - 1.0
    df["ema50"] = (close / (ema50 + 1e-8)) - 1.0
    df["ema200"] = (close / (ema200 + 1e-8)) - 1.0
    df["macd_hist_atr"] = macd_hist / (atr14 + 1e-8)
    df["bb20_width_price"] = (bb20.bollinger_hband() - bb20.bollinger_lband()) / (close + 1e-8)
    df["bb50_width_price"] = (bb50.bollinger_hband() - bb50.bollinger_lband()) / (close + 1e-8)
    df["obv_pct_change"] = obv.pct_change()

    vol20 = vol.rolling(20).mean().replace(0.0, np.nan)
    spread = (high - low) / (close + 1e-8)
    spread_mean20 = spread.rolling(20).mean().replace(0.0, np.nan)
    df["bid_ask_spread_frac"] = (spread / (spread_mean20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
    df["bid_depth_5_over_vol20"] = ((vol * 0.5) / (vol20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
    df["ask_depth_5_over_vol20"] = ((vol * 0.5) / (vol20 + 1e-8)).clip(-1.0, 1.0).fillna(0.0)
    imbalance = (((close - low) / ((high - low) + 1e-8)) * 2.0) - 1.0
    df["bid_ask_imbalance"] = imbalance.clip(-1.0, 1.0).fillna(0.0)
    df["price_dist_best_bid"] = ((close - low) / (close + 1e-8)).clip(-1.0, 1.0).fillna(0.0)

    df["eth_return_1"] = close.pct_change(1).fillna(0.0)
    df["eth_return_4"] = close.pct_change(4).fillna(0.0)
    df["eth_return_16"] = close.pct_change(16).fillna(0.0)

    return df


def build_dataset(csv_path: str, candle_interval: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    _ = candle_interval
    skip_header_row = 1 if _csv_has_header_row(csv_path) else 0
    raw_rows = pd.read_csv(csv_path, header=None, skiprows=skip_header_row, usecols=[0]).shape[0]
    if raw_rows < MIN_REQUIRED_ROWS:
        print(
            f"[edge_test] Skipping {csv_path}: insufficient data ({raw_rows} rows, minimum {MIN_REQUIRED_ROWS} required)."
        )
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), []

    t0 = time.perf_counter()
    features_source = build_lightweight_features(csv_path)
    print(f"[edge_test] Lightweight feature build completed in {time.perf_counter() - t0:.2f}s")

    feature_columns = [
        col
        for col in features_source.columns
        if col not in BASE_COLUMNS
    ]

    features_df = features_source[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    next_close = features_source["close"].shift(-1)
    labels = (next_close > features_source["close"]).astype(int)

    features_df = features_df.iloc[:-1].reset_index(drop=True)
    labels = labels.iloc[:-1].to_numpy(dtype=np.int32)

    return features_df.to_numpy(dtype=np.float32), labels, feature_columns


def _format_timeframe_label(candle_interval: int) -> str:
    if candle_interval >= 1440:
        days = candle_interval // 1440
        return f"{days}day" if days == 1 else f"{days}days"
    if candle_interval >= 60 and candle_interval % 60 == 0:
        hours = candle_interval // 60
        return f"{hours}hour" if hours == 1 else f"{hours}hours"
    return f"{candle_interval}min"


def _extract_timeframe_token(path: str) -> str | None:
    stem = Path(path).stem
    match = re.search(r"_(\d+)$", stem)
    return match.group(1) if match else None


def parse_timeframe(timeframe: str | None, csv_path: str | None = None) -> tuple[int, str]:
    if timeframe:
        token = timeframe.strip().lower()
    else:
        token = _extract_timeframe_token(csv_path or "")
        if token is None:
            raise ValueError(f"Unable to infer timeframe from filename: {csv_path}")

    normalized = re.sub(r"[^0-9a-z]", "", token)
    match = re.match(r"^(\d+)(min|m|hour|h|day|d)?$", normalized)
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe or token}")

    value = int(match.group(1))
    unit = match.group(2) or "min"
    if unit in {"m", "min"}:
        minutes = value
    elif unit in {"h", "hour"}:
        minutes = value * 60
    else:
        minutes = value * 1440

    return minutes, _format_timeframe_label(minutes)


def run_edge_test(csv_path: str, candle_interval: int, timeframe_label: str) -> EdgeResult | None:
    X, y, feature_columns = build_dataset(csv_path, candle_interval)
    if X.size == 0 or y.size == 0:
        return None

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    importance_idx = np.argsort(clf.feature_importances_)[::-1][:5]
    top_features = [(feature_columns[idx], float(clf.feature_importances_[idx])) for idx in importance_idx]

    print("=" * 88)
    print(f"CSV: {csv_path}")
    print(f"Timeframe: {timeframe_label} (candle_interval={candle_interval})")
    print(f"Samples: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Test accuracy: {acc:.4f}")
    print("Top 5 feature importances:")
    for rank, (name, importance) in enumerate(top_features, start=1):
        print(f"{rank:2d}. {name}: {importance:.6f}")

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return EdgeResult(
        csv_path=csv_path,
        timeframe_label=timeframe_label,
        samples=len(X),
        train_samples=len(X_train),
        test_samples=len(X_test),
        accuracy=float(acc),
        top_features=top_features,
    )


def find_default_csv_jobs(default_dir: str = "D:/") -> list[tuple[str, int, str]]:
    base = Path(default_dir)
    candidates = sorted(base.glob("*USD*.csv"))
    jobs: list[tuple[str, int, str]] = []
    for path in candidates:
        token = _extract_timeframe_token(str(path))
        if token is None:
            continue
        candle_interval, label = parse_timeframe(token)
        jobs.append((str(path), candle_interval, label))
    return jobs


def print_ranked_summary(results: list[EdgeResult]) -> None:
    if not results:
        print("No successful runs to summarize.")
        return

    ranked = sorted(results, key=lambda r: r.accuracy, reverse=True)
    print("\n" + "#" * 88)
    print("Final ranking by test accuracy (descending)")
    print("#" * 88)
    print(f"{'Rank':<6}{'Accuracy':<12}{'Timeframe':<12}{'Samples':<12}{'CSV'}")
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:<6}{result.accuracy:<12.4f}{result.timeframe_label:<12}"
            f"{result.samples:<12,}{result.csv_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictive edge of engineered features.")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV file for a single run. If omitted, runs a sweep over D:/*USD*.csv.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Timeframe for --csv (e.g. 5min, 1min, 1day). If omitted, inferred from filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_results: list[EdgeResult] = []

    if args.csv:
        interval, label = parse_timeframe(args.timeframe, args.csv)
        result = run_edge_test(args.csv, interval, label)
        if result is not None:
            all_results.append(result)
    else:
        jobs = find_default_csv_jobs("D:/")
        if not jobs:
            raise FileNotFoundError(
                "No files matching D:/*USD*.csv with timeframe suffix were found."
            )
        for csv_path, interval, label in jobs:
            result = run_edge_test(csv_path, interval, label)
            if result is not None:
                all_results.append(result)

    print_ranked_summary(all_results)
