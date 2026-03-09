# Polymarket Prediction Market Trading System

This repo now includes a complete end-to-end workflow for discovering Polymarket markets, collecting signals, modeling edge, backtesting, monitoring, and optionally executing bets.

## Files Added

- `polymarket_explorer.py` — fetches top 100 active/open markets from Polymarket and saves to `polymarket_markets.json`.
- `data_collector.py` — enriches markets with polling, crypto, FRED, and NewsAPI sentiment signals into `market_signals.json`.
- `probability_model.py` — builds model probabilities, computes edge and Kelly sizing, and saves `model_predictions.json`.
- `backtest_polymarket.py` — simulates historical strategy performance on resolved markets and saves `backtest_results.json`.
- `monitor_polymarket.py` — Streamlit dashboard with top opportunities, backtest stats, positions, and PnL.
- `bet_executor.py` — auto-bet executor via CLOB endpoint with logging and daily-loss circuit breaker.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the repo root:

```env
NEWSAPI_KEY=your_newsapi_key
POLYMARKET_PRIVATE_KEY=your_wallet_private_key
POLYMARKET_WALLET_ADDRESS=your_wallet_address
PORTFOLIO_VALUE=1000
```

## Run Order

Run the scripts in this order:

1. Fetch active markets
```bash
python polymarket_explorer.py
```

2. Collect market signals
```bash
python data_collector.py
```

3. Build model predictions
```bash
python probability_model.py
```

4. Backtest historical performance
```bash
python backtest_polymarket.py
```

5. Launch monitoring dashboard (auto-refresh every 5 minutes)
```bash
streamlit run monitor_polymarket.py
```

6. Execute high-edge bets (optional/live risk)
```bash
python bet_executor.py
```

## Notes

- All scripts print progress updates and handle API retries.
- `bet_executor.py` stops placing bets if daily realized losses exceed 20% of configured portfolio value.
- News sentiment requires a valid NewsAPI key.
- Crypto signal in `data_collector.py` attempts to use `checkpoints/gbm_model.pkl` if available.
