# ARTHA AI Paper-Trading System

ARTHA v18 is a daily NSE screener with a locally trained, interpretable logistic-regression ranker and a simulated paper portfolio. **It is paper-only:** it has no broker adapter and never places live orders.

## Daily workflow

1. Refresh the NSE mainboard and SME symbol universe; use the stored universe cache if NSE is unavailable.
2. Download adjusted daily OHLCV through Yahoo Finance in batches, check basic history quality and freshness, and apply a minimum average traded-value filter.
3. Build candle/trend/volume, NIFTY-relative-strength, mapped sector-relative-strength, volatility, and turnover features.
4. Create historical price-action examples with labels based on whether a 1.5R target or stop is reached first within 20 sessions. Fit a local logistic model only when the data has enough examples and both classes; validation is chronological, not random. If history is insufficient or the temporal validation AUC is below 0.52, the bot transparently falls back to rule-score ranking.
5. Queue the best setups for simulated next-session entries. In breakout regimes, the screener can also flag a compact base just below 20-day resistance as `PRE_BREAKOUT`; this is a conditional paper entry and only fills if a later daily bar crosses the stated trigger. Entries expire if no trigger/bar appears within 10 calendar days or if the opening gap is too large. The paper engine applies share sizing, cash, open-position, per-trade risk, and portfolio-heat limits; it tracks partial target exits, stops, time exits, cash, equity, and closed-trade R-multiples.

The model’s output is a **ranking score, not a calibrated probability or guarantee**. Sector classification covers a limited set of symbols and Yahoo sector-index data is best-effort; missing sector data falls back to NIFTY-relative strength. The historical training universe is based on currently listed symbols, so survivorship bias remains.

## Default paper-risk settings

- Starting capital: ₹1,00,000 (`TOTAL_CAPITAL_INR`)
- Risk budget per trade: 0.5% of current paper equity (`RISK_PER_TRADE_PCT`)
- Total open portfolio heat: 5% (`PORTFOLIO_HEAT_PCT`)
- Maximum positions: 3 (`MAX_POSITIONS_ACTIVE`)
- Maximum position value: 25% of equity (`MAX_POSITION_VALUE_PCT`)
- Minimum 20-session average traded value: ₹50 lakh (`MIN_AVG_DAILY_TURNOVER_INR`)
- Maximum modeled stop distance: 5.5% (`MAX_SETUP_RISK_PCT`)
- Entry-gap cancellation: 3% default (`MAX_PAPER_ENTRY_GAP_PCT`)
- Simulated slippage and transaction costs: configurable with `PAPER_SLIPPAGE_BPS` and `PAPER_FEE_BPS_ROUND_TRIP`

The paper ledger is stored in `artha_memory.json`. It is not a brokerage statement and currently uses daily OHLC bars, conservative stop-first handling when both barriers are touched on the same day, and approximate transaction costs. Taxes, brokerage tiers, market impact, circuit limits, order queues, and intraday fills are not modeled precisely.

## Data limitations and fail-safe behavior

- NSE or Yahoo Finance may rate-limit/deny GitHub Actions. Reports show universe and price-data coverage; a successful workflow does not mean every listing had usable prices.
- If NIFTY benchmark history is unavailable, ARTHA does not queue new equity paper entries. In a bear regime it considers only a small defensive ETF list.
- Data validation flags stale or extreme recent price moves, but Yahoo’s adjustment history can still be wrong or incomplete. No fundamentals, earnings calendar, news, corporate-action feed, or intraday confirmation is currently integrated.
- The paper system is research software, not financial advice. Validate performance across regimes and independently verify data before making any real-world decision.

## Run locally

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python artha.py
```

Set `BOT_TOKEN` and `CHAT_ID` as environment variables for Telegram reports. Without them, the report is printed to the console. Never commit credentials.

## GitHub Actions

The weekday workflow installs `requirements.txt`, runs unit tests, then executes ARTHA and persists `artha_memory.json`. Configure `BOT_TOKEN` and `CHAT_ID` as repository Actions secrets. Workflow concurrency is serialized to protect the paper ledger. The scheduled workflow runs code from the repository default branch. There is no live-order workflow or broker connection.
