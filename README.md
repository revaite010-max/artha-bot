# ARTHA Bot

ARTHA is a daily NSE/BSE equity screener that refreshes its listed-stock universe from the exchanges, classifies the broad market regime, reports every covered stock whose trailing two-month return clears the configured threshold, ranks separate entry setups, and can send a report through Telegram. The momentum leaderboard is not capped at the number of actionable trade setups. Historical prices are sourced from Yahoo Finance via `yfinance`.

## Run locally

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python artha.py
```

Without credentials, the report is printed to the console. To send it to Telegram, set `BOT_TOKEN` and `CHAT_ID` as environment variables. Do not commit credentials to the repository.

## Configuration

The capital, portfolio heat, maximum number of active picks, and momentum threshold are configured near the top of `artha.py`. On each run, the bot refreshes the NSE mainboard/SME equity CSVs and BSE active-equity scrip master, then deduplicates dual-listed companies by ISIN. The GitHub Actions workflow runs on weekdays and can also be started manually. Add `BOT_TOKEN` and `CHAT_ID` as repository Actions secrets to enable Telegram delivery.

The screener requests one year of history so the 200-day moving average has adequate warm-up data. Its separate momentum leaderboard uses the last 42 trading sessions (about two months) and lists every covered stock with a return of at least `MIN_TWO_MONTH_RETURN_PCT` (5% by default); adjust the constants near the top of `artha.py` to change this. This historical screen still runs when market-regime data is unavailable, while new entry setups remain fail-closed. Position sizing for actionable setups is capped by both available capital and portfolio-wide stop-loss risk. Each run logs data coverage and download errors to make feed/execution problems visible.

## Adaptive learning and memory

ARTHA currently uses Yahoo Finance via `yfinance` for historical prices. It stores up to 12 daily market-session snapshots (a maximum of 300 mixed ranked/exploration stocks per session, with OHLCV, features, regime, and source metadata) in `artha_memory.json`. Once five later market sessions are available, it labels each stored observation using a simulated stop/target outcome and updates a local logistic-regression model. The learned probabilities help rank the momentum leaders and rule-qualified setups. The report shows market-universe counts, data coverage, model sample count, and warm-up/adaptive status. Exchange listings are refreshed from official masters; when a refresh fails, the last successfully cached universe is used and the stale-cache warning is shown.

This is a transparent, local statistical learning component—not a hosted generative-AI service. At first it uses prior weights and needs outcomes to learn; predictions are uncertain and are not validated guarantees. Outcomes are paper simulations based on daily bars, not actual fills. The bot does not connect to a broker or place orders; human review and independent validation are required before any trade.

## Important

ARTHA is an experimental screening tool, not an investment adviser or broker. Yahoo Finance data can be delayed, incomplete, or incorrect. Signals and example stop-loss/target levels are not guarantees; independently verify data, liquidity, corporate actions, and suitability before making any investment decision. The bot does not place orders or continuously monitor stops.
