# ARTHA Bot

ARTHA is a daily NSE equity screener. Each run refreshes the NSE mainboard and SME equity symbol lists, deduplicates symbols, downloads one year of Yahoo Finance history in batches, and applies the current market-regime strategy plus a price-action selection layer. The report shows the number of NSE symbols loaded and the number with usable OHLCV data.

## Universe and data coverage

- NSE mainboard source: `EQUITY_L.csv`.
- NSE SME source: `SME_EQUITY_L.csv`.
- Historical prices: Yahoo Finance through `yfinance` (`.NS` tickers).
- The refreshed NSE universe is cached in `artha_memory.json`. If NSE is temporarily unreachable, the bot uses its last cached universe. On a first run with no cache, it uses the older built-in NSE list and clearly labels that run as a **partial fallback** rather than claiming it scanned the full exchange.
- Newly listed, suspended, illiquid, or Yahoo-unavailable securities may not have a full year of OHLCV data and are counted as unavailable in the scan report.

The NSE website and Yahoo Finance can rate-limit or deny automated requests. Check the logged universe source and data-coverage counts after each run; a successful workflow run does not necessarily mean every listed stock had usable price history.

## Price-action selection

For eligible bullish regimes, ARTHA additionally checks daily candles for a 20-session resistance breakout with a strong close and volume, a bullish engulfing candle, or a rejection from recent support. These patterns can create an independent price-action setup or add confirmation to the regime strategy. Every candidate still needs a defined stop with no more than 5% modeled risk; price patterns are not guarantees.

## Run locally

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python artha.py
```

Set `BOT_TOKEN` and `CHAT_ID` as environment variables for Telegram delivery. Without them, the report is printed to the console. Do not commit credentials.

## GitHub Actions

The workflow runs on weekdays and can be started manually. Configure `BOT_TOKEN` and `CHAT_ID` as repository Actions secrets. It runs the unit tests before scanning, then persists NSE-universe cache and bot memory. The scheduled workflow runs code from the repository's default branch.

## Risk notice

ARTHA is an experimental screening tool, not an investment adviser or broker. Market data can be delayed, incomplete, or incorrect. Recommendations and stop/target values are not guarantees. Verify prices, liquidity, corporate actions, and suitability independently. The bot does not place orders or monitor stops.
