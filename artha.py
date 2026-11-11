"""
ARTHA v17.0 - NSE-Wide Research & Risk Engine
Auto-Switches Between:
 1. Bull Mode     --> Breakouts & VCP Patterns
 2. Sideways Mode --> Dip-Buying & Mean Reversion (Quality Bounces)
 3. Bear Mode      --> Defensive Assets & Gold ETFs
 4. Research Layer  --> Walk-forward testing, relative strength, liquidity & data quality
 5. Risk Layer      --> Portfolio heat, risk-budget sizing & tracked outcomes
"""

import io
import json
import math
import os
import tempfile
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

# ============================================================
# 🔑 CONFIGURATION & ENVIRONMENT
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v17.0"
BOT_TAGLINE = "NSE-Wide Price Action, Research & Risk Screener"

MEMORY_FILE = "artha_memory.json"

TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL_INR", "100000"))
MAX_PORTFOLIO_HEAT = float(os.environ.get("MAX_PORTFOLIO_HEAT_PCT", "5"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.5"))
MAX_POSITION_VALUE_PCT = float(os.environ.get("MAX_POSITION_VALUE_PCT", "25"))
MIN_AVG_DAILY_TURNOVER_INR = float(os.environ.get("MIN_AVG_DAILY_TURNOVER_INR", "20000000"))
MAX_POSITIONS_ACTIVE = 3
MAX_OUTCOME_BARS = 20
BACKTEST_COST_BPS = float(os.environ.get("BACKTEST_COST_BPS", "20"))

# ============================================================
# 🧠 MEMORY MANAGEMENT SYSTEM
# ============================================================
def load_memory():
    defaults = {
        "version": BOT_VERSION.lstrip("v"),
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_picks": 0,
        "total_wins": 0,
        "total_losses": 0,
        "pending_evaluations": {},
        "completed_trades": [],
        "expired_signals": [],
        "failed_stocks": {},
        "nse_universe": [],
        "nse_universe_updated_at": None,
        "nse_source_counts": {},
        "research_summary": {},
        "mode_stats": {
            "BREAKOUT": {"wins": 0, "losses": 0},
            "DIP_BUYING": {"wins": 0, "losses": 0},
            "DEFENSIVE": {"wins": 0, "losses": 0}
        }
    }
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            if isinstance(memory, dict):
                for key, value in defaults.items():
                    if key not in memory or (value is not None and
                                             not isinstance(memory[key], type(value))):
                        memory[key] = value
                return memory
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read memory file ({exc}); starting with fresh memory.")
    return defaults

def save_memory(memory):
    """Atomically persist picks and the cached NSE universe."""
    memory["version"] = BOT_VERSION.lstrip("v")
    memory["last_updated"] = datetime.now().isoformat()
    directory = os.path.dirname(os.path.abspath(MEMORY_FILE))
    fd, temp_path = tempfile.mkstemp(prefix="artha-memory-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, MEMORY_FILE)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================================================
# 🧮 PURE PYTHON TECHNICAL INDICATORS
# ============================================================
def get_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def get_sma(s, n): return s.rolling(n).mean()

def get_rsi(s, n=14):
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def get_macd(s):
    ema12 = get_ema(s, 12)
    ema26 = get_ema(s, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def get_bb(s, n=20, std=2):
    sma = s.rolling(n).mean()
    stdev = s.rolling(n).std()
    return sma + (std * stdev), sma, sma - (std * stdev)


def normalize_yfinance_columns(frame):
    """Normalize price column names for both single- and multi-ticker yfinance data."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        fields = {"open", "high", "low", "close", "adj close", "volume"}
        normalized.columns = [
            next((str(part).strip().lower() for part in column
                  if str(part).strip().lower() in fields), str(column[0]).lower())
            .replace(" ", "_") for column in normalized.columns
        ]
    else:
        normalized.columns = [str(column).strip().lower().replace(" ", "_")
                              for column in normalized.columns]
    return normalized


def extract_ticker_frame(downloaded, ticker, requested_tickers=None):
    """Extract an individual symbol from a batched yfinance DataFrame."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        target = ticker.strip().upper()
        for level in range(downloaded.columns.nlevels):
            matches = {str(value).strip().upper(): value
                       for value in downloaded.columns.get_level_values(level)}
            if target in matches:
                try:
                    frame = downloaded.xs(matches[target], axis=1, level=level, drop_level=True)
                    return normalize_yfinance_columns(frame)
                except (KeyError, ValueError):
                    return pd.DataFrame()
        return pd.DataFrame()
    if requested_tickers and (len(requested_tickers) != 1 or
                              requested_tickers[0].upper() != ticker.upper()):
        return pd.DataFrame()
    return normalize_yfinance_columns(downloaded)


def has_usable_price_history(frame, minimum_rows=200, require_volume=True):
    required = {"open", "high", "low", "close"}
    if require_volume:
        required.add("volume")
    if frame is None or len(frame) < minimum_rows or not required.issubset(frame.columns):
        return False
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if prices.isna().any().any() or (prices <= 0).any().any():
        return False
    if require_volume:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        if volume.isna().any() or (volume < 0).any():
            return False
    open_, high, low, close = (prices[name] for name in ("open", "high", "low", "close"))
    if (high < pd.concat([open_, close, low], axis=1).max(axis=1)).any():
        return False
    if (low > pd.concat([open_, close, high], axis=1).min(axis=1)).any():
        return False
    return True


def average_daily_turnover(frame, window=20):
    """Mean traded value in INR; used as a conservative liquidity floor."""
    if frame is None or len(frame) < window or not {"close", "volume"}.issubset(frame.columns):
        return 0.0
    traded_value = (pd.to_numeric(frame["close"], errors="coerce") *
                    pd.to_numeric(frame["volume"], errors="coerce"))
    value = traded_value.tail(window).mean()
    return float(value) if math.isfinite(float(value)) else 0.0


def has_suspect_price_discontinuity(frame, threshold=0.45, recent_bars=30):
    """Flag extreme close-to-close jumps for review; adjusted Yahoo data is still fallible."""
    if frame is None or len(frame) < 2:
        return True
    close = pd.to_numeric(frame["close"], errors="coerce").tail(recent_bars)
    returns = close.pct_change().dropna()
    return bool((returns.abs() > threshold).any())


def is_history_stale(frame, max_age_days=7, today=None):
    if frame is None or frame.empty:
        return True
    try:
        last_date = pd.Timestamp(frame.index[-1]).date()
        reference = today or datetime.now().date()
        return (reference - last_date).days > max_age_days
    except (TypeError, ValueError, OverflowError):
        return True


def calc_relative_strength(stock_close, benchmark_close, lookback=60):
    """Return stock minus benchmark total return over aligned daily observations, in pct points."""
    if stock_close is None or benchmark_close is None:
        return None
    aligned = pd.concat([
        pd.to_numeric(stock_close, errors="coerce").rename("stock"),
        pd.to_numeric(benchmark_close, errors="coerce").rename("benchmark"),
    ], axis=1).dropna()
    if len(aligned) <= lookback:
        return None
    stock_return = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[-lookback - 1] - 1) * 100
    benchmark_return = (aligned["benchmark"].iloc[-1] /
                        aligned["benchmark"].iloc[-lookback - 1] - 1) * 100
    return float(stock_return - benchmark_return)


NSE_MAINBOARD_URLS = [
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://www.nseindia.com/content/equities/EQUITY_L.csv",
]
NSE_SME_URLS = [
    "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
    "https://www.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
]
NSE_MIN_MAINBOARD_ROWS = 1000
NSE_BATCH_SIZE = 40


def parse_nse_equity_master(csv_bytes):
    """Parse a current NSE equity master file into unique Yahoo Finance tickers."""
    table = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding="utf-8-sig")
    columns = {str(name).strip().upper().replace("_", " "): name for name in table.columns}
    symbol_col = columns.get("SYMBOL") or columns.get("SYMBOL NAME")
    if not symbol_col:
        raise ValueError("NSE file has no SYMBOL column")
    symbols = []
    for value in table[symbol_col].dropna().astype(str):
        symbol = value.strip().upper()
        if symbol and symbol not in {"NAN", "NONE"}:
            symbols.append(symbol)
    return sorted({f"{symbol}.NS" for symbol in symbols})


def _download_nse_master(session, urls, required=True):
    """Try official NSE mirrors with short retries before using the cached list."""
    if isinstance(urls, str):
        urls = [urls]
    errors = []
    for url in urls:
        for attempt in range(2):
            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()
                symbols = parse_nse_equity_master(response.content)
                if symbols and (not required or len(symbols) >= NSE_MIN_MAINBOARD_ROWS):
                    return symbols
                raise ValueError(f"NSE master returned only {len(symbols)} symbols: {url}")
            except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
                errors.append(f"{url}: {exc}")
                if attempt == 0:
                    time.sleep(0.5)
    detail = "; ".join(errors[-3:]) or "empty response"
    if required:
        raise RuntimeError(f"Could not download required NSE master: {detail}")
    print(f"[NSE MASTER NOTICE]: Optional SME list unavailable: {detail}")
    return []


def fetch_nse_universe():
    """Load NSE mainboard and SME equity symbols from NSE's public CSV masters."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    })
    mainboard = _download_nse_master(session, NSE_MAINBOARD_URLS, required=True)
    if len(mainboard) < NSE_MIN_MAINBOARD_ROWS:
        raise ValueError(f"NSE mainboard list unexpectedly small ({len(mainboard)} symbols)")
    sme = _download_nse_master(session, NSE_SME_URLS, required=False)
    universe = sorted(set(mainboard).union(sme))
    counts = {"mainboard": len(mainboard), "sme": len(sme), "unique_total": len(universe),
              "source": "NSE official equity masters"}
    return universe, counts


# ============================================================
def detect_market_regime(nifty=None):
    """Determine market state from NIFTY history; missing data fails closed."""
    try:
        if nifty is None:
            nifty = normalize_yfinance_columns(
                yf.download("^NSEI", period="1y", progress=False, timeout=20))
        if not has_usable_price_history(nifty, minimum_rows=200, require_volume=False):
            raise ValueError("NIFTY history is empty, short, or invalid")
        close = nifty['close'].astype(float)
        
        ema20  = get_ema(close, 20).iloc[-1]
        ema50  = get_ema(close, 50).iloc[-1]
        ema200 = get_ema(close, 200).iloc[-1]
        curr   = close.iloc[-1]
        
        change_20d = ((curr - close.iloc[-20]) / close.iloc[-20]) * 100
        rsi_val    = get_rsi(close, 14).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100

        # DECISION MATRIX
        if curr > ema20 > ema50 > ema200 and change_20d > 1.5:
            regime = "STRONG_BULL"
            strategy = "BREAKOUT"
            desc = "Full Momentum & VCP Breakouts Active"
        elif curr > ema50 and change_20d > -1.0:
            regime = "SIDEWAYS_BULL"
            strategy = "DIP_BUYING"
            desc = "Buy-The-Dip Mode Active (Quality Bounces at Support)"
        elif curr < ema50 and curr > ema200:
            regime = "SIDEWAYS_BEAR"
            strategy = "DIP_BUYING"
            desc = "Strict Mean Reversion Mode (Deep Support Bounces Only)"
        else:
            regime = "BEAR"
            strategy = "DEFENSIVE"
            desc = "Capital Protection Mode (Gold ETFs & Safe Assets Only)"

        return {
            "regime": regime,
            "strategy": strategy,
            "desc": desc,
            "nifty_price": round(curr, 2),
            "change_20d": round(change_20d, 2),
            "rsi": round(rsi_val, 1),
            "volatility": round(volatility, 2)
        }
    except Exception as exc:
        print(f"Market regime unavailable: {exc}")
        return {
            "regime": "DATA_UNAVAILABLE",
            "strategy": "NO_TRADE",
            "desc": "Market data unavailable; no entry strategy activated",
            "nifty_price": 0, "change_20d": 0, "rsi": 0, "volatility": 0
        }


SECTOR_PROXY_BY_NAME = {
    "IT": "^CNXIT", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG", "METAL": "^CNXMETAL",
    "REALTY": "^CNXREALTY", "ENERGY": "^CNXENERGY",
}
SECTOR_MEMBERS = {
    "IT": set("TCS INFY HCLTECH WIPRO TECHM LTIM PERSISTENT COFORGE MPHASIS LTTS TATAELXSI KPITTECH CYIENT ZENSAR BIRLASOFT".split()),
    "BANK": set("HDFCBANK ICICIBANK SBIN AXISBANK KOTAKBANK INDUSINDBK BANKBARODA PNB FEDERALBNK IDFCFIRSTB AUBANK CANBK UNIONBANK YESBANK".split()),
    "AUTO": set("MARUTI TATAMOTORS M&M BAJAJ-AUTO HEROMOTOCO EICHERMOT TVSMOTOR ASHOKLEY".split()),
    "PHARMA": set("SUNPHARMA DRREDDY CIPLA DIVISLAB AUROPHARMA TORNTPHARM ALKEM GLENMARK LAURUSLABS BIOCON LUPIN".split()),
    "FMCG": set("HINDUNILVR ITC NESTLEIND BRITANNIA TATACONSUM DABUR GODREJCP MARICO COLPAL EMAMILTD PATANJALI".split()),
    "METAL": set("TATASTEEL JSWSTEEL HINDALCO VEDL SAIL JINDALSTEL NMDC HINDCOPPER JSL".split()),
    "REALTY": set("DLF GODREJPROP PRESTIGE OBEROIRLTY PHOENIXLTD BRIGADE SOBHA LODHA".split()),
    "ENERGY": set("RELIANCE ONGC NTPC POWERGRID COALINDIA BPCL IOC GAIL TATAPOWER ADANIGREEN".split()),
}


def sector_for_ticker(ticker):
    symbol = ticker[:-3] if ticker.endswith(".NS") else ticker
    return next((sector for sector, members in SECTOR_MEMBERS.items()
                 if symbol in members), None)


def fetch_sector_histories():
    """Best-effort Yahoo sector-index history; missing proxies fall back to NIFTY RS."""
    tickers = list(SECTOR_PROXY_BY_NAME.values())
    try:
        batch = yf.download(tickers=tickers, period="1y", progress=False, timeout=25,
                            group_by="ticker", auto_adjust=True, threads=True)
    except Exception as exc:
        print(f"[SECTOR CONTEXT]: sector indices unavailable ({type(exc).__name__}); using NIFTY only.")
        return {}
    result = {}
    for sector, ticker in SECTOR_PROXY_BY_NAME.items():
        frame = extract_ticker_frame(batch, ticker, requested_tickers=tickers)
        if not frame.empty and "close" in frame:
            result[sector] = frame["close"].astype(float)
    return result


def qualify_relative_strength(pick, frame, benchmark_close, sector_histories):
    market_rs = calc_relative_strength(frame["close"], benchmark_close)
    if market_rs is None or market_rs < -5.0:
        return False
    sector = sector_for_ticker(pick["ticker"])
    sector_rs = None
    if sector and sector in sector_histories:
        sector_rs = calc_relative_strength(frame["close"], sector_histories[sector])
        if sector_rs is not None and sector_rs < -5.0:
            return False
    pick["market_rs_pct"] = round(market_rs, 2)
    pick["sector"] = sector or "Unmapped"
    pick["sector_rs_pct"] = round(sector_rs, 2) if sector_rs is not None else None
    if market_rs > 0:
        pick["score"] = min(100, pick["score"] + 5)
        pick["signals"].append("Outperforming NIFTY over 60 sessions")
    if sector_rs is not None and sector_rs > 0:
        pick["score"] = min(100, pick["score"] + 5)
        pick["signals"].append(f"Outperforming {sector} index over 60 sessions")
    return True

# ============================================================
# 🎯 2. STRATEGY A: BREAKOUT & VCP SCANNER (For Bull Markets)
# ============================================================
def scan_breakout(df, ticker_clean):
    try:
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume']
        
        c = len(df) - 1
        curr = close.iloc[c]
        
        ema20  = get_ema(close, 20)
        ema50  = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi    = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr    = get_atr(high, low, close, 14).iloc[c]
        
        # Hard Filters
        if not (curr > ema20.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
        if not (55 <= rsi.iloc[c] <= 72): return None
        if not (macd.iloc[c] > msig.iloc[c]): return None
        
        avg_vol = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 1
        if vol_ratio < 1.8: return None
        
        high_20 = high.iloc[-25:-2].max()
        if curr < high_20 * 0.99: return None  # Must be at or above 20-day high
        
        # Scoring
        score = 70
        signals = ["20D Breakout"]
        
        if vol_ratio >= 3.0:
            score += 15
            signals.append("Heavy Volume Surge")
        if 60 <= rsi.iloc[c] <= 68:
            score += 10
            signals.append("Sweet Spot RSI")
            
        sl = round(max(ema20.iloc[c] * 0.985, curr - (1.5 * atr)), 2)
        risk = curr - sl
        if risk <= 0 or (risk / curr) * 100 > 4.5: return None
        
        tgt1 = round(curr + (risk * 2.0), 2)
        tgt2 = round(curr + (risk * 3.5), 2)
        
        return {
            "ticker": ticker_clean,
            "strategy": "BREAKOUT",
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1),
            "sl": sl, "tgt1": tgt1, "tgt2": tgt2,
            "risk_pct": round((risk / curr) * 100, 2),
            "rr_ratio": 2.0,
            "signals": signals
        }
    except:
        return None

# ============================================================
# 🎯 3. STRATEGY B: DIP-BUYING & MEAN REVERSION (For Sideways)
# ============================================================
def scan_dip_buying(df, ticker_clean):
    """Finds high-quality companies bouncing off key support in sideways markets"""
    try:
        close = df['close']
        high  = df['high']
        low   = df['low']
        vol   = df['volume']
        
        c = len(df) - 1
        curr = close.iloc[c]
        
        ema20  = get_ema(close, 20)
        ema50  = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi    = get_rsi(close, 14)
        atr    = get_atr(high, low, close, 14).iloc[c]
        bb_up, bb_mid, bb_low = get_bb(close, 20)
        
        # 1. Long-term Trend Check: Stock MUST be structurally healthy (above 200 EMA)
        if curr < ema200.iloc[c]: return None
        
        # 2. Dip Condition: Price pulled back to 50 EMA OR touched lower Bollinger Band in last 3 days
        near_ema50 = abs(curr - ema50.iloc[c]) / curr <= 0.025
        near_bb_low = low.iloc[-3:].min() <= bb_low.iloc[-3:].max() * 1.01
        
        if not (near_ema50 or near_bb_low): return None
        
        # 3. Oversold / Reversal Signal: RSI was low (< 42) and is starting to turn UP
        rsi_oversold = rsi.iloc[c-2] < 42 or rsi.iloc[c-1] < 42 or rsi.iloc[c] < 42
        rsi_turning_up = rsi.iloc[c] > rsi.iloc[c-1]
        
        if not (rsi_oversold and rsi_turning_up): return None
        
        # 4. Confirmation Candle: Green candle bouncing off support
        green_candle = close.iloc[c] > df['open'].iloc[c]
        if not green_candle: return None
        
        # 5. Volume Confirmation: Dip-buying volume returning
        avg_vol = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 1
        
        # Scoring
        score = 75
        signals = ["Support Bounce"]
        
        if near_ema50: signals.append("50 EMA Support")
        if near_bb_low: signals.append("Bollinger Band Oversold")
        if vol_ratio > 1.4:
            score += 10
            signals.append("Volume Reversal")
            
        sl = round(min(low.iloc[-3:].min() * 0.99, curr - (1.5 * atr)), 2)
        risk = curr - sl
        if risk <= 0 or (risk / curr) * 100 > 4.0: return None
        
        # Targets for Dip-Buying: Bounce back to 20 EMA or Upper Bollinger Band
        tgt1 = round(curr + (risk * 1.8), 2)
        tgt2 = round(curr + (risk * 3.0), 2)
        
        return {
            "ticker": ticker_clean,
            "strategy": "DIP_BUYING",
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1),
            "sl": sl, "tgt1": tgt1, "tgt2": tgt2,
            "risk_pct": round((risk / curr) * 100, 2),
            "rr_ratio": 1.8,
            "signals": signals
        }
    except:
        return None


def scan_price_action(df, ticker_clean):
    """Find bullish breakouts, engulfing reversals, and support rejections from candles."""
    try:
        if df is None or len(df) < 60:
            return None
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        current, previous = float(close.iloc[-1]), float(close.iloc[-2])
        current_open = float(open_.iloc[-1])
        current_high, current_low = float(high.iloc[-1]), float(low.iloc[-1])
        if not all(math.isfinite(float(value)) for value in
                   (current, previous, current_open, current_high, current_low)):
            return None
        if current <= 0 or current_high <= current_low:
            return None

        atr = float(get_atr(high, low, close, 14).iloc[-1])
        ema50 = float(get_ema(close, 50).iloc[-1])
        if not math.isfinite(atr) or atr <= 0 or current <= ema50:
            return None

        avg_volume = float(volume.rolling(20).mean().iloc[-1])
        latest_volume = float(volume.iloc[-1])
        vol_ratio = latest_volume / avg_volume if avg_volume > 0 else 0.0
        close_location = (current - current_low) / (current_high - current_low)
        resistance = float(high.iloc[-21:-1].max())
        support = float(low.iloc[-23:-3].min())
        previous_open = float(open_.iloc[-2])
        previous_close = float(close.iloc[-2])
        signals = []

        breakout = (current > resistance * 1.002 and previous <= resistance * 1.01
                    and close_location >= 0.70 and vol_ratio >= 1.2)
        bullish_engulfing = (current > current_open and previous_close < previous_open
                             and current >= previous_open and current_open <= previous_close
                             and close_location >= 0.65 and vol_ratio >= 1.0)
        recent_support_test = float(low.iloc[-3:].min()) <= support + 0.5 * atr
        support_rejection = (recent_support_test and current > current_open
                             and current > previous and close_location >= 0.70
                             and vol_ratio >= 1.0)

        score = 68
        if breakout:
            signals.append("20-session resistance breakout")
            score += 15
        if bullish_engulfing:
            signals.append("Bullish engulfing candle")
            score += 12
        if support_rejection:
            signals.append("Support rejection / higher close")
            score += 10
        if not signals:
            return None
        if close_location >= 0.85:
            signals.append("Strong close near candle high")
            score += 5
        if vol_ratio >= 2.0:
            signals.append("Above-average volume confirmation")
            score += 5

        structural_level = resistance if breakout else support
        stop = max(structural_level - 0.35 * atr, current - 1.6 * atr)
        risk = current - stop
        risk_pct = (risk / current) * 100
        if risk <= 0 or risk_pct > 5.0:
            return None
        target1 = round(current + 1.5 * risk, 2)
        target2 = round(current + 2.5 * risk, 2)
        return {
            "ticker": ticker_clean,
            "strategy": "PRICE_ACTION",
            "score": min(round(score, 1), 100.0),
            "price": round(current, 2),
            "rsi": round(float(get_rsi(close, 14).iloc[-1]), 1),
            "vol_ratio": round(vol_ratio, 1),
            "sl": round(stop, 2), "tgt1": target1, "tgt2": target2,
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": 1.5,
            "signals": signals,
        }
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
        return None


def walk_forward_backtest_price_action(frame, benchmark_close=None, sector_close=None,
                                       max_holding=MAX_OUTCOME_BARS):
    """Fixed-rule, next-session-entry historical test; conservative stop-first daily fills."""
    empty = {"trades": 0, "wins": 0, "win_rate_pct": 0.0, "average_r": 0.0,
             "profit_factor": 0.0, "net_r": 0.0, "gross_profit_r": 0.0,
             "gross_loss_r": 0.0}
    if frame is None or len(frame) < 100 or not has_usable_price_history(
            frame, minimum_rows=100, require_volume=True):
        return empty
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    true_range = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    resistance = high.shift(1).rolling(20).max()
    support = low.shift(3).rolling(20).min()
    vol_ratio = volume / volume.rolling(20).mean()
    bar_range = (high - low).replace(0, float("nan"))
    close_location = (close - low) / bar_range
    breakout = ((close > resistance * 1.002) & (close.shift(1) <= resistance.shift(1) * 1.01)
                & (close_location >= 0.70) & (vol_ratio >= 1.2))
    engulfing = ((close > open_) & (close.shift(1) < open_.shift(1))
                 & (close >= open_.shift(1)) & (open_ <= close.shift(1))
                 & (close_location >= 0.65) & (vol_ratio >= 1.0))
    support_test = low.rolling(3).min() <= support + 0.5 * atr
    rejection = (support_test & (close > open_) & (close > close.shift(1))
                 & (close_location >= 0.70) & (vol_ratio >= 1.0))
    trend_ok = close > ema50
    traded_value = close * volume
    trend_ok &= traded_value.rolling(20).mean() >= MIN_AVG_DAILY_TURNOVER_INR
    if benchmark_close is not None:
        benchmark = pd.to_numeric(benchmark_close, errors="coerce").reindex(frame.index).ffill()
        stock_rs = close.pct_change(60) * 100
        market_rs = benchmark.pct_change(60) * 100
        trend_ok &= (stock_rs - market_rs) >= -5.0
    if sector_close is not None:
        sector = pd.to_numeric(sector_close, errors="coerce").reindex(frame.index).ffill()
        sector_rs = sector.pct_change(60) * 100
        trend_ok &= (close.pct_change(60) * 100 - sector_rs) >= -5.0
    daily_move = close.pct_change().abs()
    # Guard only on information known at the signal close; never use next-session data to filter signals.
    stable_bar = daily_move.le(0.45)
    signals = (breakout | engulfing | rejection) & trend_ok & stable_bar
    trade_rs = []
    next_available = 0
    for i in range(60, len(frame) - 1):
        if i < next_available or not bool(signals.iloc[i]):
            continue
        current = float(close.iloc[i])
        current_atr = float(atr.iloc[i])
        if not math.isfinite(current_atr) or current_atr <= 0:
            continue
        is_breakout = bool(breakout.iloc[i])
        level = float(resistance.iloc[i] if is_breakout else support.iloc[i])
        stop = max(level - 0.35 * current_atr, current - 1.6 * current_atr)
        entry = float(open_.iloc[i + 1]) * (1 + BACKTEST_COST_BPS / 20000)
        risk = entry - stop
        if risk <= 0 or risk / entry > 0.05:
            continue
        target = entry + 1.5 * risk
        exit_price, bars_held = float(close.iloc[-1]), 0
        resolved = False
        full_horizon_available = i + 1 + max_holding <= len(frame)
        end = min(len(frame), i + 1 + max_holding)
        for j in range(i + 1, end):
            bars_held += 1
            day_open, day_low, day_high = float(open_.iloc[j]), float(low.iloc[j]), float(high.iloc[j])
            # If both barriers occur intraday, assume stop was hit first (conservative daily-bar rule).
            if day_open <= stop or day_low <= stop:
                exit_price = min(day_open, stop)
                resolved = True
                break
            if day_high >= target:
                exit_price = target
                resolved = True
                break
        if not resolved:
            if not full_horizon_available:
                continue  # Right-censored trades are omitted, not treated as closed.
            j = end - 1
            exit_price = float(close.iloc[j])
        exit_price *= (1 - BACKTEST_COST_BPS / 20000)
        trade_rs.append((exit_price - entry) / risk)
        next_available = max(i + 1, j + 1)

    if not trade_rs:
        return empty
    wins = sum(value > 0 for value in trade_rs)
    gross_profit = sum(value for value in trade_rs if value > 0)
    gross_loss = abs(sum(value for value in trade_rs if value < 0))
    equity, peak, max_drawdown = 0.0, 0.0, 0.0
    for value in trade_rs:
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return {
        "trades": len(trade_rs), "wins": wins,
        "win_rate_pct": round(100 * wins / len(trade_rs), 1),
        "average_r": round(sum(trade_rs) / len(trade_rs), 3),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 99.0,
        "net_r": round(sum(trade_rs), 2), "gross_profit_r": round(gross_profit, 3),
        "gross_loss_r": round(gross_loss, 3), "max_drawdown_r": round(max_drawdown, 2),
    }


def calculate_position_size(price, stop, capital, available_heat):
    """Risk-budget shares, bounded by per-position capital and remaining portfolio heat."""
    risk_per_share = float(price) - float(stop)
    if risk_per_share <= 0 or price <= 0 or capital <= 0:
        return {"shares": 0, "risk_amount": 0.0, "allocation": 0.0}
    budget = min(capital * RISK_PER_TRADE_PCT / 100, max(0.0, available_heat))
    max_value = capital * MAX_POSITION_VALUE_PCT / 100
    shares = min(int(budget / risk_per_share), int(max_value / price))
    allocation = shares * float(price)
    return {"shares": shares, "risk_amount": round(shares * risk_per_share, 2),
            "allocation": round(allocation, 2)}


def evaluate_pending_picks(memory, price_frames):
    """Resolve tracked suggestions against subsequent daily bars, stop-first on ambiguous bars."""
    pending = memory.setdefault("pending_evaluations", {})
    completed = memory.setdefault("completed_trades", [])
    resolved_count = 0
    for pick_id, record in list(pending.items()):
        symbol = record.get("ticker", "")
        frame = price_frames.get(symbol)
        if frame is None or frame.empty:
            continue
        try:
            as_of = pd.Timestamp(record.get("as_of_date") or record.get("date")).date()
            bars = frame.loc[[pd.Timestamp(index).date() > as_of for index in frame.index]]
        except (TypeError, ValueError):
            continue
        for index, bar in bars.iterrows():
            record["bars_held"] = int(record.get("bars_held", 0)) + 1
            entry, stop, target = (float(record[k]) for k in
                                   ("entry_price", "sl", "tgt1"))
            day_open, day_low, day_high = (float(bar[k]) for k in ("open", "low", "high"))
            exit_price, outcome = None, None
            if day_open <= stop or day_low <= stop:
                exit_price, outcome = min(day_open, stop), "STOP"
            elif day_high >= target:
                exit_price, outcome = target, "TARGET1"
            elif record["bars_held"] >= MAX_OUTCOME_BARS:
                exit_price, outcome = float(bar["close"]), "TIME_EXIT"
            if outcome:
                risk = max(entry - stop, 0.01)
                record.update({"outcome": outcome, "exit_price": round(exit_price, 2),
                               "resolved_at": pd.Timestamp(index).strftime("%Y-%m-%d"),
                               "r_multiple": round((exit_price - entry) / risk, 3)})
                completed.append(record.copy())
                del pending[pick_id]
                resolved_count += 1
                break
    memory["completed_trades"] = completed[-2000:]
    return resolved_count


def expire_unverifiable_picks(memory, max_age_days=45, today=None):
    """Release risk for old picks with no usable history without inventing an outcome."""
    today = today or datetime.now().date()
    pending = memory.setdefault("pending_evaluations", {})
    expired = memory.setdefault("expired_signals", [])
    expired_count = 0
    for pick_id, record in list(pending.items()):
        try:
            signal_date = pd.Timestamp(record.get("as_of_date") or record.get("date")).date()
        except (TypeError, ValueError):
            continue
        if (today - signal_date).days > max_age_days:
            expired.append({**record, "status": "DATA_UNAVAILABLE",
                            "expired_at": today.isoformat()})
            del pending[pick_id]
            expired_count += 1
    memory["expired_signals"] = expired[-500:]
    return expired_count


def completed_trade_summary(records, lookback=30):
    recent = records[-lookback:]
    if not recent:
        return {"trades": 0, "win_rate_pct": 0.0, "average_r": 0.0}
    r_values = [float(record.get("r_multiple", 0)) for record in recent]
    return {"trades": len(recent),
            "win_rate_pct": round(100 * sum(value > 0 for value in r_values) / len(r_values), 1),
            "average_r": round(sum(r_values) / len(r_values), 3)}


def completed_trade_group_summary(records, grouping="strategy", lookback=100):
    """Show outcome quality grouped by strategy and recognizable price-action pattern."""
    grouped = {}
    for record in records[-lookback:]:
        if grouping == "strategy":
            keys = [str(record.get("strategy", "UNKNOWN"))]
        else:
            signals = [str(signal).lower() for signal in record.get("signals", [])]
            keys = []
            if any("breakout" in signal for signal in signals):
                keys.append("BREAKOUT")
            if any("engulf" in signal for signal in signals):
                keys.append("ENGULFING")
            if any("support" in signal for signal in signals):
                keys.append("SUPPORT_REJECTION")
            if not keys:
                keys.append(str(record.get("strategy", "OTHER")))
        value = float(record.get("r_multiple", 0.0))
        for key in set(keys):
            grouped.setdefault(key, []).append(value)
    return {key: {"trades": len(values),
                  "win_rate_pct": round(100 * sum(value > 0 for value in values) / len(values), 1),
                  "average_r": round(sum(values) / len(values), 3)}
            for key, values in sorted(grouped.items())}


def format_trade_groups(groups):
    if not groups:
        return "no resolved samples"
    return "; ".join(f"{name}: n={stats['trades']}, win={stats['win_rate_pct']}%, "
                      f"avg={stats['average_r']}R" for name, stats in groups.items())


def aggregate_backtest_summaries(summaries):
    trades = sum(item.get("trades", 0) for item in summaries)
    wins = sum(item.get("wins", 0) for item in summaries)
    gross_profit = sum(item.get("gross_profit_r", 0.0) for item in summaries)
    gross_loss = sum(item.get("gross_loss_r", 0.0) for item in summaries)
    net_r = sum(item.get("net_r", 0.0) for item in summaries)
    return {"trades": trades, "wins": wins,
            "win_rate_pct": round(100 * wins / trades, 1) if trades else 0.0,
            "average_r": round(net_r / trades, 3) if trades else 0.0,
            "profit_factor": (round(gross_profit / gross_loss, 2) if gross_loss else
                              99.0 if gross_profit else 0.0),
            "net_r": round(net_r, 2)}

# ============================================================
# 🎯 4. STRATEGY C: DEFENSIVE & GOLD ALLOCATOR (For Bear Markets)
# ============================================================
def scan_defensive(price_frames=None):
    """Scans defensive ETFs and optionally returns their bars for outcome tracking."""
    defensive_universe = ["GOLDBEES.NS", "SILVERBEES.NS", "GOLDSHARE.NS", "HDFCGOLD.NS"]
    picks = []
    
    for ticker in defensive_universe:
        try:
            df = normalize_yfinance_columns(yf.download(
                ticker, period="3mo", progress=False, timeout=20, auto_adjust=True))
            if price_frames is not None and not df.empty:
                price_frames[ticker.replace(".NS", "")] = df

            close = df['close']
            c = len(df) - 1
            curr = close.iloc[c]
            
            ema20 = get_ema(close, 20).iloc[c]
            rsi = get_rsi(close, 14).iloc[c]
            
            if curr > ema20 and rsi > 45:
                sl = round(curr * 0.98, 2)
                risk = curr - sl
                picks.append({
                    "ticker": ticker.replace(".NS", ""),
                    "strategy": "DEFENSIVE",
                    "score": 80.0,
                    "price": round(curr, 2),
                    "rsi": round(rsi, 1),
                    "vol_ratio": 1.0,
                    "sl": sl,
                    "tgt1": round(curr * 1.04, 2),
                    "tgt2": round(curr * 1.08, 2),
                    "risk_pct": 2.0,
                    "rr_ratio": 2.0,
                    "signals": ["Gold/Hedge Protection", "Capital Preservation"]
                })
        except:
            continue
            
    return picks

# ============================================================
# 📥 STOCK UNIVERSE GENERATOR
# ============================================================
def get_legacy_nse_universe():
    """Emergency fallback only; normal scans use the current NSE masters."""
    liquid_stocks = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","KOTAKBANK",
        "LT","AXISBANK","BAJFINANCE","MARUTI","ASIANPAINT","TITAN","SUNPHARMA",
        "TATAMOTORS","WIPRO","ULTRACEMCO","NESTLEIND","POWERGRID","NTPC","ONGC",
        "COALINDIA","JSWSTEEL","TATASTEEL","HCLTECH","TECHM","ADANIENT","ADANIPORTS",
        "BAJAJFINSV","BPCL","CIPLA","DRREDDY","EICHERMOT","GRASIM","HAVELLS",
        "HEROMOTOCO","HINDALCO","INDUSINDBK","M&M","SBIN","DIVISLAB","APOLLOHOSP",
        "LTIM","TATACONSUM","PIDILITIND","SBILIFE","SHREECEM","BRITANNIA","BAJAJ-AUTO",
        "DABUR","GODREJCP","MARICO","COLPAL","BERGEPAINT","AMBUJACEM","ACC","SIEMENS",
        "ABB","BOSCHLTD","MUTHOOTFIN","CHOLAFIN","MOTHERSON","BALKRISIND","PERSISTENT",
        "COFORGE","MPHASIS","LTTS","TATAELXSI","JUBLFOOD","DMART","IRCTC","DIXON",
        "VOLTAS","DEEPAKNTR","ATUL","PIIND","NAVINFLUOR","ALKYLAMINE","ZOMATO","NYKAA",
        "PAYTM","POLICYBZR","FEDERALBNK","IDFCFIRSTB","BANDHANBNK","RBLBANK","BANKBARODA",
        "PNB","CANBK","UNIONBANK","INDIANB","IOB","YESBANK","JKCEMENT","RAMCOCEM",
        "ADANIGREEN","TATAPOWER","JSWENERGY","CESC","TORNTPOWER","HAL","BEL","BHEL",
        "BEML","RVNL","IRFC","PFC","REC","HUDCO","IREDA","NAUKRI","INDIAMART",
        "JUSTDIAL","AFFLE","ROUTE","LALPATHLAB","METROPOLIS","MAXHEALTH","FORTIS",
        "APOLLOTYRE","MRF","CEAT","SCHAEFFLER","SKFINDIA","TIMKEN","GRINDWELL","ELGIEQUIP",
        "IPCALAB","AUROPHARMA","TORNTPHARM","ALKEM","GLENMARK","LAURUSLABS","BIOCON",
        "PFIZER","SANOFI","ABBOTINDIA","GILLETTE","3MINDIA","HONAUT","RADICO","TIPSINDLTD",
        "CDSL","BSE","CAMS","KFIN","MCX","ANGELONE","ICICIPRULI","HDFCLIFE","GICRE",
        "SHRIRAMFIN","MANAPPURAM","LICHSGFIN","MFSL","FINEORG","GALAXYSURF","TATACHEM",
        "GNFC","GSFC","CLEAN","NEOGEN","AARTI","VINATI","ASTRAL","SUPREMEIND",
        "CENTURYPLY","PRINCEPIPE","KPRMILL","WELSPUNLIV","RAYMOND","PAGEIND",
        "MANYAVAR","CAMPUS","METRO","BATA","RELAXO","DLF","GODREJPROP","PRESTIGE",
        "OBEROIRLTY","PHOENIXLTD","BRIGADE","SOBHA","LODHA","TIINDIA","LATENTVIEW",
        "RAILTEL","COCHINSHIP","MAZAGON","GRSE","DATAPATT","LICI","IEX","NHPC","SJVN",
        "JINDALSTEL","SAIL","NMDC","MOIL","VEDL","HINDCOPPER","APLAPOLLO","JSL",
        "KPITTECH","INTELLECT","BIRLASOFT","ZENSAR","TATATECH","CYIENT","TANLA",
        "INDHOTEL","LEMONTREE","CHALET","PATANJALI","EMAMILTD","JYOTHYLAB","UNOMINDA",
        "TRENT","WESTLIFE","DEVYANI","SAPPHIRE","POLYCAB","KEI","FINCABLES","VGUARD",
        "CROMPTON","AMBER","SYMPHONY","QUESS","TEAMLEASE","EXIDEIND","AMARAJABAT",
        "BHARATFORG","TVSMOTOR","ESCORTS","DALBHARAT","AUBANK","EQUITASBNK"
    ]
    return [f"{s}.NS" for s in sorted(set(liquid_stocks))]


def get_scan_universe(memory):
    """Refresh all NSE listed equity symbols, with a cached/legacy fail-soft fallback."""
    try:
        symbols, counts = fetch_nse_universe()
        memory["nse_universe"] = symbols
        memory["nse_universe_updated_at"] = datetime.now().isoformat()
        memory["nse_source_counts"] = counts
        print(f"[NSE UNIVERSE]: Refreshed all NSE equity symbols: {counts}")
        return symbols
    except Exception as exc:
        cached = memory.get("nse_universe", [])
        if cached:
            print(f"[NSE UNIVERSE WARNING]: Refresh failed ({exc}); using cached list "
                  f"from {memory.get('nse_universe_updated_at') or 'unknown date'}.")
            memory["nse_source_counts"]["source"] = "cached NSE master"
            return cached
        fallback = get_legacy_nse_universe()
        memory["nse_source_counts"] = {
            "source": "legacy fallback (partial universe)",
            "mainboard": 0, "sme": 0, "unique_total": len(fallback),
            "warning": str(exc),
        }
        print(f"[NSE UNIVERSE ERROR]: Official NSE master unavailable ({exc}); "
              f"using {len(fallback)} legacy symbols so the bot can still run.")
        return fallback

# ============================================================
# 📤 TELEGRAM MESSAGING ENGINE (Safe Text Formatting)
# ============================================================
def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing. Outputting to console.")
        print(msg)
        return
        
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        parts = [msg[i:i+3800] for i in range(0, len(msg), 3800)]
        for part in parts:
            requests.post(url, json={"chat_id": CHAT_ID, "text": part}, timeout=15)
            time.sleep(0.8)
    except Exception as e:
        print(f"Telegram Exception: {e}")

# ============================================================
# 🚀 MAIN ADAPTIVE ENGINE
# ============================================================
def main():
    print("=" * 60)
    print(f"{BOT_NAME} {BOT_VERSION} — {BOT_TAGLINE}")
    print(f"Execution Time: {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print("=" * 60)
    memory = load_memory()

    try:
        nifty = normalize_yfinance_columns(
            yf.download("^NSEI", period="1y", progress=False, timeout=25, auto_adjust=True))
    except Exception as exc:
        print(f"[MARKET DATA ERROR]: {type(exc).__name__}")
        nifty = pd.DataFrame()
    regime_info = detect_market_regime(nifty)
    strategy = regime_info["strategy"]
    benchmark_close = nifty["close"].astype(float) if "close" in nifty else None
    print(f"\n[REGIME DETECTED]: {regime_info['regime']}")
    print(f"[ACTIVE STRATEGY]: {strategy} ({regime_info['desc']})")
    sector_histories = fetch_sector_histories() if benchmark_close is not None else {}

    universe = get_scan_universe(memory)
    save_memory(memory)
    price_frames, backtest_summaries = {}, []
    results = scan_defensive(price_frames) if strategy == "DEFENSIVE" else []
    valid_data_count = invalid_data_count = download_error_count = 0
    illiquid_count = anomaly_count = stale_count = price_action_count = 0
    anomaly_symbols, stale_symbols = [], []
    download_error_types = {}
    print(f"\n[SCANNING]: {len(universe)} NSE symbols in batches of {NSE_BATCH_SIZE}; "
          f"strategy: {strategy}; minimum average turnover: Rs. {MIN_AVG_DAILY_TURNOVER_INR:,.0f}")

    processed = 0
    for start in range(0, len(universe), NSE_BATCH_SIZE):
        tickers = universe[start:start + NSE_BATCH_SIZE]
        try:
            downloaded = yf.download(
                tickers=tickers, period="1y", progress=False, timeout=30,
                group_by="ticker", auto_adjust=True, threads=True)
        except Exception as exc:
            download_error_count += len(tickers)
            error_type = type(exc).__name__
            download_error_types[error_type] = download_error_types.get(error_type, 0) + len(tickers)
            print(f"[YFINANCE BATCH ERROR]: {len(tickers)} symbols skipped ({error_type}).")
            processed += len(tickers)
            continue

        for ticker in tickers:
            processed += 1
            if processed % 200 == 0:
                print(f"  Processed {processed}/{len(universe)} | valid data: {valid_data_count} | "
                      f"setups: {len(results)}")
            df = extract_ticker_frame(downloaded, ticker, requested_tickers=tickers)
            if not has_usable_price_history(df, minimum_rows=200, require_volume=True):
                invalid_data_count += 1
                continue
            valid_data_count += 1
            ticker_clean = ticker[:-3] if ticker.endswith(".NS") else ticker
            price_frames[ticker_clean] = df
            if is_history_stale(df):
                stale_count += 1
                if len(stale_symbols) < 100:
                    stale_symbols.append(ticker_clean)
                continue

            if benchmark_close is not None:
                sector = sector_for_ticker(ticker_clean)
                sector_close = sector_histories.get(sector) if sector else None
                backtest_summaries.append(walk_forward_backtest_price_action(
                    df, benchmark_close=benchmark_close, sector_close=sector_close))

            if has_suspect_price_discontinuity(df):
                anomaly_count += 1
                if len(anomaly_symbols) < 100:
                    anomaly_symbols.append(ticker_clean)
                continue

            turnover = average_daily_turnover(df)
            if turnover < MIN_AVG_DAILY_TURNOVER_INR:
                illiquid_count += 1
                continue

            try:
                result = None
                if strategy == "BREAKOUT":
                    result = scan_breakout(df, ticker_clean)
                elif strategy == "DIP_BUYING":
                    result = scan_dip_buying(df, ticker_clean)
                if strategy in {"BREAKOUT", "DIP_BUYING"}:
                    price_action = scan_price_action(df, ticker_clean)
                    if price_action:
                        price_action_count += 1
                        if result:
                            result["score"] = min(100, result["score"] + 5)
                            result["signals"] = list(dict.fromkeys(
                                result["signals"] + price_action["signals"]))
                        else:
                            result = price_action
                if result:
                    result["avg_turnover_inr"] = round(turnover)
                    result["signals"].append("Liquidity passed 20-session turnover floor")
                    if benchmark_close is None or not qualify_relative_strength(
                            result, df, benchmark_close, sector_histories):
                        continue
                    results.append(result)
            except Exception as exc:
                error_type = type(exc).__name__
                download_error_types[error_type] = download_error_types.get(error_type, 0) + 1

    resolved_count = evaluate_pending_picks(memory, price_frames)
    expired_count = expire_unverifiable_picks(memory)
    backtest_summary = aggregate_backtest_summaries(backtest_summaries)
    memory["research_summary"] = {
        "asof": datetime.now().isoformat(),
        "price_action_walk_forward_1y": backtest_summary,
        "recent_outcomes": completed_trade_summary(memory.get("completed_trades", [])),
        "outcomes_by_strategy": completed_trade_group_summary(
            memory.get("completed_trades", []), "strategy"),
        "outcomes_by_pattern": completed_trade_group_summary(
            memory.get("completed_trades", []), "pattern"),
        "data_quality_review": {"extreme_recent_moves": anomaly_symbols, "stale_history": stale_symbols},
        "backtest_note": "Fixed rules; next-session entry; conservative stop-first; approximate 20 bps costs."
    }
    print(f"[NSE COVERAGE]: {valid_data_count}/{len(universe)} valid OHLCV; "
          f"{invalid_data_count} invalid; {download_error_count} failed batches.")
    print(f"[QUALITY FILTERS]: {illiquid_count} below turnover floor; {anomaly_count} extreme recent "
          f"price moves; {stale_count} stale histories.")
    print(f"[PRICE ACTION]: {price_action_count} patterns; {resolved_count} picks resolved; "
          f"{expired_count} unverifiable old picks expired without a performance result.")
    print(f"[WALK-FORWARD]: {backtest_summary}")
    if download_error_types:
        print(f"[SCAN ERROR TYPES]: {download_error_types}")

    results.sort(key=lambda item: item["score"], reverse=True)
    active_tickers = {record.get("ticker") for record in
                      memory.get("pending_evaluations", {}).values()}
    top_picks, used_risk = [], 0.0
    # Legacy picks lack a stored risk amount; reserve their historical 1.5% risk budget conservatively.
    pending_risk = sum(float(item.get("risk_amount", TOTAL_CAPITAL * 0.015)) for item in
                       memory.get("pending_evaluations", {}).values())
    available_heat = max(0.0, TOTAL_CAPITAL * MAX_PORTFOLIO_HEAT / 100 - pending_risk)
    for pick in results:
        if len(top_picks) >= MAX_POSITIONS_ACTIVE or pick["ticker"] in active_tickers:
            continue
        sizing = calculate_position_size(pick["price"], pick["sl"], TOTAL_CAPITAL,
                                         available_heat - used_risk)
        if sizing["shares"] < 1:
            continue
        pick.update(sizing)
        top_picks.append(pick)
        used_risk += sizing["risk_amount"]

    for pick in top_picks:
        if pick["ticker"] in price_frames and not price_frames[pick["ticker"]].empty:
            signal_date = str(pd.Timestamp(price_frames[pick["ticker"]].index[-1]).date())
        else:
            signal_date = datetime.now().date().isoformat()
        pick_id = f"{pick['ticker']}_{signal_date}"
        memory.setdefault("pending_evaluations", {})[pick_id] = {
            "ticker": pick["ticker"], "date": datetime.now().isoformat(),
            "as_of_date": signal_date, "entry_price": pick["price"],
            "sl": pick["sl"], "tgt1": pick["tgt1"], "tgt2": pick["tgt2"],
            "strategy": pick["strategy"], "score": pick["score"],
            "risk_amount": pick["risk_amount"], "shares": pick["shares"],
            "allocation": pick["allocation"], "signals": pick["signals"],
            "market_rs_pct": pick.get("market_rs_pct"),
            "sector": pick.get("sector"), "sector_rs_pct": pick.get("sector_rs_pct"),
            "avg_turnover_inr": pick.get("avg_turnover_inr"),
            "rsi": pick.get("rsi"), "vol_ratio": pick.get("vol_ratio"),
            "risk_pct": pick.get("risk_pct"), "bars_held": 0,
        }
        memory["total_picks"] = int(memory.get("total_picks", 0)) + 1
    save_memory(memory)

    today = datetime.now().strftime("%A, %d %b %Y")
    counts = memory.get("nse_source_counts", {})
    completed = memory.get("completed_trades", [])
    outcomes = completed_trade_summary(completed)
    by_strategy = completed_trade_group_summary(completed, "strategy", lookback=30)
    by_pattern = completed_trade_group_summary(completed, "pattern", lookback=30)
    msg = ("========================================\n"
           f"⚡ ARTHA {BOT_VERSION} | Pre-Market Report\n📅 {today}\n"
           "========================================\n\n"
           "[MARKET CONTEXT]\n"
           f"Regime: {regime_info['regime']} | Mode: {strategy}\n"
           f"NIFTY: Rs. {regime_info['nifty_price']} | 20D: {regime_info['change_20d']:+.2f}%\n"
           f"Universe: {counts.get('source', 'unknown')} | {len(universe)} symbols\n"
           f"Valid / invalid / failed: {valid_data_count} / {invalid_data_count} / {download_error_count}\n"
           f"Liquidity / anomaly / stale exclusions: {illiquid_count} / {anomaly_count} / {stale_count}\n\n"
           "[RESEARCH - 1Y WALK-FORWARD PRICE ACTION]\n"
           f"Trades: {backtest_summary['trades']} | Win rate: {backtest_summary['win_rate_pct']}% | "
           f"Avg: {backtest_summary['average_r']}R | PF: {backtest_summary['profit_factor']} | "
           f"Net: {backtest_summary['net_r']}R\n"
           "Estimate only; current-universe survivorship bias and approximate costs apply.\n"
           f"Recent tracked outcomes ({outcomes['trades']}): {outcomes['win_rate_pct']}% wins, "
           f"average {outcomes['average_r']}R; resolved: {resolved_count}; "
           f"expired without outcome: {expired_count}.\n"
           f"By strategy: {format_trade_groups(by_strategy)}\n"
           f"By pattern: {format_trade_groups(by_pattern)}\n\n"
           f"[SELECTION] Qualified: {len(results)} | Price action: {price_action_count} | "
           f"Selected: {len(top_picks)} | Available heat: Rs. {max(0, TOTAL_CAPITAL * MAX_PORTFOLIO_HEAT / 100 - pending_risk):,.0f}\n\n")

    if top_picks:
        msg += "[TOP RISK-SIZED SETUPS]\n"
        for rank, pick in enumerate(top_picks, 1):
            msg += (f"#{rank} {pick['ticker']} | {pick['strategy']} | Score {pick['score']}/100\n"
                    f"Entry Rs. {pick['price']} | Stop Rs. {pick['sl']} | "
                    f"Targets Rs. {pick['tgt1']} / Rs. {pick['tgt2']}\n"
                    f"Qty {pick['shares']} | Allocation Rs. {pick['allocation']:,.0f} | "
                    f"Risk Rs. {pick['risk_amount']:,.0f} ({pick['risk_pct']}%)\n"
                    f"20D turnover Rs. {pick.get('avg_turnover_inr', 0):,.0f} | "
                    f"RS vs NIFTY {pick.get('market_rs_pct', 'n/a')}% | "
                    f"{pick.get('sector', 'sector n/a')} RS {pick.get('sector_rs_pct', 'n/a')}%\n"
                    f"Signals: {', '.join(pick['signals'])}\n\n")
    else:
        msg += "No candidate cleared market, liquidity, data-quality and portfolio-risk filters today.\n\n"
    msg += ("[RISK / DATA NOTES]\n"
            f"Risk budget: {RISK_PER_TRADE_PCT}% of capital per idea; total heat cap "
            f"{MAX_PORTFOLIO_HEAT}%; max position {MAX_POSITION_VALUE_PCT}% of capital.\n"
            "Yahoo auto-adjusted OHLC is used; extreme recent jumps are excluded for review. "
            "Daily-bar outcomes assume stop-first if both barriers trade.\n"
            "Research results are not forecasts or guarantees. Verify prices and suitability independently.\n")
    send_telegram(msg)
    print(f"\n✅ ARTHA {BOT_VERSION} Run Complete!")

if __name__ == "__main__":
    main()
