"""
ARTHA v16.0 - NSE-Wide Price-Action Strategy Engine
Auto-Switches Between:
 1. Bull Mode     --> Breakouts & VCP Patterns
 2. Sideways Mode --> Dip-Buying & Mean Reversion (Quality Bounces)
 3. Bear Mode     --> Defensive Assets & Gold ETFs
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
BOT_VERSION = "v16.0"
BOT_TAGLINE = "NSE-Wide Adaptive Multi-Strategy Screener"

MEMORY_FILE = "artha_memory.json"

TOTAL_CAPITAL       = 100000  # Rs. 1,00,000 Base Capital
MAX_PORTFOLIO_HEAT  = 5       # Max 5% total risk at any time
MAX_POSITIONS_ACTIVE = 3       # Quality over quantity (Max 3 open positions)

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
        "failed_stocks": {},
        "nse_universe": [],
        "nse_universe_updated_at": None,
        "nse_source_counts": {},
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
    return True


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


def get_legacy_nse_universe():
    """Small emergency fallback retained only if NSE's master is unreachable and uncached."""
# ============================================================
def detect_market_regime():
    """Dynamically determines the market state and picks the active strategy"""
    try:
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

# ============================================================
# 🎯 4. STRATEGY C: DEFENSIVE & GOLD ALLOCATOR (For Bear Markets)
# ============================================================
def scan_defensive():
    """Scans Gold ETFs, Silver ETFs & Liquid assets during Bear Markets"""
    defensive_universe = ["GOLDBEES.NS", "SILVERBEES.NS", "GOLDSHARE.NS", "HDFCGOLD.NS"]
    picks = []
    
    for ticker in defensive_universe:
        try:
            df = yf.download(ticker, period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
                
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

    # STEP 1: DETECT MARKET REGIME & ACTIVE STRATEGY
    regime_info = detect_market_regime()
    regime      = regime_info["regime"]
    strategy    = regime_info["strategy"]
    
    print(f"\n[REGIME DETECTED]: {regime}")
    print(f"[ACTIVE STRATEGY]: {strategy} ({regime_info['desc']})")

    # STEP 2: Scan every active NSE equity in bounded price-download batches.
    universe = get_scan_universe(memory)
    # Persist a successfully refreshed NSE symbol list before the long market scan.
    save_memory(memory)
    results = scan_defensive() if strategy == "DEFENSIVE" else []
    valid_data_count = 0
    invalid_data_count = 0
    download_error_count = 0
    price_action_count = 0
    download_error_types = {}
    print(f"\n[SCANNING]: {len(universe)} NSE symbols in batches of {NSE_BATCH_SIZE}; "
          f"active setup strategy: {strategy}")

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

            try:
                ticker_clean = ticker[:-3] if ticker.endswith(".NS") else ticker
                if strategy == "BREAKOUT":
                    result = scan_breakout(df, ticker_clean)
                elif strategy == "DIP_BUYING":
                    result = scan_dip_buying(df, ticker_clean)
                else:
                    result = None

                # Price action is an additional selection path and confirmation layer.
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
                    results.append(result)
            except Exception as exc:
                error_type = type(exc).__name__
                download_error_types[error_type] = download_error_types.get(error_type, 0) + 1

    print(f"[NSE COVERAGE]: {valid_data_count}/{len(universe)} stocks had valid one-year OHLCV; "
          f"{invalid_data_count} empty/short/invalid; {download_error_count} in failed batches.")
    print(f"[PRICE ACTION]: {price_action_count} stocks passed candle/structure checks.")
    if download_error_types:
        print(f"[SCAN ERROR TYPES]: {download_error_types}")

    # Rank & Pick Top Setups
    results.sort(key=lambda x: x["score"], reverse=True)
    top_picks = results[:MAX_POSITIONS_ACTIVE]

    # Save picks to Memory for tracking
    for p in top_picks:
        pick_id = f"{p['ticker']}_{datetime.now().strftime('%Y%m%d')}"
        memory["pending_evaluations"][pick_id] = {
            "ticker": p['ticker'],
            "date": datetime.now().isoformat(),
            "entry_price": p['price'],
            "sl": p['sl'],
            "tgt1": p['tgt1'],
            "strategy": p['strategy'],
            "score": p['score']
        }
        memory["total_picks"] += 1

    save_memory(memory)

    # STEP 3: CONSTRUCT TELEGRAM REPORT
    today = datetime.now().strftime("%A, %d %b %Y")
    
    msg = ""
    msg += "========================================\n"
    msg += f"⚡ ARTHA {BOT_VERSION} | Pre-Market Report\n"
    msg += f"📅 {today}\n"
    msg += "========================================\n\n"

    msg += "[🌍 MARKET REGIME ENGINE]\n"
    msg += f"State        : {regime}\n"
    msg += f"Active Mode  : {strategy}\n"
    msg += f"Nifty Price  : {regime_info['nifty_price']}\n"
    msg += f"20D Momentum : {regime_info['change_20d']:+.2f}%\n"
    msg += f"System Mode  : {regime_info['desc']}\n\n"

    counts = memory.get("nse_source_counts", {})
    msg += "[🇮🇳 NSE UNIVERSE & DATA COVERAGE]\n"
    msg += f"Universe source : {counts.get('source', 'unknown')}\n"
    msg += f"Mainboard / SME : {counts.get('mainboard', 0)} / {counts.get('sme', 0)}\n"
    msg += f"NSE symbols     : {len(universe)}\n"
    msg += f"Valid 1Y OHLCV  : {valid_data_count}\n"
    msg += f"Invalid / failed: {invalid_data_count} / {download_error_count}\n"
    msg += f"Universe updated: {memory.get('nse_universe_updated_at') or 'fallback / not refreshed'}\n\n"

    msg += "[📊 SCAN SUMMARY]\n"
    msg += f"Active Strategy : {strategy}\n"
    msg += f"Setups Qualified: {len(results)}\n"
    msg += f"Price Action     : {price_action_count} confirmed signals\n"
    msg += f"Top Selected    : {len(top_picks)}\n\n"

    if top_picks:
        msg += "----------------------------------------\n"
        msg += f"[🏆 TOP {len(top_picks)} ACTIONABLE SETUPS]\n"
        msg += "----------------------------------------\n\n"

        for i, p in enumerate(top_picks, 1):
            # Calculate Position Sizing (Risk 1.5% of Rs. 1L = Rs. 1500)
            risk_per_share = p['price'] - p['sl']
            shares = int(1500 / risk_per_share) if risk_per_share > 0 else 1
            shares = max(1, shares)
            pos_value = shares * p['price']

            msg += f"#{i} {p['ticker']} | {p['strategy']} MODE\n"
            msg += f"Score      : {p['score']}/100\n"
            msg += f"CMP        : Rs. {p['price']}\n"
            msg += f"RSI        : {p['rsi']} | Vol Ratio: {p['vol_ratio']}x\n\n"

            msg += "TRADE SETUP:\n"
            msg += f"  Entry    : Rs. {p['price']}\n"
            msg += f"  StopLoss : Rs. {p['sl']} (-{p['risk_pct']}%)\n"
            msg += f"  Target 1 : Rs. {p['tgt1']} (+{round((p['tgt1']-p['price'])/p['price']*100, 1)}%)\n"
            msg += f"  Target 2 : Rs. {p['tgt2']} (+{round((p['tgt2']-p['price'])/p['price']*100, 1)}%)\n\n"

            msg += "POSITION SIZING (Rs. 1L Capital):\n"
            msg += f"  Qty      : {shares} shares\n"
            msg += f"  Allocation: Rs. {pos_value:,.0f}\n\n"

            if p['signals']:
                msg += f"Signals    : {', '.join(p['signals'])}\n"
            msg += "----------------------------------------\n\n"
    else:
        msg += "[📉 NO ACTIONABLE SETUPS TODAY]\n"
        msg += "No stock met the current strategy and price-action filters.\n"
        msg += "No trade is suggested when confirmation is missing.\n\n"

    msg += "[📐 DISCIPLINE RULES]\n"
    msg += "1. Enter ONLY if setup holds after 9:45 AM.\n"
    msg += "2. Strictly respect the StopLoss.\n"
    msg += "3. Book 50% profits at Target 1, trail rest.\n\n"

    msg += "========================================\n"
    msg += f"{BOT_NAME} {BOT_VERSION} | {BOT_TAGLINE}\n"

    send_telegram(msg)
    print(f"\n✅ ARTHA {BOT_VERSION} Run Complete!")

if __name__ == "__main__":
    main()
