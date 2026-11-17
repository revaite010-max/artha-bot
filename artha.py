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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 🔑 CONFIGURATION & ENVIRONMENT
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v18.0"
BOT_TAGLINE = "NSE-Wide AI Paper-Trading System"

MEMORY_FILE = "artha_memory.json"

MAX_POSITIONS_ACTIVE = int(os.environ.get("MAX_POSITIONS_ACTIVE", "3"))
BREAKOUT_MIN_VOLUME_RATIO = float(os.environ.get("BREAKOUT_MIN_VOLUME_RATIO", "1.2"))
PRICE_ACTION_MIN_VOLUME_RATIO = float(os.environ.get("PRICE_ACTION_MIN_VOLUME_RATIO", "0.8"))
DIP_RSI_OVERSOLD_LEVEL = float(os.environ.get("DIP_RSI_OVERSOLD_LEVEL", "48"))
MAX_SETUP_RISK_PCT = float(os.environ.get("MAX_SETUP_RISK_PCT", "5.5"))
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL_INR", "100000"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.5"))
PORTFOLIO_HEAT_PCT = float(os.environ.get("PORTFOLIO_HEAT_PCT", "5"))
MAX_POSITION_VALUE_PCT = float(os.environ.get("MAX_POSITION_VALUE_PCT", "25"))
PAPER_SLIPPAGE_BPS = float(os.environ.get("PAPER_SLIPPAGE_BPS", "10"))
PAPER_FEE_BPS_ROUND_TRIP = float(os.environ.get("PAPER_FEE_BPS_ROUND_TRIP", "20"))
MAX_PAPER_ENTRY_GAP_PCT = float(os.environ.get("MAX_PAPER_ENTRY_GAP_PCT", "3"))
PAPER_HOLD_BARS = 20
MAX_ENTRY_WAIT_CALENDAR_DAYS = 10
MIN_AVG_DAILY_TURNOVER_INR = float(os.environ.get("MIN_AVG_DAILY_TURNOVER_INR", "5000000"))
AI_MIN_SAMPLES = int(os.environ.get("AI_MIN_SAMPLES", "200"))
AI_MAX_SAMPLES = int(os.environ.get("AI_MAX_SAMPLES", "50000"))
AI_MIN_CLASS_SAMPLES = 20
AI_MIN_VALIDATION_AUC = float(os.environ.get("AI_MIN_VALIDATION_AUC", "0.52"))
AI_FEATURES = ["rsi14", "volume_ratio", "close_location", "resistance_distance_pct",
               "ema50_distance_pct", "ema200_distance_pct", "momentum20_pct",
               "relative_strength60_pct", "sector_relative_strength60_pct",
               "atr_pct", "log_turnover"]

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
        "ai_training_summary": {},
        "paper_account": {},
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
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        return False
    return True


def is_history_stale(frame, max_age_days=7):
    if frame is None or frame.empty:
        return True
    try:
        return (datetime.now().date() - pd.Timestamp(frame.index[-1]).date()).days > max_age_days
    except (TypeError, ValueError):
        return True


def has_suspect_price_discontinuity(frame, threshold=0.45, recent_bars=30):
    if frame is None or len(frame) < 2:
        return True
    returns = pd.to_numeric(frame["close"], errors="coerce").tail(recent_bars).pct_change().dropna()
    return bool((returns.abs() > threshold).any())


def average_daily_turnover(frame, window=20):
    if frame is None or len(frame) < window:
        return 0.0
    turnover = (pd.to_numeric(frame["close"], errors="coerce") *
                pd.to_numeric(frame["volume"], errors="coerce")).tail(window).mean()
    return float(turnover) if math.isfinite(float(turnover)) else 0.0


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
def detect_market_regime(nifty=None):
    """Determine market state; missing benchmark data fails closed."""
    try:
        if nifty is None:
            nifty = normalize_yfinance_columns(
                yf.download("^NSEI", period="1y", progress=False, timeout=20,
                            auto_adjust=True))
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
def scan_breakout(df, ticker_clean, diagnostics=None):
    """Balanced breakout filter; qualifying strength is scored rather than over-filtered."""
    def reject(reason):
        if diagnostics is not None:
            key = f"BREAKOUT:{reason}"
            diagnostics[key] = diagnostics.get(key, 0) + 1
        return None

    try:
        if df is None or len(df) < 200:
            return reject("short_history")
        close, high, low, vol = (df[name].astype(float) for name in
                                 ("close", "high", "low", "volume"))
        c = len(df) - 1
        curr = float(close.iloc[c])
        ema20, ema50, ema200 = (get_ema(close, n).iloc[c] for n in (20, 50, 200))
        rsi = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr = float(get_atr(high, low, close, 14).iloc[c])
        if not (curr > ema50 > ema200 and curr >= ema20 * 0.985):
            return reject("trend")
        if not (50 <= rsi.iloc[c] <= 76):
            return reject("rsi")
        # Allow an improving MACD just before its signal-line cross.
        if macd.iloc[c] <= msig.iloc[c] and macd.iloc[c] <= macd.iloc[c-1]:
            return reject("macd")
        avg_vol = float(vol.rolling(20).mean().iloc[c])
        vol_ratio = float(vol.iloc[c] / avg_vol) if avg_vol > 0 else 0.0
        if vol_ratio < BREAKOUT_MIN_VOLUME_RATIO:
            return reject("volume")
        high_20 = float(high.iloc[-21:-1].max())
        if curr < high_20 * 0.985:
            return reject("resistance")
        score, signals = 65, ["20D breakout / resistance proximity"]
        if curr >= high_20:
            score += 5
            signals[0] = "20D Breakout"
        if vol_ratio >= 1.8:
            score += 10
            signals.append("Volume confirmation")
        if 55 <= rsi.iloc[c] <= 70:
            score += 8
            signals.append("Momentum RSI")
        if macd.iloc[c] > msig.iloc[c]:
            score += 5
            signals.append("MACD bullish")
        if not math.isfinite(atr) or atr <= 0:
            return reject("atr")
        sl = round(max(ema20 * 0.985, curr - 1.5 * atr), 2)
        risk = curr - sl
        if risk <= 0 or (risk / curr) * 100 > MAX_SETUP_RISK_PCT:
            return reject("stop_risk")
        return {
            "ticker": ticker_clean, "strategy": "BREAKOUT", "score": round(score, 1),
            "price": round(curr, 2), "rsi": round(float(rsi.iloc[c]), 1),
            "vol_ratio": round(vol_ratio, 1), "sl": sl,
            "tgt1": round(curr + 2.0 * risk, 2),
            "tgt2": round(curr + 3.0 * risk, 2),
            "risk_pct": round(risk / curr * 100, 2), "rr_ratio": 2.0,
            "signals": signals,
        }
    except Exception:
        return reject("calculation_error")

# ============================================================
# 🎯 3. STRATEGY B: DIP-BUYING & MEAN REVERSION (For Sideways)
# ============================================================
def scan_dip_buying(df, ticker_clean, diagnostics=None):
    """Identify support rebounds with a wider tolerance and explicit confirmation."""
    def reject(reason):
        if diagnostics is not None:
            key = f"DIP_BUYING:{reason}"
            diagnostics[key] = diagnostics.get(key, 0) + 1
        return None

    try:
        if df is None or len(df) < 200:
            return reject("short_history")
        close, high, low, vol = (df[name].astype(float) for name in
                                 ("close", "high", "low", "volume"))
        open_ = df["open"].astype(float)
        c = len(df) - 1
        curr = float(close.iloc[c])
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi = get_rsi(close, 14)
        atr = float(get_atr(high, low, close, 14).iloc[c])
        _, _, bb_low = get_bb(close, 20)
        if curr < float(ema200.iloc[c]) * 0.985:
            return reject("long_trend")
        near_ema50 = abs(curr - float(ema50.iloc[c])) / curr <= 0.04
        near_bb_low = float(low.iloc[-3:].min()) <= float(bb_low.iloc[-3:].max()) * 1.02
        if not (near_ema50 or near_bb_low):
            return reject("support_proximity")
        recent_rsi = rsi.iloc[-5:]
        oversold_recently = bool((recent_rsi < DIP_RSI_OVERSOLD_LEVEL).any())
        rsi_turning = float(rsi.iloc[c]) > float(rsi.iloc[c-1])
        close_improving = float(close.iloc[c]) > float(close.iloc[c-1])
        if not (oversold_recently and (rsi_turning or close_improving)):
            return reject("reversal")
        green_candle = float(close.iloc[c]) > float(open_.iloc[c])
        close_location = ((float(close.iloc[c]) - float(low.iloc[c])) /
                          max(float(high.iloc[c] - low.iloc[c]), 1e-9))
        if not (green_candle or close_improving) or close_location < 0.50:
            return reject("candle_confirmation")
        avg_vol = float(vol.rolling(20).mean().iloc[c])
        vol_ratio = float(vol.iloc[c] / avg_vol) if avg_vol > 0 else 0.0
        score, signals = 68, ["Support rebound"]
        if near_ema50:
            signals.append("Near 50 EMA")
        if near_bb_low:
            signals.append("Lower Bollinger support")
        if green_candle:
            score += 5
            signals.append("Bullish close")
        if vol_ratio >= 1.2:
            score += 7
            signals.append("Volume returning")
        if not math.isfinite(atr) or atr <= 0:
            return reject("atr")
        sl = round(min(float(low.iloc[-3:].min()) * 0.99, curr - 1.5 * atr), 2)
        risk = curr - sl
        if risk <= 0 or risk / curr * 100 > MAX_SETUP_RISK_PCT:
            return reject("stop_risk")
        return {
            "ticker": ticker_clean, "strategy": "DIP_BUYING", "score": round(score, 1),
            "price": round(curr, 2), "rsi": round(float(rsi.iloc[c]), 1),
            "vol_ratio": round(vol_ratio, 1), "sl": sl,
            "tgt1": round(curr + 1.8 * risk, 2),
            "tgt2": round(curr + 3.0 * risk, 2),
            "risk_pct": round(risk / curr * 100, 2), "rr_ratio": 1.8,
            "signals": signals,
        }
    except Exception:
        return reject("calculation_error")


def scan_prebreakout(df, ticker_clean, diagnostics=None):
    """Find a tight, non-extended base near resistance and queue only a trigger entry."""
    def reject(reason):
        if diagnostics is not None:
            key = f"PRE_BREAKOUT:{reason}"
            diagnostics[key] = diagnostics.get(key, 0) + 1
        return None

    try:
        if df is None or len(df) < 200:
            return reject("short_history")
        close, high, low, volume = (pd.to_numeric(df[name], errors="coerce")
                                    for name in ("close", "high", "low", "volume"))
        current = float(close.iloc[-1])
        resistance = float(high.iloc[-21:-1].max())
        atr = float(get_atr(high, low, close, 14).iloc[-1])
        ema20, ema50, ema200 = (float(get_ema(close, period).iloc[-1])
                                for period in (20, 50, 200))
        rsi = float(get_rsi(close, 14).iloc[-1])
        average_volume = float(volume.rolling(20).mean().iloc[-1])
        volume_ratio = float(volume.iloc[-1] / average_volume) if average_volume > 0 else 0.0
        if not all(math.isfinite(value) for value in
                   (current, resistance, atr, ema20, ema50, ema200, rsi, volume_ratio)):
            return reject("invalid_features")
        distance_pct = (resistance - current) / resistance * 100
        momentum20 = (current / float(close.iloc[-21]) - 1) * 100
        if not 0.3 <= distance_pct <= 2.5:
            return reject("not_near_resistance")
        if not (current > ema50 > ema200 and current >= ema20 * 0.98):
            return reject("trend")
        if not (48 <= rsi <= 70 and atr / current * 100 <= 4.0):
            return reject("momentum_or_volatility")
        if not (-2 <= momentum20 <= 8 and current / ema20 <= 1.045):
            return reject("extended_move")
        if volume_ratio > 1.8:
            return reject("late_volume_spike")

        trigger = round(resistance * 1.002, 2)
        stop = round(max(resistance - 1.5 * atr, current - 2.0 * atr), 2)
        risk = trigger - stop
        if risk <= 0 or risk / trigger * 100 > MAX_SETUP_RISK_PCT:
            return reject("stop_risk")
        score = 70 + (5 if volume_ratio >= 0.8 else 0) + (5 if distance_pct <= 1.2 else 0)
        return {
            "ticker": ticker_clean, "strategy": "PRE_BREAKOUT",
            "score": min(score, 85), "price": round(current, 2),
            "entry_trigger": trigger, "rsi": round(rsi, 1),
            "vol_ratio": round(volume_ratio, 1), "sl": stop,
            "tgt1": round(trigger + 1.5 * risk, 2),
            "tgt2": round(trigger + 2.5 * risk, 2),
            "risk_pct": round(risk / trigger * 100, 2), "rr_ratio": 1.5,
            "signals": [f"Base {distance_pct:.1f}% below 20D resistance",
                        "Conditional entry only above trigger", "Not extended above 20D average"],
        }
    except Exception:
        return reject("calculation_error")


def scan_price_action(df, ticker_clean, diagnostics=None):
    """Balanced candle-structure scan for breakouts, engulfing bars, and support bounces."""
    def reject(reason):
        if diagnostics is not None:
            key = f"PRICE_ACTION:{reason}"
            diagnostics[key] = diagnostics.get(key, 0) + 1
        return None

    try:
        if df is None or len(df) < 60:
            return reject("short_history")
        close, open_, high, low, volume = (pd.to_numeric(df[name], errors="coerce")
                                           for name in ("close", "open", "high", "low", "volume"))
        current, previous = float(close.iloc[-1]), float(close.iloc[-2])
        current_open, current_high, current_low = (float(open_.iloc[-1]), float(high.iloc[-1]),
                                                   float(low.iloc[-1]))
        if not all(math.isfinite(value) for value in
                   (current, previous, current_open, current_high, current_low)):
            return reject("invalid_bar")
        if current <= 0 or current_high <= current_low:
            return reject("invalid_range")
        atr = float(get_atr(high, low, close, 14).iloc[-1])
        ema50 = float(get_ema(close, 50).iloc[-1])
        if not math.isfinite(atr) or atr <= 0:
            return reject("atr")
        if current < ema50 * 0.985:
            return reject("trend")
        avg_volume = float(volume.rolling(20).mean().iloc[-1])
        latest_volume = float(volume.iloc[-1])
        vol_ratio = latest_volume / avg_volume if avg_volume > 0 else 0.0
        close_location = (current - current_low) / (current_high - current_low)
        resistance = float(high.iloc[-21:-1].max())
        support = float(low.iloc[-23:-3].min())
        previous_open, previous_close = float(open_.iloc[-2]), float(close.iloc[-2])
        breakout = (current > resistance * 1.001 and previous <= resistance * 1.02
                    and close_location >= 0.60 and vol_ratio >= PRICE_ACTION_MIN_VOLUME_RATIO)
        bullish_engulfing = (current > current_open and previous_close < previous_open
                             and current >= previous_open and current_open <= previous_close
                             and close_location >= 0.58 and vol_ratio >= PRICE_ACTION_MIN_VOLUME_RATIO)
        support_test = float(low.iloc[-3:].min()) <= support + 0.75 * atr
        support_rejection = (support_test and current > previous and close_location >= 0.58
                             and vol_ratio >= PRICE_ACTION_MIN_VOLUME_RATIO)
        signals, score = [], 65
        if breakout:
            signals.append("20-session resistance breakout")
            score += 15
        if bullish_engulfing:
            signals.append("Bullish engulfing candle")
            score += 12
        if support_rejection:
            signals.append("Support bounce / higher close")
            score += 10
        if not signals:
            return reject("no_pattern")
        if close_location >= 0.80:
            signals.append("Strong close near candle high")
            score += 5
        if vol_ratio >= 1.5:
            signals.append("Above-average volume")
            score += 5
        structural_level = resistance if breakout else support
        stop = max(structural_level - 0.35 * atr, current - 1.6 * atr)
        risk = current - stop
        risk_pct = risk / current * 100
        if risk <= 0 or risk_pct > MAX_SETUP_RISK_PCT:
            return reject("stop_risk")
        return {
            "ticker": ticker_clean, "strategy": "PRICE_ACTION",
            "score": min(round(score, 1), 100.0), "price": round(current, 2),
            "rsi": round(float(get_rsi(close, 14).iloc[-1]), 1),
            "vol_ratio": round(vol_ratio, 1), "sl": round(stop, 2),
            "tgt1": round(current + 1.5 * risk, 2),
            "tgt2": round(current + 2.5 * risk, 2),
            "risk_pct": round(risk_pct, 2), "rr_ratio": 1.5, "signals": signals,
        }
    except Exception:
        return reject("calculation_error")

# ============================================================
# 🎯 4. STRATEGY C: DEFENSIVE & GOLD ALLOCATOR (For Bear Markets)
# ============================================================
def scan_defensive(price_frames=None):
    """Scans defensive ETFs and supplies their histories to the paper simulator."""
    defensive_universe = ["GOLDBEES.NS", "SILVERBEES.NS", "GOLDSHARE.NS", "HDFCGOLD.NS"]
    picks = []
    
    for ticker in defensive_universe:
        try:
            df = normalize_yfinance_columns(yf.download(
                ticker, period="6mo", progress=False, timeout=20, auto_adjust=True))
            if df.empty or "close" not in df:
                continue
            if price_frames is not None:
                price_frames[ticker.replace(".NS", "")] = df
            if (not has_usable_price_history(df, minimum_rows=50, require_volume=True) or
                    is_history_stale(df) or has_suspect_price_discontinuity(df) or
                    average_daily_turnover(df) < MIN_AVG_DAILY_TURNOVER_INR):
                continue
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
                    "as_of_date": pd.Timestamp(df.index[-1]).date().isoformat(),
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

SECTOR_INDEX_TICKERS = {
    "IT": "^CNXIT", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG", "METAL": "^CNXMETAL",
    "REALTY": "^CNXREALTY", "ENERGY": "^CNXENERGY",
}
SECTOR_SYMBOLS = {
    "IT": set("TCS INFY HCLTECH WIPRO TECHM LTIM PERSISTENT COFORGE MPHASIS LTTS TATAELXSI KPITTECH".split()),
    "BANK": set("HDFCBANK ICICIBANK SBIN AXISBANK KOTAKBANK INDUSINDBK BANKBARODA PNB FEDERALBNK IDFCFIRSTB AUBANK CANBK".split()),
    "AUTO": set("MARUTI TATAMOTORS M&M BAJAJ-AUTO HEROMOTOCO EICHERMOT TVSMOTOR ASHOKLEY".split()),
    "PHARMA": set("SUNPHARMA DRREDDY CIPLA DIVISLAB AUROPHARMA TORNTPHARM ALKEM GLENMARK LAURUSLABS BIOCON".split()),
    "FMCG": set("HINDUNILVR ITC NESTLEIND BRITANNIA TATACONSUM DABUR GODREJCP MARICO COLPAL".split()),
    "METAL": set("TATASTEEL JSWSTEEL HINDALCO VEDL SAIL JINDALSTEL NMDC HINDCOPPER JSL".split()),
    "REALTY": set("DLF GODREJPROP PRESTIGE OBEROIRLTY PHOENIXLTD BRIGADE SOBHA LODHA".split()),
    "ENERGY": set("RELIANCE ONGC NTPC POWERGRID COALINDIA BPCL IOC GAIL TATAPOWER ADANIGREEN".split()),
}


def sector_for_ticker(ticker):
    symbol = ticker[:-3] if ticker.endswith(".NS") else ticker
    return next((sector for sector, members in SECTOR_SYMBOLS.items() if symbol in members), None)


def fetch_sector_histories():
    tickers = list(SECTOR_INDEX_TICKERS.values())
    try:
        downloaded = yf.download(tickers=tickers, period="1y", progress=False,
                                  timeout=25, group_by="ticker", auto_adjust=True,
                                  threads=True)
    except Exception as exc:
        print(f"[SECTOR DATA]: unavailable ({type(exc).__name__}); using NIFTY-relative strength only.")
        return {}
    histories = {}
    for sector, ticker in SECTOR_INDEX_TICKERS.items():
        frame = extract_ticker_frame(downloaded, ticker, requested_tickers=tickers)
        if not frame.empty and "close" in frame:
            histories[sector] = frame["close"].astype(float)
    return histories


def build_price_action_feature_frame(frame, benchmark_close=None, sector_close=None):
    """Leakage-conscious daily features; all columns on date t use information through t only."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    true_range = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    ema50, ema200 = get_ema(close, 50), get_ema(close, 200)
    resistance = high.shift(1).rolling(20).max()
    support = low.shift(3).rolling(20).min()
    avg_volume = volume.rolling(20).mean()
    close_location = (close - low) / (high - low).replace(0, np.nan)
    benchmark_return = pd.Series(np.nan, index=frame.index)
    if benchmark_close is not None:
        benchmark = pd.to_numeric(benchmark_close, errors="coerce").reindex(frame.index).ffill()
        benchmark_return = benchmark.pct_change(60) * 100
    sector_return = benchmark_return
    if sector_close is not None:
        sector = pd.to_numeric(sector_close, errors="coerce").reindex(frame.index).ffill()
        sector_return = sector.pct_change(60) * 100
    stock_return20 = (close / close.shift(20) - 1) * 100
    stock_return60 = (close / close.shift(60) - 1) * 100
    features = pd.DataFrame(index=frame.index)
    features["rsi14"] = get_rsi(close, 14)
    features["volume_ratio"] = volume / avg_volume.replace(0, np.nan)
    features["close_location"] = close_location
    features["resistance_distance_pct"] = (close / resistance - 1) * 100
    features["ema50_distance_pct"] = (close / ema50 - 1) * 100
    features["ema200_distance_pct"] = (close / ema200 - 1) * 100
    features["momentum20_pct"] = stock_return20
    features["relative_strength60_pct"] = stock_return60 - benchmark_return
    features["sector_relative_strength60_pct"] = stock_return60 - sector_return
    features["atr_pct"] = atr / close * 100
    features["log_turnover"] = np.log1p((close * volume).rolling(20).mean() / 1_000_000)
    previous_bear = close.shift(1) < open_.shift(1)
    engulfing = ((close > open_) & previous_bear & (close >= open_.shift(1)) &
                 (open_ <= close.shift(1)) & (close_location >= 0.55))
    breakout = (close >= resistance * 0.985) & (close_location >= 0.55)
    support_test = low.rolling(3).min() <= support + 0.75 * atr
    bounce = support_test & (close > close.shift(1)) & (close_location >= 0.55)
    trend = (close >= ema50 * 0.985) & (close >= ema200 * 0.97)
    turnover20 = (close * volume).rolling(20).mean()
    features["_turnover20"] = turnover20
    features["_atr"] = atr
    features["_resistance"] = resistance
    features["_support"] = support
    features["_breakout"] = breakout
    features["_candidate"] = (trend & (features["volume_ratio"] >= 0.6) &
                              (turnover20 >= MIN_AVG_DAILY_TURNOVER_INR) &
                              (breakout | engulfing | bounce))
    return features.replace([np.inf, -np.inf], np.nan)


def build_labeled_training_samples(frame, benchmark_close=None, ticker="", feature_frame=None,
                                  sector_close=None):
    """Label historical candle setups by stop/target ordering in the next 20 sessions."""
    features = feature_frame if feature_frame is not None else build_price_action_feature_frame(
        frame, benchmark_close, sector_close)
    close, open_, high, low = (pd.to_numeric(frame[name], errors="coerce")
                               for name in ("close", "open", "high", "low"))
    samples = []
    last_signal = len(frame) - PAPER_HOLD_BARS - 1
    # Thinning to alternate sessions reduces highly overlapping labels and model cost.
    for i in range(200, last_signal + 1, 2):
        row = features.iloc[i]
        if not bool(row["_candidate"]) or row[AI_FEATURES].isna().any():
            continue
        close_i, atr_i = float(close.iloc[i]), float(row["_atr"])
        structure = float(row["_resistance"] if bool(row["_breakout"]) else row["_support"])
        if not all(math.isfinite(value) for value in (close_i, atr_i, structure)) or atr_i <= 0:
            continue
        stop = max(structure - 0.35 * atr_i, close_i - 1.6 * atr_i)
        entry = float(open_.iloc[i + 1]) * (1 + PAPER_SLIPPAGE_BPS / 10000)
        risk = entry - stop
        if risk <= 0 or risk / entry * 100 > MAX_SETUP_RISK_PCT:
            continue
        target = entry + 1.5 * risk
        label, resolved = None, False
        for j in range(i + 1, i + PAPER_HOLD_BARS + 1):
            if float(open_.iloc[j]) <= stop or float(low.iloc[j]) <= stop:
                label, resolved = 0, True
                break
            if float(high.iloc[j]) >= target:
                label, resolved = 1, True
                break
        if not resolved:
            final_return = float(close.iloc[i + PAPER_HOLD_BARS]) / entry - 1
            label = int(final_return > 2 * PAPER_SLIPPAGE_BPS / 10000)
        sample = row[AI_FEATURES].astype(float).to_dict()
        sample.update({"label": int(label), "signal_date": pd.Timestamp(frame.index[i]).date().isoformat(),
                       "ticker": ticker})
        samples.append(sample)
    return samples


def train_local_model(samples):
    """Chronologically validate and fit a local logistic ranking model; no random split."""
    summary = {"status": "INSUFFICIENT_HISTORY", "samples": len(samples),
               "validation_auc": None, "validation_samples": 0}
    if len(samples) < AI_MIN_SAMPLES:
        return None, summary
    data = pd.DataFrame(samples).sort_values("signal_date")
    raw_sample_count = len(data)
    if len(data) > AI_MAX_SAMPLES:
        stride = math.ceil(len(data) / AI_MAX_SAMPLES)
        data = data.iloc[::stride].copy()
    summary["fit_samples"] = len(data)
    summary["raw_samples"] = raw_sample_count
    if data["label"].nunique() < 2:
        summary["status"] = "ONE_CLASS_ONLY"
        return None, summary
    unique_dates = sorted(data["signal_date"].unique())
    if len(unique_dates) < 5:
        summary["status"] = "INSUFFICIENT_DATE_SPAN"
        return None, summary
    split_index = max(1, int(len(unique_dates) * 0.8))
    split_date = unique_dates[split_index]
    # Labels look forward 20 sessions; purge the overlapping horizon before validation.
    purge_index = max(0, split_index - PAPER_HOLD_BARS)
    training_end = unique_dates[purge_index]
    training = data[data["signal_date"] < training_end]
    validation = data[data["signal_date"] >= split_date]
    summary["purged_training_dates"] = split_index - purge_index
    summary["training_samples"] = len(training)
    summary["validation_samples"] = len(validation)
    if (training["label"].nunique() < 2 or validation["label"].nunique() < 2 or
            min(training["label"].value_counts().min(), validation["label"].value_counts().min())
            < AI_MIN_CLASS_SAMPLES):
        summary["status"] = "INSUFFICIENT_CLASS_BALANCE"
        return None, summary
    try:
        validation_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=600, class_weight="balanced", random_state=17))
        validation_model.fit(training[AI_FEATURES], training["label"])
        validation_prob = validation_model.predict_proba(validation[AI_FEATURES])[:, 1]
        auc = float(roc_auc_score(validation["label"], validation_prob))
        summary["validation_auc"] = round(auc, 3)
        summary["positive_rate"] = round(float(data["label"].mean()), 3)
        if auc < AI_MIN_VALIDATION_AUC:
            summary["status"] = "NO_VALIDATED_EDGE_RULE_RANKING_FALLBACK"
            return None, summary
        final_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=600, class_weight="balanced", random_state=17))
        final_model.fit(data[AI_FEATURES], data["label"])
        summary["status"] = "ACTIVE"
        return final_model, summary
    except (ValueError, FloatingPointError) as exc:
        summary["status"] = f"TRAINING_ERROR_{type(exc).__name__}"
        return None, summary


def ai_probability(model, pick, frame, benchmark_close):
    if model is None:
        return None
    if pick.get("ai_features"):
        values = pd.Series(pick["ai_features"], index=AI_FEATURES, dtype=float)
    else:
        row = build_price_action_feature_frame(frame, benchmark_close).iloc[-1]
        values = row[AI_FEATURES].astype(float)
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        return float(model.predict_proba(values.to_frame().T)[0, 1])
    except (ValueError, TypeError):
        return None


def ensure_paper_account(memory):
    account = memory.get("paper_account")
    if not isinstance(account, dict):
        account = {}
        memory["paper_account"] = account
    account.setdefault("starting_cash", TOTAL_CAPITAL)
    account.setdefault("cash", TOTAL_CAPITAL)
    account.setdefault("positions", {})
    account.setdefault("pending_entries", [])
    account.setdefault("closed_trades", [])
    account.setdefault("rejected_entries", [])
    account.setdefault("equity_history", [])
    try:
        account["cash"] = float(account["cash"])
        account["starting_cash"] = float(account["starting_cash"])
    except (TypeError, ValueError):
        account["cash"] = account["starting_cash"] = TOTAL_CAPITAL
        account["positions"] = {}
        account["pending_entries"] = []
    for key, expected in (("positions", dict), ("pending_entries", list),
                          ("closed_trades", list), ("rejected_entries", list),
                          ("equity_history", list)):
        if not isinstance(account.get(key), expected):
            account[key] = expected()
    return account


def paper_equity(account, price_frames=None):
    market_value = 0.0
    for ticker, position in account["positions"].items():
        frame = (price_frames or {}).get(ticker)
        price = (float(frame["close"].iloc[-1]) if frame is not None and not frame.empty
                 else float(position.get("last_price", position["entry_price"])))
        market_value += int(position["remaining_shares"]) * price
    return float(account["cash"]) + market_value


def _paper_sell(account, position, quantity, raw_price):
    quantity = min(int(quantity), int(position["remaining_shares"]))
    fill_price = float(raw_price) * (1 - PAPER_SLIPPAGE_BPS / 10000)
    proceeds = fill_price * quantity
    fee = proceeds * PAPER_FEE_BPS_ROUND_TRIP / 20000
    account["cash"] += proceeds - fee
    position["realized_pnl"] += (fill_price - float(position["entry_price"])) * quantity - fee
    position["remaining_shares"] -= quantity
    position["last_exit_price"] = fill_price
    return fill_price


def _paper_close_position(account, ticker, position, date, outcome, raw_exit):
    if position["remaining_shares"] > 0:
        _paper_sell(account, position, position["remaining_shares"], raw_exit)
    risk = max(float(position.get("initial_risk_amount", 0.0)), 0.01)
    position.update({"outcome": outcome, "exit_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                     "net_pnl": round(position["realized_pnl"], 2),
                     "r_multiple": round(position["realized_pnl"] / risk, 3)})
    account["closed_trades"].append(position.copy())
    account["positions"].pop(ticker, None)


def _process_paper_bar(account, ticker, position, date, bar):
    day_open, day_low, day_high, day_close = (float(bar[key]) for key in
                                               ("open", "low", "high", "close"))
    position["bars_held"] = int(position.get("bars_held", 0)) + 1
    position["last_price"] = day_close
    position["last_processed_date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
    stop = float(position["stop"])
    if day_open <= stop or day_low <= stop:
        exit_at = min(day_open, stop)
        outcome = "STOP_AFTER_TARGET1" if position.get("target1_hit") else "STOP"
        _paper_close_position(account, ticker, position, date, outcome, exit_at)
        return
    if not position.get("target1_hit") and day_high >= float(position["target1"]):
        quantity = max(1, int(position["remaining_shares"]) // 2)
        _paper_sell(account, position, quantity, float(position["target1"]))
        position["target1_hit"] = True
        position["stop"] = max(float(position["stop"]), float(position["entry_price"]))
        if position["remaining_shares"] == 0:
            _paper_close_position(account, ticker, position, date, "TARGET1", float(position["target1"]))
            return
    if position.get("target1_hit") and position["remaining_shares"] > 0 and \
            day_high >= float(position["target2"]):
        _paper_close_position(account, ticker, position, date, "TARGET2", float(position["target2"]))
        return
    if position["bars_held"] >= PAPER_HOLD_BARS:
        _paper_close_position(account, ticker, position, date, "TIME_EXIT", day_close)


def process_paper_account(memory, price_frames):
    """Simulate pending next-session entries and manage open positions; never places real orders."""
    account = ensure_paper_account(memory)
    processed_entries = closed_count = rejected_count = 0
    pending = []
    for order in account["pending_entries"]:
        ticker = order["ticker"]
        signal_date = pd.Timestamp(order["signal_date"]).date()
        latest_available_date = max((pd.Timestamp(date).date() for frame in price_frames.values()
                                      if frame is not None and not frame.empty
                                      for date in frame.index), default=signal_date)
        if (latest_available_date - signal_date).days > MAX_ENTRY_WAIT_CALENDAR_DAYS:
            order["status"] = "EXPIRED_NO_NEXT_SESSION_DATA"
            account["rejected_entries"].append(order)
            rejected_count += 1
            continue
        frame = price_frames.get(ticker)
        if frame is None or frame.empty:
            pending.append(order)
            continue
        future_bars = [(date, row) for date, row in frame.iterrows()
                       if pd.Timestamp(date).date() > signal_date]
        if not future_bars:
            pending.append(order)
            continue
        trigger = order.get("entry_trigger")
        if trigger is not None:
            trigger = float(trigger)
            trigger_bars = [(date, row) for date, row in future_bars
                            if float(row["high"]) >= trigger]
            if not trigger_bars:
                pending.append(order)
                continue
            entry_date, entry_bar = trigger_bars[0]
            raw_open = max(float(entry_bar["open"]), trigger)
        else:
            entry_date, entry_bar = future_bars[0]
            raw_open = float(entry_bar["open"])
        if ticker in account["positions"] or len(account["positions"]) >= MAX_POSITIONS_ACTIVE:
            order["status"] = "REJECTED_POSITION_LIMIT"
            account["rejected_entries"].append(order)
            rejected_count += 1
            continue
        fill = raw_open * (1 + PAPER_SLIPPAGE_BPS / 10000)
        reference = float(order["reference_close"])
        if reference <= 0 or abs(fill / reference - 1) * 100 > MAX_PAPER_ENTRY_GAP_PCT:
            order["status"] = "REJECTED_ENTRY_GAP"
            account["rejected_entries"].append(order)
            rejected_count += 1
            continue
        stop = float(order["stop"])
        per_share_risk = fill - stop
        if per_share_risk <= 0 or per_share_risk / fill * 100 > MAX_SETUP_RISK_PCT:
            order["status"] = "REJECTED_GAP_RISK"
            account["rejected_entries"].append(order)
            rejected_count += 1
            continue
        equity = paper_equity(account, price_frames)
        open_risk = sum(max(0.0, int(pos["remaining_shares"]) *
                            (float(pos["entry_price"]) - float(pos["stop"])))
                        for pos in account["positions"].values())
        heat_room = max(0.0, equity * PORTFOLIO_HEAT_PCT / 100 - open_risk)
        budget = min(equity * RISK_PER_TRADE_PCT / 100, heat_room)
        cash_per_share = fill * (1 + PAPER_FEE_BPS_ROUND_TRIP / 20000)
        quantity = min(int(budget / per_share_risk),
                       int(equity * MAX_POSITION_VALUE_PCT / 100 / fill),
                       int(float(account["cash"]) / cash_per_share))
        if quantity < 1:
            order["status"] = "REJECTED_RISK_OR_CASH_LIMIT"
            account["rejected_entries"].append(order)
            rejected_count += 1
            continue
        target1 = round(fill + float(order.get("rr1", 1.5)) * per_share_risk, 2)
        target2 = round(fill + float(order.get("rr2", 2.5)) * per_share_risk, 2)
        position = {**order, "target1": target1, "target2": target2,
                    "entry_price": round(fill, 4), "entry_date":
                    pd.Timestamp(entry_date).strftime("%Y-%m-%d"), "shares": quantity,
                    "remaining_shares": quantity, "initial_risk_amount": quantity * per_share_risk,
                    "realized_pnl": 0.0, "bars_held": 0, "target1_hit": False,
                    "last_price": float(entry_bar["close"]),
                    "realized_pnl": 0.0,
                    "last_processed_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d")}
        entry_fee = fill * quantity * PAPER_FEE_BPS_ROUND_TRIP / 20000
        position["realized_pnl"] = -entry_fee
        account["cash"] -= fill * quantity + entry_fee
        account["positions"][ticker] = position
        processed_entries += 1
        _process_paper_bar(account, ticker, position, entry_date, entry_bar)
        if ticker not in account["positions"]:
            closed_count += 1

    account["pending_entries"] = pending
    for ticker, position in list(account["positions"].items()):
        frame = price_frames.get(ticker)
        if frame is None or frame.empty:
            continue
        last_date = pd.Timestamp(position.get("last_processed_date", position["entry_date"])).date()
        for date, bar in frame.iterrows():
            if pd.Timestamp(date).date() <= last_date:
                continue
            _process_paper_bar(account, ticker, position, date, bar)
            if ticker not in account["positions"]:
                closed_count += 1
                break
            last_date = pd.Timestamp(date).date()

    if account["positions"]:
        for ticker, position in account["positions"].items():
            frame = price_frames.get(ticker)
            if frame is not None and not frame.empty:
                position["last_price"] = float(frame["close"].iloc[-1])
    equity = paper_equity(account, price_frames)
    point = {"date": datetime.now().date().isoformat(),
             "equity": round(equity, 2), "cash": round(float(account["cash"]), 2),
             "open_positions": len(account["positions"])}
    if account["equity_history"] and account["equity_history"][-1].get("date") == point["date"]:
        account["equity_history"][-1] = point
    else:
        account["equity_history"].append(point)
    account["equity_history"] = account["equity_history"][-3650:]
    account["closed_trades"] = account["closed_trades"][-2000:]
    account["rejected_entries"] = account["rejected_entries"][-500:]
    account["cash"] = round(float(account["cash"]), 2)
    return {"entered": processed_entries, "closed": closed_count, "rejected": rejected_count,
            "equity": round(equity, 2), "realized_pnl": round(
                sum(float(trade.get("net_pnl", 0.0)) for trade in account["closed_trades"]), 2)}


def queue_paper_entries(account, picks, signal_dates):
    open_tickers = set(account["positions"])
    queued_tickers = {item["ticker"] for item in account["pending_entries"]}
    for pick in picks:
        ticker = pick["ticker"]
        if ticker in open_tickers or ticker in queued_tickers:
            continue
        signal_date = signal_dates.get(ticker) or pick.get("as_of_date") or \
            datetime.now().date().isoformat()
        account["pending_entries"].append({
            "ticker": ticker, "signal_date": signal_date, "reference_close": pick["price"],
            "entry_trigger": pick.get("entry_trigger"),
            "stop": pick["sl"], "target1": pick["tgt1"], "target2": pick["tgt2"],
            "rr1": (pick["tgt1"] - pick.get("entry_trigger", pick["price"])) /
                   max(pick.get("entry_trigger", pick["price"]) - pick["sl"], 0.01),
            "rr2": (pick["tgt2"] - pick.get("entry_trigger", pick["price"])) /
                   max(pick.get("entry_trigger", pick["price"]) - pick["sl"], 0.01),
            "strategy": pick["strategy"], "rule_score": pick["score"],
            "ai_score": pick.get("ai_score"), "ai_features": pick.get("ai_features", {}),
            "signals": pick.get("signals", []), "status": "QUEUED_NEXT_SESSION",
        })
        queued_tickers.add(ticker)


def paper_performance_summary(account, lookback=30):
    closed = account.get("closed_trades", [])[-lookback:]
    returns = [float(trade.get("r_multiple", 0.0)) for trade in closed]
    equity_values = [float(point.get("equity", TOTAL_CAPITAL))
                     for point in account.get("equity_history", [])]
    peak, max_drawdown = float(account.get("starting_cash", TOTAL_CAPITAL)), 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak * 100)
    return {"trades": len(closed),
            "win_rate_pct": round(100 * sum(value > 0 for value in returns) / len(returns), 1)
            if returns else 0.0,
            "average_r": round(sum(returns) / len(returns), 3) if returns else 0.0,
            "max_drawdown_pct": round(max_drawdown, 2)}


def paper_group_performance(account, lookback=100):
    grouped = {"strategy": {}, "pattern": {}}
    for trade in account.get("closed_trades", [])[-lookback:]:
        r_value = float(trade.get("r_multiple", 0.0))
        strategy = str(trade.get("strategy", "UNKNOWN"))
        grouped["strategy"].setdefault(strategy, []).append(r_value)
        signals = [str(signal).lower() for signal in trade.get("signals", [])]
        patterns = []
        if any("breakout" in signal for signal in signals):
            patterns.append("BREAKOUT")
        if any("engulf" in signal for signal in signals):
            patterns.append("ENGULFING")
        if any("support" in signal or "bounce" in signal for signal in signals):
            patterns.append("SUPPORT_BOUNCE")
        for pattern in set(patterns):
            grouped["pattern"].setdefault(pattern, []).append(r_value)
    return {kind: {name: {"n": len(values),
                          "win_pct": round(100 * sum(value > 0 for value in values) / len(values), 1),
                          "avg_r": round(sum(values) / len(values), 3)}
                   for name, values in sorted(group.items())}
            for kind, group in grouped.items()}


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
    print("=" * 64)
    print(f"{BOT_NAME} {BOT_VERSION} — {BOT_TAGLINE}")
    print(f"Run time: {datetime.now().strftime('%d %b %Y %I:%M %p')} | PAPER TRADING ONLY")
    print("=" * 64)
    memory = load_memory()
    account = ensure_paper_account(memory)

    try:
        nifty = normalize_yfinance_columns(yf.download(
            "^NSEI", period="1y", progress=False, timeout=25, auto_adjust=True))
    except Exception as exc:
        print(f"[MARKET DATA ERROR]: {type(exc).__name__}")
        nifty = pd.DataFrame()
    regime_info = detect_market_regime(nifty)
    strategy = regime_info["strategy"]
    benchmark_close = nifty["close"].astype(float) if "close" in nifty else None
    sector_histories = fetch_sector_histories() if benchmark_close is not None else {}
    print(f"[REGIME]: {regime_info['regime']} | {strategy} — {regime_info['desc']}")

    universe = get_scan_universe(memory)
    save_memory(memory)
    price_frames = {}
    tracked_tickers = set(account["positions"]) | {
        entry.get("ticker") for entry in account["pending_entries"]}
    results = scan_defensive(price_frames) if strategy == "DEFENSIVE" else []
    training_samples, filter_rejections = [], {}
    valid_data_count = invalid_data_count = download_error_count = 0
    stale_count = anomaly_count = illiquid_count = price_action_count = strategy_setup_count = 0
    download_error_types = {}
    signal_dates = {}
    print(f"[SCAN]: {len(universe)} NSE symbols; {NSE_BATCH_SIZE} symbols per Yahoo batch.")

    processed = 0
    for start in range(0, len(universe), NSE_BATCH_SIZE):
        tickers = universe[start:start + NSE_BATCH_SIZE]
        try:
            downloaded = yf.download(tickers=tickers, period="1y", progress=False,
                                     timeout=30, group_by="ticker", auto_adjust=True,
                                     threads=True)
        except Exception as exc:
            download_error_count += len(tickers)
            kind = type(exc).__name__
            download_error_types[kind] = download_error_types.get(kind, 0) + len(tickers)
            continue

        for ticker in tickers:
            processed += 1
            df = extract_ticker_frame(downloaded, ticker, requested_tickers=tickers)
            if not has_usable_price_history(df, minimum_rows=200, require_volume=True):
                invalid_data_count += 1
                continue
            valid_data_count += 1
            clean = ticker[:-3] if ticker.endswith(".NS") else ticker
            signal_dates[clean] = str(pd.Timestamp(df.index[-1]).date())
            if clean in tracked_tickers:
                price_frames[clean] = df
            if is_history_stale(df):
                stale_count += 1
                continue
            anomalous = has_suspect_price_discontinuity(df)
            if anomalous:
                anomaly_count += 1
            turnover = average_daily_turnover(df)
            if turnover < MIN_AVG_DAILY_TURNOVER_INR:
                illiquid_count += 1
            sector = sector_for_ticker(clean)
            sector_close = sector_histories.get(sector) if sector else None
            features = None
            if benchmark_close is not None:
                features = build_price_action_feature_frame(df, benchmark_close, sector_close)
                if turnover >= MIN_AVG_DAILY_TURNOVER_INR and not anomalous:
                    training_samples.extend(build_labeled_training_samples(
                        df, benchmark_close, ticker=clean, feature_frame=features,
                        sector_close=sector_close))

            try:
                result = None
                if strategy == "BREAKOUT":
                    result = scan_breakout(df, clean, diagnostics=filter_rejections)
                elif strategy == "DIP_BUYING":
                    result = scan_dip_buying(df, clean, diagnostics=filter_rejections)
                if result:
                    strategy_setup_count += 1
                if strategy in {"BREAKOUT", "DIP_BUYING"}:
                    pattern = scan_price_action(df, clean, diagnostics=filter_rejections)
                    if pattern:
                        price_action_count += 1
                        if result:
                            result["score"] = min(100, result["score"] + 5)
                            result["signals"] = list(dict.fromkeys(result["signals"] + pattern["signals"]))
                        else:
                            result = pattern
                    if result is None and strategy == "BREAKOUT":
                        result = scan_prebreakout(df, clean, diagnostics=filter_rejections)
                if result:
                    if anomalous:
                        filter_rejections["DATA:extreme_recent_move"] = \
                            filter_rejections.get("DATA:extreme_recent_move", 0) + 1
                        continue
                    if turnover < MIN_AVG_DAILY_TURNOVER_INR:
                        filter_rejections["DATA:low_turnover"] = \
                            filter_rejections.get("DATA:low_turnover", 0) + 1
                        continue
                    if benchmark_close is None or features is None:
                        continue
                    final_features = features.iloc[-1]
                    result["avg_turnover_inr"] = round(turnover)
                    result["market_rs_pct"] = round(float(final_features["relative_strength60_pct"]), 2)
                    sector_rs = final_features["sector_relative_strength60_pct"]
                    result["sector"] = sector or "NIFTY fallback"
                    result["sector_rs_pct"] = round(float(sector_rs), 2) if pd.notna(sector_rs) else None
                    if result["market_rs_pct"] > 0:
                        result["score"] += 4
                        result["signals"].append("Outperforming NIFTY 60D")
                    elif result["market_rs_pct"] < -10:
                        result["score"] -= 5
                        result["signals"].append("Lagging NIFTY 60D")
                    if result["sector_rs_pct"] is not None and result["sector_rs_pct"] > 0:
                        result["score"] += 3
                        result["signals"].append(f"Outperforming {sector} 60D")
                    result["ai_features"] = final_features[AI_FEATURES].replace(
                        [np.inf, -np.inf], np.nan).fillna(0.0).astype(float).to_dict()
                    results.append(result)
                    price_frames[clean] = df
            except Exception as exc:
                kind = type(exc).__name__
                download_error_types[kind] = download_error_types.get(kind, 0) + 1

    model, ai_summary = train_local_model(training_samples) if benchmark_close is not None else (
        None, {"status": "NO_BENCHMARK_DATA", "samples": 0, "validation_auc": None})
    ai_summary["samples"] = len(training_samples)
    memory["ai_training_summary"] = {k: v for k, v in ai_summary.items()
                                      if k not in {"model", "pipeline"}}

    for pick in results:
        frame = price_frames.get(pick["ticker"])
        probability = (ai_probability(model, pick, frame, benchmark_close)
                       if frame is not None and pick.get("strategy") != "DEFENSIVE" else None)
        pick["ai_score"] = round(probability, 4) if probability is not None else None
    ranking_mode = "LOCAL_LOGISTIC_ML" if model is not None else "RULE_SCORE_FALLBACK"
    if model is not None:
        results.sort(key=lambda pick: (pick.get("ai_score") or 0.0, pick["score"]), reverse=True)
    else:
        results.sort(key=lambda pick: pick["score"], reverse=True)

    paper_activity = process_paper_account(memory, price_frames)
    account = ensure_paper_account(memory)
    queued_tickers = {entry.get("ticker") for entry in account["pending_entries"]}
    open_tickers = set(account["positions"])
    slots = max(0, MAX_POSITIONS_ACTIVE - len(open_tickers) - len(queued_tickers))
    top_picks = []
    for pick in results:
        if len(top_picks) >= slots:
            break
        if pick["ticker"] in queued_tickers or pick["ticker"] in open_tickers:
            continue
        top_picks.append(pick)
    queue_paper_entries(account, top_picks, signal_dates)
    paper_equity_now = paper_equity(account, price_frames)
    save_memory(memory)

    print(f"[COVERAGE]: {valid_data_count}/{len(universe)} valid OHLCV; "
          f"{invalid_data_count} invalid; {download_error_count} batch failures; "
          f"{stale_count} stale; {anomaly_count} extreme-move stocks flagged.")
    print(f"[AI]: {ai_summary} | ranker={ranking_mode}")
    print(f"[PAPER]: entered={paper_activity['entered']}, closed={paper_activity['closed']}, "
          f"rejected={paper_activity['rejected']}, queued={len(account['pending_entries'])}, "
          f"open={len(account['positions'])}, equity=Rs. {paper_equity_now:,.2f}")
    if filter_rejections:
        top_rejections = sorted(filter_rejections.items(), key=lambda item: item[1], reverse=True)[:6]
        print("[FILTER REJECTIONS]: " + ", ".join(f"{name}={count}" for name, count in top_rejections))
    if download_error_types:
        print(f"[SCAN ERROR TYPES]: {download_error_types}")

    closed_summary = paper_performance_summary(account)
    group_summary = paper_group_performance(account)
    strategy_stats = [f"{name} n={stats['n']} avg={stats['avg_r']}R"
                      for name, stats in group_summary["strategy"].items() if stats["n"] >= 3]
    today = datetime.now().strftime("%A, %d %b %Y")
    counts = memory.get("nse_source_counts", {})
    msg = ("========================================\n"
           f"⚡ ARTHA {BOT_VERSION} | AI PAPER-TRADING REPORT\n📅 {today}\n"
           "========================================\n\n"
           "PAPER ONLY — no broker connected, no real orders are placed.\n\n"
           "[MARKET / SCAN]\n"
           f"Regime: {regime_info['regime']} | Mode: {strategy}\n"
           f"Universe: {counts.get('source', 'unknown')} | {len(universe)} symbols\n"
           f"Valid / invalid / failed: {valid_data_count} / {invalid_data_count} / {download_error_count}\n"
           f"Stale / extreme-move flags: {stale_count} / {anomaly_count}\n"
           f"Strategy setups: {strategy_setup_count} | Price-action patterns: {price_action_count}\n\n"
           "[LOCAL ML RANKER]\n"
           f"Status: {ai_summary.get('status')} | matured samples: {ai_summary.get('samples', 0)}\n"
           f"Temporal validation AUC: {ai_summary.get('validation_auc', 'n/a')} | "
           f"ranking mode: {ranking_mode}\n"
           "AI scores are research estimates, not calibrated win probabilities.\n\n"
           "[PAPER ACCOUNT]\n"
           f"Equity: Rs. {paper_equity_now:,.2f} | Cash: Rs. {account['cash']:,.2f} | "
           f"Open positions: {len(account['positions'])}/{MAX_POSITIONS_ACTIVE}\n"
           f"Queued for next session: {len(account['pending_entries'])} | "
           f"Entered / closed / rejected this run: {paper_activity['entered']} / "
           f"{paper_activity['closed']} / {paper_activity['rejected']}\n"
           f"Recent closed paper trades ({closed_summary['trades']}): "
           f"{closed_summary['win_rate_pct']}% wins, avg {closed_summary['average_r']}R | "
           f"max drawdown {closed_summary['max_drawdown_pct']}%\n"
           f"Closed paper P&L (retained ledger): Rs. {paper_activity['realized_pnl']:,.2f}\n"
           f"Strategy average R (n≥3): {', '.join(strategy_stats) or 'not enough samples'}\n\n")
    if top_picks:
        msg += "[NEXT-SESSION PAPER QUEUE — NOT AN ORDER]\n"
        for i, pick in enumerate(top_picks, 1):
            ai_text = f"{pick['ai_score']:.3f}" if pick.get("ai_score") is not None else "not active"
            trigger_text = (f"conditional trigger Rs. {pick['entry_trigger']} | "
                            if pick.get("entry_trigger") is not None else "")
            msg += (f"#{i} {pick['ticker']} | {pick['strategy']} | rule {pick['score']}/100 | "
                    f"ML score {ai_text}\n"
                    f"Reference Rs. {pick['price']} | {trigger_text}stop Rs. {pick['sl']} | "
                    f"targets Rs. {pick['tgt1']} / Rs. {pick['tgt2']}\n"
                    f"Signals: {', '.join(pick['signals'])}\n\n")
    elif strategy == "NO_TRADE":
        msg += "No paper entries queued: NIFTY benchmark data unavailable; fail-safe mode.\n"
    elif strategy == "DEFENSIVE" and not top_picks:
        msg += "No defensive ETF passed this run; no paper entry queued.\n"
    else:
        msg += "No setup passed this run; paper account remains in cash/holds existing positions.\n"
    if filter_rejections:
        common = sorted(filter_rejections.items(), key=lambda item: item[1], reverse=True)[:5]
        msg += "\n[TOP REJECTION REASONS] " + ", ".join(
            f"{name.split(':')[-1]} {count}" for name, count in common) + "\n"
    msg += ("\n[DEFAULT RISK SETTINGS]\n"
            f"Capital Rs. {TOTAL_CAPITAL:,.0f}; risk/trade {RISK_PER_TRADE_PCT}%; "
            f"portfolio heat {PORTFOLIO_HEAT_PCT}%; position value cap {MAX_POSITION_VALUE_PCT}%.\n"
            "Validate paper performance over multiple market regimes before considering any live use.\n")
    send_telegram(msg)
    print(f"✅ ARTHA {BOT_VERSION} paper run complete.")


if __name__ == "__main__":
    main()
