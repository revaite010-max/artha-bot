"""
ARTHA v17.0 - Full NSE+BSE Universe + Adaptive Learning
Auto-Switches Between:
 1. Bull Mode     --> Breakouts & VCP Patterns
 2. Sideways Mode --> Dip-Buying & Mean Reversion (Quality Bounces)
 3. Bear Mode     --> Defensive Assets & Gold ETFs

Persists daily OHLCV/features and trains an explainable online logistic model
from five-session simulated outcomes. It does not place real orders.
"""

import hashlib
import io
import json
import math
import os
import re
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
BOT_TAGLINE = "Adaptive Learning Trading Research Bot"

MEMORY_FILE = "artha_memory.json"

TOTAL_CAPITAL       = 100000  # Rs. 1,00,000 Base Capital
MAX_PORTFOLIO_HEAT  = 5       # Max 5% total risk at any time
MAX_POSITIONS_ACTIVE = 3       # Max 3 actionable trade setups
MOMENTUM_LOOKBACK_BARS = 42    # Approximately two months of NSE trading sessions
MIN_TWO_MONTH_RETURN_PCT = 5.0 # Include every universe stock meeting this return threshold
AI_FORWARD_SESSIONS = 5
AI_MIN_TRAINING_SAMPLES = 30
MAX_MARKET_SNAPSHOTS = 12  # Five forward sessions plus a small holiday/retry buffer
MAX_SNAPSHOT_STOCKS = 300  # Bound persisted training data when scanning the full exchange universe
MAX_AI_OUTCOMES = 2000
YFINANCE_BATCH_SIZE = 50
MARKET_DATA_SOURCE = "Yahoo Finance via yfinance"
AI_PRIOR_WEIGHTS = [0.45, 0.30, 0.20, 0.30, -0.25, 0.05, 0.25]
AI_PRIOR_BIAS = -0.15

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
        "market_snapshots": [],
        "exchange_universe": [],
        "exchange_universe_updated_at": None,
        "exchange_universe_counts": {},
        "ai_outcomes": [],
        "ai_model": {"weights": AI_PRIOR_WEIGHTS.copy(), "bias": AI_PRIOR_BIAS,
                     "sample_count": 0, "updated_at": None},
        "mode_stats": {
            "BREAKOUT": {"wins": 0, "losses": 0},
            "DIP_BUYING": {"wins": 0, "losses": 0},
            "DEFENSIVE": {"wins": 0, "losses": 0}
        }
    }
    if not os.path.exists(MEMORY_FILE):
        return defaults

    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        if not isinstance(memory, dict):
            raise ValueError("memory root must be a JSON object")
        for key, value in defaults.items():
            if key not in memory or not isinstance(memory[key], type(value)):
                memory[key] = value
        model = memory.get("ai_model")
        if (not isinstance(model, dict) or not isinstance(model.get("weights"), list)
                or len(model["weights"]) != len(AI_PRIOR_WEIGHTS)):
            memory["ai_model"] = defaults["ai_model"].copy()
        else:
            model.setdefault("bias", AI_PRIOR_BIAS)
            model.setdefault("sample_count", 0)
            model.setdefault("updated_at", None)
        return memory
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        # Keep a damaged file for diagnosis rather than silently overwriting it.
        print(f"Could not load {MEMORY_FILE}: {exc}. Starting with fresh memory.")
        return defaults

def save_memory(memory):
    """Write atomically so a terminated run cannot leave truncated JSON."""
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
    """Flatten yfinance's version-dependent single-ticker column layout."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        price_fields = {"open", "high", "low", "close", "adj close", "volume"}
        columns = []
        for column in normalized.columns:
            parts = [str(part).strip().lower() for part in column]
            field = next((part for part in parts if part in price_fields), parts[0])
            columns.append(field.replace(" ", "_"))
        normalized.columns = columns
    else:
        normalized.columns = [str(column).strip().lower().replace(" ", "_")
                              for column in normalized.columns]
    return normalized


def has_usable_price_history(frame, minimum_rows=200, require_volume=False):
    required = {"open", "high", "low", "close"}
    if require_volume:
        required.add("volume")
    if frame is None or len(frame) < minimum_rows or not required.issubset(frame.columns):
        return False
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if not (prices.notna().all().all() and (prices > 0).all().all()):
        return False
    if require_volume:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        return bool(volume.notna().all() and (volume >= 0).all())
    return True


def size_positions(picks):
    """Size positions within both per-trade risk and portfolio heat/capital limits."""
    remaining_risk = TOTAL_CAPITAL * MAX_PORTFOLIO_HEAT / 100
    remaining_capital = TOTAL_CAPITAL
    sized = []
    for pick in picks:
        risk_per_share = pick["price"] - pick["sl"]
        if risk_per_share <= 0 or pick["price"] <= 0:
            continue
        trade_risk_budget = min(TOTAL_CAPITAL * 0.015, remaining_risk)
        shares = min(int(trade_risk_budget / risk_per_share),
                     int(remaining_capital / pick["price"]))
        if shares < 1:
            continue
        allocation = shares * pick["price"]
        risk_amount = shares * risk_per_share
        sized.append({"pick": pick, "shares": shares, "allocation": allocation,
                      "risk_amount": risk_amount})
        remaining_capital -= allocation
        remaining_risk -= risk_amount
    return sized


def scan_two_month_momentum(frame, ticker):
    """Return a stock's trailing ~2-month performance when it clears the configured bar."""
    bars_back = MOMENTUM_LOOKBACK_BARS
    if frame is None or len(frame) <= bars_back or "close" not in frame.columns:
        return None
    close = pd.to_numeric(frame["close"], errors="coerce")
    current = close.iloc[-1]
    prior = close.iloc[-(bars_back + 1)]
    if pd.isna(current) or pd.isna(prior) or current <= 0 or prior <= 0:
        return None
    return_pct = (current / prior - 1) * 100
    if return_pct < MIN_TWO_MONTH_RETURN_PCT:
        return None
    return {"ticker": ticker, "price": round(float(current), 2),
            "return_2m_pct": round(float(return_pct), 2),
            "start_date": str(frame.index[-(bars_back + 1)].date()),
            "end_date": str(frame.index[-1].date())}


def _clip(value, low=-2.0, high=2.0):
    return max(low, min(high, float(value)))


def build_ai_features(frame, market_return_20d=0.0):
    """Build a fixed-scale, explainable feature vector from daily OHLCV data."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    current = float(close.iloc[-1])
    ret_42 = (current / float(close.iloc[-43]) - 1) * 100
    ret_20 = (current / float(close.iloc[-21]) - 1) * 100
    rsi = float(get_rsi(close, 14).iloc[-1])
    ema50 = float(get_ema(close, 50).iloc[-1])
    atr = float(get_atr(frame["high"], frame["low"], close, 14).iloc[-1])
    if "volume" in frame:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        avg_volume = float(volume.rolling(20).mean().iloc[-1])
        latest_volume = float(volume.iloc[-1])
    else:
        avg_volume = latest_volume = float("nan")
    volume_ratio = (latest_volume / avg_volume
                    if pd.notna(latest_volume) and pd.notna(avg_volume) and avg_volume > 0
                    else 1.0)
    features = [
        _clip(ret_42 / 20),
        _clip(ret_20 / 15),
        _clip((rsi - 50) / 25),
        _clip(((current / ema50) - 1) * 100 / 10),
        _clip((atr / current) * 100 / 6, 0, 2),
        _clip(volume_ratio / 2.5 - 1, -1, 2),
        _clip(float(market_return_20d) / 10),
    ]
    if not all(pd.notna(value) for value in features):
        raise ValueError("AI features contain invalid market data")
    return features


def predict_ai_probability(features, model):
    """Estimate five-session success probability using the persistent logistic model."""
    weights = model.get("weights", AI_PRIOR_WEIGHTS)
    bias = float(model.get("bias", AI_PRIOR_BIAS))
    logit = bias + sum(float(w) * float(x) for w, x in zip(weights, features))
    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def train_ai_model(model, features, label, learning_rate=0.03, l2=0.001):
    """One online logistic-regression update from a matured paper-trade outcome."""
    probability = predict_ai_probability(features, model)
    error = float(label) - probability
    weights = model.setdefault("weights", AI_PRIOR_WEIGHTS.copy())
    model["bias"] = float(model.get("bias", AI_PRIOR_BIAS)) + learning_rate * error
    model["weights"] = [
        max(-5.0, min(5.0, float(weight) + learning_rate *
                      (error * float(feature) - l2 * float(weight))))
        for weight, feature in zip(weights, features)
    ]
    model["sample_count"] = int(model.get("sample_count", 0)) + 1
    model["updated_at"] = datetime.now().isoformat()
    return probability


def append_market_snapshot(memory, snapshot):
    """Persist a bounded, mixed exploration/ranked sample of daily market features."""
    stocks = snapshot.get("stocks", {})
    if len(stocks) > MAX_SNAPSHOT_STOCKS:
        ranked = sorted(stocks, key=lambda ticker: stocks[ticker].get("ai_probability", 0),
                        reverse=True)
        ranked_count = MAX_SNAPSHOT_STOCKS // 2
        selected = set(ranked[:ranked_count])
        remainder = [ticker for ticker in stocks if ticker not in selected]
        remainder.sort(key=lambda ticker: hashlib.sha256(
            f"{snapshot['session_date']}:{ticker}".encode()).hexdigest())
        selected.update(remainder[:MAX_SNAPSHOT_STOCKS - len(selected)])
        snapshot["stocks"] = {ticker: stocks[ticker] for ticker in sorted(selected)}
    snapshot["sampled_stock_count"] = len(snapshot.get("stocks", {}))

    snapshots = memory.setdefault("market_snapshots", [])
    session_date = snapshot["session_date"]
    existing = next((item for item in snapshots if item.get("session_date") == session_date), None)
    if existing:
        snapshot["trained"] = existing.get("trained", False)
        snapshots[snapshots.index(existing)] = snapshot
    else:
        snapshots.append(snapshot)
    snapshots.sort(key=lambda item: item.get("session_date", ""))
    memory["market_snapshots"] = snapshots[-MAX_MARKET_SNAPSHOTS:]


def learn_from_matured_snapshots(memory):
    """Label outcomes after five later market sessions and update the online model."""
    snapshots = memory.setdefault("market_snapshots", [])
    model = memory.setdefault("ai_model", {"weights": AI_PRIOR_WEIGHTS.copy(),
                                            "bias": AI_PRIOR_BIAS, "sample_count": 0})
    outcomes = memory.setdefault("ai_outcomes", [])
    learned = 0
    for index, snapshot in enumerate(snapshots):
        if snapshot.get("trained") or len(snapshots) - index - 1 < AI_FORWARD_SESSIONS:
            continue
        future = snapshots[index + 1:index + 1 + AI_FORWARD_SESSIONS]
        for ticker, entry in snapshot.get("stocks", {}).items():
            entry_price = float(entry.get("close", 0))
            features = entry.get("features")
            if entry_price <= 0 or not isinstance(features, list) or len(features) != len(AI_PRIOR_WEIGHTS):
                continue
            future_rows = [day.get("stocks", {}).get(ticker) for day in future]
            future_rows = [row for row in future_rows if row and float(row.get("close", 0)) > 0]
            if len(future_rows) < 3:
                continue

            stop = float(entry.get("stop", entry_price * 0.98))
            target = float(entry.get("target", entry_price * 1.02))
            exit_price = float(future_rows[-1]["close"])
            reason = "five_session_exit"
            label = int(exit_price > entry_price)
            for row in future_rows:
                low = float(row.get("low", row["close"]))
                high = float(row.get("high", row["close"]))
                # If both levels appear in one daily bar, conservatively count the stop first.
                if low <= stop:
                    exit_price, label, reason = stop, 0, "stop_loss"
                    break
                if high >= target:
                    exit_price, label, reason = target, 1, "target_hit"
                    break
            predicted_probability = train_ai_model(model, features, label)
            outcomes.append({"session_date": snapshot.get("session_date"),
                             "ticker": ticker, "predicted_probability": round(predicted_probability, 4),
                             "label": label,
                             "return_pct": round((exit_price / entry_price - 1) * 100, 2),
                             "exit_reason": reason})
            learned += 1
        snapshot["trained"] = True

    if len(outcomes) > MAX_AI_OUTCOMES:
        memory["ai_outcomes"] = outcomes[-MAX_AI_OUTCOMES:]
    return learned

# ============================================================
# 🎯 1. ADAPTIVE MARKET REGIME ENGINE
# ============================================================
def detect_market_regime():
    """Determine market state; missing/invalid data must never enable risk-taking."""
    try:
        nifty = normalize_yfinance_columns(
            yf.download("^NSEI", period="1y", progress=False, timeout=15))
        if not has_usable_price_history(nifty):
            raise ValueError("insufficient or invalid NIFTY price history")

        close = nifty["close"].astype(float)
        ema20 = get_ema(close, 20).iloc[-1]
        ema50 = get_ema(close, 50).iloc[-1]
        ema200 = get_ema(close, 200).iloc[-1]
        curr = close.iloc[-1]
        change_20d = ((curr - close.iloc[-20]) / close.iloc[-20]) * 100
        rsi_val = get_rsi(close, 14).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100
        if not all(pd.notna(value) for value in
                   (ema20, ema50, ema200, curr, change_20d, rsi_val, volatility)):
            raise ValueError("market indicators are not ready")

        if curr > ema20 > ema50 > ema200 and change_20d > 1.5:
            regime, strategy, desc = "STRONG_BULL", "BREAKOUT", "Full Momentum & VCP Breakouts Active"
        elif curr > ema50 and change_20d > -1.0:
            regime, strategy, desc = "SIDEWAYS_BULL", "DIP_BUYING", "Buy-The-Dip Mode Active (Quality Bounces at Support)"
        elif curr < ema50 and curr > ema200:
            regime, strategy, desc = "SIDEWAYS_BEAR", "DIP_BUYING", "Strict Mean Reversion Mode (Deep Support Bounces Only)"
        else:
            regime, strategy, desc = "BEAR", "DEFENSIVE", "Capital Protection Mode (Gold ETFs & Safe Assets Only)"

        return {"regime": regime, "strategy": strategy, "desc": desc,
                "nifty_price": round(curr, 2), "change_20d": round(change_20d, 2),
                "rsi": round(rsi_val, 1), "volatility": round(volatility, 2),
                "as_of_date": str(nifty.index[-1].date())}
    except Exception as exc:
        print(f"Market regime unavailable: {exc}")
        return {"regime": "DATA_UNAVAILABLE", "strategy": "NO_TRADE",
                "desc": "Market data unavailable; no new positions will be suggested",
                "nifty_price": 0, "change_20d": 0, "rsi": 0, "volatility": 0,
                "as_of_date": ""}

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

# ============================================================
# 🎯 4. STRATEGY C: DEFENSIVE & GOLD ALLOCATOR (For Bear Markets)
# ============================================================
def scan_defensive():
    """Scans Gold ETFs, Silver ETFs & Liquid assets during Bear Markets"""
    defensive_universe = ["GOLDBEES.NS", "SILVERBEES.NS", "GOLDSHARE.NS", "HDFCGOLD.NS"]
    picks = []
    
    for ticker in defensive_universe:
        try:
            df = normalize_yfinance_columns(
                yf.download(ticker, period="1y", progress=False, timeout=15))
            if not has_usable_price_history(df, require_volume=True):
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
NSE_EQUITY_MASTER_URLS = [
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
]
BSE_EQUITY_MASTER_URLS = [
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData_new/w?Group=&Scripcode=&segment=Equity&status=Active&scripName=",
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?segment=Equity&status=Active",
]


def _normalise_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _row_value(row, *names):
    lookup = {_normalise_header(key): value for key, value in row.items()}
    for name in names:
        value = lookup.get(_normalise_header(name))
        if value is not None and not pd.isna(value) and str(value).strip().lower() not in {"", "nan", "none", "null", "-"}:
            return str(value).strip()
    return ""


def _clean_isin(value):
    isin = str(value or "").strip().upper()
    return isin if re.fullmatch(r"IN[A-Z0-9]{10}", isin) else ""


def parse_nse_master(csv_content):
    """Parse an NSE equity/SME master CSV into normalized securities."""
    table = pd.read_csv(io.BytesIO(csv_content), dtype=str, encoding="utf-8-sig")
    records = []
    for row in table.to_dict(orient="records"):
        symbol = _row_value(row, "SYMBOL", "SYMBOL NAME")
        if not symbol:
            continue
        isin = _clean_isin(_row_value(row, "ISIN NUMBER", "ISIN", "ISIN_NUMBER"))
        records.append({"symbol": symbol, "ticker": f"{symbol}.NS", "exchange": "NSE",
                        "isin": isin})
    return records


def _extract_bse_rows(payload):
    if isinstance(payload, list):
        return payload if not payload or isinstance(payload[0], dict) else []
    if isinstance(payload, dict):
        list_values = [value for value in payload.values()
                       if isinstance(value, list) and value and isinstance(value[0], dict)]
        if list_values:
            return max(list_values, key=len)
        for value in payload.values():
            if isinstance(value, dict):
                rows = _extract_bse_rows(value)
                if rows:
                    return rows
    return []


def parse_bse_master(payload):
    """Parse BSE's active equity scrip-master response."""
    records = []
    for row in _extract_bse_rows(payload):
        status = _row_value(row, "Status", "SCRIP_STATUS").lower()
        segment = _row_value(row, "Segment", "SEGMENT").lower()
        if status and status != "active":
            continue
        if segment and "equity" not in segment:
            continue
        code = _row_value(row, "SCRIP_CD", "SCRIPCODE", "Security Code")
        if code.endswith(".0"):
            code = code[:-2]
        if not code or not code.isdigit():
            continue
        symbol = _row_value(row, "scrip_id", "SYMBOL", "Security Id") or code
        isin = _clean_isin(_row_value(row, "ISIN_NUMBER", "ISIN No", "ISIN"))
        records.append({"symbol": symbol, "ticker": f"{code}.BO", "exchange": "BSE",
                        "isin": isin})
    return records


def merge_exchange_universes(nse_records, bse_records):
    """Deduplicate dual-listed shares by ISIN; keep the NSE Yahoo ticker when possible."""
    instruments = {}
    isin_to_key = {}
    for record in nse_records:
        isin = record.get("isin", "")
        key = f"ISIN:{isin}" if isin else f"NSE:{record['ticker']}"
        current = instruments.setdefault(key, {
            "ticker": record["ticker"], "symbol": record["symbol"],
            "isin": isin, "exchanges": []})
        if "NSE" not in current["exchanges"]:
            current["exchanges"].append("NSE")
        if isin:
            isin_to_key[isin] = key

    for record in bse_records:
        isin = record.get("isin", "")
        key = isin_to_key.get(isin) if isin else None
        if key is None:
            key = f"ISIN:{isin}" if isin else f"BSE:{record['ticker']}"
            instruments.setdefault(key, {
                "ticker": record["ticker"], "symbol": record["symbol"],
                "isin": isin, "exchanges": []})
            if isin:
                isin_to_key[isin] = key
        current = instruments[key]
        if "BSE" not in current["exchanges"]:
            current["exchanges"].append("BSE")

    return sorted(instruments.values(), key=lambda item: item["ticker"])


def _exchange_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/json,text/plain,*/*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    })
    return session


def fetch_exchange_universe():
    """Fetch NSE mainboard+SME and BSE active equity lists; fail on partial masters."""
    session = _exchange_http_session()
    nse_records = []
    for url in NSE_EQUITY_MASTER_URLS:
        response = session.get(url, timeout=30,
                               headers={"Referer": "https://www.nseindia.com/"})
        response.raise_for_status()
        parsed = parse_nse_master(response.content)
        if not parsed:
            raise ValueError(f"NSE master returned no equities: {url}")
        nse_records.extend(parsed)

    try:
        session.get("https://www.bseindia.com/", timeout=20)
    except requests.RequestException:
        pass  # API requests below still get browser-style headers and their own timeout.
    bse_error = None
    bse_records = []
    for url in BSE_EQUITY_MASTER_URLS:
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            bse_records = parse_bse_master(response.json())
            if len(bse_records) < 1000:
                raise ValueError("BSE master response is empty or unexpectedly small")
            break
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            bse_error = exc
    if len(nse_records) < 1000:
        raise ValueError("NSE masters returned fewer than 1,000 equities")
    if not bse_records:
        raise RuntimeError(f"Unable to fetch the BSE active-equity master: {bse_error}")

    instruments = merge_exchange_universes(nse_records, bse_records)
    if len(instruments) < 3000:
        raise ValueError(f"Merged exchange universe unexpectedly small: {len(instruments)}")
    counts = {"nse_master_rows": len(nse_records), "bse_master_rows": len(bse_records),
              "unique_securities": len(instruments),
              "dual_listed": sum(1 for item in instruments if len(item["exchanges"]) > 1)}
    return instruments, counts


def get_scan_universe(memory):
    """Refresh the complete listed-equity universe, using persisted cache only on source failure."""
    try:
        instruments, counts = fetch_exchange_universe()
        memory["exchange_universe"] = instruments
        memory["exchange_universe_counts"] = counts
        memory["exchange_universe_updated_at"] = datetime.now().isoformat()
        print(f"[UNIVERSE]: Refreshed full NSE+BSE equity masters: {counts}")
        return instruments
    except Exception as exc:
        cached = [item for item in memory.get("exchange_universe", [])
                  if isinstance(item, dict) and item.get("ticker") and item.get("symbol")]
        if cached:
            print(f"[UNIVERSE WARNING]: Refresh failed ({exc}); using cached list from "
                  f"{memory.get('exchange_universe_updated_at') or 'unknown date'}.")
            return cached
        raise RuntimeError(f"Could not build the full NSE+BSE universe and no cache exists: {exc}") from exc


def extract_ticker_frame(downloaded, ticker, requested_tickers=None):
    """Extract one symbol from yfinance's batched, version-dependent MultiIndex output."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        target = ticker.strip().upper()
        for level in range(downloaded.columns.nlevels):
            values = downloaded.columns.get_level_values(level)
            matches = {str(value).strip().upper(): value for value in values}
            if target in matches:
                try:
                    return normalize_yfinance_columns(
                        downloaded.xs(matches[target], axis=1, level=level, drop_level=True))
                except (KeyError, ValueError):
                    pass
        return pd.DataFrame()
    if requested_tickers and (len(requested_tickers) != 1 or
                              requested_tickers[0].upper() != ticker.upper()):
        return pd.DataFrame()
    return normalize_yfinance_columns(downloaded)


def iter_chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


# ============================================================
# 📤 TELEGRAM MESSAGING ENGINE (Safe Text Formatting)
# ============================================================
def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing. Outputting to console.")
        print(msg)
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    parts = [msg[i:i + 3800] for i in range(0, len(msg), 3800)]
    try:
        with requests.Session() as session:
            for part in parts:
                response = session.post(url, json={"chat_id": CHAT_ID, "text": part}, timeout=15)
                response.raise_for_status()
                time.sleep(0.8)
        return True
    except requests.RequestException as exc:
        # Do not print the exception: its URL can contain the secret bot token.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"Telegram delivery failed{f' (HTTP {status})' if status else ''}.")
        return False

# ============================================================
# 🚀 MAIN ADAPTIVE ENGINE
# ============================================================
def main():
    print("=" * 60)
    print(f"{BOT_NAME} {BOT_VERSION} — {BOT_TAGLINE}")
    print(f"Execution Time: {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print("=" * 60)

    memory = load_memory()
    learned_samples = learn_from_matured_snapshots(memory)

    # STEP 1: DETECT MARKET REGIME & ACTIVE STRATEGY
    regime_info = detect_market_regime()
    regime      = regime_info["regime"]
    strategy    = regime_info["strategy"]
    
    print(f"\n[REGIME DETECTED]: {regime}")
    print(f"[ACTIVE STRATEGY]: {strategy} ({regime_info['desc']})")

    # STEP 2: Scan the full universe for two-month winners on every run.
    # This independent momentum list is not gated by the market-regime strategy.
    universe = get_scan_universe(memory)
    # Persist a successful exchange-master refresh before the lengthy all-market scan.
    save_memory(memory)
    results = []
    momentum_leaders = []
    valid_data_count = 0
    invalid_data_count = 0
    download_error_types = {}
    stock_snapshot_records = {}
    model = memory["ai_model"]
    market_return_20d = regime_info.get("change_20d", 0.0)
    snapshot_date = regime_info.get("as_of_date", "")
    print(f"\n[SCANNING]: Checking {len(universe)} stocks for 2-month momentum"
          f" and {strategy} setups...")

    processed_count = 0
    for chunk in iter_chunks(universe, YFINANCE_BATCH_SIZE):
        tickers = [item["ticker"] for item in chunk]
        try:
            downloaded = yf.download(
                tickers=tickers, period="1y", progress=False, timeout=30,
                group_by="ticker", auto_adjust=True, threads=True)
        except Exception as exc:
            error_type = type(exc).__name__
            download_error_types[error_type] = download_error_types.get(error_type, 0) + len(chunk)
            print(f"[BATCH DOWNLOAD ERROR] {len(chunk)} symbols: {error_type}")
            continue

        for instrument in chunk:
            ticker = instrument["ticker"]
            processed_count += 1
            if processed_count % 250 == 0:
                print(f"  Scanned {processed_count}/{len(universe)} | "
                      f"2M leaders: {len(momentum_leaders)} | setups: {len(results)}")
            df = extract_ticker_frame(downloaded, ticker, requested_tickers=tickers)
            if not has_usable_price_history(
                    df, minimum_rows=MOMENTUM_LOOKBACK_BARS + 1, require_volume=False):
                invalid_data_count += 1
                continue
            valid_data_count += 1

            try:
                features = build_ai_features(df, market_return_20d)
                ai_probability = predict_ai_probability(features, model)
                leader = scan_two_month_momentum(df, ticker)
                if leader:
                    leader["ai_probability"] = ai_probability
                    momentum_leaders.append(leader)

                current = float(df["close"].iloc[-1])
                atr = float(get_atr(df["high"], df["low"], df["close"], 14).iloc[-1])
                if not snapshot_date:
                    snapshot_date = str(df.index[-1].date())
                stop_pct = min(max(1.5 * atr / current, 0.02), 0.06)
                stock_snapshot_records[ticker] = {
                    "open": round(float(df["open"].iloc[-1]), 4),
                    "high": round(float(df["high"].iloc[-1]), 4),
                    "low": round(float(df["low"].iloc[-1]), 4),
                    "close": round(current, 4),
                    "volume": (round(float(df["volume"].iloc[-1]), 2)
                               if "volume" in df and pd.notna(df["volume"].iloc[-1]) else 0.0),
                    "features": features,
                    "ai_probability": round(ai_probability, 5),
                    "stop": round(current * (1 - stop_pct), 4),
                    "target": round(current * (1 + stop_pct), 4),
                }
            except (ValueError, TypeError, IndexError, ZeroDivisionError):
                invalid_data_count += 1
                continue

            # EMA200-based entry strategies need a full year; momentum leaders do not.
            enough_trend_history = has_usable_price_history(df, require_volume=True)
            if strategy == "BREAKOUT" and enough_trend_history:
                result = scan_breakout(df, ticker)
            elif strategy == "DIP_BUYING" and enough_trend_history:
                result = scan_dip_buying(df, ticker)
            else:
                result = None
            if result:
                result["ai_probability"] = ai_probability
                results.append(result)

    if strategy == "DEFENSIVE":
        print("[DEFENSIVE MODE]: Also scanning Gold ETFs & Protection Assets...")
        results.extend(scan_defensive())
    elif strategy == "NO_TRADE":
        print("[NO-TRADE MODE]: No entry setups due to unavailable market-regime data;"
              " historical momentum leaders will still be reported.")

    momentum_leaders.sort(key=lambda item: (item.get("ai_probability", 0),
                                            item["return_2m_pct"]), reverse=True)
    print(f"[DATA COVERAGE]: {valid_data_count}/{len(universe)} stocks valid; "
          f"{invalid_data_count} empty/invalid; {sum(download_error_types.values())} download errors; "
          f"{len(momentum_leaders)} stocks gained at least {MIN_TWO_MONTH_RETURN_PCT:.1f}%"
          f" over {MOMENTUM_LOOKBACK_BARS} sessions.")
    if download_error_types:
        print(f"[DOWNLOAD ERROR TYPES]: {download_error_types}")

    if stock_snapshot_records:
        if not snapshot_date:
            snapshot_date = datetime.now().date().isoformat()
        append_market_snapshot(memory, {
            "session_date": snapshot_date,
            "source": MARKET_DATA_SOURCE,
            "regime": regime,
            "nifty_close": regime_info.get("nifty_price", 0),
            "nifty_return_20d_pct": market_return_20d,
            "stocks": stock_snapshot_records,
            "trained": False,
        })
    print(f"[AI MODEL]: {model.get('sample_count', 0)} matured paper outcomes; "
          f"learned {learned_samples} new labels this run; "
          f"model {'learning' if model.get('sample_count', 0) < AI_MIN_TRAINING_SAMPLES else 'adaptive'}.")

    # Rank picks, then enforce portfolio-wide risk and capital limits.
    results.sort(key=lambda x: (x.get("ai_probability", 0), x["score"]), reverse=True)
    candidate_picks = results[:MAX_POSITIONS_ACTIVE]
    sized_picks = size_positions(candidate_picks)
    top_picks = [position["pick"] for position in sized_picks]

    # Record at most one evaluation per ticker/day, even if a workflow is retried.
    today_key = datetime.now().strftime('%Y%m%d')
    for p in top_picks:
        pick_id = f"{p['ticker']}_{today_key}"
        if pick_id in memory["pending_evaluations"]:
            continue
        memory["pending_evaluations"][pick_id] = {
            "ticker": p['ticker'],
            "date": datetime.now().isoformat(),
            "entry_price": p['price'],
            "sl": p['sl'],
            "tgt1": p['tgt1'],
            "tgt2": p['tgt2'],
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

    msg += "[🧠 AI LEARNING ENGINE]\n"
    msg += f"Market data  : {MARKET_DATA_SOURCE}\n"
    msg += f"Learned labels: {model.get('sample_count', 0)} matured 5-session outcomes\n"
    msg += f"Status       : {'warm-up' if model.get('sample_count', 0) < AI_MIN_TRAINING_SAMPLES else 'online-adaptive'}\n"
    msg += f"Memory       : {len(memory.get('market_snapshots', []))} daily snapshots retained\n"
    msg += (f"Universe     : {len(universe)} securities; NSE source rows "
            f"{memory.get('exchange_universe_counts', {}).get('nse_master_rows', 0)}, "
            f"BSE source rows {memory.get('exchange_universe_counts', {}).get('bse_master_rows', 0)}; "
            f"refreshed {memory.get('exchange_universe_updated_at') or 'unknown'}\n")
    recent_outcomes = memory.get("ai_outcomes", [])[-100:]
    if recent_outcomes:
        hit_rate = sum(int(item.get("label", 0)) for item in recent_outcomes) / len(recent_outcomes)
        msg += f"Recent paper hit rate: {hit_rate:.0%} ({len(recent_outcomes)} outcomes)\n"
    else:
        msg += "Recent paper hit rate: collecting first outcomes\n"
    msg += "Method       : local logistic model; simulated paper outcomes, no order execution\n\n"

    msg += "[📊 SCAN SUMMARY]\n"
    msg += f"Active Strategy : {strategy}\n"
    msg += f"Setups Qualified: {len(results)}\n"
    msg += f"Actionable Picks: {len(top_picks)} (risk/capital capped)\n"
    msg += f"2M Leaders      : {len(momentum_leaders)} stocks at or above "
    msg += f"+{MIN_TWO_MONTH_RETURN_PCT:.1f}%\n"
    msg += f"Data Coverage   : {valid_data_count}/{len(universe)} valid; "
    msg += f"{invalid_data_count} invalid, {sum(download_error_types.values())} download errors\n\n"

    msg += "----------------------------------------\n"
    msg += "[📈 TWO-MONTH MOMENTUM LEADERS]\n"
    msg += f"Trailing {MOMENTUM_LOOKBACK_BARS} trading sessions; all universe stocks "
    msg += f"with return >= +{MIN_TWO_MONTH_RETURN_PCT:.1f}% (no top-N cap).\n"
    msg += "Historical performance screen only, not a guarantee of future returns.\n\n"
    if momentum_leaders:
        for i, leader in enumerate(momentum_leaders, 1):
            msg += (f"{i}. {leader['ticker']} | +{leader['return_2m_pct']:.2f}% 2M | "
                    f"AI 5-session: {leader['ai_probability']:.0%} | "
                    f"CMP Rs. {leader['price']} | {leader['start_date']} to "
                    f"{leader['end_date']}\n")
    else:
        msg += (f"No stocks met the +{MIN_TWO_MONTH_RETURN_PCT:.1f}% threshold "
                "in the available universe.\n")
    msg += "\n"

    if top_picks:
        msg += "----------------------------------------\n"
        msg += f"[🏆 TOP {len(top_picks)} ACTIONABLE SETUPS]\n"
        msg += "----------------------------------------\n\n"

        for i, position in enumerate(sized_picks, 1):
            p = position["pick"]
            shares = position["shares"]
            pos_value = position["allocation"]

            msg += f"#{i} {p['ticker']} | {p['strategy']} MODE\n"
            msg += f"Rule Score : {p['score']}/100\n"
            msg += f"AI 5D Prob : {p.get('ai_probability', 0):.0%}\n"
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
        msg += "No separate entry setup passed the current risk filters.\n"
        msg += "See the two-month momentum list above for historical leaders.\n\n"

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
