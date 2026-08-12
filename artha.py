"""
ARTHA v13.0 - VCP Master Edition
4 Scanners: Safe + VCP + Momentum + Pro
Based on Mark Minervini's SEPA methodology
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

BOT_NAME = "⚡ ARTHA"
BOT_VERSION = "v13.0"
BOT_TAGLINE = "VCP Master Edition"

MEMORY_FILE = "artha_memory.json"

# Capital Allocation (Rebalanced)
TOTAL_CAPITAL = 100000
CAPITAL_SAFE = 60000       # 60% - Safe large caps
CAPITAL_VCP = 20000        # 20% - VCP picks (NEW)
CAPITAL_MOMENTUM = 15000   # 15% - Momentum
CAPITAL_PRO = 5000         # 5% - Pro trades

# Risk Limits
SAFE_MAX_RISK = 1.5
VCP_MAX_RISK = 2.0
MOMENTUM_MAX_RISK = 2.5
PRO_MAX_RISK = 2.0

# Score Thresholds
SAFE_MIN_SCORE = 85
VCP_MIN_SCORE = 80
MOMENTUM_MIN_SCORE = 75
PRO_MIN_SCORE = 80

# ============================================================
# 🧠 MEMORY
# ============================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                mem = json.load(f)
                for key in ['ml_features', 'user_feedback', 'active_positions',
                           'confirmation_queue', 'failed_stocks', 'winning_patterns',
                           'yesterday_gainers', 'watchlist', 'vcp_watchlist']:
                    if key not in mem:
                        mem[key] = {} if 'positions' in key or 'feedback' in key or 'queue' in key or 'stocks' in key or 'patterns' in key or 'gainers' in key or 'watchlist' in key else []
                return mem
        except:
            pass
    
    return {
        "version": "13.0",
        "created": datetime.now().isoformat(),
        "total_picks": 0,
        "total_wins": 0,
        "total_losses": 0,
        "safe_wins": 0, "safe_losses": 0,
        "vcp_wins": 0, "vcp_losses": 0,
        "momentum_wins": 0, "momentum_losses": 0,
        "pro_wins": 0, "pro_losses": 0,
        "pending_evaluations": {},
        "confirmation_queue": {},
        "completed_trades": [],
        "sector_performance": {},
        "learning_insights": [],
        "user_feedback": {},
        "failed_stocks": {},
        "winning_patterns": {},
        "yesterday_gainers": [],
        "watchlist": {},
        "vcp_watchlist": {},
        "ml_features": [],
        "weights": {"trend": 1.0, "volume": 1.0, "rsi": 1.0, "sector": 1.0, "momentum": 1.0}
    }

def save_memory(memory):
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

# ============================================================
# 🧮 CORE MATH
# ============================================================

def get_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def get_sma(s, n): return s.rolling(n).mean()

def get_rsi(s, n=14):
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_macd(s):
    ema12 = get_ema(s, 12)
    ema26 = get_ema(s, 26)
    return ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()

def get_atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ============================================================
# 🌍 MARKET REGIME
# ============================================================

def detect_market_regime():
    try:
        nifty = yf.download("^NSEI", period="6mo", progress=False)
        close = nifty['Close'].squeeze()
        ema20 = get_ema(close, 20).iloc[-1]
        ema50 = get_ema(close, 50).iloc[-1]
        ema200 = get_ema(close, 200).iloc[-1]
        curr = close.iloc[-1]
        change_5d = ((curr - close.iloc[-5]) / close.iloc[-5]) * 100
        change_20d = ((curr - close.iloc[-20]) / close.iloc[-20]) * 100
        rsi = get_rsi(close, 14).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100
        
        if curr > ema20 > ema50 > ema200 and change_20d > 3 and rsi > 55:
            regime, allowed = "STRONG_BULL", True
        elif curr > ema20 > ema50 and change_20d > 0:
            regime, allowed = "BULL", True
        elif curr < ema50 and change_20d < -3:
            regime, allowed = "BEAR", False
        elif volatility > 2:
            regime, allowed = "VOLATILE", False
        else:
            regime, allowed = "SIDEWAYS", False
        
        return {
            "regime": regime, "trade_allowed": allowed,
            "change_5d": round(change_5d, 2), "change_20d": round(change_20d, 2),
            "rsi": round(rsi, 1), "volatility": round(volatility, 2),
            "price": round(curr, 2)
        }
    except:
        return {"regime": "UNKNOWN", "trade_allowed": False}

# ============================================================
# 🎨 SECTOR ANALYSIS
# ============================================================

NIFTY_SECTORS = {
    "IT": "^CNXIT", "Banking": "^NSEBANK", "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY", "Finance": "^CNXFIN"
}

def analyze_sectors():
    sector_scores = {}
    for sector, symbol in NIFTY_SECTORS.items():
        try:
            df = yf.download(symbol, period="2mo", progress=False, timeout=10)
            if len(df) < 30: continue
            close = df['Close'].squeeze()
            ret_5d = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100
            ret_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            ema20 = close.ewm(span=20).mean().iloc[-1]
            trend_ok = close.iloc[-1] > ema20
            score = (ret_5d * 0.6) + (ret_20d * 0.4) + (5 if trend_ok else 0)
            sector_scores[sector] = {"score": round(score, 2), "ret_5d": round(ret_5d, 2)}
        except:
            continue
    
    ranked = sorted(sector_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    return {
        "hot_sectors": [s[0] for s in ranked[:3]],
        "cold_sectors": [s[0] for s in ranked[-3:]],
        "all_scores": sector_scores
    }

def detect_stock_sector(ticker):
    sectors = {
        "IT": ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","COFORGE","MPHASIS","PERSISTENT","LTTS","TATAELXSI","KPITTECH","OFSS","SONATSOFTW"],
        "Banking": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK","FEDERALBNK","IDFCFIRSTB","BANKBARODA","PNB","RBLBANK","YESBANK"],
        "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","TVSMOTOR","EXIDEIND","BOSCHLTD","BALKRISIND","MOTHERSON","MRF","APOLLOTYRE"],
        "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","ALKEM","BIOCON","TORNTPHARM","GLENMARK","IPCALAB","LAURUSLABS"],
        "FMCG": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","GODREJCP","MARICO","COLPAL","TATACONSUM","EMAMILTD"],
        "Metal": ["TATASTEEL","JSWSTEEL","HINDALCO","VEDL","COALINDIA","NMDC","SAIL","JINDALSTEL","APLAPOLLO"],
        "Energy": ["RELIANCE","ONGC","IOC","BPCL","GAIL","NTPC","POWERGRID","TATAPOWER","ADANIGREEN","ADANIPOWER"],
        "Finance": ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SHRIRAMFIN","MANAPPURAM","LICHSGFIN","HDFCLIFE","SBILIFE","PFC","RECLTD","IRFC"]
    }
    for sector, stocks in sectors.items():
        if ticker in stocks:
            return sector
    return "Other"

# ============================================================
# 📥 UNIVERSES
# ============================================================

SAFE_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS",
    "AXISBANK.NS","HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS",
    "ASIANPAINT.NS","SUNPHARMA.NS","HCLTECH.NS","TITAN.NS",
    "NTPC.NS","M&M.NS","TATAMOTORS.NS","ULTRACEMCO.NS",
    "WIPRO.NS","NESTLEIND.NS","TATASTEEL.NS","POWERGRID.NS",
    "BAJAJFINSV.NS","TECHM.NS","HDFCLIFE.NS","SBILIFE.NS",
    "ADANIENT.NS","COALINDIA.NS","GRASIM.NS","HINDALCO.NS",
    "DIVISLAB.NS","JSWSTEEL.NS","DRREDDY.NS","CIPLA.NS",
    "APOLLOHOSP.NS","EICHERMOT.NS","BRITANNIA.NS","HEROMOTOCO.NS",
    "TATACONSUM.NS","INDUSINDBK.NS","BAJAJ-AUTO.NS","ADANIPORTS.NS",
    "ONGC.NS","BPCL.NS","LTIM.NS","SHRIRAMFIN.NS"
]

def get_expanded_universe():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        return [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().unique().tolist()]
    except:
        return SAFE_UNIVERSE

# ============================================================
# 🎯 SCANNER 1: SAFE PICKS
# ============================================================

def safe_scanner(ticker, nifty_close, memory, sector_data, regime):
    try:
        ticker_clean = ticker.replace(".NS", "")
        if is_recently_failed(ticker_clean, memory): return None
        
        df = yf.download(ticker, period="6mo", progress=False, timeout=10)
        if len(df) < 100: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        open_ = df['Open'].squeeze()
        
        if vol.mean() < 500000: return None
        if close.iloc[-1] < 100: return None
        
        ema9 = get_ema(close, 9)
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr = get_atr(high, low, close, 14)
        
        c = len(df) - 1
        curr = close.iloc[c]
        atr_val = atr.iloc[c]
        
        if not (curr > ema9.iloc[c] > ema20.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
        if ema20.iloc[c] <= ema20.iloc[c-5]: return None
        if not (55 <= rsi.iloc[c] <= 68): return None
        if not (macd.iloc[c] > msig.iloc[c] > 0): return None
        
        avg_vol_20 = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol_20 if avg_vol_20 > 0 else 0
        if vol_ratio < 1.5: return None
        
        recent_5d_high = high.iloc[-5:].max()
        prev_20d_high = high.iloc[-25:-5].max()
        if recent_5d_high <= prev_20d_high: return None
        
        dist_from_ema20 = ((curr - ema20.iloc[c]) / ema20.iloc[c]) * 100
        if dist_from_ema20 > 8 or dist_from_ema20 < 0: return None
        
        if close.iloc[c] <= open_.iloc[c]: return None
        
        stock_sector = detect_stock_sector(ticker_clean)
        if stock_sector in sector_data['cold_sectors']: return None
        
        score = 25
        signals = ["Perfect Trend"]
        
        if vol_ratio >= 3: score += 20; signals.append("Huge Volume")
        elif vol_ratio >= 2: score += 15
        else: score += 10
        
        if 58 <= rsi.iloc[c] <= 65: score += 15; signals.append("Ideal RSI")
        else: score += 10
        
        if stock_sector in sector_data['hot_sectors'][:1]: score += 15; signals.append("Top Sector")
        elif stock_sector in sector_data['hot_sectors']: score += 10; signals.append("Hot Sector")
        
        change_5d = ((curr - close.iloc[-5]) / close.iloc[-5]) * 100
        if change_5d > 3: score += 10
        
        if nifty_close is not None and len(nifty_close) > 20:
            stock_ret = (close.iloc[-1] / close.iloc[-20]) - 1
            nifty_ret = (nifty_close.iloc[-1] / nifty_close.iloc[-20]) - 1
            if stock_ret > nifty_ret * 1.5: score += 10; signals.append("Outperforming")
        
        if score < SAFE_MIN_SCORE: return None
        
        recent_low = low.iloc[-3:].min()
        atr_stop = curr - (atr_val * 1.2)
        ema_stop = ema20.iloc[c] * 0.99
        sl_candidates = [s for s in [recent_low * 0.995, atr_stop, ema_stop] if s < curr and (curr - s) / curr <= 0.03]
        if not sl_candidates: return None
        sl = max(sl_candidates)
        
        risk = curr - sl
        risk_pct = (risk / curr) * 100
        if risk_pct > SAFE_MAX_RISK: return None
        
        tgt1 = curr + (risk * 2.0)
        tgt2 = curr + (risk * 3.5)
        
        risk_amount = CAPITAL_SAFE * (SAFE_MAX_RISK / 100)
        shares = int(risk_amount / risk)
        position_value = shares * curr
        
        max_position = CAPITAL_SAFE * 0.2
        if position_value > max_position:
            shares = int(max_position / curr)
            position_value = shares * curr
        
        if shares < 1: return None
        
        return {
            "ticker": ticker_clean, "scanner": "SAFE",
            "score": round(score, 1), "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1), "vol_ratio": round(vol_ratio, 1),
            "sector": stock_sector, "sl": round(sl, 2),
            "tgt1": round(tgt1, 2), "tgt2": round(tgt2, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "shares": shares, "position_value": round(position_value, 0),
            "change_5d": round(change_5d, 2), "signals": signals[:5],
            "category": "LARGE_MID_CAP"
        }
    except:
        return None

# ============================================================
# 🎯 SCANNER 2: VCP SCANNER (NEW - Minervini Method)
# ============================================================

def find_contractions(df):
    """Find volatility contractions"""
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()
    close = df['Close'].squeeze()
    
    contractions = []
    lookback = 3
    
    # Find swing points
    swings = []
    for i in range(lookback, len(close) - lookback):
        # Swing high
        is_high = True
        for j in range(1, lookback + 1):
            if high.iloc[i] < high.iloc[i-j] or high.iloc[i] < high.iloc[i+j]:
                is_high = False
                break
        if is_high:
            swings.append({"type": "H", "index": i, "price": high.iloc[i]})
        
        # Swing low
        is_low = True
        for j in range(1, lookback + 1):
            if low.iloc[i] > low.iloc[i-j] or low.iloc[i] > low.iloc[i+j]:
                is_low = False
                break
        if is_low:
            swings.append({"type": "L", "index": i, "price": low.iloc[i]})
    
    # Find H->L->H patterns (contractions)
    for i in range(len(swings) - 2):
        if swings[i]['type'] == 'H' and swings[i+1]['type'] == 'L':
            high_price = swings[i]['price']
            low_price = swings[i+1]['price']
            depth = ((high_price - low_price) / high_price) * 100
            
            if depth > 0 and depth < 50:  # Valid contraction (0-50%)
                start_idx = swings[i]['index']
                end_idx = swings[i+1]['index']
                
                if start_idx < end_idx:
                    vol_avg = volume.iloc[start_idx:end_idx+1].mean()
                    
                    contractions.append({
                        "start": start_idx,
                        "end": end_idx,
                        "high": high_price,
                        "low": low_price,
                        "depth": depth,
                        "volume_avg": vol_avg
                    })
    
    return contractions

def vcp_scanner(ticker, nifty_close, memory):
    """VCP Scanner based on Minervini's SEPA methodology"""
    try:
        ticker_clean = ticker.replace(".NS", "")
        if is_recently_failed(ticker_clean, memory): return None
        
        df = yf.download(ticker, period="1y", progress=False, timeout=10)
        if len(df) < 200: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        open_ = df['Open'].squeeze()
        
        # Basic filters
        avg_vol = volume.mean()
        if avg_vol < 200000: return None
        if close.iloc[-1] < 50: return None
        if close.iloc[-1] > 15000: return None
        
        curr = close.iloc[-1]
        
        # ═══════════════════════════════════
        # 1. TREND TEMPLATE (Minervini's 8 rules)
        # ═══════════════════════════════════
        
        ema50 = get_ema(close, 50).iloc[-1]
        ema150 = get_ema(close, 150).iloc[-1]
        ema200 = get_ema(close, 200).iloc[-1]
        
        ema200_month_ago = get_ema(close, 200).iloc[-30] if len(close) >= 230 else ema200
        
        high_52w = high.iloc[-252:].max() if len(close) >= 252 else high.max()
        low_52w = low.iloc[-252:].min() if len(close) >= 252 else low.min()
        
        checks = {
            "above_150ema": curr > ema150,
            "above_200ema": curr > ema200,
            "150_above_200": ema150 > ema200,
            "200_rising": ema200 > ema200_month_ago,
            "50_above_150": ema50 > ema150,
            "above_50ema": curr > ema50,
            "25pct_above_low": curr > low_52w * 1.25,
            "within_25pct_high": curr > high_52w * 0.75
        }
        
        trend_score = sum(checks.values())
        if trend_score < 6: return None
        
        # ═══════════════════════════════════
        # 2. CONTRACTION DETECTION
        # ═══════════════════════════════════
        
        # Analyze last 3 months
        recent_data = df.iloc[-60:]
        contractions = find_contractions(recent_data)
        
        if len(contractions) < 2: return None
        
        # Take most recent contractions
        recent_contractions = sorted(contractions, key=lambda x: x['end'])[-4:]
        
        if len(recent_contractions) < 2: return None
        
        # Check if contractions are getting smaller
        depths = [c['depth'] for c in recent_contractions]
        contractions_shrinking = True
        for i in range(1, len(depths)):
            if depths[i] > depths[i-1] * 1.1:  # Allow small tolerance
                contractions_shrinking = False
                break
        
        if not contractions_shrinking: return None
        
        # ═══════════════════════════════════
        # 3. VOLUME ANALYSIS
        # ═══════════════════════════════════
        
        # Volume should be drying up
        vol_periods = [c['volume_avg'] for c in recent_contractions]
        volume_drying = True
        for i in range(1, len(vol_periods)):
            if vol_periods[i] > vol_periods[i-1] * 1.3:
                volume_drying = False
                break
        
        # ═══════════════════════════════════
        # 4. PIVOT POINT & BREAKOUT
        # ═══════════════════════════════════
        
        latest_contraction = recent_contractions[-1]
        pivot_range = latest_contraction['depth']
        
        # Tight range = closer to breakout
        if pivot_range > 15: return None  # Too wide
        
        recent_high = high.iloc[-10:].max()
        distance_from_high = ((recent_high - curr) / recent_high) * 100
        near_pivot = distance_from_high < 3
        
        # Check breakout
        avg_vol_50 = volume.iloc[-50:].mean()
        curr_vol = volume.iloc[-1]
        vol_ratio = curr_vol / avg_vol_50 if avg_vol_50 > 0 else 0
        
        just_broke_out = (
            curr > recent_high * 0.99 and
            vol_ratio >= 1.5 and
            close.iloc[-1] > open_.iloc[-1]
        )
        
        # ═══════════════════════════════════
        # 5. RS RATING
        # ═══════════════════════════════════
        
        rs_rating = 70
        if nifty_close is not None:
            try:
                stock_ret_63d = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
                stock_ret_126d = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0
                
                nifty_ret_63d = (nifty_close.iloc[-1] / nifty_close.iloc[-63] - 1) * 100 if len(nifty_close) >= 63 else 0
                nifty_ret_126d = (nifty_close.iloc[-1] / nifty_close.iloc[-126] - 1) * 100 if len(nifty_close) >= 126 else 0
                
                rs_63 = stock_ret_63d - nifty_ret_63d
                rs_126 = stock_ret_126d - nifty_ret_126d
                
                rs_rating = 70 + (rs_63 * 0.6) + (rs_126 * 0.4)
                rs_rating = max(1, min(99, rs_rating))
            except:
                pass
        
        if rs_rating < 70: return None
        
        # ═══════════════════════════════════
        # 6. RSI CHECK
        # ═══════════════════════════════════
        
        rsi = get_rsi(close, 14).iloc[-1]
        if not (50 <= rsi <= 75): return None
        
        # ═══════════════════════════════════
        # SCORING
        # ═══════════════════════════════════
        
        score = 0
        signals = []
        
        # Trend template (40 points)
        score += trend_score * 5
        signals.append(f"Trend {trend_score}/8")
        
        # Contractions (20 points)
        if len(recent_contractions) >= 4:
            score += 20
            signals.append(f"{len(recent_contractions)} Contractions")
        elif len(recent_contractions) == 3:
            score += 17
        elif len(recent_contractions) == 2:
            score += 15
        
        # Volume drying (15 points)
        if volume_drying:
            score += 15
            signals.append("Volume Drying")
        
        # Tight range (15 points)
        if pivot_range < 5:
            score += 15
            signals.append(f"Tight ({pivot_range:.1f}%)")
        elif pivot_range < 8:
            score += 12
        elif pivot_range < 12:
            score += 8
        
        # Near pivot (10 points)
        if near_pivot:
            score += 10
            signals.append("Near Pivot")
        
        # Breakout in progress (bonus)
        if just_broke_out:
            score += 15
            signals.append(f"BREAKOUT ({vol_ratio:.1f}x vol)")
        
        # RS Rating bonus
        if rs_rating >= 90:
            score += 15
            signals.append(f"RS {rs_rating:.0f}")
        elif rs_rating >= 80:
            score += 10
        
        # Distance from 52W high
        dist_52w = ((high_52w - curr) / curr) * 100
        if dist_52w < 5:
            score += 10
            signals.append("Near 52W High")
        
        if score < VCP_MIN_SCORE: return None
        
        # ═══════════════════════════════════
        # CALCULATE LEVELS
        # ═══════════════════════════════════
        
        # Entry
        entry = curr if just_broke_out else recent_high
        
        # SL: Below latest contraction low
        latest_low = latest_contraction['low']
        sl = latest_low * 0.98
        
        risk = entry - sl
        if risk <= 0: return None
        
        risk_pct = (risk / entry) * 100
        if risk_pct > 8: return None  # VCP allows wider stops
        
        # Targets (VCP produces big moves)
        tgt1 = entry + (risk * 2.5)
        tgt2 = entry + (risk * 5)
        tgt3 = entry + (risk * 8)  # Extended target
        
        # Position sizing
        risk_amount = CAPITAL_VCP * (VCP_MAX_RISK / 100)
        shares = int(risk_amount / risk)
        position_value = shares * entry
        
        max_position = CAPITAL_VCP * 0.4
        if position_value > max_position:
            shares = int(max_position / entry)
            position_value = shares * entry
        
        if shares < 1: return None
        
        return {
            "ticker": ticker_clean,
            "scanner": "VCP",
            "score": round(score, 1),
            "price": round(curr, 2),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2),
            "tgt3": round(tgt3, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - entry) / risk, 2),
            "trend_score": trend_score,
            "contractions": len(recent_contractions),
            "pivot_range": round(pivot_range, 2),
            "rs_rating": round(rs_rating, 0),
            "vol_ratio": round(vol_ratio, 1),
            "breaking_out": just_broke_out,
            "distance_from_high": round(distance_from_high, 2),
            "rsi": round(rsi, 1),
            "sector": detect_stock_sector(ticker_clean),
            "shares": shares,
            "position_value": round(position_value, 0),
            "signals": signals[:6],
            "category": "VCP_PATTERN"
        }
    except Exception as e:
        return None

# ============================================================
# 🎯 SCANNER 3: MOMENTUM (Small/Mid Caps)
# ============================================================

def momentum_scanner(ticker, nifty_close, memory, regime):
    try:
        ticker_clean = ticker.replace(".NS", "")
        if is_recently_failed(ticker_clean, memory): return None
        
        df = yf.download(ticker, period="3mo", progress=False, timeout=10)
        if len(df) < 30: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        open_ = df['Open'].squeeze()
        
        avg_vol = vol.mean()
        curr = close.iloc[-1]
        
        if curr < 30 or curr > 5000: return None
        if avg_vol < 100000: return None
        
        if avg_vol >= 1000000 and curr >= 100:
            category, min_vol_ratio, max_risk = "MID_CAP", 2, 2.0
        else:
            category, min_vol_ratio, max_risk = "SMALL_CAP", 3, 2.5
        
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        rsi = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr = get_atr(high, low, close, 14)
        
        c = len(df) - 1
        atr_val = atr.iloc[c]
        
        if not (curr > ema20.iloc[c]): return None
        
        recent_3d_high = high.iloc[-3:].max()
        prev_20d_high = high.iloc[-25:-3].max()
        if recent_3d_high <= prev_20d_high: return None
        
        avg_vol_20 = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol_20 if avg_vol_20 > 0 else 0
        if vol_ratio < min_vol_ratio: return None
        
        change_5d = ((curr - close.iloc[-5]) / close.iloc[-5]) * 100
        if change_5d < 3: return None
        
        if not (50 <= rsi.iloc[c] <= 80): return None
        
        dist_from_ema20 = ((curr - ema20.iloc[c]) / ema20.iloc[c]) * 100
        if dist_from_ema20 > 15: return None
        
        if close.iloc[c] <= open_.iloc[c]: return None
        
        if high.iloc[c] > low.iloc[c]:
            close_pos = (close.iloc[c] - low.iloc[c]) / (high.iloc[c] - low.iloc[c])
            if close_pos < 0.5: return None
        
        score = 20
        signals = [f"{category} Momentum"]
        
        if vol_ratio >= 5: score += 25; signals.append(f"Massive Vol ({vol_ratio:.1f}x)")
        elif vol_ratio >= 3: score += 20; signals.append("High Vol")
        else: score += 15
        
        if change_5d > 10: score += 20; signals.append(f"Strong Momentum ({change_5d:.1f}%)")
        elif change_5d > 5: score += 15
        else: score += 10
        
        breakout_pct = ((recent_3d_high - prev_20d_high) / prev_20d_high) * 100
        if breakout_pct > 5: score += 15; signals.append(f"Breakout +{breakout_pct:.1f}%")
        elif breakout_pct > 2: score += 10
        
        if 60 <= rsi.iloc[c] <= 70: score += 15; signals.append("Ideal RSI")
        elif 55 <= rsi.iloc[c] <= 75: score += 10
        
        if macd.iloc[c] > msig.iloc[c] > 0: score += 10; signals.append("MACD Bullish")
        
        if score < MOMENTUM_MIN_SCORE: return None
        
        recent_low = low.iloc[-5:].min()
        atr_stop = curr - (atr_val * 2)
        ema_stop = ema20.iloc[c] * 0.97
        sl_candidates = [s for s in [recent_low, atr_stop, ema_stop] if s < curr]
        if not sl_candidates: return None
        sl = max(sl_candidates)
        
        risk = curr - sl
        risk_pct = (risk / curr) * 100
        if risk_pct > max_risk: return None
        
        tgt1 = curr + (risk * 2.0)
        tgt2 = curr + (risk * 3.5)
        tgt3 = curr + (risk * 5.0)
        
        risk_amount = CAPITAL_MOMENTUM * (max_risk / 100)
        shares = int(risk_amount / risk)
        position_value = shares * curr
        
        max_position = CAPITAL_MOMENTUM * 0.4
        if position_value > max_position:
            shares = int(max_position / curr)
            position_value = shares * curr
        
        if shares < 1: return None
        
        return {
            "ticker": ticker_clean, "scanner": "MOMENTUM",
            "score": round(score, 1), "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1), "vol_ratio": round(vol_ratio, 1),
            "sector": detect_stock_sector(ticker_clean),
            "sl": round(sl, 2), "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2), "tgt3": round(tgt3, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "shares": shares, "position_value": round(position_value, 0),
            "change_5d": round(change_5d, 2),
            "breakout_pct": round(breakout_pct, 2),
            "signals": signals[:5], "category": category
        }
    except:
        return None

# ============================================================
# 🎯 SCANNER 4: PRO PICKS (Yesterday's Gainers)
# ============================================================

def get_yesterday_gainers():
    try:
        gainers = []
        universe = SAFE_UNIVERSE[:50] + get_expanded_universe()[:150]
        universe = list(set(universe))
        
        for ticker in universe:
            try:
                df = yf.download(ticker, period="5d", progress=False, timeout=5)
                if len(df) < 2: continue
                
                yesterday_close = df['Close'].iloc[-2]
                day_before_close = df['Close'].iloc[-3] if len(df) >= 3 else df['Close'].iloc[-2]
                
                change_pct = ((yesterday_close - day_before_close) / day_before_close) * 100
                
                if 5 <= change_pct <= 15:
                    vol_ratio = df['Volume'].iloc[-2] / df['Volume'].iloc[-10:-2].mean() if len(df) > 2 else 0
                    if vol_ratio >= 2:
                        gainers.append({
                            "ticker": ticker,
                            "change": round(change_pct, 2),
                            "vol_ratio": round(vol_ratio, 1),
                            "price": round(yesterday_close, 2)
                        })
            except:
                continue
        
        return sorted(gainers, key=lambda x: x['change'], reverse=True)[:20]
    except:
        return []

def pro_scanner(ticker_data, memory):
    try:
        ticker = ticker_data['ticker']
        ticker_clean = ticker.replace(".NS", "")
        
        if is_recently_failed(ticker_clean, memory): return None
        
        df = yf.download(ticker, period="10d", progress=False, timeout=10)
        if len(df) < 5: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        curr = close.iloc[-1]
        yesterday_high = ticker_data['price']
        
        change_from_high = ((curr - yesterday_high) / yesterday_high) * 100
        if change_from_high < -3 or change_from_high > 2: return None
        
        recent_low = low.iloc[-3:].min()
        earlier_low = low.iloc[-6:-3].min() if len(low) > 6 else low.iloc[-3]
        if recent_low < earlier_low * 0.98: return None
        
        recent_vol = vol.iloc[-2:].mean()
        yesterday_vol = vol.iloc[-3] if len(vol) > 3 else vol.iloc[-2]
        vol_declining = recent_vol < yesterday_vol
        
        ema20 = get_ema(close, 20)
        rsi = get_rsi(close, 14)
        atr = get_atr(high, low, close, 14)
        
        if not (curr > ema20.iloc[-1] * 0.98): return None
        if not (50 <= rsi.iloc[-1] <= 75): return None
        
        score = 40
        signals = ["Yesterday's Gainer", "Consolidating"]
        
        if ticker_data['change'] >= 10:
            score += 20; signals.append(f"Strong ({ticker_data['change']}%)")
        elif ticker_data['change'] >= 7:
            score += 15
        else:
            score += 10
        
        if vol_declining: score += 15; signals.append("Healthy Pullback")
        if ticker_data['vol_ratio'] >= 3: score += 15; signals.append("Institutional")
        if 55 <= rsi.iloc[-1] <= 70: score += 10; signals.append("RSI Support")
        
        if score < PRO_MIN_SCORE: return None
        
        atr_val = atr.iloc[-1]
        sl = max(recent_low * 0.99, curr - (atr_val * 1.5))
        risk = curr - sl
        if risk <= 0: return None
        
        risk_pct = (risk / curr) * 100
        if risk_pct > PRO_MAX_RISK: return None
        
        tgt1 = curr + (risk * 2.0)
        tgt2 = curr + (risk * 3.5)
        
        risk_amount = CAPITAL_PRO * (PRO_MAX_RISK / 100)
        shares = int(risk_amount / risk)
        position_value = shares * curr
        
        max_position = CAPITAL_PRO * 0.5
        if position_value > max_position:
            shares = int(max_position / curr)
            position_value = shares * curr
        
        if shares < 1: return None
        
        return {
            "ticker": ticker_clean, "scanner": "PRO",
            "score": round(score, 1), "price": round(curr, 2),
            "yesterday_change": ticker_data['change'],
            "rsi": round(rsi.iloc[-1], 1),
            "sector": detect_stock_sector(ticker_clean),
            "sl": round(sl, 2), "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2), "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "shares": shares, "position_value": round(position_value, 0),
            "consolidating": True, "signals": signals[:5],
            "category": "PRO_TREND"
        }
    except:
        return None

# ============================================================
# 🎯 HELPER FUNCTIONS
# ============================================================

def is_recently_failed(ticker, memory):
    if ticker in memory.get('failed_stocks', {}):
        failed_date = datetime.fromisoformat(memory['failed_stocks'][ticker]['failed_date'])
        if (datetime.now() - failed_date).days < 30:
            return True
    return False

def add_to_confirmation_queue(memory, pick, scanner_type):
    queue_id = f"{pick['ticker']}_{scanner_type}_{datetime.now().strftime('%Y%m%d')}"
    memory['confirmation_queue'][queue_id] = {
        "ticker": pick['ticker'],
        "detected_date": datetime.now().isoformat(),
        "detected_price": pick['price'],
        "sl": pick['sl'],
        "tgt1": pick['tgt1'],
        "tgt2": pick['tgt2'],
        "score": pick['score'],
        "sector": pick.get('sector', 'Other'),
        "scanner": scanner_type,
        "status": "PENDING"
    }
    return memory

def check_confirmations(memory):
    confirmed = []
    to_remove = []
    
    for qid, item in memory['confirmation_queue'].items():
        try:
            days_old = (datetime.now() - datetime.fromisoformat(item['detected_date'])).days
            scanner = item.get('scanner', 'SAFE')
            
            required_days = 2 if scanner in ['SAFE', 'VCP'] else 1
            
            if days_old < required_days: continue
            
            ticker = item['ticker']
            df = yf.download(f"{ticker}.NS", period="10d", progress=False, timeout=10)
            if len(df) < 3: continue
            
            current = df['Close'].iloc[-1]
            detected = item['detected_price']
            
            still_holding = current >= detected * 0.98
            no_reversal = current > detected * 0.97
            volume_ok = df['Volume'].iloc[-2:].mean() > df['Volume'].iloc[-20:-2].mean() * 0.5
            
            confirmations = sum([still_holding, no_reversal, volume_ok])
            
            if confirmations >= 2:
                item['status'] = "CONFIRMED"
                item['confirmed_price'] = current
                confirmed.append(item)
                to_remove.append(qid)
            elif days_old > 4:
                to_remove.append(qid)
                memory['failed_stocks'][ticker] = {
                    "failed_date": datetime.now().isoformat(),
                    "reason": f"Failed {scanner} confirmation"
                }
        except:
            continue
    
    for qid in to_remove:
        del memory['confirmation_queue'][qid]
    
    return memory, confirmed

# ============================================================
# 📤 TELEGRAM
# ============================================================

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        parts = [msg[i:i+3800] for i in range(0, len(msg), 3800)]
        for part in parts:
            requests.post(url, json={"chat_id": CHAT_ID, "text": part}, timeout=15)
            time.sleep(0.8)
    except:
        pass

# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    print("=" * 60)
    print(f"ARTHA {BOT_VERSION} - {BOT_TAGLINE}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("=" * 60)
    
    memory = load_memory()
    total_eval = memory['total_wins'] + memory['total_losses']
    wr = (memory['total_wins'] / total_eval * 100) if total_eval > 0 else 0
    
    send_telegram(f"⚡ ARTHA {BOT_VERSION} starting VCP Master scan...")
    
    # Market regime
    print("\n[REGIME]")
    regime = detect_market_regime()
    print(f"  {regime['regime']}")
    
    if not regime['trade_allowed']:
        msg = f"⚡ ARTHA {BOT_VERSION}\n📅 {datetime.now().strftime('%A, %d %b %Y')}\n\n"
        msg += f"[🛑 NO TRADING TODAY]\n"
        msg += f"Regime: {regime['regime']}\n"
        msg += f"Nifty: {regime.get('price', 0)}\n"
        save_memory(memory)
        send_telegram(msg)
        return
    
    # Sectors
    print("\n[SECTORS]")
    sector_data = analyze_sectors()
    print(f"  Hot: {sector_data['hot_sectors']}")
    
    # Confirmations
    print("\n[CONFIRM]")
    memory, confirmed_picks = check_confirmations(memory)
    print(f"  Confirmed: {len(confirmed_picks)}")
    
    # Get Nifty
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except:
        nifty_close = None
    
    # ═══════════════════════════════════
    # SCANNER 1: SAFE
    # ═══════════════════════════════════
    print("\n[SAFE SCANNER]")
    safe_picks = []
    for i, t in enumerate(SAFE_UNIVERSE):
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(SAFE_UNIVERSE)} | Found: {len(safe_picks)}")
        r = safe_scanner(t, nifty_close, memory, sector_data, regime)
        if r: safe_picks.append(r)
    safe_picks.sort(key=lambda x: x["score"], reverse=True)
    safe_picks = safe_picks[:2]
    print(f"  Found: {len(safe_picks)}")
    
    # ═══════════════════════════════════
    # SCANNER 2: VCP (NEW)
    # ═══════════════════════════════════
    print("\n[VCP SCANNER]")
    expanded_universe = get_expanded_universe()
    vcp_picks = []
    
    # Scan expanded universe for VCP patterns
    for i, t in enumerate(expanded_universe[:400]):  # Limit for speed
        if (i+1) % 50 == 0:
            print(f"  {i+1}/400 | Found: {len(vcp_picks)}")
        r = vcp_scanner(t, nifty_close, memory)
        if r: vcp_picks.append(r)
    
    vcp_picks.sort(key=lambda x: x["score"], reverse=True)
    vcp_picks = vcp_picks[:3]  # Top 3 VCP picks
    print(f"  Found: {len(vcp_picks)}")
    
    # ═══════════════════════════════════
    # SCANNER 3: MOMENTUM
    # ═══════════════════════════════════
    print("\n[MOMENTUM SCANNER]")
    momentum_universe = [t for t in expanded_universe if t not in SAFE_UNIVERSE][:300]
    momentum_picks = []
    for i, t in enumerate(momentum_universe):
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(momentum_universe)} | Found: {len(momentum_picks)}")
        r = momentum_scanner(t, nifty_close, memory, regime)
        if r: momentum_picks.append(r)
    momentum_picks.sort(key=lambda x: x["score"], reverse=True)
    momentum_picks = momentum_picks[:2]
    print(f"  Found: {len(momentum_picks)}")
    
    # ═══════════════════════════════════
    # SCANNER 4: PRO
    # ═══════════════════════════════════
    print("\n[PRO SCANNER]")
    yesterday_gainers = get_yesterday_gainers()
    pro_picks = []
    for gainer in yesterday_gainers:
        r = pro_scanner(gainer, memory)
        if r: pro_picks.append(r)
    pro_picks.sort(key=lambda x: x["score"], reverse=True)
    pro_picks = pro_picks[:2]
    print(f"  Selected: {len(pro_picks)}")
    
    # Add all to confirmation queue
    for pick in safe_picks:
        memory = add_to_confirmation_queue(memory, pick, "SAFE")
    for pick in vcp_picks:
        memory = add_to_confirmation_queue(memory, pick, "VCP")
    for pick in momentum_picks:
        memory = add_to_confirmation_queue(memory, pick, "MOMENTUM")
    for pick in pro_picks:
        memory = add_to_confirmation_queue(memory, pick, "PRO")
    
    # Save confirmed
    for cp in confirmed_picks:
        pick_id = f"{cp['ticker']}_{datetime.now().strftime('%Y%m%d')}"
        memory["pending_evaluations"][pick_id] = {
            "ticker": cp['ticker'],
            "date": datetime.now().isoformat(),
            "entry_price": cp['confirmed_price'],
            "sl": cp['sl'], "tgt1": cp['tgt1'], "tgt2": cp['tgt2'],
            "score": cp['score'], "sector": cp['sector'],
            "scanner": cp['scanner']
        }
        memory["total_picks"] += 1
    
    save_memory(memory)
    
    # ═══════════════════════════════════
    # BUILD REPORT
    # ═══════════════════════════════════
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"{BOT_TAGLINE}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    # Brain status
    msg += f"🧠 [BRAIN v13.0]\n"
    msg += f"Total: {memory['total_picks']} | WR: {wr:.1f}%\n"
    msg += f"Safe: {memory.get('safe_wins', 0)}W/{memory.get('safe_losses', 0)}L\n"
    msg += f"VCP: {memory.get('vcp_wins', 0)}W/{memory.get('vcp_losses', 0)}L\n"
    msg += f"Momentum: {memory.get('momentum_wins', 0)}W/{memory.get('momentum_losses', 0)}L\n"
    msg += f"Pro: {memory.get('pro_wins', 0)}W/{memory.get('pro_losses', 0)}L\n\n"
    
    # Market
    msg += f"[🌍 MARKET]\n"
    msg += f"Regime: {regime['regime']}\n"
    msg += f"Nifty: {regime.get('price', 0)}\n"
    msg += f"20D: {regime.get('change_20d', 0):+.2f}%\n\n"
    
    # Sectors
    msg += f"[🔥 SECTORS]\n"
    msg += f"Hot: {', '.join(sector_data['hot_sectors'])}\n\n"
    
    # CONFIRMED PICKS
    if confirmed_picks:
        msg += "=" * 40 + "\n"
        msg += "[✅ CONFIRMED - TRADE READY]\n"
        msg += "=" * 40 + "\n\n"
        
        for cp in confirmed_picks:
            emoji = "🛡️" if cp['scanner'] == "SAFE" else "⭐" if cp['scanner'] == "VCP" else "🚀" if cp['scanner'] == "MOMENTUM" else "🎯"
            msg += f"{emoji} {cp['ticker']} [{cp['scanner']}]\n"
            msg += f"  Entry: Rs.{cp['confirmed_price']}\n"
            msg += f"  SL: Rs.{cp['sl']}\n"
            msg += f"  T1: Rs.{cp['tgt1']}\n"
            msg += "-" * 40 + "\n\n"
    
    # SAFE PICKS
    if safe_picks:
        msg += "=" * 40 + "\n"
        msg += f"[🛡️ SAFE PICKS - {len(safe_picks)}]\n"
        msg += f"Large/Mid Caps | 60% Capital | Rs.{CAPITAL_SAFE:,}\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(safe_picks):
            msg += f"#{i+1} {s['ticker']} | Score: {s['score']}/100\n"
            msg += f"🏢 {s['sector']}\n"
            msg += f"📊 RSI: {s['rsi']} | Vol: {s['vol_ratio']}x\n"
            msg += f"💰 Price: Rs.{s['price']}\n"
            msg += f"🎯 SL: Rs.{s['sl']} ({s['risk_pct']}%)\n"
            msg += f"🎯 T1: Rs.{s['tgt1']} (1:{s['rr_ratio']})\n"
            msg += f"💼 {s['shares']} shares (Rs.{s['position_value']:,})\n"
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[🛡️ SAFE: None today]\n\n"
    
    # ⭐ VCP PICKS (NEW - HIGHLIGHTED)
    if vcp_picks:
        msg += "=" * 40 + "\n"
        msg += f"[⭐ VCP MASTER PICKS - {len(vcp_picks)}]\n"
        msg += f"Minervini Method | 20% Capital | Rs.{CAPITAL_VCP:,}\n"
        msg += "🏆 HIGHEST QUALITY SETUPS\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(vcp_picks):
            msg += f"#{i+1} {s['ticker']} | Score: {s['score']}/135\n"
            msg += f"🏢 {s['sector']}\n\n"
            
            msg += "📊 VCP QUALITY:\n"
            msg += f"  Trend Template: {s['trend_score']}/8 ✅\n"
            msg += f"  Contractions: {s['contractions']}\n"
            msg += f"  Tight Range: {s['pivot_range']}%\n"
            msg += f"  RS Rating: {s['rs_rating']}/99\n"
            msg += f"  Dist from High: {s['distance_from_high']}%\n"
            
            if s['breaking_out']:
                msg += f"  🚀 BREAKING OUT NOW! ({s['vol_ratio']}x vol)\n"
            
            msg += f"\n💰 SETUP:\n"
            msg += f"  Entry: Rs.{s['entry']}\n"
            msg += f"  SL: Rs.{s['sl']} ({s['risk_pct']}%)\n"
            msg += f"  T1: Rs.{s['tgt1']} (1:{s['rr_ratio']})\n"
            msg += f"  T2: Rs.{s['tgt2']}\n"
            msg += f"  T3: Rs.{s['tgt3']} (Extended)\n"
            msg += f"💼 {s['shares']} shares (Rs.{s['position_value']:,})\n"
            
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[⭐ VCP: None today - Rare pattern]\n\n"
    
    # MOMENTUM PICKS
    if momentum_picks:
        msg += "=" * 40 + "\n"
        msg += f"[🚀 MOMENTUM - {len(momentum_picks)}]\n"
        msg += f"Small/Mid Caps | 15% Capital | Rs.{CAPITAL_MOMENTUM:,}\n"
        msg += "⚠️ HIGHER RISK\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(momentum_picks):
            msg += f"#{i+1} {s['ticker']} | Score: {s['score']}/100\n"
            msg += f"🏢 {s['sector']} | {s['category']}\n"
            msg += f"📊 RSI: {s['rsi']} | Vol: {s['vol_ratio']}x\n"
            msg += f"📈 5D: {s['change_5d']:+.2f}%\n"
            msg += f"💰 Price: Rs.{s['price']}\n"
            msg += f"🎯 SL: Rs.{s['sl']} ({s['risk_pct']}%)\n"
            msg += f"🎯 T1: Rs.{s['tgt1']} (1:{s['rr_ratio']})\n"
            msg += f"💼 {s['shares']} shares (Rs.{s['position_value']:,})\n"
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[🚀 MOMENTUM: None today]\n\n"
    
    # PRO PICKS
    if pro_picks:
        msg += "=" * 40 + "\n"
        msg += f"[🎯 PRO PICKS - {len(pro_picks)}]\n"
        msg += f"Yesterday's Gainers | 5% Capital | Rs.{CAPITAL_PRO:,}\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(pro_picks):
            msg += f"#{i+1} {s['ticker']} | Score: {s['score']}/100\n"
            msg += f"🏢 {s['sector']}\n"
            msg += f"📊 RSI: {s['rsi']}\n"
            msg += f"📈 Yesterday: +{s['yesterday_change']:.2f}%\n"
            msg += f"💰 Price: Rs.{s['price']}\n"
            msg += f"🎯 SL: Rs.{s['sl']} ({s['risk_pct']}%)\n"
            msg += f"🎯 T1: Rs.{s['tgt1']} (1:{s['rr_ratio']})\n"
            msg += f"💼 {s['shares']} shares (Rs.{s['position_value']:,})\n"
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[🎯 PRO: None today]\n\n"
    
    # Rules
    msg += "=" * 40 + "\n"
    msg += "[💰 CAPITAL ALLOCATION]\n"
    msg += f"🛡️ Safe: Rs.{CAPITAL_SAFE:,} (60%)\n"
    msg += f"⭐ VCP: Rs.{CAPITAL_VCP:,} (20%) - PRIORITY\n"
    msg += f"🚀 Momentum: Rs.{CAPITAL_MOMENTUM:,} (15%)\n"
    msg += f"🎯 Pro: Rs.{CAPITAL_PRO:,} (5%)\n\n"
    
    msg += "[⚠️ TRADING RULES]\n"
    msg += "1. VCP picks = HIGHEST priority\n"
    msg += "2. Safe picks = Reliable base\n"
    msg += "3. Wait 2 days for confirmation\n"
    msg += "4. Book 40% at T1, trail rest\n"
    msg += "5. Never override stop loss\n\n"
    
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION}"
    
    send_telegram(msg)
    print("\n[DONE] v13.0 scan complete!")

if __name__ == "__main__":
    main()
