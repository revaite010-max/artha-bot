import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 🔑 Settings
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v8.0"
BOT_TAGLINE = "Smart Entry Edition"

# ============================================================
# 🧮 MATH LIBRARY
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
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

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

# ============================================================
# 🎯 MARKET REGIME DETECTOR (NEW!)
# ============================================================
def detect_market_regime():
    """Detect if market is BULL, BEAR, or SIDEWAYS"""
    try:
        nifty = yf.download("^NSEI", period="6mo", progress=False)
        close = nifty['Close'].squeeze()
        
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        
        curr = close.iloc[-1]
        curr_ema20 = ema20.iloc[-1]
        curr_ema50 = ema50.iloc[-1]
        curr_ema200 = ema200.iloc[-1]
        
        # Check trend structure
        strong_bull = curr > curr_ema20 > curr_ema50 > curr_ema200
        moderate_bull = curr > curr_ema20 > curr_ema50
        sideways = curr > curr_ema200 and abs(curr - curr_ema20) / curr < 0.02
        bear = curr < curr_ema50 < curr_ema200
        
        # Recent momentum
        change_20d = ((curr - close.iloc[-20]) / close.iloc[-20]) * 100
        
        if strong_bull and change_20d > 3:
            regime = "STRONG_BULL"
            trade_allowed = True
            confidence = 100
        elif moderate_bull and change_20d > 0:
            regime = "BULL"
            trade_allowed = True
            confidence = 75
        elif sideways:
            regime = "SIDEWAYS"
            trade_allowed = False
            confidence = 40
        elif bear:
            regime = "BEAR"
            trade_allowed = False
            confidence = 20
        else:
            regime = "UNCERTAIN"
            trade_allowed = False
            confidence = 30
        
        return {
            "regime": regime,
            "trade_allowed": trade_allowed,
            "confidence": confidence,
            "nifty_change_20d": round(change_20d, 2),
            "nifty_price": round(curr, 2)
        }
    except Exception as e:
        return {"regime": "UNKNOWN", "trade_allowed": False, "confidence": 0, "nifty_change_20d": 0, "nifty_price": 0}

# ============================================================
# 🎯 EARLY BREAKOUT DETECTOR (NEW!)
# ============================================================
def detect_early_breakout(df):
    """Detect breakouts in EARLY stage (Day 1-3), not late"""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Check last 5 days
        last_5_high = high.iloc[-5:].max()
        prev_20_high = high.iloc[-25:-5].max()
        
        # Was consolidating before?
        consolidation_range = (high.iloc[-25:-5].max() - low.iloc[-25:-5].min()) / low.iloc[-25:-5].min() * 100
        was_consolidating = consolidation_range < 10  # Less than 10% range
        
        # Just broke out?
        just_broke_out = last_5_high > prev_20_high
        
        # Days since breakout
        days_since_breakout = 0
        for i in range(1, 6):
            if high.iloc[-i] > prev_20_high:
                days_since_breakout = i
                break
        
        # Volume expanding on breakout
        avg_vol_prev = volume.iloc[-25:-5].mean()
        recent_vol = volume.iloc[-5:].mean()
        volume_expanding = recent_vol > avg_vol_prev * 1.5
        
        # Perfect early setup
        is_early = was_consolidating and just_broke_out and days_since_breakout <= 3 and volume_expanding
        
        return {
            "is_early_breakout": is_early,
            "days_since_breakout": days_since_breakout,
            "was_consolidating": was_consolidating,
            "volume_expanding": volume_expanding,
            "prev_range_pct": round(consolidation_range, 2)
        }
    except:
        return None

# ============================================================
# 🎯 PULLBACK OPPORTUNITY DETECTOR (NEW!)
# ============================================================
def detect_pullback_entry(df):
    """Detect pullback to support (safer entry)"""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        ema20 = get_ema(close, 20).iloc[-1]
        ema50 = get_ema(close, 50).iloc[-1]
        curr = close.iloc[-1]
        
        # Recent high (5-15 days ago)
        recent_high = high.iloc[-15:-3].max()
        
        # Currently pulling back to EMA20?
        dist_from_high = ((recent_high - curr) / recent_high) * 100
        near_ema20 = abs(curr - ema20) / curr < 0.02  # Within 2% of EMA20
        
        # Bouncing from support?
        bouncing = curr > low.iloc[-3:].min() and low.iloc[-1] > low.iloc[-2]
        
        # Volume declining on pullback (healthy)
        pullback_vol = df['Volume'].iloc[-3:].mean()
        prev_vol = df['Volume'].iloc[-15:-3].mean()
        healthy_pullback = pullback_vol < prev_vol
        
        is_pullback_entry = (
            dist_from_high < 8 and  # Pulled back but not too much
            dist_from_high > 2 and  # At least 2% pullback
            near_ema20 and          # Near support
            bouncing and             # Starting to bounce
            healthy_pullback         # Low volume on decline
        )
        
        return {
            "is_pullback_entry": is_pullback_entry,
            "dist_from_high": round(dist_from_high, 2),
            "near_support": near_ema20,
            "healthy_pullback": healthy_pullback
        }
    except:
        return None

# ============================================================
# 🎯 MULTI-TIMEFRAME CONFIRMATION (NEW!)
# ============================================================
def multi_timeframe_confirmation(ticker):
    """Check if multiple timeframes agree"""
    try:
        # Weekly
        weekly = yf.download(ticker, period="2y", interval="1wk", progress=False, timeout=10)
        if len(weekly) < 30: return None
        
        w_close = weekly['Close'].squeeze()
        w_ema20 = get_ema(w_close, 20)
        weekly_bullish = w_close.iloc[-1] > w_ema20.iloc[-1]
        
        # Daily (already have this)
        daily = yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=10)
        d_close = daily['Close'].squeeze()
        d_ema20 = get_ema(d_close, 20)
        d_ema50 = get_ema(d_close, 50)
        daily_bullish = d_close.iloc[-1] > d_ema20.iloc[-1] > d_ema50.iloc[-1]
        
        # Hourly (last 5 days)
        hourly = yf.download(ticker, period="5d", interval="1h", progress=False, timeout=10)
        if len(hourly) < 20: 
            hourly_bullish = daily_bullish  # Fallback
        else:
            h_close = hourly['Close'].squeeze()
            h_ema20 = get_ema(h_close, 20)
            hourly_bullish = h_close.iloc[-1] > h_ema20.iloc[-1]
        
        alignment_score = sum([weekly_bullish, daily_bullish, hourly_bullish])
        
        return {
            "weekly_bullish": weekly_bullish,
            "daily_bullish": daily_bullish,
            "hourly_bullish": hourly_bullish,
            "alignment": f"{alignment_score}/3",
            "all_aligned": alignment_score == 3
        }
    except:
        return None

# ============================================================
# 🎯 VOLUME QUALITY ANALYZER (NEW!)
# ============================================================
def analyze_volume_quality(df):
    """Distinguish accumulation from distribution"""
    try:
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Last 10 days analysis
        recent_close = close.iloc[-10:]
        recent_vol = volume.iloc[-10:]
        avg_vol = volume.rolling(50).mean().iloc[-1]
        
        # Count up-days and down-days with high volume
        up_days_high_vol = 0
        down_days_high_vol = 0
        
        for i in range(-10, 0):
            if volume.iloc[i] > avg_vol * 1.3:
                if close.iloc[i] > close.iloc[i-1]:
                    up_days_high_vol += 1
                else:
                    down_days_high_vol += 1
        
        # Accumulation = high volume on up days
        # Distribution = high volume on down days
        if up_days_high_vol >= 3 and down_days_high_vol <= 1:
            quality = "ACCUMULATION"
            score = 10
        elif up_days_high_vol > down_days_high_vol:
            quality = "MILD_ACCUMULATION"
            score = 7
        elif up_days_high_vol == down_days_high_vol:
            quality = "NEUTRAL"
            score = 5
        else:
            quality = "DISTRIBUTION"
            score = 0
        
        return {
            "quality": quality,
            "score": score,
            "up_days_high_vol": up_days_high_vol,
            "down_days_high_vol": down_days_high_vol
        }
    except:
        return {"quality": "UNKNOWN", "score": 5}

# ============================================================
# 🎯 SMART SCANNER (v8.0)
# ============================================================
def smart_scan_stock(ticker, nifty_close=None):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=10)
        if df is None or len(df) < 100: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        if close.iloc[-1] < 30 or vol.mean() < 50000: return None
        
        # Basic technicals
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi = get_rsi(close, 14)
        macd, msig, _ = get_macd(close)
        atr = get_atr(high, low, close, 14)
        
        c = len(df) - 1
        curr = close.iloc[c]
        
        # HARD FILTERS (Stricter than before)
        if not (curr > ema20.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
        if not (55 <= rsi.iloc[c] <= 70): return None  # Tighter RSI range
        if not (macd.iloc[c] > msig.iloc[c] and macd.iloc[c] > 0): return None
        
        # NEW: Early breakout OR pullback entry
        early = detect_early_breakout(df)
        pullback = detect_pullback_entry(df)
        
        # Must be EARLY breakout (Day 1-3) OR healthy PULLBACK
        if not (early and early['is_early_breakout']) and not (pullback and pullback['is_pullback_entry']):
            return None  # Skip late breakouts!
        
        # NEW: Volume quality check
        vol_analysis = analyze_volume_quality(df)
        if vol_analysis['quality'] == 'DISTRIBUTION':
            return None  # Skip if distribution
        
        # NEW: Multi-timeframe check
        mtf = multi_timeframe_confirmation(ticker)
        if mtf and not mtf['all_aligned']:
            return None  # Skip if timeframes don't agree
        
        # SMART SCORING (Only high-probability setups)
        score = 0
        signals = []
        
        # Entry type bonus
        if early and early['is_early_breakout']:
            score += 25
            signals.append(f"Early Breakout (Day {early['days_since_breakout']})")
        
        if pullback and pullback['is_pullback_entry']:
            score += 25
            signals.append("Healthy Pullback Entry")
        
        # Volume quality
        score += vol_analysis['score']
        if vol_analysis['quality'] == 'ACCUMULATION':
            signals.append("Institutional Accumulation")
        
        # RSI in sweet spot
        if 58 <= rsi.iloc[c] <= 65:
            score += 15
            signals.append("Optimal RSI Zone")
        elif 55 <= rsi.iloc[c] <= 70:
            score += 8
        
        # Multi-timeframe alignment
        if mtf and mtf['all_aligned']:
            score += 20
            signals.append("MTF Aligned")
        
        # Distance from EMA (not too extended)
        dist_from_ema20 = ((curr - ema20.iloc[c]) / ema20.iloc[c]) * 100
        if dist_from_ema20 < 5:
            score += 15
            signals.append("Not Extended")
        elif dist_from_ema20 < 8:
            score += 8
        else:
            return None  # Too extended = risky
        
        # RS Rating vs Nifty
        if nifty_close is not None and len(nifty_close) > 63:
            stock_perf = close.iloc[-1] / close.iloc[-63]
            nifty_perf = nifty_close.iloc[-1] / nifty_close.iloc[-63]
            rs = (stock_perf / nifty_perf) * 100
            if rs > 110:
                score += 15
                signals.append(f"Strong RS ({rs:.0f})")
            elif rs > 100:
                score += 8
        else:
            rs = 100
        
        # Only accept HIGH SCORES (stricter)
        if score < 75:
            return None
        
        # SMART TARGETS (based on ATR + resistance)
        # SL: Below EMA20 or recent low
        recent_low = low.iloc[-5:].min()
        sl_ema = ema20.iloc[c] * 0.98
        sl = max(recent_low * 0.99, sl_ema)
        
        risk = curr - sl
        if risk <= 0: return None
        
        # Conservative targets (higher win rate)
        tgt1 = curr + (risk * 1.5)  # 1.5R
        tgt2 = curr + (risk * 2.5)  # 2.5R
        
        risk_pct = (risk / curr) * 100
        if risk_pct > 4: return None  # Skip high-risk setups
        
        return {
            "ticker": ticker.replace(".NS", ""),
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "entry_type": "PULLBACK" if pullback and pullback['is_pullback_entry'] else "EARLY_BREAKOUT",
            "sl": round(sl, 2),
            "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "vol_quality": vol_analysis['quality'],
            "mtf": mtf['alignment'] if mtf else "N/A",
            "rs_rating": round(rs, 0),
            "days_since_breakout": early['days_since_breakout'] if early else 0,
            "signals": signals[:5]
        }
    except Exception as e:
        return None

# ============================================================
# 📥 GET STOCKS
# ============================================================
def get_all_tickers():
    print("Fetching NSE stocks...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        symbols = df["SYMBOL"].dropna().unique().tolist()
        return [f"{s.strip()}.NS" for s in symbols]
    except:
        try:
            url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/nse_stocks.csv"
            df = pd.read_csv(url)
            return [f"{s.strip()}.NS" for s in df["Symbol"].dropna().unique().tolist()]
        except:
            return ["RELIANCE.NS","TCS.NS","INFY.NS"]

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
        print("[TELEGRAM] Sent")
    except Exception as e:
        print(f"[ERROR] {e}")

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"ARTHA {BOT_VERSION} - {BOT_TAGLINE}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("=" * 60)
    
    send_telegram(f"[STARTING] ARTHA {BOT_VERSION} smart scan...")
    
    # STEP 1: Check Market Regime (CRITICAL!)
    print("\n[1/4] Checking Market Regime...")
    regime = detect_market_regime()
    print(f"  Regime: {regime['regime']} | Trade Allowed: {regime['trade_allowed']}")
    
    # If market is BEAR or SIDEWAYS, don't recommend trades
    if not regime['trade_allowed']:
        msg = "=" * 40 + "\n"
        msg += f"⚡ ARTHA {BOT_VERSION}\n"
        msg += f"{BOT_TAGLINE}\n"
        msg += f"📅 {datetime.now().strftime('%A, %d %b %Y')}\n"
        msg += "=" * 40 + "\n\n"
        msg += "[⚠️ MARKET REGIME WARNING]\n\n"
        msg += f"Current Regime: {regime['regime']}\n"
        msg += f"Nifty 20D Change: {regime['nifty_change_20d']:+.2f}%\n"
        msg += f"Confidence: {regime['confidence']}%\n\n"
        msg += "🚫 NO TRADES RECOMMENDED TODAY\n\n"
        msg += "REASON:\n"
        
        if regime['regime'] == 'BEAR':
            msg += "Market in downtrend. Most breakouts will fail.\n"
            msg += "Even 'good' setups get sold into.\n"
        elif regime['regime'] == 'SIDEWAYS':
            msg += "Market is choppy. Breakouts are traps.\n"
            msg += "Wait for clear trend to emerge.\n"
        else:
            msg += "Market direction unclear. Preserve capital.\n"
        
        msg += "\n💡 ARTHA SAYS:\n"
        msg += "The best trade today is NO TRADE.\n"
        msg += "Wait for market to align with your bias.\n\n"
        msg += "Come back tomorrow. Cash is a position too.\n\n"
        msg += "=" * 40 + "\n"
        msg += f"{BOT_NAME} {BOT_VERSION}"
        
        send_telegram(msg)
        return
    
    # STEP 2: Get Nifty benchmark
    print("\n[2/4] Fetching Nifty...")
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except:
        nifty_close = None
    
    # STEP 3: Smart Scan
    print("\n[3/4] Smart Scanning Stocks...")
    tickers = get_all_tickers()
    print(f"Scanning {len(tickers)} stocks with SMART logic...\n")
    
    results = []
    count = 0
    for t in tickers:
        count += 1
        if count % 300 == 0:
            print(f"  Progress: {count}/{len(tickers)} | Found: {len(results)}")
        r = smart_scan_stock(t, nifty_close)
        if r:
            results.append(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Take only TOP 3 (Quality over quantity)
    top3 = results[:3]
    print(f"\n[FOUND] {len(results)} HIGH-QUALITY setups")
    print(f"[SELECTED] Top 3 for delivery")
    
    # STEP 4: Build Report
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = ""
    msg += "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"{BOT_TAGLINE}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    # Market Regime
    msg += "[🌍 MARKET REGIME]\n"
    msg += f"Status: {regime['regime']}\n"
    msg += f"Nifty: {regime['nifty_price']} ({regime['nifty_change_20d']:+.2f}% 20D)\n"
    msg += f"Confidence: {regime['confidence']}%\n"
    msg += f"Trading: {'✅ ALLOWED' if regime['trade_allowed'] else '❌ BLOCKED'}\n\n"
    
    msg += "[🎯 SCAN LOGIC]\n"
    msg += "✓ Early breakouts only (Day 1-3)\n"
    msg += "✓ Pullback entries (safer)\n"
    msg += "✓ Multi-timeframe aligned\n"
    msg += "✓ Institutional accumulation\n"
    msg += "✓ Max 4% risk per trade\n"
    msg += "✓ Only top 3 picks (quality)\n\n"
    
    msg += f"[📊 SCAN STATS]\n"
    msg += f"Stocks scanned: {len(tickers)}\n"
    msg += f"Passed filters: {len(results)}\n"
    msg += f"Selected: Top 3\n\n"
    
    if top3:
        msg += "=" * 40 + "\n"
        msg += "[🏆 TOP 3 SMART PICKS]\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(top3):
            grade = "A+" if s['score'] >= 90 else "A" if s['score'] >= 80 else "B+"
            
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/100)\n"
            msg += f"Entry Type: {s['entry_type']}\n\n"
            
            msg += f"📊 QUALITY METRICS:\n"
            msg += f"  RSI: {s['rsi']} (Sweet spot)\n"
            msg += f"  Volume: {s['vol_quality']}\n"
            msg += f"  MTF Alignment: {s['mtf']}\n"
            msg += f"  RS vs Nifty: {s['rs_rating']}/100\n"
            
            if s['days_since_breakout'] > 0:
                msg += f"  Breakout Age: {s['days_since_breakout']} day(s)\n"
            
            msg += f"\n💰 TRADE SETUP:\n"
            msg += f"  Entry: Rs.{s['price']}\n"
            msg += f"  Stop Loss: Rs.{s['sl']} (Risk: {s['risk_pct']}%)\n"
            msg += f"  Target 1: Rs.{s['tgt1']} (1.5R)\n"
            msg += f"  Target 2: Rs.{s['tgt2']} (2.5R)\n"
            msg += f"  R:R = 1:{s['rr_ratio']}\n\n"
            
            if s['signals']:
                msg += f"✨ Signals: {', '.join(s['signals'])}\n"
            
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[❌ NO HIGH-QUALITY SETUPS TODAY]\n\n"
        msg += "REASON:\n"
        msg += "No stocks met our STRICT criteria:\n"
        msg += "• Early breakout OR pullback\n"
        msg += "• Multi-timeframe alignment\n"
        msg += "• Institutional accumulation\n"
        msg += "• Low risk (<4%)\n\n"
        msg += "This is GOOD - No forced trades!\n"
        msg += "Wait for perfect setup tomorrow.\n\n"
    
    msg += "=" * 40 + "\n"
    msg += "[📐 EXECUTION RULES]\n"
    msg += "1. Enter only if price holds by 10:30 AM\n"
    msg += "2. Risk only 1-2% capital per trade\n"
    msg += "3. Book 50% at T1, trail rest\n"
    msg += "4. Move SL to entry after T1 hit\n"
    msg += "5. Exit if breakout fails within 2 days\n\n"
    
    msg += "[💡 KEY DIFFERENCE FROM v7.0]\n"
    msg += "v7.0: 5 picks, many late/failing\n"
    msg += "v8.0: 3 picks, all early/confirmed\n\n"
    msg += "Fewer trades = Higher win rate\n"
    msg += "Quality > Quantity always wins\n\n"
    
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION} | Educational only"
    
    send_telegram(msg)
    print("\n[DONE] Smart report sent!")

if __name__ == "__main__":
    main()
