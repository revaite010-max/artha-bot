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
BOT_VERSION = "v5.0"
BOT_TAGLINE = "Institutional Grade Intelligence"

# ============================================================
# 🧮 ADVANCED MATH LIBRARY
# ============================================================
def get_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def get_sma(s, n):
    return s.rolling(n).mean()

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
    hist = macd - signal
    return macd, signal, hist

def get_bb(s, n=20, std=2):
    sma = s.rolling(n).mean()
    stdev = s.rolling(n).std()
    upper = sma + (std * stdev)
    lower = sma - (std * stdev)
    return upper, sma, lower

def get_atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def get_adx(high, low, close, n=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    return adx, plus_di, minus_di

def get_stochastic(high, low, close, n=14):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(3).mean()
    return k, d

def get_obv(close, volume):
    """On-Balance Volume - Institutional accumulation indicator"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def get_mfi(high, low, close, volume, n=14):
    """Money Flow Index - Volume weighted RSI"""
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(), 0).rolling(n).sum()
    negative = money_flow.where(typical < typical.shift(), 0).rolling(n).sum()
    mfi = 100 - (100 / (1 + positive/negative))
    return mfi

def get_cci(high, low, close, n=20):
    """Commodity Channel Index"""
    typical = (high + low + close) / 3
    sma = typical.rolling(n).mean()
    mad = typical.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical - sma) / (0.015 * mad)
    return cci

def get_williams_r(high, low, close, n=14):
    """Williams %R"""
    highest = high.rolling(n).max()
    lowest = low.rolling(n).min()
    return -100 * (highest - close) / (highest - lowest)

def get_roc(close, n=12):
    """Rate of Change - Momentum"""
    return ((close - close.shift(n)) / close.shift(n)) * 100

def get_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()

def get_supertrend(high, low, close, atr_period=10, multiplier=3):
    """Supertrend Indicator"""
    atr = get_atr(high, low, close, atr_period)
    hl_avg = (high + low) / 2
    upper = hl_avg + (multiplier * atr)
    lower = hl_avg - (multiplier * atr)
    return upper, lower

# ============================================================
# 📊 PATTERN DETECTION
# ============================================================
def detect_vcp(close, high, low, lookback=50):
    """Volatility Contraction Pattern (Minervini)"""
    try:
        recent = close.iloc[-lookback:]
        ranges = []
        for i in range(0, lookback, 10):
            segment_high = high.iloc[-(lookback-i):-(lookback-i-10)].max() if (lookback-i-10) > 0 else high.iloc[-10:].max()
            segment_low = low.iloc[-(lookback-i):-(lookback-i-10)].min() if (lookback-i-10) > 0 else low.iloc[-10:].min()
            ranges.append((segment_high - segment_low) / segment_low * 100)
        # VCP = decreasing volatility ranges
        if len(ranges) >= 3 and ranges[-1] < ranges[-2] < ranges[-3]:
            return True
        return False
    except:
        return False

def detect_cup_handle(close, lookback=60):
    """Cup and Handle Pattern"""
    try:
        data = close.iloc[-lookback:]
        left_high = data.iloc[:15].max()
        bottom = data.iloc[15:45].min()
        right_high = data.iloc[45:].max()
        # Cup: two highs roughly equal with dip
        if abs(left_high - right_high) / left_high < 0.05 and bottom < left_high * 0.85:
            return True
        return False
    except:
        return False

def detect_breakout_type(close, high, low, volume):
    """Classify breakout quality"""
    try:
        curr = close.iloc[-1]
        prev_high_20 = high.iloc[-21:-1].max()
        prev_high_50 = high.iloc[-51:-1].max()
        avg_vol = volume.iloc[-20:].mean()
        curr_vol = volume.iloc[-1]
        
        if curr > prev_high_50 and curr_vol > avg_vol * 2:
            return "🔥 STRONG 50-Day Breakout"
        elif curr > prev_high_20 and curr_vol > avg_vol * 2:
            return "⚡ 20-Day Breakout"
        elif curr > prev_high_20:
            return "📈 Weak Breakout (Low Vol)"
        else:
            return "📊 No Breakout"
    except:
        return "N/A"

def calculate_rs_rating(stock_close, nifty_close):
    """Relative Strength vs Nifty (like IBD RS Rating)"""
    try:
        periods = [63, 126, 189, 252]  # 3M, 6M, 9M, 12M
        stock_perf = []
        nifty_perf = []
        for p in periods:
            if len(stock_close) > p and len(nifty_close) > p:
                stock_perf.append(stock_close.iloc[-1] / stock_close.iloc[-p])
                nifty_perf.append(nifty_close.iloc[-1] / nifty_close.iloc[-p])
        if not stock_perf:
            return 50
        # Weighted: recent quarters matter more
        weights = [0.40, 0.20, 0.20, 0.20][:len(stock_perf)]
        stock_score = sum(s*w for s,w in zip(stock_perf, weights))
        nifty_score = sum(n*w for n,w in zip(nifty_perf, weights))
        rs = (stock_score / nifty_score) * 100
        return min(max(rs, 0), 100)
    except:
        return 50

def detect_institutional_activity(volume, close):
    """Detect institutional accumulation via volume patterns"""
    try:
        avg_vol = volume.rolling(50).mean()
        recent_high_vol_days = 0
        for i in range(-10, 0):
            if volume.iloc[i] > avg_vol.iloc[i] * 1.5 and close.iloc[i] > close.iloc[i-1]:
                recent_high_vol_days += 1
        if recent_high_vol_days >= 4:
            return "🏦 HEAVY Institutional Buying"
        elif recent_high_vol_days >= 2:
            return "💼 Moderate Institutional Activity"
        else:
            return "📊 Normal Activity"
    except:
        return "N/A"

def detect_base_quality(close, high, low, lookback=30):
    """Analyze base formation quality"""
    try:
        data_close = close.iloc[-lookback:]
        data_high = high.iloc[-lookback:]
        data_low = low.iloc[-lookback:]
        range_pct = ((data_high.max() - data_low.min()) / data_low.min()) * 100
        volatility = data_close.pct_change().std() * 100
        
        if range_pct < 12 and volatility < 1.5:
            return {"quality": "TIGHT", "score": 10}
        elif range_pct < 20 and volatility < 2.5:
            return {"quality": "GOOD", "score": 7}
        elif range_pct < 30:
            return {"quality": "LOOSE", "score": 4}
        else:
            return {"quality": "WIDE", "score": 0}
    except:
        return {"quality": "N/A", "score": 0}

# ============================================================
# 📥 GET STOCKS (Dynamic)
# ============================================================
def get_all_tickers():
    print("📥 Fetching NSE stock list...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        symbols = df["SYMBOL"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"   ✅ Loaded {len(tickers)} NSE stocks")
        return tickers
    except:
        pass
    
    try:
        url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/nse_stocks.csv"
        df = pd.read_csv(url)
        symbols = df["Symbol"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"   ✅ Loaded {len(tickers)} via mirror")
        return tickers
    except:
        return ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"]

# ============================================================
# 🚀 IPO TRACKER
# ============================================================
def get_ipo_data():
    print("🔍 Fetching IPO data...")
    ipo_msg = "*🚀 IPO INTELLIGENCE*\n"
    
    try:
        url = "https://www.chittorgarh.com/ipo/ipo_list.asp"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'table'})
        if tables:
            ipo_msg += "\n*📅 Upcoming/Open:*\n"
            for row in tables[0].find_all('tr')[1:5]:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    name = cols[0].text.strip()[:28]
                    date = cols[1].text.strip() if len(cols) > 1 else "N/A"
                    ipo_msg += f"• {name}\n  📅 {date}\n"
    except:
        ipo_msg += "  ⚠️ Data unavailable\n"
    
    try:
        url = "https://www.investorgain.com/report/live-ipo-gmp/331/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table')
        if table:
            ipo_msg += "\n*💎 GMP Signal:*\n"
            for row in table.find_all('tr')[1:4]:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    name = cols[0].text.strip()[:22]
                    gmp = cols[3].text.strip() if len(cols) > 3 else "N/A"
                    ipo_msg += f"• {name}: {gmp}\n"
    except:
        pass
    
    return ipo_msg + "\n"

# ============================================================
# 🌍 ADVANCED GLOBAL CONTEXT
# ============================================================
def get_global_context():
    lines = []
    score = 0
    max_score = 8
    
    try:
        syms = ["^GSPC","^IXIC","^VIX","^INDIAVIX","DX-Y.NYB","CL=F","GC=F","^NSEI"]
        data = yf.download(syms, period="30d", progress=False)
        close = data["Close"]
        
        # S&P 500 trend
        try:
            v = close["^GSPC"].dropna().iloc[-1]
            a20 = close["^GSPC"].dropna().rolling(20).mean().iloc[-1]
            a5 = close["^GSPC"].dropna().rolling(5).mean().iloc[-1]
            if v > a20 and a5 > a20:
                score += 1
                lines.append("  S&P 500  : ✅ Strong Uptrend")
            elif v > a20:
                score += 0.5
                lines.append("  S&P 500  : 🟡 Above 20DMA")
            else:
                lines.append("  S&P 500  : ❌ Below 20DMA")
        except: pass
        
        # Nasdaq
        try:
            v = close["^IXIC"].dropna().iloc[-1]
            a = close["^IXIC"].dropna().rolling(20).mean().iloc[-1]
            if v > a: score += 1; lines.append("  Nasdaq   : ✅ Bullish")
            else: lines.append("  Nasdaq   : ❌ Weak")
        except: pass
        
        # US VIX
        try:
            vix = close["^VIX"].dropna().iloc[-1]
            if vix < 15: score += 1; lines.append(f"  US VIX   : ✅ {vix:.1f} (Calm)")
            elif vix < 20: score += 0.5; lines.append(f"  US VIX   : 🟡 {vix:.1f}")
            else: lines.append(f"  US VIX   : ❌ {vix:.1f} (Fear)")
        except: pass
        
        # India VIX
        try:
            ivix = close["^INDIAVIX"].dropna().iloc[-1]
            if ivix < 13: score += 1; lines.append(f"  IND VIX  : ✅ {ivix:.1f} (Calm)")
            elif ivix < 16: score += 0.5; lines.append(f"  IND VIX  : 🟡 {ivix:.1f}")
            else: lines.append(f"  IND VIX  : ❌ {ivix:.1f} (Fear)")
        except: pass
        
        # DXY
        try:
            dxy = close["DX-Y.NYB"].dropna().pct_change().iloc[-1] * 100
            if dxy < 0: score += 1; lines.append(f"  DXY      : ✅ {dxy:+.2f}% (Weak $)")
            elif dxy < 0.3: score += 0.5; lines.append(f"  DXY      : 🟡 {dxy:+.2f}%")
            else: lines.append(f"  DXY      : ❌ {dxy:+.2f}% (Strong $)")
        except: pass
        
        # Crude
        try:
            crude = close["CL=F"].dropna().pct_change().iloc[-1] * 100
            crude_val = close["CL=F"].dropna().iloc[-1]
            if crude < 1: score += 1; lines.append(f"  Crude    : ✅ ${crude_val:.0f} ({crude:+.1f}%)")
            else: lines.append(f"  Crude    : ⚠️ ${crude_val:.0f} ({crude:+.1f}%)")
        except: pass
        
        # Gold
        try:
            gold_val = close["GC=F"].dropna().iloc[-1]
            gold_chg = close["GC=F"].dropna().pct_change(5).iloc[-1] * 100
            lines.append(f"  Gold     : ${gold_val:.0f} ({gold_chg:+.1f}% 7D)")
            if gold_chg < 2: score += 1
        except: pass
        
        # Nifty trend
        try:
            nifty = close["^NSEI"].dropna()
            nifty_curr = nifty.iloc[-1]
            nifty_20 = nifty.rolling(20).mean().iloc[-1]
            nifty_50 = nifty.rolling(50).mean().iloc[-1]
            if nifty_curr > nifty_20 > nifty_50:
                score += 1
                lines.append(f"  Nifty    : ✅ Strong Uptrend")
            elif nifty_curr > nifty_20:
                score += 0.5
                lines.append(f"  Nifty    : 🟡 Above 20DMA")
            else:
                lines.append(f"  Nifty    : ❌ Weakening")
        except: pass
        
    except Exception as e:
        lines.append(f"  ⚠️ {str(e)[:40]}")
    
    pct = (score / max_score) * 100
    if pct >= 75: verdict = f"🟢 STRONG ({score:.1f}/{max_score})"
    elif pct >= 50: verdict = f"🟡 NEUTRAL ({score:.1f}/{max_score})"
    else: verdict = f"🔴 WEAK ({score:.1f}/{max_score})"
    
    return {"lines": lines, "score": score, "verdict": verdict, "pct": pct}

# ============================================================
# 🎯 ADVANCED STOCK SCANNER (100-Point System)
# ============================================================
def scan_stock(ticker, nifty_close=None):
    try:
        df = yf.download(ticker, period="1y", interval="1d", 
                         progress=False, timeout=10)
        if df is None or len(df) < 200:
            return None
        
        close  = df['Close'].squeeze()
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        vol    = df['Volume'].squeeze()
        open_  = df['Open'].squeeze()
        
        # Basic filters
        if close.iloc[-1] < 30 or vol.mean() < 50000:
            return None
        
        # ── CALCULATE ALL INDICATORS ──
        ema9   = get_ema(close, 9)
        ema21  = get_ema(close, 21)
        ema50  = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        sma50  = get_sma(close, 50)
        rsi    = get_rsi(close, 14)
        macd, msig, mhist = get_macd(close)
        bb_up, bb_mid, bb_low = get_bb(close, 20)
        atr = get_atr(high, low, close, 14)
        adx, plus_di, minus_di = get_adx(high, low, close)
        obv = get_obv(close, vol)
        mfi = get_mfi(high, low, close, vol)
        roc = get_roc(close, 12)
        
        c = len(df) - 1
        curr = close.iloc[c]
        avg_vol = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 0
        
        # ── HARD FILTERS (Must pass all) ──
        if not (curr > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]):
            return None
        if not (vol_ratio >= 1.5):
            return None
        if not (50 <= rsi.iloc[c] <= 80):
            return None
        if not (adx.iloc[c] > 20):
            return None
        if not (macd.iloc[c] > msig.iloc[c]):
            return None
        
        # ── 100-POINT SCORING SYSTEM ──
        score = 0
        signals = []
        
        # 1. TREND STRENGTH (20 points)
        if curr > ema9.iloc[c] > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]:
            score += 20; signals.append("Perfect EMA Stack")
        elif curr > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]:
            score += 15
        else:
            score += 10
        
        # 2. VOLUME QUALITY (15 points)
        if vol_ratio >= 3.5:
            score += 15; signals.append("Massive Volume")
        elif vol_ratio >= 2.5:
            score += 12
        elif vol_ratio >= 2.0:
            score += 8
        else:
            score += 5
        
        # 3. MOMENTUM (15 points)
        if 60 <= rsi.iloc[c] <= 70:
            score += 10; signals.append("Ideal RSI Zone")
        elif 55 <= rsi.iloc[c] <= 75:
            score += 7
        else:
            score += 3
        
        if roc.iloc[c] > 10:
            score += 5; signals.append("Strong Momentum")
        elif roc.iloc[c] > 5:
            score += 3
        
        # 4. RELATIVE STRENGTH (15 points)
        if nifty_close is not None:
            rs_rating = calculate_rs_rating(close, nifty_close)
            if rs_rating >= 90:
                score += 15; signals.append("Top 10% Performer")
            elif rs_rating >= 80:
                score += 12
            elif rs_rating >= 70:
                score += 8
            else:
                score += 4
        else:
            rs_rating = 50
            score += 5
        
        # 5. INSTITUTIONAL FLOW (10 points)
        obv_slope = (obv.iloc[c] - obv.iloc[c-20]) / abs(obv.iloc[c-20]) * 100 if obv.iloc[c-20] != 0 else 0
        if obv_slope > 20:
            score += 10; signals.append("Heavy Accumulation")
        elif obv_slope > 10:
            score += 6
        else:
            score += 3
        
        # 6. TREND STRENGTH ADX (10 points)
        if adx.iloc[c] >= 30 and plus_di.iloc[c] > minus_di.iloc[c]:
            score += 10; signals.append("Very Strong Trend")
        elif adx.iloc[c] >= 25:
            score += 7
        else:
            score += 4
        
        # 7. 52-WEEK PROXIMITY (5 points)
        high_52w = high.rolling(252).max().iloc[c]
        dist_52w = (high_52w - curr) / curr * 100
        if dist_52w <= 3:
            score += 5; signals.append("Near 52W High")
        elif dist_52w <= 8:
            score += 3
        else:
            score += 1
        
        # 8. BASE QUALITY (5 points)
        base = detect_base_quality(close, high, low)
        score += base["score"] / 2
        if base["quality"] == "TIGHT":
            signals.append("Tight Base")
        
        # 9. MFI + BOLLINGER (5 points)
        if not pd.isna(mfi.iloc[c]) and 50 <= mfi.iloc[c] <= 80:
            score += 3
        if curr >= bb_up.iloc[c] * 0.98:
            score += 2; signals.append("BB Breakout")
        
        # ── ADVANCED METRICS ──
        # ATR-based stop loss (dynamic)
        atr_stop = round(curr - (1.5 * atr.iloc[c]), 2)
        ema_stop = round(ema21.iloc[c] * 0.98, 2)
        sl = max(atr_stop, ema_stop)  # Use tighter of the two
        
        # Targets based on ATR
        tgt1 = round(curr + (2 * atr.iloc[c]), 2)
        tgt2 = round(curr + (4 * atr.iloc[c]), 2)
        tgt3 = round(curr + (6 * atr.iloc[c]), 2)
        
        # Risk-Reward
        risk = curr - sl
        reward = tgt1 - curr
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        risk_pct = round((risk / curr) * 100, 2)
        
        # Position sizing (assuming 1L capital, 1% risk = 1000)
        position_size = round(1000 / risk) if risk > 0 else 0
        position_value = round(position_size * curr, 0)
        
        # Institutional detection
        inst_signal = detect_institutional_activity(vol, close)
        
        # Breakout type
        breakout = detect_breakout_type(close, high, low, vol)
        
        # VCP pattern
        vcp = detect_vcp(close, high, low)
        if vcp:
            signals.append("VCP Pattern")
            score += 3
        
        # Cup & Handle
        cup = detect_cup_handle(close)
        if cup:
            signals.append("Cup & Handle")
            score += 3
        
        return {
            "ticker": ticker.replace(".NS",""),
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "adx": round(adx.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1),
            "rs_rating": round(rs_rating, 0),
            "mfi": round(mfi.iloc[c], 1) if not pd.isna(mfi.iloc[c]) else 0,
            "roc": round(roc.iloc[c], 1),
            "atr": round(atr.iloc[c], 2),
            "sl": sl,
            "tgt1": tgt1,
            "tgt2": tgt2,
            "tgt3": tgt3,
            "rr_ratio": rr_ratio,
            "risk_pct": risk_pct,
            "position_size": position_size,
            "position_value": position_value,
            "dist_52w": round(dist_52w, 1),
            "inst_signal": inst_signal,
            "breakout_type": breakout,
            "base_quality": base["quality"],
            "signals": signals[:4]  # Top 4 signals
        }
    except Exception as e:
        return None

# ============================================================
# 📤 TELEGRAM
# ============================================================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
        for part in parts:
            requests.post(url, json={
                "chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown"
            }, timeout=15)
            time.sleep(1)
        print("✅ Telegram sent")
    except Exception as e:
        print(f"❌ {e}")

# ============================================================
# 🚀 MAIN ENGINE
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"⚡ ARTHA {BOT_VERSION} - Institutional Grade Scanner")
    print(f"⏰ {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'='*60}\n")
    
    # Global context
    print("🌍 Scanning global markets...")
    ctx = get_global_context()
    print(f"   {ctx['verdict']}")
    
    # Get Nifty for RS calculation
    print("\n📈 Fetching Nifty benchmark...")
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except:
        nifty_close = None
    
    # IPO data
    ipo_data = get_ipo_data()
    
    # Get all stocks
    tickers = get_all_tickers()
    print(f"\n🇮🇳 Scanning {len(tickers)} stocks with 100-point system...\n")
    
    results = []
    for i, t in enumerate(tickers):
        if (i+1) % 200 == 0:
            print(f"   Progress: {i+1}/{len(tickers)} | Qualified: {len(results)}")
        r = scan_stock(t, nifty_close)
        if r and r["score"] >= 60:  # Only high-quality setups
            results.append(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"\n✅ Elite Breakouts (Score ≥60): {len(results)}")
    
    # Build message
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*{BOT_NAME} {BOT_VERSION}*\n"
    msg += f"_{BOT_TAGLINE}_\n"
    msg += f"📅 {today}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    msg += f"*🌍 GLOBAL PULSE*\n"
    for line in ctx["lines"]:
        msg += f"{line}\n"
    msg += f"  Verdict: {ctx['verdict']}\n\n"
    
    msg += ipo_data
    
    msg += f"*📊 SCAN SUMMARY*\n"
    msg += f"  Universe: {len(tickers)} stocks\n"
    msg += f"  Elite Setups: {len(results)}\n"
    msg += f"  Min Score: 60/100\n\n"
    
    if top5:
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"*🏆 TOP 5 ELITE PICKS*\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, s in enumerate(top5):
            grade = "A+" if s['score'] >= 85 else "A" if s['score'] >= 75 else "B+"
            
            msg += f"*{medals[i]} {s['ticker']}* | Grade: *{grade}* ({s['score']}/100)\n\n"
            
            msg += f"💰 *Price*: ₹{s['price']}\n"
            msg += f"📊 *Metrics*:\n"
            msg += f"  • RSI: {s['rsi']} | ADX: {s['adx']}\n"
            msg += f"  • RS Rating: {s['rs_rating']:.0f}/100\n"
            msg += f"  • MFI: {s['mfi']} | ROC: {s['roc']:+.1f}%\n"
            msg += f"  • Volume: {s['vol_ratio']}x average\n"
            msg += f"  • ATR: ₹{s['atr']}\n\n"
            
            msg += f"📍 *Position*:\n"
            msg += f"  • {s['dist_52w']:.1f}% from 52W High\n"
            msg += f"  • Base: {s['base_quality']}\n"
            msg += f"  • {s['breakout_type']}\n"
            msg += f"  • {s['inst_signal']}\n\n"
            
            msg += f"🎯 *Trade Setup*:\n"
            msg += f"  • Entry: ₹{s['price']}\n"
            msg += f"  • SL: ₹{s['sl']} (Risk: {s['risk_pct']}%)\n"
            msg += f"  • T1: ₹{s['tgt1']} | T2: ₹{s['tgt2']} | T3: ₹{s['tgt3']}\n"
            msg += f"  • R:R Ratio: 1:{s['rr_ratio']}\n\n"
            
            msg += f"💼 *Position Size* (₹1L, 1% risk):\n"
            msg += f"  • Shares: {s['position_size']}\n"
            msg += f"  • Value: ₹{s['position_value']:,.0f}\n\n"
            
            if s['signals']:
                msg += f"✨ *Signals*: {', '.join(s['signals'])}\n"
            
            msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    else:
        msg += f"*📉 No elite setups today*\n"
        msg += f"_Market not offering high-probability trades_\n\n"
    
    # Trading Rules
    msg += f"*📐 EXECUTION RULES*\n"
    msg += f"1️⃣ Enter only if breakout holds by 10:30 AM\n"
    msg += f"2️⃣ Never risk more than 1-2% per trade\n"
    msg += f"3️⃣ Max 5 open positions\n"
    msg += f"4️⃣ Book 40% at T1, 30% at T2, trail rest\n"
    msg += f"5️⃣ Move SL to entry after T1 hit\n\n"
    
    # Grade legend
    msg += f"*📖 GRADE MEANING*\n"
    msg += f"  A+ (85+): Very High Conviction\n"
    msg += f"  A (75-84): High Conviction\n"
    msg += f"  B+ (60-74): Moderate Conviction\n\n"
    
    # Market advice
    if ctx["pct"] >= 75:
        advice = "🟢 _All systems green. Deploy full capital._"
    elif ctx["pct"] >= 50:
        advice = "🟡 _Mixed signals. Trade selective. Half size._"
    else:
        advice = "🔴 _Markets weak. Sit on hands. Capital first._"
    
    msg += f"*💡 ARTHA SAYS:*\n{advice}\n\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"_{BOT_NAME} {BOT_VERSION}_\n"
    msg += f"_Educational only. Not SEBI advice._"
    
    send_telegram(msg)
    print("\n🎉 ARTHA Complete!")

if __name__ == "__main__":
    main()
