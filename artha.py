import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 🔑 Settings
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v5.1"
BOT_TAGLINE = "Institutional Grade Intelligence"

# ============================================================
# 🧮 MATH LIBRARY
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

def get_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def get_mfi(high, low, close, volume, n=14):
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(), 0).rolling(n).sum()
    negative = money_flow.where(typical < typical.shift(), 0).rolling(n).sum()
    mfi = 100 - (100 / (1 + positive/negative))
    return mfi

def get_roc(close, n=12):
    return ((close - close.shift(n)) / close.shift(n)) * 100

def calculate_rs_rating(stock_close, nifty_close):
    try:
        periods = [63, 126, 189, 252]
        stock_perf = []
        nifty_perf = []
        for p in periods:
            if len(stock_close) > p and len(nifty_close) > p:
                stock_perf.append(stock_close.iloc[-1] / stock_close.iloc[-p])
                nifty_perf.append(nifty_close.iloc[-1] / nifty_close.iloc[-p])
        if not stock_perf: return 50
        weights = [0.40, 0.20, 0.20, 0.20][:len(stock_perf)]
        stock_score = sum(s*w for s,w in zip(stock_perf, weights))
        nifty_score = sum(n*w for n,w in zip(nifty_perf, weights))
        rs = (stock_score / nifty_score) * 100
        return min(max(rs, 0), 100)
    except:
        return 50

def detect_base_quality(close, high, low, lookback=30):
    try:
        data_close = close.iloc[-lookback:]
        data_high = high.iloc[-lookback:]
        data_low = low.iloc[-lookback:]
        range_pct = ((data_high.max() - data_low.min()) / data_low.min()) * 100
        volatility = data_close.pct_change().std() * 100
        if range_pct < 12 and volatility < 1.5: return {"quality": "TIGHT", "score": 10}
        elif range_pct < 20 and volatility < 2.5: return {"quality": "GOOD", "score": 7}
        elif range_pct < 30: return {"quality": "LOOSE", "score": 4}
        else: return {"quality": "WIDE", "score": 0}
    except: return {"quality": "N/A", "score": 0}

# ============================================================
# 📥 STOCKS
# ============================================================
def get_all_tickers():
    print("Fetching NSE stocks...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        symbols = df["SYMBOL"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"   Loaded {len(tickers)} stocks")
        return tickers
    except: pass
    
    try:
        url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/nse_stocks.csv"
        df = pd.read_csv(url)
        symbols = df["Symbol"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"   Loaded {len(tickers)} via mirror")
        return tickers
    except: pass
    
    print("   Using backup list")
    return ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

# ============================================================
# 🚀 IPO TRACKER
# ============================================================
def get_ipo_data():
    print("Fetching IPO data...")
    ipo_msg = "[IPO INTELLIGENCE]\n"
    
    try:
        url = "https://www.chittorgarh.com/ipo/ipo_list.asp"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'table'})
        if tables:
            ipo_msg += "\n[Upcoming/Open IPOs]\n"
            for row in tables[0].find_all('tr')[1:5]:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    name = cols[0].text.strip()[:28]
                    date = cols[1].text.strip() if len(cols) > 1 else "N/A"
                    ipo_msg += f"- {name} | Date: {date}\n"
    except:
        ipo_msg += "- Data unavailable\n"
    
    try:
        url = "https://www.investorgain.com/report/live-ipo-gmp/331/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table')
        if table:
            ipo_msg += "\n[GMP Signal - Higher = Better Listing]\n"
            for row in table.find_all('tr')[1:4]:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    name = cols[0].text.strip()[:22]
                    gmp = cols[3].text.strip() if len(cols) > 3 else "N/A"
                    ipo_msg += f"- {name}: GMP {gmp}\n"
            ipo_msg += "\n"
    except: pass
    
    return ipo_msg + "\n"

# ============================================================
# 🌍 GLOBAL CONTEXT
# ============================================================
def get_global_context():
    lines = []
    score = 0
    max_score = 8
    try:
        syms = ["^GSPC","^IXIC","^VIX","^INDIAVIX","DX-Y.NYB","CL=F","GC=F","^NSEI"]
        data = yf.download(syms, period="30d", progress=False)
        close = data["Close"]
        
        try:
            v = close["^GSPC"].dropna().iloc[-1]; a20 = close["^GSPC"].dropna().rolling(20).mean().iloc[-1]
            if v > a20: score += 1; lines.append("  S&P500 : BULLISH")
            else: lines.append("  S&P500 : WEAK")
        except: pass
        
        try:
            v = close["^IXIC"].dropna().iloc[-1]; a = close["^IXIC"].dropna().rolling(20).mean().iloc[-1]
            if v > a: score += 1; lines.append("  Nasdaq  : BULLISH")
            else: lines.append("  Nasdaq  : WEAK")
        except: pass
        
        try:
            vix = close["^VIX"].dropna().iloc[-1]
            if vix < 16: score += 1; lines.append(f"  US VIX  : LOW ({vix:.1f})")
            elif vix < 20: score += 0.5; lines.append(f"  US VIX  : MODERATE ({vix:.1f})")
            else: lines.append(f"  US VIX  : HIGH ({vix:.1f})")
        except: pass
        
        try:
            ivix = close["^INDIAVIX"].dropna().iloc[-1]
            if ivix < 14: score += 1; lines.append(f"  IND VIX : LOW ({ivix:.1f})")
            elif ivix < 17: score += 0.5; lines.append(f"  IND VIX : MODERATE ({ivix:.1f})")
            else: lines.append(f"  IND VIX : HIGH ({ivix:.1f})")
        except: pass
        
        try:
            dxy = close["DX-Y.NYB"].dropna().pct_change().iloc[-1] * 100
            if dxy < 0.3: score += 1; lines.append(f"  DXY     : STABLE ({dxy:+.1f}%)")
            else: lines.append(f"  DXY     : RISING ({dxy:+.1f}%)")
        except: pass
        
        try:
            crude_val = close["CL=F"].dropna().iloc[-1]
            crude_chg = close["CL=F"].dropna().pct_change().iloc[-1] * 100
            if crude_chg < 2: score += 1; lines.append(f"  Crude   : ${crude_val:.0f} ({crude_chg:+.1f}%)")
            else: lines.append(f"  Crude   : ${crude_val:.0f} ({crude_chg:+.1f}%) SPIKE")
        except: pass
        
        try:
            gold_val = close["GC=F"].dropna().iloc[-1]
            gold_chg = close["GC=F"].dropna().pct_change(5).iloc[-1] * 100
            lines.append(f"  Gold    : ${gold_val:.0f} ({gold_chg:+.1f}% 7D)")
            if gold_chg < 2: score += 1
        except: pass
        
        try:
            nifty = close["^NSEI"].dropna(); nc = nifty.iloc[-1]
            n20 = nifty.rolling(20).mean().iloc[-1]; n50 = nifty.rolling(50).mean().iloc[-1]
            if nc > n20 > n50: score += 1; lines.append("  NIFTY   : STRONG UPTREND")
            elif nc > n20: score += 0.5; lines.append("  NIFTY   : ABOVE 20DMA")
            else: lines.append("  NIFTY   : WEAKENING")
        except: pass
        
    except Exception as e:
        lines.append(f"  Error: {str(e)[:40]}")
    
    pct = (score / max_score) * 100
    if pct >= 75: verdict = f"STRONG ({score}/{max_score})"
    elif pct >= 50: verdict = f"NEUTRAL ({score}/{max_score})"
    else: verdict = f"WEAK ({score}/{max_score})"
    
    return {"lines": lines, "score": score, "verdict": verdict, "pct": pct}

# ============================================================
# 🎯 SCANNER
# ============================================================
def scan_stock(ticker, nifty_close=None):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=10)
        if df is None or len(df) < 200: return None
        
        close = df['Close'].squeeze(); high = df['High'].squeeze()
        low = df['Low'].squeeze(); vol = df['Volume'].squeeze(); open_ = df['Open'].squeeze()
        
        if close.iloc[-1] < 30 or vol.mean() < 50000: return None
        
        ema9 = get_ema(close, 9); ema21 = get_ema(close, 21); ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200); rsi = get_rsi(close, 14)
        macd, msig, mhist = get_macd(close); bb_up, bb_mid, bb_low = get_bb(close, 20)
        atr = get_atr(high, low, close, 14); adx, plus_di, minus_di = get_adx(high, low, close)
        obv = get_obv(close, vol); mfi = get_mfi(high, low, close, vol); roc = get_roc(close, 12)
        
        c = len(df) - 1; curr = close.iloc[c]; avg_vol = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 0
        
        # HARD FILTERS
        if not (curr > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
        if not (vol_ratio >= 1.5): return None
        if not (50 <= rsi.iloc[c] <= 80): return None
        if not (adx.iloc[c] > 20): return None
        if not (macd.iloc[c] > msig.iloc[c]): return None
        
        # SCORING (100 points max)
        score = 0; signals = []
        
        # Trend stack (20)
        if curr > ema9.iloc[c] > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]: score += 20; signals.append("Perfect EMA Stack")
        elif curr > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]: score += 15
        else: score += 10
        
        # Volume (15)
        if vol_ratio >= 3.5: score += 15; signals.append("Massive Volume")
        elif vol_ratio >= 2.5: score += 12
        elif vol_ratio >= 2.0: score += 8
        else: score += 5
        
        # Momentum (15)
        if 60 <= rsi.iloc[c] <= 70: score += 10; signals.append("Ideal RSI Zone")
        elif 55 <= rsi.iloc[c] <= 75: score += 7
        else: score += 3
        if roc.iloc[c] > 10: score += 5; signals.append("Strong Momentum")
        elif roc.iloc[c] > 5: score += 3
        
        # RS Rating vs Nifty (15)
        if nifty_close is not None:
            rs_rating = calculate_rs_rating(close, nifty_close)
            if rs_rating >= 90: score += 15; signals.append(f"Top 10% Performer (RS:{rs_rating:.0f})")
            elif rs_rating >= 80: score += 12
            elif rs_rating >= 70: score += 8
            else: score += 4
        else: rs_rating = 50; score += 5
        
        # Institutional Flow (10)
        obv_slope = (obv.iloc[c] - obv.iloc[c-20]) / abs(obv.iloc[c-20]) * 100 if obv.iloc[c-20] != 0 else 0
        if obv_slope > 20: score += 10; signals.append("Heavy Accumulation")
        elif obv_slope > 10: score += 6
        else: score += 3
        
        # ADX Strength (10)
        if adx.iloc[c] >= 30 and plus_di.iloc[c] > minus_di.iloc[c]: score += 10; signals.append("Very Strong Trend")
        elif adx.iloc[c] >= 25: score += 7
        else: score += 4
        
        # 52W Proximity (5)
        high_52w = high.rolling(252).max().iloc[c]
        dist_52w = (high_52w - curr) / curr * 100
        if dist_52w <= 3: score += 5; signals.append("Near 52W High")
        elif dist_52w <= 8: score += 3
        else: score += 1
        
        # Base Quality (5)
        base = detect_base_quality(close, high, low)
        score += base["score"] / 2
        if base["score"] == 10: signals.append("Tight Base")
        
        # MFI/BB (5)
        if not pd.isna(mfi.iloc[c]) and 50 <= mfi.iloc[c] <= 80: score += 3
        if curr >= bb_up.iloc[c] * 0.98: score += 2; signals.append("BB Breakout")
        
        # ATR SL & Targets
        atr_stop = round(curr - (1.5 * atr.iloc[c]), 2)
        ema_stop = round(ema21.iloc[c] * 0.98, 2)
        sl = max(atr_stop, ema_stop)
        tgt1 = round(curr + (2 * atr.iloc[c]), 2)
        tgt2 = round(curr + (4 * atr.iloc[c]), 2)
        tgt3 = round(curr + (6 * atr.iloc[c]), 2)
        
        risk = curr - sl; reward = tgt1 - curr
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        risk_pct = round((risk / curr) * 100, 2)
        position_size = round(1000 / risk) if risk > 0 else 0
        position_value = round(position_size * curr, 0)
        
        inst_signal = "Normal Activity"
        if vol_ratio >= 2.5 and rsi.iloc[c] > 60: inst_signal = "Institutional Buying Detected"
        
        breakout_type = "Consolidation Breakout" if curr >= bb_up.iloc[c] * 0.98 else "Normal Move"
        
        return {
            "ticker": ticker.replace(".NS",""), "score": round(score, 1),
            "price": round(curr, 2), "rsi": round(rsi.iloc[c], 1), "adx": round(adx.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1), "rs_rating": round(rs_rating, 0),
            "mfi": round(mfi.iloc[c], 1) if not pd.isna(mfi.iloc[c]) else 0,
            "roc": round(roc.iloc[c], 1), "atr": round(atr.iloc[c], 2),
            "sl": sl, "tgt1": tgt1, "tgt2": tgt2, "tgt3": tgt3,
            "rr_ratio": rr_ratio, "risk_pct": risk_pct,
            "position_size": position_size, "position_value": position_value,
            "dist_52w": round(dist_52w, 1), "inst_signal": inst_signal,
            "breakout_type": breakout_type, "base_quality": base["quality"],
            "signals": signals[:4]
        }
    except Exception as e:
        return None

# ============================================================
# 📤 TELEGRAM SENDER (SAFE VERSION - NO MARKDOWN)
# ============================================================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        # Split into safe chunks (Telegram limit is 4096, using 3800 for safety)
        max_len = 3800
        parts = [msg[i:i+max_len] for i in range(0, len(msg), max_len)]
        
        for i, part in enumerate(parts, 1):
            payload = {
                "chat_id": CHAT_ID,
                "text": part
                # NO parse_mode here!
            }
            
            r = requests.post(url, json=payload, timeout=25)
            status_code = r.status_code
            
            print(f"[TELEGRAM] Part {i}/{len(parts)} - Status: {status_code}")
            
            if status_code == 200:
                print("[TELEGRAM] Success")
            else:
                print(f"[TELEGRAM] Error response: {r.text[:300]}")
            
            time.sleep(0.8)  # Small delay between parts
        
        print("[TELEGRAM] All parts sent successfully!")
        
    except Exception as e:
        print(f"[TELEGRAM ERROR] Failed: {e}")

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("="*60)
    print(f"ARTHAT {BOT_VERSION}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("="*60)
    
    # Test message first
    send_telegram("[TEST] ARTHA bot is alive. Full report coming soon...")
    
    print("\n[STEP 1] Global Context...")
    ctx = get_global_context()
    print(f"[GLOBAL] {ctx['verdict']}")
    
    print("\n[STEP 2] Fetching Nifty benchmark...")
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except: nifty_close = None
    
    print("\n[STEP 3] IPO Data...")
    ipo_data = get_ipo_data()
    
    print("\n[STEP 4] Getting Stock List...")
    tickers = get_all_tickers()
    print(f"\n[STEP 5] Scanning {len(tickers)} stocks...\n")
    
    results = []
    count = 0
    for t in tickers:
        count += 1
        if count % 300 == 0: print(f"... Scanned {count}/{len(tickers)}, Found: {len(results)}")
        r = scan_stock(t, nifty_close)
        if r and r["score"] >= 55: results.append(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"\n[DONE] Elite Setups: {len(results)} out of {len(tickers)}")
    
    # BUILD PLAIN TEXT MESSAGE (Safe for Telegram)
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = ""
    msg += "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    msg += "[GLOBAL MARKETS]\n"
    for line in ctx["lines"]:
        msg += line + "\n"
    msg += f"Verdict: {ctx['verdict']}\n\n"
    
    msg += ipo_data
    
    msg += f"[SCAN SUMMARY]\n"
    msg += f"Universe: {len(tickers)} stocks\n"
    msg += f"Elite Setups: {len(results)}\n"
    msg += f"Min Score: 55/100\n\n"
    
    if top5:
        msg += "-" * 40 + "\n"
        msg += "[TOP 5 ELITE PICKS]\n"
        msg += "-" * 40 + "\n\n"
        
        ranks = ["#1", "#2", "#3", "#4", "#5"]
        grades = ["A+", "A+", "A", "B+", "B+"]
        
        for i, s in enumerate(top5):
            grade = "A+" if s['score'] >= 85 else "A" if s['score'] >= 75 else "B+"
            
            msg += f"{ranks[i]} {s['ticker']} | GRADE: {grade} ({s['score']}/100)\n\n"
            
            msg += f"PRICE: Rs.{s['price']}\n"
            msg += f"RSI: {s['rsi']} | ADX: {s['adx']} | ROC: {s['roc']:+.1f}%\n"
            msg += f"RS Rating (vs Nifty): {s['rs_rating']}/100\n"
            msg += f"MFI: {s['mfi']} | Volume: {s['vol_ratio']}x Avg\n"
            msg += f"ATR: Rs.{s['atr']}\n\n"
            
            msg += f"POSITION:\n"
            msg += f"- Distance from 52W High: {s['dist_52w']:.1f}%\n"
            msg += f"- Base Quality: {s['base_quality']}\n"
            msg += f"- Type: {s['breakout_type']}\n"
            msg += f"- Signal: {s['inst_signal']}\n\n"
            
            msg += f"TRADE SETUP:\n"
            msg += f"- Entry: Rs.{s['price']}\n"
            msg += f"- Stop Loss: Rs.{s['sl']} (Risk: {s['risk_pct']}%)\n"
            msg += f"- Target 1: Rs.{s['tgt1']}\n"
            msg += f"- Target 2: Rs.{s['tgt2']}\n"
            msg += f"- Target 3: Rs.{s['tgt3']}\n"
            msg += f"- Risk:Reward Ratio: 1:{s['rr_ratio']}\n\n"
            
            msg += f"POSITION SIZE (Rs.1L Capital, 1% Risk):\n"
            msg += f"- Shares: {s['position_size']}\n"
            msg += f"- Value: Rs.{s['position_value']:,.0f}\n\n"
            
            if s['signals']:
                sig_str = ", ".join(s['signals'])
                msg += f"SIGNALS: {sig_str}\n"
            
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[NO ELITE SETUPS TODAY]\n"
        msg += "Market not offering high-probability trades.\n\n"
    
    msg += "[EXECUTION RULES]\n"
    msg += "1. Enter only if breakout holds by 10:30 AM\n"
    msg += "2. Risk max 1-2%% per trade\n"
    msg += "3. Max 5 open positions\n"
    msg += "4. Book 40%% at T1, 30%% at T2, trail rest\n"
    msg += "5. Move SL to entry after T1 hit\n\n"
    
    msg += "[GRADE LEGEND]\n"
    msg += "A+ (85+): Very High Conviction\n"
    msg += "A  (75-84): High Conviction\n"
    msg += "B+ (60-74): Moderate Conviction\n\n"
    
    advice = ""
    if ctx["pct"] >= 75: advice = "ADVICE: All systems green. Deploy full capital."
    elif ctx["pct"] >= 50: advice = "ADVICE: Mixed signals. Trade selective. Half size."
    else: advice = "ADVICE: Markets weak. Sit on hands. Capital first."
    
    msg += f"[ARTHAT SAYS] {advice}\n\n"
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION} | Educational only\n"
    
    send_telegram(msg)
    print("\n[DONE] Message sent!")

if __name__ == "__main__":
    main()
