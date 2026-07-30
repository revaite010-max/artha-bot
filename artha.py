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

# 🔑 Settings
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v7.0"
BOT_TAGLINE = "Multi-Asset Intelligence Pro"

# ============================================================
# 🌍 ASSET UNIVERSE
# ============================================================
COMMODITIES = {
    "Gold (COMEX)"      : "GC=F",
    "Silver (COMEX)"    : "SI=F",
    "Crude Oil WTI"     : "CL=F",
    "Brent Crude"       : "BZ=F",
    "Natural Gas"       : "NG=F",
    "Copper"            : "HG=F",
    "Platinum"          : "PL=F",
    "Palladium"         : "PA=F"
}

INDIAN_ETFS = {
    "Gold BEES"         : "GOLDBEES.NS",
    "Silver BEES"       : "SILVERBEES.NS",
    "Gold Shares"       : "GOLDSHARE.NS",
    "Nippon Silver ETF" : "SILVER.NS",
    "HDFC Gold ETF"     : "HDFCGOLD.NS",
    "SBI Gold ETF"      : "SETFGOLD.NS"
}

CRYPTO = {
    "Bitcoin"      : "BTC-USD",
    "Ethereum"     : "ETH-USD",
    "BNB"          : "BNB-USD",
    "Solana"       : "SOL-USD",
    "XRP"          : "XRP-USD",
    "Cardano"      : "ADA-USD",
    "Avalanche"    : "AVAX-USD",
    "Polkadot"     : "DOT-USD",
    "Chainlink"    : "LINK-USD",
    "Polygon"      : "MATIC-USD"
}

US_TOP_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "ORCL", "CRM", "ADBE", "INTC", "QCOM", "AVGO", "TSM",
    "JPM", "V", "MA", "PYPL"
]

# ============================================================
# 🧮 MATH LIBRARY (Same as v6.0)
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

def get_bb(s, n=20, std=2):
    sma = s.rolling(n).mean()
    stdev = s.rolling(n).std()
    return sma + (std * stdev), sma, sma - (std * stdev)

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
    return dx.rolling(n).mean(), plus_di, minus_di

def get_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def get_roc(close, n=12):
    return ((close - close.shift(n)) / close.shift(n)) * 100

def calculate_rs_rating(stock_close, benchmark_close):
    try:
        periods = [63, 126, 189, 252]
        stock_perf, bench_perf = [], []
        for p in periods:
            if len(stock_close) > p and len(benchmark_close) > p:
                stock_perf.append(stock_close.iloc[-1] / stock_close.iloc[-p])
                bench_perf.append(benchmark_close.iloc[-1] / benchmark_close.iloc[-p])
        if not stock_perf: return 50
        weights = [0.40, 0.20, 0.20, 0.20][:len(stock_perf)]
        stock_score = sum(s*w for s,w in zip(stock_perf, weights))
        bench_score = sum(n*w for n,w in zip(bench_perf, weights))
        rs = (stock_score / bench_score) * 100
        return min(max(rs, 0), 100)
    except: return 50

# ============================================================
# 📥 GET NSE STOCKS
# ============================================================
def get_nse_tickers():
    print("Fetching NSE stocks...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        symbols = df["SYMBOL"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"  NSE: {len(tickers)} stocks")
        return tickers
    except: pass
    
    try:
        url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/nse_stocks.csv"
        df = pd.read_csv(url)
        symbols = df["Symbol"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"  NSE: {len(tickers)} stocks (via mirror)")
        return tickers
    except:
        return ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"]

# ============================================================
# 📥 GET BSE STOCKS (NEW)
# ============================================================
def get_bse_tickers():
    print("Fetching BSE stocks...")
    try:
        # Try BSE 500 list
        url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/bse_stocks.csv"
        df = pd.read_csv(url)
        symbols = df["Symbol"].dropna().unique().tolist()
        tickers = [f"{s.strip()}.BO" for s in symbols[:500]]
        print(f"  BSE: {len(tickers)} stocks")
        return tickers
    except:
        pass
    
    # Backup BSE exclusive list
    bse_exclusive = [
        "BAJAJCON","VSTIND","3MINDIA","GILLETTE","PFIZER","SANOFI","ABBOTINDIA",
        "HONAUT","TIPSINDLTD","SAREGAMA","NAVNETEDUL","EIDPARRY","JYOTHYLAB",
        "SYMPHONY","TTKPRESTIG","WHIRLPOOL","VOLTAS","BLUESTARCO","HAWKINCOOK",
        "AKZOINDIA","BALPHARMA","LGBBROSLTD","MAHINDCIE","MENONBE","NELCO",
        "OMAXAUTO","PANAMAPET","QUINTEGRA","RANEENGINE","SAKUMA","TAINWALCHM",
        "UDAIPURCEM","VBINDUSTRI","WABAG","XLENERGY","YASHOPTICS","ZOTA",
        "ADORWELD","BAJAJHIND","CENTUM","DANLAW","ELECTHERM","JAINSTUDIO",
        "KABRAEXTRU","LGBFORGE","MANGCHEFER","NATCOPHARM","OMKARCHEM","PRICOL",
        "RANEHOLDNG","SBEC","TIRUMALCHM","UNITEDDR","VIVIDHA","WELSPUNIND",
        "XCHANGING","YUKEN","ZENITHEXPO","AARTIIND","BLKASHYAP","CENTUM"
    ]
    tickers = [f"{s}.BO" for s in bse_exclusive]
    print(f"  BSE: {len(tickers)} stocks (backup list)")
    return tickers

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
            v = close["^GSPC"].dropna().iloc[-1]
            a20 = close["^GSPC"].dropna().rolling(20).mean().iloc[-1]
            if v > a20: score += 1; lines.append("  S&P500 : BULLISH")
            else: lines.append("  S&P500 : WEAK")
        except: pass
        
        try:
            v = close["^IXIC"].dropna().iloc[-1]
            a = close["^IXIC"].dropna().rolling(20).mean().iloc[-1]
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
            else: lines.append(f"  Crude   : ${crude_val:.0f} SPIKE")
        except: pass
        
        try:
            gold_val = close["GC=F"].dropna().iloc[-1]
            gold_chg = close["GC=F"].dropna().pct_change(5).iloc[-1] * 100
            lines.append(f"  Gold    : ${gold_val:.0f} ({gold_chg:+.1f}% 7D)")
            if gold_chg < 2: score += 1
        except: pass
        
        try:
            nifty = close["^NSEI"].dropna()
            nc = nifty.iloc[-1]
            n20 = nifty.rolling(20).mean().iloc[-1]
            n50 = nifty.rolling(50).mean().iloc[-1]
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
# 🎯 GENERIC ASSET SCANNER (Works for ALL asset types)
# ============================================================
def scan_asset(ticker, asset_type="stock", benchmark_close=None):
    try:
        # Different periods for different assets
        period = "6mo" if asset_type in ["crypto", "commodity"] else "1y"
        min_length = 50 if asset_type in ["crypto", "commodity"] else 100
        
        df = yf.download(ticker, period=period, progress=False, timeout=10)
        if df is None or len(df) < min_length: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series([0]*len(df))
        open_ = df['Open'].squeeze()
        
        # Skip very cheap stocks (except crypto)
        if asset_type == "stock" and close.iloc[-1] < 30: return None
        if asset_type == "stock" and vol.mean() < 50000: return None
        
        # Calculate indicators
        ema9 = get_ema(close, 9)
        ema21 = get_ema(close, 21)
        ema50 = get_ema(close, 50)
        rsi = get_rsi(close, 14)
        macd, msig, mhist = get_macd(close)
        bb_up, bb_mid, bb_low = get_bb(close, 20)
        atr = get_atr(high, low, close, 14)
        adx, plus_di, minus_di = get_adx(high, low, close)
        roc = get_roc(close, 12)
        
        c = len(df) - 1
        curr = close.iloc[c]
        avg_vol = vol.rolling(20).mean().iloc[c] if vol.mean() > 0 else 1
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 1
        
        # Different filters for different assets
        if asset_type == "stock":
            ema200 = get_ema(close, 200)
            if not (curr > ema21.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
            if not (vol_ratio >= 1.5): return None
            if not (50 <= rsi.iloc[c] <= 80): return None
            if not (adx.iloc[c] > 20): return None
        else:
            # Relaxed filters for crypto/commodity
            if not (curr > ema21.iloc[c] > ema50.iloc[c]): return None
            if not (50 <= rsi.iloc[c] <= 80): return None
            if not (macd.iloc[c] > msig.iloc[c]): return None
        
        # Scoring (100 points)
        score = 0
        signals = []
        
        # Trend
        if asset_type == "stock" and curr > ema9.iloc[c] > ema21.iloc[c] > ema50.iloc[c]:
            score += 20; signals.append("Perfect Stack")
        elif curr > ema21.iloc[c] > ema50.iloc[c]:
            score += 15
        else:
            score += 10
        
        # Volume (if applicable)
        if vol.mean() > 0:
            if vol_ratio >= 3: score += 15; signals.append("High Volume")
            elif vol_ratio >= 2: score += 10
            elif vol_ratio >= 1.5: score += 5
        
        # Momentum
        if 60 <= rsi.iloc[c] <= 70: score += 15; signals.append("Ideal RSI")
        elif 55 <= rsi.iloc[c] <= 75: score += 10
        else: score += 5
        
        if roc.iloc[c] > 15: score += 10; signals.append("Strong Momentum")
        elif roc.iloc[c] > 5: score += 5
        
        # ADX
        if adx.iloc[c] >= 30: score += 10; signals.append("Strong Trend")
        elif adx.iloc[c] >= 20: score += 5
        
        # RS Rating
        if benchmark_close is not None:
            rs_rating = calculate_rs_rating(close, benchmark_close)
            if rs_rating >= 90: score += 15; signals.append("Top 10%")
            elif rs_rating >= 75: score += 10
            elif rs_rating >= 60: score += 5
        else:
            rs_rating = 50
        
        # 52W proximity (stocks/commodities only)
        if asset_type != "crypto":
            try:
                high_52w = high.rolling(min(252, len(df))).max().iloc[c]
                dist_52w = (high_52w - curr) / curr * 100
                if dist_52w <= 3: score += 10; signals.append("Near High")
                elif dist_52w <= 8: score += 5
            except:
                dist_52w = 0
        else:
            dist_52w = 0
        
        # Bollinger breakout
        if curr >= bb_up.iloc[c] * 0.98:
            score += 5; signals.append("BB Breakout")
        
        # SL & Targets
        atr_stop = round(curr - (1.5 * atr.iloc[c]), 4)
        ema_stop = round(ema21.iloc[c] * 0.97, 4)
        sl = max(atr_stop, ema_stop)
        
        # Different target multipliers
        tgt_mult = {
            "stock": (2, 4, 6),
            "commodity": (1.5, 3, 4.5),
            "crypto": (3, 6, 10)
        }
        m1, m2, m3 = tgt_mult.get(asset_type, (2, 4, 6))
        
        tgt1 = round(curr + (m1 * atr.iloc[c]), 4)
        tgt2 = round(curr + (m2 * atr.iloc[c]), 4)
        tgt3 = round(curr + (m3 * atr.iloc[c]), 4)
        
        risk = curr - sl
        reward = tgt1 - curr
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        risk_pct = round((risk / curr) * 100, 2)
        
        return {
            "ticker": ticker.replace(".NS","").replace(".BO","").replace("-USD","").replace("=F",""),
            "type": asset_type,
            "score": round(score, 1),
            "price": round(curr, 4 if asset_type == "crypto" else 2),
            "rsi": round(rsi.iloc[c], 1),
            "adx": round(adx.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1) if vol.mean() > 0 else 0,
            "rs_rating": round(rs_rating, 0),
            "roc": round(roc.iloc[c], 1),
            "atr": round(atr.iloc[c], 4 if asset_type == "crypto" else 2),
            "sl": sl, "tgt1": tgt1, "tgt2": tgt2, "tgt3": tgt3,
            "rr_ratio": rr_ratio, "risk_pct": risk_pct,
            "dist_52w": round(dist_52w, 1),
            "signals": signals[:4]
        }
    except Exception as e:
        return None

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
            ipo_msg += "\nUpcoming/Open IPOs:\n"
            for row in tables[0].find_all('tr')[1:5]:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    name = cols[0].text.strip()[:28]
                    date = cols[1].text.strip() if len(cols) > 1 else "N/A"
                    ipo_msg += f"- {name} | {date}\n"
    except:
        ipo_msg += "- Data unavailable\n"
    return ipo_msg + "\n"

# ============================================================
# 📤 TELEGRAM
# ============================================================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        max_len = 3800
        parts = [msg[i:i+max_len] for i in range(0, len(msg), max_len)]
        for i, part in enumerate(parts, 1):
            payload = {"chat_id": CHAT_ID, "text": part}
            r = requests.post(url, json=payload, timeout=25)
            print(f"[TELEGRAM] Part {i}/{len(parts)} - Status: {r.status_code}")
            time.sleep(0.8)
        print("[TELEGRAM] Sent")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"ARTHA {BOT_VERSION} - {BOT_TAGLINE}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("=" * 60)
    
    send_telegram(f"[STARTING] ARTHA {BOT_VERSION} multi-asset scan...")
    
    # ── Global Context ──
    print("\n[1/7] Global Context...")
    ctx = get_global_context()
    print(f"  {ctx['verdict']}")
    
    # ── Fetch Benchmarks ──
    print("\n[2/7] Benchmarks...")
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except: nifty_close = None
    
    try:
        sp500 = yf.download("^GSPC", period="1y", progress=False)
        sp500_close = sp500['Close'].squeeze()
    except: sp500_close = None
    
    try:
        btc = yf.download("BTC-USD", period="6mo", progress=False)
        btc_close = btc['Close'].squeeze()
    except: btc_close = None
    
    # ── Scan NSE ──
    print("\n[3/7] Scanning NSE stocks...")
    nse_tickers = get_nse_tickers()
    nse_results = []
    for i, t in enumerate(nse_tickers):
        if (i+1) % 300 == 0: print(f"  NSE: {i+1}/{len(nse_tickers)} | Found: {len(nse_results)}")
        r = scan_asset(t, "stock", nifty_close)
        if r and r["score"] >= 55: nse_results.append(r)
    nse_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  NSE: {len(nse_results)} elite setups")
    
    # ── Scan BSE ──
    print("\n[4/7] Scanning BSE stocks...")
    bse_tickers = get_bse_tickers()
    bse_results = []
    for i, t in enumerate(bse_tickers):
        if (i+1) % 100 == 0: print(f"  BSE: {i+1}/{len(bse_tickers)} | Found: {len(bse_results)}")
        r = scan_asset(t, "stock", nifty_close)
        if r and r["score"] >= 50: bse_results.append(r)
    bse_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  BSE: {len(bse_results)} elite setups")
    
    # ── Scan Commodities ──
    print("\n[5/7] Scanning Commodities...")
    commodity_results = []
    for name, ticker in COMMODITIES.items():
        r = scan_asset(ticker, "commodity")
        if r and r["score"] >= 45:
            r["display_name"] = name
            commodity_results.append(r)
    for name, ticker in INDIAN_ETFS.items():
        r = scan_asset(ticker, "stock", nifty_close)
        if r and r["score"] >= 45:
            r["display_name"] = name
            commodity_results.append(r)
    commodity_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Commodities: {len(commodity_results)} setups")
    
    # ── Scan Crypto ──
    print("\n[6/7] Scanning Crypto...")
    crypto_results = []
    for name, ticker in CRYPTO.items():
        r = scan_asset(ticker, "crypto", btc_close)
        if r and r["score"] >= 45:
            r["display_name"] = name
            crypto_results.append(r)
    crypto_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Crypto: {len(crypto_results)} setups")
    
    # ── Scan US Stocks ──
    print("\n[7/7] Scanning US Stocks...")
    us_results = []
    for ticker in US_TOP_STOCKS:
        r = scan_asset(ticker, "stock", sp500_close)
        if r and r["score"] >= 55: us_results.append(r)
    us_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  US: {len(us_results)} setups")
    
    # ── IPO Data ──
    ipo_data = get_ipo_data()
    
    # Get top picks
    top_nse = nse_results[:5]
    top_bse = bse_results[:3]
    top_commodity = commodity_results[:2]
    top_crypto = crypto_results[:2]
    top_us = us_results[:2]
    
    total_scanned = len(nse_tickers) + len(bse_tickers) + len(COMMODITIES) + len(INDIAN_ETFS) + len(CRYPTO) + len(US_TOP_STOCKS)
    total_found = len(nse_results) + len(bse_results) + len(commodity_results) + len(crypto_results) + len(us_results)
    
    # ── BUILD MESSAGE ──
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = ""
    msg += "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"{BOT_TAGLINE}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    # Global
    msg += "[🌍 GLOBAL MARKETS]\n"
    for line in ctx["lines"]: msg += line + "\n"
    msg += f"Verdict: {ctx['verdict']}\n\n"
    
    # Coverage stats
    msg += "[📊 MULTI-ASSET COVERAGE]\n"
    msg += f"Total Scanned: {total_scanned:,} assets\n"
    msg += f"NSE: {len(nse_tickers)} | BSE: {len(bse_tickers)}\n"
    msg += f"Commodities: {len(COMMODITIES)+len(INDIAN_ETFS)} | Crypto: {len(CRYPTO)} | US: {len(US_TOP_STOCKS)}\n"
    msg += f"Elite Setups Found: {total_found}\n\n"
    
    msg += ipo_data
    
    # NSE Picks
    if top_nse:
        msg += "=" * 40 + "\n"
        msg += "[🇮🇳 TOP 5 NSE BREAKOUTS]\n"
        msg += "=" * 40 + "\n\n"
        for i, s in enumerate(top_nse):
            grade = "A+" if s['score'] >= 90 else "A" if s['score'] >= 80 else "B+"
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/100)\n"
            msg += f"  Price: Rs.{s['price']} | RSI: {s['rsi']} | Vol: {s['vol_ratio']}x\n"
            msg += f"  RS: {s['rs_rating']}/100 | ROC: {s['roc']:+.1f}%\n"
            msg += f"  SL: Rs.{s['sl']} | T1: Rs.{s['tgt1']} | T2: Rs.{s['tgt2']}\n"
            msg += f"  R:R = 1:{s['rr_ratio']} | Risk: {s['risk_pct']}%\n"
            if s['signals']: msg += f"  Signals: {', '.join(s['signals'])}\n"
            msg += "\n"
    
    # BSE Picks
    if top_bse:
        msg += "=" * 40 + "\n"
        msg += "[🏢 TOP 3 BSE BREAKOUTS]\n"
        msg += "=" * 40 + "\n\n"
        for i, s in enumerate(top_bse):
            grade = "A+" if s['score'] >= 85 else "A" if s['score'] >= 75 else "B+"
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/100)\n"
            msg += f"  Price: Rs.{s['price']} | RSI: {s['rsi']}\n"
            msg += f"  SL: Rs.{s['sl']} | T1: Rs.{s['tgt1']}\n"
            msg += f"  R:R = 1:{s['rr_ratio']}\n\n"
    
    # Commodities
    if top_commodity:
        msg += "=" * 40 + "\n"
        msg += "[🥇 TOP COMMODITY BREAKOUTS]\n"
        msg += "=" * 40 + "\n\n"
        for i, s in enumerate(top_commodity):
            name = s.get('display_name', s['ticker'])
            msg += f"#{i+1} {name}\n"
            msg += f"  Score: {s['score']}/100\n"
            msg += f"  Price: ${s['price']} | RSI: {s['rsi']}\n"
            msg += f"  SL: ${s['sl']} | Target: ${s['tgt1']}\n"
            msg += f"  Trade via: MCX / ETFs (GOLDBEES/SILVERBEES)\n\n"
    
    # Crypto
    if top_crypto:
        msg += "=" * 40 + "\n"
        msg += "[🪙 TOP CRYPTO BREAKOUTS]\n"
        msg += "=" * 40 + "\n\n"
        for i, s in enumerate(top_crypto):
            name = s.get('display_name', s['ticker'])
            msg += f"#{i+1} {name}\n"
            msg += f"  Score: {s['score']}/100\n"
            msg += f"  Price: ${s['price']} | RSI: {s['rsi']}\n"
            msg += f"  SL: ${s['sl']} | T1: ${s['tgt1']} | T2: ${s['tgt2']}\n"
            msg += f"  Trade via: CoinDCX, WazirX, Binance\n\n"
    
    # US Stocks
    if top_us:
        msg += "=" * 40 + "\n"
        msg += "[🇺🇸 TOP US STOCK BREAKOUTS]\n"
        msg += "=" * 40 + "\n\n"
        for i, s in enumerate(top_us):
            msg += f"#{i+1} {s['ticker']}\n"
            msg += f"  Score: {s['score']}/100\n"
            msg += f"  Price: ${s['price']} | RSI: {s['rsi']}\n"
            msg += f"  RS vs S&P: {s['rs_rating']}/100\n"
            msg += f"  SL: ${s['sl']} | T1: ${s['tgt1']}\n"
            msg += f"  Trade via: Vested, INDmoney, Groww US\n\n"
    
    # Rules
    msg += "=" * 40 + "\n"
    msg += "[EXECUTION RULES]\n"
    msg += "1. Enter only if breakout holds after market open\n"
    msg += "2. Risk max 1-2% per trade\n"
    msg += "3. Max 5 open positions across assets\n"
    msg += "4. Book 40% at T1, trail rest\n"
    msg += "5. NSE/BSE: Delivery-based swing trades\n"
    msg += "6. Commodities: MCX or Gold/Silver ETFs\n"
    msg += "7. Crypto: Only 5-10% of portfolio\n"
    msg += "8. US Stocks: Long-term via Vested\n\n"
    
    msg += "[GRADE LEGEND]\n"
    msg += "A+ (85+): Very High Conviction\n"
    msg += "A  (75-84): High Conviction\n"
    msg += "B+ (55-74): Moderate\n\n"
    
    # Advice
    if ctx["pct"] >= 75:
        advice = "All systems green. Deploy across assets."
    elif ctx["pct"] >= 50:
        advice = "Mixed signals. Focus on strongest picks."
    else:
        advice = "Markets weak. Consider commodities as hedge."
    
    msg += f"[ARTHA SAYS] {advice}\n\n"
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION} | Educational only\n"
    
    send_telegram(msg)
    print("\n[DONE] Multi-asset report sent!")

if __name__ == "__main__":
    main()
