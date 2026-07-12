# ============================================================
# ⚡ ARTHA v3.2 - GitHub Actions Edition
# Automated Daily NSE/BSE + Global + Crypto Scanner
# ============================================================

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
np.NaN = np.nan
import requests
import warnings
import os
import io
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# 🔑 Load secrets from GitHub environment
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_TAGLINE = "Smart Money. Delivered Daily."
BOT_VERSION = "v3.2"

# ============================================================
# 📥 GET NSE STOCK LIST
# ============================================================
def get_nse_stocks():
    print("📥 Fetching NSE stocks...")
    
    # Try nsepython first
    try:
        from nsepython import nse_eq_symbols
        symbols = nse_eq_symbols()
        tickers = [f"{s}.NS" for s in symbols]
        print(f"   ✅ NSE via nsepython: {len(tickers)} stocks")
        return tickers
    except Exception as e:
        print(f"   ⚠️ nsepython failed: {str(e)[:50]}")
    
    # Try NSE official
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        symbols = df["SYMBOL"].dropna().tolist()
        tickers = [f"{s.strip()}.NS" for s in symbols]
        print(f"   ✅ NSE via official: {len(tickers)} stocks")
        return tickers
    except Exception as e:
        print(f"   ⚠️ NSE official failed: {str(e)[:50]}")
    
    # Backup Nifty 500
    print("   ⚠️ Using Nifty 500 backup")
    return get_backup_list()

def get_backup_list():
    stocks = [
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
        "KAYNES","SYRMA","IPCALAB","AUROPHARMA","TORNTPHARM","ALKEM","GLENMARK",
        "LAURUSLABS","GRANULES","BIOCON","STRIDES","PFIZER","SANOFI","ABBOTINDIA",
        "GILLETTE","3MINDIA","HONAUT","RADICO","TIPSINDLTD","SAREGAMA","CDSL","BSE",
        "CAMS","KFIN","MCX","ANGELONE","ICICIPRULI","HDFCLIFE","GICRE","NIACL","IIFL",
        "SHRIRAMFIN","MANAPPURAM","LICHSGFIN","MFSL","FINEORG","GALAXYSURF","TATACHEM",
        "GNFC","GSFC","CLEAN","NEOGEN","AARTI","VINATI","ASTRAL","SUPREMEIND","NILKAMAL",
        "CERA","SOMANY","CENTURYPLY","GREENPANEL","PRINCEPIPE","KPRMILL","WELSPUNLIV",
        "RAYMOND","PAGEIND","VSTIND","MANYAVAR","CAMPUS","METRO","BATA","RELAXO","DLF",
        "GODREJPROP","PRESTIGE","OBEROIRLTY","PHOENIXLTD","BRIGADE","SOBHA","MAHLIFE",
        "SUNTECK","LODHA","TIINDIA","LATENTVIEW","RAILTEL","GPPL","COCHINSHIP","MAZAGON",
        "GRSE","DATAPATT","INOX","SPORTKING","RALLIS","DHANUKA","BAYER","LICI","IEX",
        "NHPC","SJVN","ADANIPOWER","JINDALSTEL","SAIL","NMDC","MOIL","VEDL","HINDCOPPER",
        "WELCORP","APLAPOLLO","JSL","RATNAMANI","JINDALSAW","MAHSEAMLES","POWERINDIA",
        "KPITTECH","INTELLECT","SONATSOFTW","BIRLASOFT","ZENSAR","MASTEK","RATEGAIN",
        "MAPMYINDIA","TATATECH","CYIENT","NEWGEN","TANLA","INDHOTEL","LEMONTREE","CHALET",
        "EIHOTEL","TAJGVK","ASTERDM","KRBL","PATANJALI","ZYDUSWELL","EMAMILTD","BAJAJCON",
        "JYOTHYLAB","SUPRAJIT","ENDURANCE","JAMNAAUTO","LUMAXIND","MINDACORP","ZFCVINDIA",
        "UNOMINDA","ABFRL","VMART","TRENT","WESTLIFE","DEVYANI","SAPPHIRE","OFSS","NIITLTD",
        "POLYCAB","KEI","FINCABLES","VGUARD","CROMPTON","BAJAJELEC","WHIRLPOOL","BLUESTARCO",
        "AMBER","SYMPHONY","QUESS","TEAMLEASE","EPL","POLYPLEX","GARFIBRES","JBMA","GABRIEL",
        "SUBROS","SUNDRMFAST","EXIDEIND","AMARAJABAT","GREAVESCOT","BHARATFORG","TVSMOTOR",
        "ATULAUTO","ESCORTS","SWARAJENG","ACE","LMW","DALBHARAT","INDIACEM","STARCEMENT",
        "APLLTD","CANFINHOME","REPCOHOME","APTUS","AAVAS","HOMEFIRST","FIVESTAR","POONAWALLA",
        "CREDITACC","UJJIVAN","EQUITASBNK","AUBANK","CSBBANK","DCBBANK","SOUTHBANK",
        "KARURVYSYA","TMB","CUB","KTKBANK"
    ]
    return [f"{s}.NS" for s in stocks]

# ============================================================
# 📤 TELEGRAM
# ============================================================
def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        max_len = 4000
        parts = [message[i:i+max_len] for i in range(0, len(message), max_len)]
        for part in parts:
            payload = {"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown"}
            r = requests.post(url, json=payload, timeout=15)
            time.sleep(1)
        print("✅ Telegram sent")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ============================================================
# 🌍 GLOBAL CONTEXT
# ============================================================
def get_global_context():
    ctx = {"score":0, "total":6, "lines":[], "verdict":"", 
           "crypto_trend":"N/A", "commodity_trend":"N/A", "forex_trend":"N/A"}
    try:
        syms = ["^GSPC","^IXIC","^VIX","DX-Y.NYB","CL=F","^INDIAVIX"]
        data = yf.download(syms, period="30d", progress=False)
        close = data["Close"]

        try:
            v = close["^GSPC"].dropna().iloc[-1]
            a = close["^GSPC"].dropna().rolling(20).mean().iloc[-1]
            if v > a: ctx["score"]+=1; ctx["lines"].append("  S&P 500  : ✅ Above 20DMA")
            else: ctx["lines"].append("  S&P 500  : ❌ Below 20DMA")
        except: pass

        try:
            v = close["^IXIC"].dropna().iloc[-1]
            a = close["^IXIC"].dropna().rolling(20).mean().iloc[-1]
            if v > a: ctx["score"]+=1; ctx["lines"].append("  Nasdaq   : ✅ Above 20DMA")
            else: ctx["lines"].append("  Nasdaq   : ❌ Below 20DMA")
        except: pass

        try:
            vix = close["^VIX"].dropna().iloc[-1]
            if vix < 16: ctx["score"]+=1; ctx["lines"].append(f"  US VIX   : ✅ {vix:.1f}")
            else: ctx["lines"].append(f"  US VIX   : ⚠️ {vix:.1f}")
        except: pass

        try:
            ivix = close["^INDIAVIX"].dropna().iloc[-1]
            if ivix < 14: ctx["score"]+=1; ctx["lines"].append(f"  India VIX: ✅ {ivix:.1f}")
            else: ctx["lines"].append(f"  India VIX: ⚠️ {ivix:.1f}")
        except: pass

        try:
            dxy = close["DX-Y.NYB"].dropna().pct_change().iloc[-1] * 100
            if dxy <= 0.3: ctx["score"]+=1; ctx["lines"].append(f"  DXY      : ✅ {dxy:+.2f}%")
            else: ctx["lines"].append(f"  DXY      : ❌ {dxy:+.2f}%")
        except: pass

        try:
            crude = close["CL=F"].dropna().pct_change().iloc[-1] * 100
            if crude < 2.0: ctx["score"]+=1; ctx["lines"].append(f"  Crude    : ✅ {crude:+.2f}%")
            else: ctx["lines"].append(f"  Crude    : ❌ {crude:+.2f}%")
        except: pass

        try:
            btc = yf.download("BTC-USD", period="7d", progress=False)
            btc_chg = btc["Close"].dropna().pct_change(5).iloc[-1] * 100
            ctx["crypto_trend"] = f"BTC {btc_chg:+.1f}% (7D)"
        except: pass

        try:
            gold = yf.download("GC=F", period="7d", progress=False)
            gold_px = gold["Close"].dropna().iloc[-1]
            gold_chg = gold["Close"].dropna().pct_change(5).iloc[-1] * 100
            ctx["commodity_trend"] = f"Gold ${gold_px:.0f} ({gold_chg:+.1f}%)"
        except: pass

        try:
            inr = yf.download("USDINR=X", period="5d", progress=False)
            inr_val = inr["Close"].dropna().iloc[-1]
            ctx["forex_trend"] = f"USD/INR ₹{inr_val:.2f}"
        except: pass

    except Exception as e:
        ctx["lines"].append(f"  ⚠️ Error: {str(e)[:40]}")
        ctx["score"] = 3

    ctx["verdict"] = ("🟢 STRONG" if ctx["score"] >= 5 else
                      "🟢 BULLISH" if ctx["score"] == 4 else
                      "🟡 NEUTRAL" if ctx["score"] == 3 else
                      "🔴 BEARISH")
    return ctx

def get_fii_proxy():
    try:
        nifty = yf.download("^NSEI", period="10d", progress=False)
        c = nifty["Close"].dropna()
        last3 = c.iloc[-3:].mean()
        prev3 = c.iloc[-6:-3].mean()
        if last3 > prev3 * 1.005:
            return {"status":"BULLISH","emoji":"✅","note":"Nifty trending up"}
        elif last3 > prev3 * 0.997:
            return {"status":"NEUTRAL","emoji":"⚠️","note":"Sideways"}
        else:
            return {"status":"BEARISH","emoji":"❌","note":"Nifty falling"}
    except:
        return {"status":"NEUTRAL","emoji":"⚠️","note":"N/A"}

# ============================================================
# 📊 STOCK SCANNER
# ============================================================
def scan_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=10)
        if df is None or len(df) < 100: return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if df["close"].dropna().iloc[-1] < 10: return None
        if df["volume"].mean() < 10000: return None

        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["ema200"] = ta.ema(df["close"], length=200)
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"])
        df["macd"] = macd.iloc[:,0]
        df["msig"] = macd.iloc[:,2]
        adx = ta.adx(df["high"], df["low"], df["close"])
        df["adx"] = adx.iloc[:,0]
        df["dmp"] = adx.iloc[:,1]
        df["dmn"] = adx.iloc[:,2]
        bb = ta.bbands(df["close"], length=20)
        df["bb_up"] = bb.iloc[:,3]
        df.dropna(inplace=True)
        if len(df) < 5: return None

        c = df.iloc[-1]
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = c["volume"] / avg_vol if avg_vol > 0 else 0
        high_20 = df["high"].rolling(20).max().iloc[-2]

        if not all([
            c["close"] > c["ema21"] > c["ema50"] > c["ema200"],
            c["close"] >= high_20 * 0.98,
            vol_ratio >= 2.0,
            55 <= c["rsi"] <= 78,
            c["macd"] > c["msig"] and c["macd"] > 0,
            c["adx"] > 20 and c["dmp"] > c["dmn"],
            c["close"] >= c["bb_up"] * 0.97,
            c["close"] > c["open"],
            c["close"] <= c["ema21"] * 1.18
        ]): return None

        score = 0
        score += 3 if vol_ratio >= 3.5 else 2 if vol_ratio >= 2.5 else 1
        score += 3 if 60 <= c["rsi"] <= 72 else 1
        score += 3 if c["adx"] >= 30 else 1
        high_52w = df["high"].rolling(252).max().iloc[-1]
        dist_52w = (high_52w - c["close"]) / c["close"]
        score += 3 if dist_52w <= 0.03 else 2 if dist_52w <= 0.08 else 1

        sl = round(min(c["ema21"], df["low"].rolling(5).min().iloc[-1]) * 0.99, 2)
        return {
            "ticker": ticker.replace(".NS","").replace(".BO",""),
            "exchange": "NSE" if ".NS" in ticker else "BSE",
            "score": score,
            "price": round(c["close"],2),
            "rsi": round(c["rsi"],1),
            "adx": round(c["adx"],1),
            "vol_ratio": round(vol_ratio,1),
            "sl": sl,
            "tgt1": round(c["close"]*1.08,2),
            "tgt2": round(c["close"]*1.15,2),
            "risk_pct": round(((c["close"]-sl)/c["close"])*100,1),
            "dist_52w": round(dist_52w*100,1)
        }
    except:
        return None

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print(f"\n{'='*55}")
    print(f"⚡ ARTHA v3.2 GitHub Actions Edition")
    print(f"⏰ {datetime.now().strftime('%d %b %Y %I:%M %p')}")
    print(f"{'='*55}\n")

    print("🌍 Global markets...")
    global_ctx = get_global_context()
    print(f"   Score: {global_ctx['score']}/6")

    print("\n🏦 Smart money...")
    fii_ctx = get_fii_proxy()
    print(f"   {fii_ctx['status']}")

    stocks = get_nse_stocks()
    print(f"\n🇮🇳 Scanning {len(stocks)} stocks...\n")

    results = []
    for i, ticker in enumerate(stocks):
        if (i+1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(stocks)} | Found: {len(results)}")
        res = scan_stock(ticker)
        if res:
            results.append(res)

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    print(f"\n✅ Complete: {len(results)} breakouts found")

    today = datetime.now().strftime("%A, %d %b %Y")
    msg = f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*{BOT_NAME} | Pre-Market Report*\n"
    msg += f"📅 {today}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    msg += f"*🌍 GLOBAL PULSE*\n"
    for line in global_ctx["lines"]:
        msg += f"{line}\n"
    msg += f"  Verdict: {global_ctx['verdict']}\n\n"

    msg += f"*📡 MACRO*\n"
    msg += f"  Crypto: {global_ctx['crypto_trend']}\n"
    msg += f"  Gold  : {global_ctx['commodity_trend']}\n"
    msg += f"  Rupee : {global_ctx['forex_trend']}\n\n"

    msg += f"*🏦 FII*: {fii_ctx['emoji']} {fii_ctx['status']}\n"
    msg += f"  {fii_ctx['note']}\n\n"

    msg += f"*📊 Scanned*: {len(stocks)} stocks | Found: {len(results)}\n\n"

    if top5:
        msg += f"*🏆 TOP 5 BREAKOUTS*\n\n"
        medals = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣"]
        for i, s in enumerate(top5):
            msg += f"*{medals[i]} {s['ticker']}* [{s['exchange']}] Score:{s['score']}\n"
            msg += f"  💰 ₹{s['price']} | RSI:{s['rsi']} ADX:{s['adx']} Vol:{s['vol_ratio']}x\n"
            msg += f"  📍 {s['dist_52w']}% from 52W High\n"
            msg += f"  🛑 SL:₹{s['sl']} (Risk {s['risk_pct']}%)\n"
            msg += f"  🎯 T1:₹{s['tgt1']} T2:₹{s['tgt2']}\n\n"
    else:
        msg += f"*📉 No breakouts today*\n\n"

    msg += f"*📐 RISK*: 1-2% per trade | Max 5 positions\n\n"
    msg += f"_{BOT_NAME} {BOT_VERSION} | {BOT_TAGLINE}_\n"
    msg += f"_Educational only. Not SEBI advice._"

    send_telegram(msg)
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
