"""
ARTHA v14.0 - Adaptive Multi-Strategy Engine
Auto-Switches Between:
 1. Bull Mode     --> Breakouts & VCP Patterns
 2. Sideways Mode --> Dip-Buying & Mean Reversion (Quality Bounces)
 3. Bear Mode     --> Defensive Assets & Gold ETFs
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

# ============================================================
# 🔑 CONFIGURATION & ENVIRONMENT
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID   = os.environ.get("CHAT_ID", "")

BOT_NAME    = "⚡ ARTHA"
BOT_VERSION = "v14.0"
BOT_TAGLINE = "Adaptive Multi-Strategy AI"

MEMORY_FILE = "artha_memory.json"

TOTAL_CAPITAL       = 100000  # Rs. 1,00,000 Base Capital
MAX_PORTFOLIO_HEAT  = 5       # Max 5% total risk at any time
MAX_POSITIONS_ACTIVE = 3       # Quality over quantity (Max 3 open positions)

# ============================================================
# 🧠 MEMORY MANAGEMENT SYSTEM
# ============================================================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                mem = json.load(f)
                for key in ['pending_evaluations', 'completed_trades', 'failed_stocks', 'mode_stats']:
                    if key not in mem:
                        mem[key] = {} if 'evaluations' in key or 'failed' in key or 'stats' in key else []
                return mem
        except:
            pass
    
    return {
        "version": "14.0",
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_picks": 0,
        "total_wins": 0,
        "total_losses": 0,
        "pending_evaluations": {},
        "completed_trades": [],
        "failed_stocks": {},
        "mode_stats": {
            "BREAKOUT": {"wins": 0, "losses": 0},
            "DIP_BUYING": {"wins": 0, "losses": 0},
            "DEFENSIVE": {"wins": 0, "losses": 0}
        }
    }

def save_memory(memory):
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

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

# ============================================================
# 🎯 1. ADAPTIVE MARKET REGIME ENGINE
# ============================================================
def detect_market_regime():
    """Dynamically determines the market state and picks the active strategy"""
    try:
        nifty = yf.download("^NSEI", period="6mo", progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = [c[0].lower() for c in nifty.columns]
        else:
            nifty.columns = [c.lower() for c in nifty.columns]
            
        close = nifty['close']
        
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
    except Exception as e:
        return {
            "regime": "SIDEWAYS_BULL",
            "strategy": "DIP_BUYING",
            "desc": "Fallback Mode (Dip Buying Active)",
            "nifty_price": 0, "change_20d": 0, "rsi": 50, "volatility": 1.5
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
def get_scan_universe():
    """Fetches NSE 200 Liquid Stocks for Scanning"""
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
    return [f"{s}.NS" for s in set(liquid_stocks)]

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

    # STEP 2: SCAN MARKET ACCORDING TO ACTIVE STRATEGY
    results = []

    if strategy == "DEFENSIVE":
        print("\n[DEFENSIVE MODE]: Scanning Gold ETFs & Protection Assets...")
        results = scan_defensive()
    else:
        universe = get_scan_universe()
        print(f"\n[SCANNING]: Running {strategy} filters on {len(universe)} stocks...")
        
        count = 0
        for ticker in universe:
            count += 1
            if count % 50 == 0:
                print(f"  Scanned {count}/{len(universe)} | Found: {len(results)}")
                
            try:
                df = yf.download(ticker, period="6mo", progress=False, timeout=8)
                if df is None or len(df) < 60: continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                ticker_clean = ticker.replace(".NS", "")

                # Run Strategy Based on Regime
                if strategy == "BREAKOUT":
                    res = scan_breakout(df, ticker_clean)
                elif strategy == "DIP_BUYING":
                    res = scan_dip_buying(df, ticker_clean)
                else:
                    res = None

                if res:
                    results.append(res)
            except:
                continue

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

    msg += "[📊 SCAN SUMMARY]\n"
    msg += f"Active Strategy : {strategy}\n"
    msg += f"Setups Qualified: {len(results)}\n"
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
        msg += "No stocks met strict quality bounds for current mode.\n"
        msg += "Capital Protection Active.\n\n"

    msg += "[📐 DISCIPLINE RULES]\n"
    msg += "1. Enter ONLY if setup holds after 9:45 AM.\n"
    msg += "2. Strictly respect the StopLoss.\n"
    msg += "3. Book 50% profits at Target 1, trail rest.\n\n"

    msg += "========================================\n"
    msg += f"{BOT_NAME} {BOT_VERSION} | {BOT_TAGLINE}\n"

    send_telegram(msg)
    print("\n✅ ARTHA v14.0 Run Complete!")

if __name__ == "__main__":
    main()
