import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

# ML Imports (with fallback)
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: sklearn not available. ML features disabled.")

# ============================================================
# 🔑 CONFIGURATION
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

# Broker Credentials (Optional)
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

BOT_NAME = "⚡ ARTHA"
BOT_VERSION = "v10.0"
BOT_TAGLINE = "Institutional-Grade AI System"

MEMORY_FILE = "artha_memory.json"

# Portfolio Configuration
TOTAL_CAPITAL = 100000  # Rs.1L default (user can change)
MAX_PORTFOLIO_HEAT = 6  # Max 6% total portfolio risk at once
MAX_SECTOR_EXPOSURE = 2  # Max 2 stocks per sector
MAX_CORRELATED_POSITIONS = 2

# ============================================================
# 🧠 MEMORY SYSTEM (Enhanced)
# ============================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                mem = json.load(f)
                # Ensure new fields exist
                if 'ml_features' not in mem:
                    mem['ml_features'] = []
                if 'sentiment_history' not in mem:
                    mem['sentiment_history'] = {}
                if 'user_feedback' not in mem:
                    mem['user_feedback'] = {}
                if 'active_positions' not in mem:
                    mem['active_positions'] = {}
                return mem
        except:
            pass
    
    return {
        "version": "10.0",
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_picks": 0,
        "total_wins": 0,
        "total_losses": 0,
        "pending_evaluations": {},
        "completed_trades": [],
        "sector_performance": {},
        "pattern_performance": {},
        "score_range_performance": {},
        "rsi_range_performance": {},
        "learning_insights": [],
        "ml_features": [],
        "sentiment_history": {},
        "user_feedback": {},
        "active_positions": {},
        "weights": {
            "trend": 1.0, "volume": 1.0, "rsi": 1.0,
            "sector_bonus": 1.0, "early_breakout": 1.0,
            "pullback": 1.0, "sentiment": 1.0, "institutional": 1.0
        },
        "ml_model_trained": False,
        "last_ml_train": None
    }

def save_memory(memory):
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

# ============================================================
# 📰 MODULE 1: NEWS SENTIMENT ANALYSIS
# ============================================================

# Simple VADER-like sentiment word lists
POSITIVE_WORDS = [
    'growth', 'profit', 'surge', 'jump', 'rally', 'gain', 'rise', 'up', 'high',
    'strong', 'bullish', 'positive', 'beat', 'exceed', 'record', 'upgrade',
    'buy', 'outperform', 'award', 'launch', 'expansion', 'acquire', 'merger',
    'dividend', 'bonus', 'success', 'boost', 'soar', 'climb', 'advance',
    'breakthrough', 'winning', 'excellent', 'strong', 'robust', 'solid',
    'expansion', 'orders', 'contract', 'deal', 'partnership', 'approval'
]

NEGATIVE_WORDS = [
    'loss', 'fall', 'drop', 'decline', 'crash', 'plunge', 'weak', 'bearish',
    'negative', 'miss', 'downgrade', 'sell', 'underperform', 'scandal',
    'fraud', 'probe', 'investigation', 'penalty', 'fine', 'lawsuit', 'debt',
    'bankruptcy', 'default', 'warning', 'concern', 'risk', 'slump', 'tumble',
    'crisis', 'trouble', 'issue', 'problem', 'delay', 'cancel', 'reject',
    'downgrade', 'exit', 'resign', 'layoff', 'closure', 'suspension'
]

def get_google_news_sentiment(ticker_name):
    """Fetch and analyze news sentiment from Google News"""
    try:
        # Google News RSS feed
        query = quote(f"{ticker_name} stock NSE India")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(r.content, 'xml')
        items = soup.find_all('item')[:10]  # Latest 10 news
        
        if not items:
            return {"sentiment": "NEUTRAL", "score": 0, "count": 0, "confidence": 0}
        
        total_score = 0
        pos_count = 0
        neg_count = 0
        
        for item in items:
            title = item.find('title').text.lower() if item.find('title') else ""
            
            # Skip if not related
            if ticker_name.lower() not in title:
                continue
            
            pos_hits = sum(1 for word in POSITIVE_WORDS if word in title)
            neg_hits = sum(1 for word in NEGATIVE_WORDS if word in title)
            
            score = pos_hits - neg_hits
            total_score += score
            
            if score > 0:
                pos_count += 1
            elif score < 0:
                neg_count += 1
        
        total_analyzed = pos_count + neg_count
        
        if total_analyzed == 0:
            return {"sentiment": "NEUTRAL", "score": 0, "count": 0, "confidence": 0}
        
        avg_score = total_score / len(items)
        
        # Determine sentiment
        if avg_score >= 1.5:
            sentiment = "VERY_BULLISH"
        elif avg_score >= 0.5:
            sentiment = "BULLISH"
        elif avg_score <= -1.5:
            sentiment = "VERY_BEARISH"
        elif avg_score <= -0.5:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        confidence = min(100, (total_analyzed / 10) * 100)
        
        return {
            "sentiment": sentiment,
            "score": round(avg_score, 2),
            "count": total_analyzed,
            "positive": pos_count,
            "negative": neg_count,
            "confidence": round(confidence, 0)
        }
    except Exception as e:
        return {"sentiment": "NEUTRAL", "score": 0, "count": 0, "confidence": 0}

# ============================================================
# 🐋 MODULE 1B: INSTITUTIONAL ACTIVITY MONITOR
# ============================================================

def check_bulk_deals(ticker_name):
    """Check NSE bulk deals for institutional activity"""
    try:
        # NSE Bulk deals page
        url = "https://www.nseindia.com/api/historical/bulk-deals"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        }
        
        # Try to fetch (NSE may block, so we handle gracefully)
        session = requests.Session()
        session.headers.update(headers)
        
        # Try alternative source - moneycontrol
        mc_url = f"https://www.moneycontrol.com/stocks/marketstats/bulk-deals.php"
        r = session.get(mc_url, timeout=10)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            # Simple check if stock appears in recent bulk deals
            text = r.text.lower()
            if ticker_name.lower() in text:
                # Try to determine buy vs sell
                idx = text.find(ticker_name.lower())
                nearby = text[max(0, idx-500):idx+500]
                
                buy_count = nearby.count('buy')
                sell_count = nearby.count('sell')
                
                if buy_count > sell_count:
                    return {"activity": "INSTITUTIONAL_BUYING", "score": 10}
                elif sell_count > buy_count:
                    return {"activity": "INSTITUTIONAL_SELLING", "score": -15}
        
        return {"activity": "NORMAL", "score": 0}
    except:
        return {"activity": "UNKNOWN", "score": 0}

def check_fii_dii_flow():
    """Check overall FII/DII flow"""
    try:
        # Fallback: Use Nifty trend as proxy
        nifty = yf.download("^NSEI", period="5d", progress=False)
        change_3d = ((nifty['Close'].iloc[-1] - nifty['Close'].iloc[-4]) / nifty['Close'].iloc[-4]) * 100
        
        if change_3d > 1:
            return {"flow": "FII_BUYING", "trend": "POSITIVE", "score": 10}
        elif change_3d < -1:
            return {"flow": "FII_SELLING", "trend": "NEGATIVE", "score": -10}
        else:
            return {"flow": "MIXED", "trend": "NEUTRAL", "score": 0}
    except:
        return {"flow": "UNKNOWN", "trend": "NEUTRAL", "score": 0}

# ============================================================
# 🛡️ MODULE 2: RISK MANAGEMENT (Kelly + ATR + Correlation)
# ============================================================

def calculate_kelly_position_size(win_rate, avg_win, avg_loss, capital):
    """Kelly Criterion for optimal position sizing"""
    try:
        if avg_loss == 0 or win_rate == 0:
            return capital * 0.02  # Default 2%
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        win_rate_dec = win_rate / 100
        loss_rate = 1 - win_rate_dec
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        
        kelly_fraction = (b * win_rate_dec - loss_rate) / b
        
        # Use HALF Kelly for safety (Kelly can be aggressive)
        kelly_fraction = max(0.01, min(0.05, kelly_fraction * 0.5))
        
        return capital * kelly_fraction
    except:
        return capital * 0.02

def calculate_atr_position_size(entry_price, atr, capital, risk_percent=1.5):
    """ATR-based volatility position sizing"""
    try:
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr * 1.5  # 1.5 ATR stop
        
        if stop_distance <= 0:
            return {"shares": 0, "value": 0, "risk": 0}
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * entry_price
        actual_risk = shares * stop_distance
        
        # Cap at 20% of capital
        max_position = capital * 0.2
        if position_value > max_position:
            shares = int(max_position / entry_price)
            position_value = shares * entry_price
            actual_risk = shares * stop_distance
        
        return {
            "shares": shares,
            "value": round(position_value, 2),
            "risk": round(actual_risk, 2),
            "risk_percent": round((actual_risk / capital) * 100, 2)
        }
    except:
        return {"shares": 0, "value": 0, "risk": 0}

def check_correlation_guard(new_pick, existing_picks, memory):
    """Prevent over-concentration in correlated stocks"""
    if not existing_picks:
        return {"allowed": True, "reason": "First pick"}
    
    new_sector = new_pick.get('sector', 'Unknown')
    
    # Count existing picks in same sector
    sector_count = sum(1 for p in existing_picks if p.get('sector') == new_sector)
    
    if sector_count >= MAX_SECTOR_EXPOSURE:
        return {"allowed": False, "reason": f"Max {MAX_SECTOR_EXPOSURE} stocks per sector reached ({new_sector})"}
    
    # Check total portfolio heat
    total_risk = sum(p.get('risk_percent', 0) for p in existing_picks) + new_pick.get('risk_percent', 0)
    
    if total_risk > MAX_PORTFOLIO_HEAT:
        return {"allowed": False, "reason": f"Portfolio heat too high ({total_risk:.1f}% > {MAX_PORTFOLIO_HEAT}%)"}
    
    return {"allowed": True, "reason": "Passed all correlation checks"}

# ============================================================
# 🤖 MODULE 3: MACHINE LEARNING ENGINE
# ============================================================

def extract_ml_features(df, ticker_name):
    """Extract features for ML model"""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        c = len(df) - 1
        
        # Technical features
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))
        
        # Volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std().iloc[c] * 100
        
        # Volume ratio
        vol_ratio = vol.iloc[c] / vol.rolling(20).mean().iloc[c]
        
        # Price position
        price_position = (close.iloc[c] - close.rolling(20).min().iloc[c]) / (close.rolling(20).max().iloc[c] - close.rolling(20).min().iloc[c])
        
        # Momentum
        roc_5 = ((close.iloc[c] - close.iloc[c-5]) / close.iloc[c-5]) * 100
        roc_20 = ((close.iloc[c] - close.iloc[c-20]) / close.iloc[c-20]) * 100
        
        # Distance from EMAs
        dist_ema20 = ((close.iloc[c] - ema20.iloc[c]) / ema20.iloc[c]) * 100
        dist_ema50 = ((close.iloc[c] - ema50.iloc[c]) / ema50.iloc[c]) * 100
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[c]
        atr_pct = (atr / close.iloc[c]) * 100
        
        return {
            "rsi": float(rsi.iloc[c]) if not pd.isna(rsi.iloc[c]) else 50,
            "volatility": float(volatility) if not pd.isna(volatility) else 2,
            "vol_ratio": float(vol_ratio) if not pd.isna(vol_ratio) else 1,
            "price_position": float(price_position) if not pd.isna(price_position) else 0.5,
            "roc_5": float(roc_5) if not pd.isna(roc_5) else 0,
            "roc_20": float(roc_20) if not pd.isna(roc_20) else 0,
            "dist_ema20": float(dist_ema20) if not pd.isna(dist_ema20) else 0,
            "dist_ema50": float(dist_ema50) if not pd.isna(dist_ema50) else 0,
            "atr_pct": float(atr_pct) if not pd.isna(atr_pct) else 2
        }
    except Exception as e:
        return None

def train_ml_model(memory):
    """Train ML model on historical trade features"""
    if not ML_AVAILABLE:
        return None
    
    features_data = memory.get('ml_features', [])
    completed = memory.get('completed_trades', [])
    
    if len(features_data) < 30 or len(completed) < 30:
        return None
    
    try:
        # Match features with outcomes
        X = []
        y = []
        
        for trade in completed:
            trade_features = None
            for fd in features_data:
                if fd.get('ticker') == trade['ticker'] and fd.get('date', '')[:10] == trade.get('date', '')[:10]:
                    trade_features = fd.get('features')
                    break
            
            if trade_features:
                X.append([
                    trade_features.get('rsi', 50),
                    trade_features.get('volatility', 2),
                    trade_features.get('vol_ratio', 1),
                    trade_features.get('price_position', 0.5),
                    trade_features.get('roc_5', 0),
                    trade_features.get('roc_20', 0),
                    trade_features.get('dist_ema20', 0),
                    trade_features.get('dist_ema50', 0),
                    trade_features.get('atr_pct', 2)
                ])
                y.append(1 if trade['outcome'] == 'WIN' else 0)
        
        if len(X) < 20:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Train gradient boosting classifier
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        return model
    except Exception as e:
        print(f"ML training error: {e}")
        return None

def predict_ml_probability(model, features):
    """Predict win probability using trained model"""
    if model is None or features is None:
        return 50  # Default neutral
    
    try:
        X = np.array([[
            features.get('rsi', 50),
            features.get('volatility', 2),
            features.get('vol_ratio', 1),
            features.get('price_position', 0.5),
            features.get('roc_5', 0),
            features.get('roc_20', 0),
            features.get('dist_ema20', 0),
            features.get('dist_ema50', 0),
            features.get('atr_pct', 2)
        ]])
        
        prob = model.predict_proba(X)[0][1] * 100
        return round(prob, 1)
    except:
        return 50

# ============================================================
# 🔌 MODULE 4: BROKER INTEGRATION (Zerodha Kite)
# ============================================================

def place_kite_order(symbol, quantity, price, order_type="LIMIT", transaction="BUY"):
    """Place order via Zerodha Kite API"""
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        return {"success": False, "message": "Kite credentials not configured"}
    
    try:
        url = "https://api.kite.trade/orders/regular"
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {KITE_API_KEY}:{KITE_ACCESS_TOKEN}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "tradingsymbol": symbol,
            "exchange": "NSE",
            "transaction_type": transaction,
            "order_type": order_type,
            "quantity": quantity,
            "product": "CNC",  # Delivery
            "validity": "DAY",
            "price": price
        }
        
        r = requests.post(url, headers=headers, data=data, timeout=15)
        result = r.json()
        
        if result.get("status") == "success":
            return {
                "success": True,
                "order_id": result["data"]["order_id"],
                "message": f"Order placed: {transaction} {quantity} {symbol}"
            }
        else:
            return {"success": False, "message": result.get("message", "Unknown error")}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ============================================================
# 📱 MODULE 4B: INTERACTIVE TELEGRAM
# ============================================================

def send_telegram_with_buttons(msg, buttons=None):
    """Send Telegram message with inline buttons"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        
        # Split long messages
        parts = [msg[i:i+3800] for i in range(0, len(msg), 3800)]
        
        for i, part in enumerate(parts):
            payload = {
                "chat_id": CHAT_ID,
                "text": part
            }
            
            # Add buttons only to last message
            if i == len(parts) - 1 and buttons:
                payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
            
            r = requests.post(url, json=payload, timeout=15)
            time.sleep(0.8)
        
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def send_telegram(msg):
    """Simple send without buttons"""
    return send_telegram_with_buttons(msg)

def create_pick_buttons(ticker):
    """Create interactive buttons for a pick"""
    return [
        [
            {"text": "✅ Win", "callback_data": f"win_{ticker}"},
            {"text": "❌ Loss", "callback_data": f"loss_{ticker}"}
        ],
        [
            {"text": "📊 Details", "callback_data": f"details_{ticker}"},
            {"text": "🔄 Retest", "callback_data": f"retest_{ticker}"}
        ]
    ]

def create_menu_buttons():
    """Main menu buttons"""
    return [
        [
            {"text": "📊 Memory Report", "callback_data": "memory_report"},
            {"text": "📈 Performance", "callback_data": "performance"}
        ],
        [
            {"text": "🎯 Active Positions", "callback_data": "positions"},
            {"text": "🧠 Insights", "callback_data": "insights"}
        ]
    ]

def check_telegram_updates(memory):
    """Check for user callbacks/commands"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        r = requests.get(url, timeout=10).json()
        
        for update in r.get('result', [])[-20:]:  # Last 20 updates
            # Handle callback queries (button presses)
            if 'callback_query' in update:
                cb = update['callback_query']
                data = cb['data']
                
                if data.startswith('win_'):
                    ticker = data.replace('win_', '')
                    handle_user_feedback(ticker, 'WIN', memory)
                
                elif data.startswith('loss_'):
                    ticker = data.replace('loss_', '')
                    handle_user_feedback(ticker, 'LOSS', memory)
                
                elif data == 'memory_report':
                    send_memory_report(memory)
                
                elif data == 'performance':
                    send_performance_report(memory)
                
                elif data == 'insights':
                    send_insights_report(memory)
                
                # Answer callback to remove loading
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                    json={"callback_query_id": cb['id']},
                    timeout=5
                )
            
            # Handle text commands
            elif 'message' in update and 'text' in update['message']:
                text = update['message']['text'].strip().lower()
                
                if text == '/memory':
                    send_memory_report(memory)
                elif text == '/performance':
                    send_performance_report(memory)
                elif text == '/insights':
                    send_insights_report(memory)
                elif text == '/help':
                    send_help_message()
        
        return memory
    except Exception as e:
        print(f"Callback check error: {e}")
        return memory

def handle_user_feedback(ticker, outcome, memory):
    """Handle user feedback on picks"""
    memory['user_feedback'][ticker] = {
        "outcome": outcome,
        "date": datetime.now().isoformat()
    }
    
    if outcome == 'WIN':
        memory['total_wins'] += 1
        confirmation = f"✅ Marked {ticker} as WIN"
    else:
        memory['total_losses'] += 1
        confirmation = f"❌ Marked {ticker} as LOSS"
    
    send_telegram(confirmation)
    save_memory(memory)

def send_memory_report(memory):
    """Send detailed memory report"""
    total = memory['total_wins'] + memory['total_losses']
    wr = (memory['total_wins'] / total * 100) if total > 0 else 0
    
    msg = "🧠 [ARTHA MEMORY REPORT]\n"
    msg += "=" * 30 + "\n\n"
    msg += f"📅 Bot Age: {(datetime.now() - datetime.fromisoformat(memory['created'])).days} days\n"
    msg += f"🎯 Total Picks: {memory['total_picks']}\n"
    msg += f"✅ Wins: {memory['total_wins']}\n"
    msg += f"❌ Losses: {memory['total_losses']}\n"
    msg += f"📊 Win Rate: {wr:.1f}%\n"
    msg += f"⏳ Pending: {len(memory['pending_evaluations'])}\n\n"
    
    if memory.get('learning_insights'):
        msg += "💡 KEY INSIGHTS:\n"
        for insight in memory['learning_insights'][:5]:
            msg += f"  {insight}\n"
    
    send_telegram(msg)

def send_performance_report(memory):
    """Send performance breakdown"""
    msg = "📈 [PERFORMANCE BREAKDOWN]\n"
    msg += "=" * 30 + "\n\n"
    
    if memory.get('sector_performance'):
        msg += "🏆 SECTOR PERFORMANCE:\n"
        sectors = sorted(
            memory['sector_performance'].items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )
        for sec, data in sectors[:8]:
            if data.get('count', 0) >= 2:
                msg += f"  {sec}: {data.get('win_rate', 0)}% WR ({data['count']} trades)\n"
    
    send_telegram(msg)

def send_insights_report(memory):
    """Send all learned insights"""
    msg = "🧠 [ALL LEARNED INSIGHTS]\n"
    msg += "=" * 30 + "\n\n"
    
    insights = memory.get('learning_insights', [])
    if insights:
        for i, insight in enumerate(insights, 1):
            msg += f"{i}. {insight}\n"
    else:
        msg += "Still learning... Need more trades.\n"
    
    send_telegram(msg)

def send_help_message():
    """Send help menu"""
    msg = "⚡ [ARTHA COMMANDS]\n"
    msg += "=" * 30 + "\n\n"
    msg += "📱 TELEGRAM COMMANDS:\n"
    msg += "/memory - Full memory report\n"
    msg += "/performance - Sector performance\n"
    msg += "/insights - All learned insights\n"
    msg += "/help - This menu\n\n"
    msg += "🎯 INLINE BUTTONS:\n"
    msg += "✅ Win - Mark trade as winner\n"
    msg += "❌ Loss - Mark trade as loser\n"
    msg += "📊 Details - Get more info\n"
    msg += "🔄 Retest - Re-analyze stock\n"
    send_telegram(msg)

# ============================================================
# 🧮 CORE TECHNICAL FUNCTIONS
# ============================================================

def get_ema(s, n): return s.ewm(span=n, adjust=False).mean()
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
        change_20d = ((curr - close.iloc[-20]) / close.iloc[-20]) * 100
        
        if curr > ema20 > ema50 > ema200 and change_20d > 3:
            return {"regime": "STRONG_BULL", "trade_allowed": True, "change_20d": round(change_20d, 2), "price": round(curr, 2)}
        elif curr > ema20 > ema50 and change_20d > 0:
            return {"regime": "BULL", "trade_allowed": True, "change_20d": round(change_20d, 2), "price": round(curr, 2)}
        elif curr < ema50:
            return {"regime": "BEAR", "trade_allowed": False, "change_20d": round(change_20d, 2), "price": round(curr, 2)}
        else:
            return {"regime": "SIDEWAYS", "trade_allowed": False, "change_20d": round(change_20d, 2), "price": round(curr, 2)}
    except:
        return {"regime": "UNKNOWN", "trade_allowed": False, "change_20d": 0, "price": 0}

# ============================================================
# 🏢 SECTOR DETECTION
# ============================================================

def detect_sector(ticker):
    sectors = {
        "IT": ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","COFORGE","MPHASIS","PERSISTENT","LTTS"],
        "Banking": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK","FEDERALBNK","IDFCFIRSTB","BANKBARODA","PNB"],
        "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","TVSMOTOR","EXIDEIND","BOSCHLTD","BALKRISIND"],
        "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","ALKEM","BIOCON","TORNTPHARM","GLENMARK"],
        "FMCG": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","GODREJCP","MARICO","COLPAL","TATACONSUM"],
        "Metal": ["TATASTEEL","JSWSTEEL","HINDALCO","VEDL","COALINDIA","NMDC","SAIL","JINDALSTEL"],
        "Energy": ["RELIANCE","ONGC","IOC","BPCL","GAIL","NTPC","POWERGRID","TATAPOWER","ADANIGREEN","ADANIPOWER"],
        "Finance": ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SHRIRAMFIN","MANAPPURAM","LICHSGFIN"],
        "Infra": ["LT","GMRINFRA","RVNL","IRB","NCC","HFCL","IRFC","IREDA"]
    }
    for sector, stocks in sectors.items():
        if ticker in stocks:
            return sector
    return "Other"

# ============================================================
# 📥 GET STOCKS
# ============================================================

def get_all_tickers():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url, headers=headers, timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        return [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().unique().tolist()]
    except:
        try:
            url = "https://raw.githubusercontent.com/gauravsdeshmukh/StockDataAnalysis/main/nse_stocks.csv"
            df = pd.read_csv(url)
            return [f"{s.strip()}.NS" for s in df["Symbol"].dropna().unique().tolist()]
        except:
            return ["RELIANCE.NS","TCS.NS","INFY.NS"]

# ============================================================
# 🎯 ULTRA-SMART SCANNER (v10.0)
# ============================================================

def ultra_smart_scan(ticker, nifty_close, memory, ml_model, fii_flow):
    try:
        df = yf.download(ticker, period="6mo", progress=False, timeout=10)
        if len(df) < 100: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        if close.iloc[-1] < 30 or vol.mean() < 50000: return None
        
        # Technical indicators
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr = get_atr(high, low, close, 14)
        
        c = len(df) - 1
        curr = close.iloc[c]
        atr_val = atr.iloc[c]
        
        # Hard filters
        if not (curr > ema20.iloc[c] > ema50.iloc[c] > ema200.iloc[c]): return None
        if not (55 <= rsi.iloc[c] <= 70): return None
        if not (macd.iloc[c] > msig.iloc[c] and macd.iloc[c] > 0): return None
        
        # Entry type detection
        high_20 = high.iloc[-25:-5].max()
        recent_5d_high = high.iloc[-5:].max()
        is_early_breakout = recent_5d_high > high_20
        
        recent_high = high.iloc[-15:-3].max()
        dist_from_high = ((recent_high - curr) / recent_high) * 100
        is_pullback = 2 <= dist_from_high <= 8 and abs(curr - ema20.iloc[c]) / curr < 0.02
        
        if not (is_early_breakout or is_pullback): return None
        
        entry_type = "PULLBACK" if is_pullback else "EARLY_BREAKOUT"
        
        # Volume
        avg_vol = vol.rolling(20).mean().iloc[c]
        vol_ratio = vol.iloc[c] / avg_vol if avg_vol > 0 else 0
        
        # BASE SCORING
        score = 0
        signals = []
        
        # Trend
        score += 20 * memory["weights"]["trend"]
        signals.append("Trend OK")
        
        # Volume
        if vol_ratio >= 3:
            score += 15 * memory["weights"]["volume"]
            signals.append("High Vol")
        elif vol_ratio >= 2:
            score += 10 * memory["weights"]["volume"]
        
        # RSI
        if 60 <= rsi.iloc[c] <= 65:
            score += 15 * memory["weights"]["rsi"]
            signals.append("Optimal RSI")
        elif 55 <= rsi.iloc[c] <= 70:
            score += 10 * memory["weights"]["rsi"]
        
        # Entry type
        if is_early_breakout:
            score += 20 * memory["weights"]["early_breakout"]
        if is_pullback:
            score += 20 * memory["weights"]["pullback"]
        
        # Sector performance (from memory)
        ticker_clean = ticker.replace(".NS", "")
        sector = detect_sector(ticker_clean)
        if sector in memory["sector_performance"]:
            sec_data = memory["sector_performance"][sector]
            if sec_data.get('count', 0) >= 3:
                if sec_data.get('win_rate', 0) >= 65:
                    score += 10
                    signals.append(f"Hot Sector")
                elif sec_data.get('win_rate', 0) <= 35:
                    return None  # Skip weak sectors
        
        # MODULE 1: SENTIMENT ANALYSIS
        sentiment_data = get_google_news_sentiment(ticker_clean)
        if sentiment_data['count'] > 0:
            if sentiment_data['sentiment'] == "VERY_BULLISH":
                score += 15 * memory["weights"]["sentiment"]
                signals.append("Very Bullish News")
            elif sentiment_data['sentiment'] == "BULLISH":
                score += 8 * memory["weights"]["sentiment"]
                signals.append("Positive News")
            elif sentiment_data['sentiment'] == "VERY_BEARISH":
                return None  # Skip stocks with very bearish news
            elif sentiment_data['sentiment'] == "BEARISH":
                score -= 15
                signals.append("Bearish News")
        
        # MODULE 1B: INSTITUTIONAL ACTIVITY
        bulk_data = check_bulk_deals(ticker_clean)
        if bulk_data['score'] > 0:
            score += bulk_data['score'] * memory["weights"]["institutional"]
            signals.append("Inst. Buying")
        elif bulk_data['score'] < 0:
            return None  # Skip stocks with institutional selling
        
        # FII/DII flow bonus
        if fii_flow['score'] > 0:
            score += 5
        elif fii_flow['score'] < 0:
            score -= 5
        
        # MODULE 3: ML PREDICTION
        ml_features = extract_ml_features(df, ticker_clean)
        ml_probability = 50
        
        if ml_model and ml_features:
            ml_probability = predict_ml_probability(ml_model, ml_features)
            
            # Skip if ML predicts low win probability
            if ml_probability < 45:
                return None
            
            # Boost score based on ML confidence
            if ml_probability >= 70:
                score += 20
                signals.append(f"ML: {ml_probability}%")
            elif ml_probability >= 60:
                score += 10
                signals.append(f"ML: {ml_probability}%")
        
        # Only accept high scores
        if score < 75: return None
        
        # RS Rating
        rs = 100
        if nifty_close is not None and len(nifty_close) > 63:
            stock_perf = close.iloc[-1] / close.iloc[-63]
            nifty_perf = nifty_close.iloc[-1] / nifty_close.iloc[-63]
            rs = (stock_perf / nifty_perf) * 100
            if rs > 110:
                score += 10
                signals.append(f"Strong RS")
        
        # MODULE 2: POSITION SIZING
        # Kelly Criterion (if enough data)
        total_evaluated = memory['total_wins'] + memory['total_losses']
        if total_evaluated >= 20:
            win_rate = (memory['total_wins'] / total_evaluated) * 100
            avg_win = 5  # Approximate avg win %
            avg_loss = 3  # Approximate avg loss %
            kelly_size = calculate_kelly_position_size(win_rate, avg_win, avg_loss, TOTAL_CAPITAL)
        else:
            kelly_size = TOTAL_CAPITAL * 0.02
        
        # ATR-based sizing
        atr_sizing = calculate_atr_position_size(curr, atr_val, TOTAL_CAPITAL, 1.5)
        
        # Use smaller of the two (conservative)
        recommended_value = min(kelly_size, atr_sizing['value'])
        recommended_shares = int(recommended_value / curr) if curr > 0 else 0
        
        # Targets
        recent_low = low.iloc[-5:].min()
        sl = max(recent_low * 0.99, ema20.iloc[c] * 0.98)
        risk = curr - sl
        if risk <= 0: return None
        
        risk_pct = (risk / curr) * 100
        if risk_pct > 4: return None
        
        tgt1 = curr + (risk * 1.5)
        tgt2 = curr + (risk * 2.5)
        
        return {
            "ticker": ticker_clean,
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1),
            "entry_type": entry_type,
            "sector": sector,
            "sl": round(sl, 2),
            "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "rs_rating": round(rs, 0),
            "atr": round(atr_val, 2),
            "recommended_shares": recommended_shares,
            "recommended_value": round(recommended_value, 0),
            "sentiment": sentiment_data['sentiment'],
            "sentiment_score": sentiment_data['score'],
            "institutional": bulk_data['activity'],
            "ml_probability": ml_probability,
            "features": ml_features,
            "signals": signals[:6]
        }
    except Exception as e:
        return None

# ============================================================
# 🎓 EVALUATE PAST PICKS & LEARN
# ============================================================

def evaluate_past_picks(memory):
    to_remove = []
    new_completed = []
    
    for pick_id, pick in memory["pending_evaluations"].items():
        try:
            days_old = (datetime.now() - datetime.fromisoformat(pick["date"])).days
            if days_old < 5: continue
            
            ticker = f"{pick['ticker']}.NS"
            df = yf.download(ticker, period="1mo", progress=False, timeout=10)
            if len(df) < 5: continue
            
            current_price = df['Close'].iloc[-1]
            entry_price = pick['entry_price']
            sl_price = pick['sl']
            tgt1_price = pick['tgt1']
            
            highest = df['High'].iloc[-days_old:].max() if days_old <= len(df) else df['High'].max()
            lowest = df['Low'].iloc[-days_old:].min() if days_old <= len(df) else df['Low'].min()
            
            hit_target = highest >= tgt1_price
            hit_sl = lowest <= sl_price
            
            if hit_target and not hit_sl:
                outcome = "WIN"
                actual_return = ((tgt1_price - entry_price) / entry_price) * 100
            elif hit_sl:
                outcome = "LOSS"
                actual_return = ((sl_price - entry_price) / entry_price) * 100
            else:
                pct_return = ((current_price - entry_price) / entry_price) * 100
                outcome = "WIN" if pct_return > 2 else "LOSS" if pct_return < -1 else "NEUTRAL"
                actual_return = pct_return
            
            completed = {
                "ticker": pick['ticker'],
                "date": pick['date'],
                "entry": entry_price,
                "sl": sl_price,
                "tgt1": tgt1_price,
                "outcome": outcome,
                "actual_return": round(actual_return, 2),
                "days_held": days_old,
                "score": pick['score'],
                "sector": pick.get('sector', 'Unknown'),
                "entry_type": pick.get('entry_type', 'Unknown'),
                "rsi": pick.get('rsi', 0)
            }
            
            new_completed.append(completed)
            to_remove.append(pick_id)
            
            if outcome == "WIN": memory["total_wins"] += 1
            elif outcome == "LOSS": memory["total_losses"] += 1
        except:
            continue
    
    memory["completed_trades"].extend(new_completed)
    for pid in to_remove:
        del memory["pending_evaluations"][pid]
    
    if len(memory["completed_trades"]) > 500:
        memory["completed_trades"] = memory["completed_trades"][-500:]
    
    return memory, new_completed

def learn_from_trades(memory):
    trades = memory["completed_trades"]
    if len(trades) < 10: return memory
    
    # Sector analysis
    sector_stats = {}
    for t in trades:
        sec = t.get('sector', 'Unknown')
        if sec not in sector_stats:
            sector_stats[sec] = {'wins': 0, 'losses': 0, 'count': 0}
        sector_stats[sec]['count'] += 1
        if t['outcome'] == 'WIN': sector_stats[sec]['wins'] += 1
        elif t['outcome'] == 'LOSS': sector_stats[sec]['losses'] += 1
    
    for sec, stats in sector_stats.items():
        if stats['count'] > 0:
            stats['win_rate'] = round((stats['wins'] / stats['count']) * 100, 1)
    
    memory["sector_performance"] = sector_stats
    
    # Insights
    insights = []
    if sector_stats:
        best = max(sector_stats.items(), key=lambda x: x[1].get('win_rate', 0) if x[1]['count'] >= 3 else 0)
        if best[1].get('count', 0) >= 3:
            insights.append(f"🏆 Best sector: {best[0]} ({best[1]['win_rate']}% WR)")
    
    if memory["total_wins"] + memory["total_losses"] > 0:
        wr = (memory["total_wins"] / (memory["total_wins"] + memory["total_losses"])) * 100
        insights.append(f"📈 Overall WR: {wr:.1f}%")
    
    memory["learning_insights"] = insights
    return memory

# ============================================================
# 🚀 MAIN ORCHESTRATOR
# ============================================================

def main():
    print("=" * 60)
    print(f"ARTHA {BOT_VERSION} - {BOT_TAGLINE}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("=" * 60)
    
    # Load memory
    memory = load_memory()
    print(f"\n[MEMORY] {memory['total_picks']} picks | {memory['total_wins']}W/{memory['total_losses']}L")
    
    # Check for user callbacks
    memory = check_telegram_updates(memory)
    
    # Send starting message
    send_telegram(f"⚡ ARTHA {BOT_VERSION} starting comprehensive scan...")
    
    # Evaluate past picks
    memory, new_completed = evaluate_past_picks(memory)
    memory = learn_from_trades(memory)
    
    # Train ML model
    print("\n[ML] Training model...")
    ml_model = train_ml_model(memory) if ML_AVAILABLE else None
    if ml_model:
        print("  ML model trained successfully")
        memory['ml_model_trained'] = True
    
    # Market regime
    regime = detect_market_regime()
    print(f"\n[REGIME] {regime['regime']}")
    
    if not regime['trade_allowed']:
        msg = f"⚡ ARTHA {BOT_VERSION}\n"
        msg += f"📅 {datetime.now().strftime('%A, %d %b %Y')}\n\n"
        msg += "[⚠️ NO TRADES TODAY]\n"
        msg += f"Regime: {regime['regime']}\n"
        msg += f"Nifty: {regime['price']} ({regime['change_20d']:+.2f}% 20D)\n\n"
        msg += "🧠 MEMORY STATUS:\n"
        msg += f"Tracked: {memory['total_picks']} picks\n"
        wr = (memory['total_wins']/(memory['total_wins']+memory['total_losses'])*100) if (memory['total_wins']+memory['total_losses'])>0 else 0
        msg += f"WR: {wr:.1f}%\n\n"
        
        save_memory(memory)
        send_telegram_with_buttons(msg, create_menu_buttons())
        return
    
    # Get benchmark
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except:
        nifty_close = None
    
    # Get FII flow
    fii_flow = check_fii_dii_flow()
    print(f"[FII] {fii_flow['flow']}")
    
    # Scan stocks
    print("\n[SCANNING] Ultra-smart scan in progress...")
    tickers = get_all_tickers()
    results = []
    
    for i, t in enumerate(tickers):
        if (i+1) % 300 == 0:
            print(f"  {i+1}/{len(tickers)} | Found: {len(results)}")
        r = ultra_smart_scan(t, nifty_close, memory, ml_model, fii_flow)
        if r: results.append(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply correlation guard
    final_picks = []
    for pick in results:
        guard = check_correlation_guard(pick, final_picks, memory)
        if guard['allowed']:
            final_picks.append(pick)
        if len(final_picks) >= 3: break
    
    print(f"\n[FINAL] {len(final_picks)} picks after correlation guard")
    
    # Save picks to memory
    for pick in final_picks:
        pick_id = f"{pick['ticker']}_{datetime.now().strftime('%Y%m%d')}"
        memory["pending_evaluations"][pick_id] = {
            "ticker": pick['ticker'],
            "date": datetime.now().isoformat(),
            "entry_price": pick['price'],
            "sl": pick['sl'],
            "tgt1": pick['tgt1'],
            "tgt2": pick['tgt2'],
            "score": pick['score'],
            "sector": pick['sector'],
            "entry_type": pick['entry_type'],
            "rsi": pick['rsi']
        }
        memory["total_picks"] += 1
        
        # Save features for ML
        if pick.get('features'):
            memory["ml_features"].append({
                "ticker": pick['ticker'],
                "date": datetime.now().isoformat(),
                "features": pick['features']
            })
    
    # Keep last 500 features
    if len(memory["ml_features"]) > 500:
        memory["ml_features"] = memory["ml_features"][-500:]
    
    save_memory(memory)
    
    # Build report
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"{BOT_TAGLINE}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    # Brain status
    total_eval = memory['total_wins'] + memory['total_losses']
    wr = (memory['total_wins'] / total_eval * 100) if total_eval > 0 else 0
    
    msg += "🧠 [BRAIN STATUS]\n"
    msg += f"Picks: {memory['total_picks']} | WR: {wr:.1f}%\n"
    msg += f"ML Model: {'✅ Active' if ml_model else '⏳ Learning'}\n"
    msg += f"Sentiment: ✅ Active\n"
    msg += f"Risk Mgmt: ✅ Active\n\n"
    
    # Market
    msg += f"[🌍 MARKET]\n"
    msg += f"Regime: {regime['regime']}\n"
    msg += f"Nifty: {regime['price']}\n"
    msg += f"FII Flow: {fii_flow['flow']}\n\n"
    
    # Picks
    if final_picks:
        msg += "=" * 40 + "\n"
        msg += "[🎯 TOP 3 SMART PICKS]\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(final_picks):
            grade = "A+" if s['score'] >= 100 else "A" if s['score'] >= 85 else "B+"
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/120)\n"
            msg += f"🏢 {s['sector']} | {s['entry_type']}\n\n"
            
            msg += "📊 TECHNICAL:\n"
            msg += f"  Price: Rs.{s['price']} | RSI: {s['rsi']}\n"
            msg += f"  Volume: {s['vol_ratio']}x | RS: {s['rs_rating']}\n"
            msg += f"  ATR: Rs.{s['atr']}\n\n"
            
            msg += "🧠 AI ANALYSIS:\n"
            msg += f"  ML Win Prob: {s['ml_probability']}%\n"
            msg += f"  News: {s['sentiment']}\n"
            msg += f"  Institutional: {s['institutional']}\n\n"
            
            msg += "💰 SMART SIZING:\n"
            msg += f"  Shares: {s['recommended_shares']}\n"
            msg += f"  Value: Rs.{s['recommended_value']:,}\n"
            msg += f"  Risk: {s['risk_pct']}%\n\n"
            
            msg += "🎯 TRADE SETUP:\n"
            msg += f"  Entry: Rs.{s['price']}\n"
            msg += f"  SL: Rs.{s['sl']}\n"
            msg += f"  T1: Rs.{s['tgt1']} | T2: Rs.{s['tgt2']}\n"
            msg += f"  R:R = 1:{s['rr_ratio']}\n\n"
            
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[❌ NO PICKS TODAY]\n"
        msg += "No stocks passed all v10.0 filters:\n"
        msg += "• Technical setup\n"
        msg += "• Sentiment check\n"
        msg += "• Institutional activity\n"
        msg += "• ML prediction\n"
        msg += "• Risk management\n\n"
    
    msg += "💡 Tap buttons below to interact!\n\n"
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION}"
    
    # Send with interactive buttons
    if final_picks:
        buttons = []
        for pick in final_picks:
            buttons.extend(create_pick_buttons(pick['ticker']))
        buttons.extend(create_menu_buttons())
        send_telegram_with_buttons(msg, buttons)
    else:
        send_telegram_with_buttons(msg, create_menu_buttons())
    
    print("\n[DONE] v10.0 scan complete!")

if __name__ == "__main__":
    main()
