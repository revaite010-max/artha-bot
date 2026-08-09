"""
ARTHA v11.0 - Ultimate Trading System
Features: Confirmation Delay + Backtesting + Sector Rotation + 
          Enhanced ML + Advanced Risk Management
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
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

# ML Imports
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configuration
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

BOT_NAME = "⚡ ARTHA"
BOT_VERSION = "v11.0"
BOT_TAGLINE = "Ultimate Trading System"

MEMORY_FILE = "artha_memory.json"

# Enhanced Portfolio Config
TOTAL_CAPITAL = 100000
MAX_PORTFOLIO_HEAT = 5  # Reduced from 6% for safety
MAX_SECTOR_EXPOSURE = 2
BASE_RISK_PER_TRADE = 1.0  # Base 1%, adjusted by regime

# ============================================================
# 🧠 ENHANCED MEMORY SYSTEM
# ============================================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                mem = json.load(f)
                for key in ['ml_features', 'sentiment_history', 'user_feedback',
                           'active_positions', 'backtest_results', 'sector_history',
                           'confirmation_queue']:
                    if key not in mem:
                        mem[key] = {} if key.endswith('_history') or key == 'user_feedback' or key == 'active_positions' or key == 'backtest_results' or key == 'confirmation_queue' else []
                if 'sector_rotation_data' not in mem:
                    mem['sector_rotation_data'] = {}
                return mem
        except:
            pass
    
    return {
        "version": "11.0",
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_picks": 0,
        "total_wins": 0,
        "total_losses": 0,
        "pending_evaluations": {},
        "confirmation_queue": {},  # Stocks awaiting confirmation
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
        "backtest_results": {},
        "sector_rotation_data": {},
        "weights": {
            "trend": 1.0, "volume": 1.0, "rsi": 1.0,
            "sector_bonus": 1.0, "early_breakout": 1.0,
            "pullback": 1.0, "sentiment": 1.0, "institutional": 1.0,
            "sector_rotation": 1.0, "multi_timeframe": 1.0
        },
        "ml_model_trained": False,
        "last_ml_train": None
    }

def save_memory(memory):
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

# ============================================================
# 🧮 CORE MATH FUNCTIONS
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
# 🎯 IMPROVEMENT #1: CONFIRMATION DELAY
# ============================================================

def add_to_confirmation_queue(memory, ticker, price, sl, tgt1, tgt2, score, sector, entry_type):
    """Add stock to 2-day confirmation queue"""
    queue_id = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
    
    memory['confirmation_queue'][queue_id] = {
        "ticker": ticker,
        "detected_date": datetime.now().isoformat(),
        "detected_price": price,
        "detected_sl": sl,
        "detected_tgt1": tgt1,
        "detected_tgt2": tgt2,
        "score": score,
        "sector": sector,
        "entry_type": entry_type,
        "confirmation_days": 0,
        "status": "PENDING_CONFIRMATION"
    }
    return memory

def check_confirmations(memory):
    """Check pending confirmations - must hold for 2 days"""
    confirmed_picks = []
    to_remove = []
    
    for queue_id, item in memory['confirmation_queue'].items():
        try:
            days_old = (datetime.now() - datetime.fromisoformat(item['detected_date'])).days
            
            ticker = item['ticker']
            df = yf.download(f"{ticker}.NS", period="10d", progress=False, timeout=10)
            
            if len(df) < days_old + 1:
                continue
            
            current_price = df['Close'].iloc[-1]
            detected_price = item['detected_price']
            
            # Confirmation criteria
            still_above_breakout = current_price >= detected_price * 0.98
            volume_holding = df['Volume'].iloc[-days_old:].mean() >= df['Volume'].iloc[-20:-days_old].mean() * 0.8
            no_reversal = current_price >= detected_price * 0.97
            
            confirmations = sum([still_above_breakout, volume_holding, no_reversal])
            
            if days_old >= 2:
                if confirmations >= 2:
                    # CONFIRMED - Add to active picks
                    item['status'] = "CONFIRMED"
                    item['confirmed_date'] = datetime.now().isoformat()
                    item['confirmed_price'] = current_price
                    item['confirmations_passed'] = confirmations
                    confirmed_picks.append(item)
                    to_remove.append(queue_id)
                else:
                    # FAILED - Remove
                    to_remove.append(queue_id)
                    print(f"  {ticker}: Failed confirmation ({confirmations}/3)")
        except:
            continue
    
    # Remove processed items
    for qid in to_remove:
        del memory['confirmation_queue'][qid]
    
    return memory, confirmed_picks

# ============================================================
# 🎯 IMPROVEMENT #2: BACKTESTING ENGINE
# ============================================================

def backtest_ticker(ticker, period="1y"):
    """Backtest strategy on historical data"""
    try:
        df = yf.download(f"{ticker}.NS", period=period, progress=False, timeout=10)
        if len(df) < 100:
            return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        ema20 = get_ema(close, 20)
        ema50 = get_ema(close, 50)
        ema200 = get_ema(close, 200)
        rsi = get_rsi(close, 14)
        macd, msig = get_macd(close)
        atr = get_atr(high, low, close, 14)
        
        trades = []
        position = None
        
        for i in range(60, len(df)):
            curr_close = close.iloc[i]
            
            # Check for entry
            if position is None:
                # Entry conditions
                if (curr_close > ema20.iloc[i] > ema50.iloc[i] > ema200.iloc[i] and
                    55 <= rsi.iloc[i] <= 70 and
                    macd.iloc[i] > msig.iloc[i] and
                    high.iloc[i] > high.iloc[i-20:i].max()):
                    
                    entry_price = curr_close
                    stop_loss = ema20.iloc[i] * 0.98
                    target = entry_price + (2 * atr.iloc[i])
                    
                    position = {
                        "entry_date": df.index[i],
                        "entry": entry_price,
                        "sl": stop_loss,
                        "tgt": target
                    }
            
            # Check for exit
            elif position is not None:
                if high.iloc[i] >= position['tgt']:
                    pnl = ((position['tgt'] - position['entry']) / position['entry']) * 100
                    trades.append({
                        "outcome": "WIN",
                        "return": round(pnl, 2),
                        "days": (df.index[i] - position['entry_date']).days
                    })
                    position = None
                elif low.iloc[i] <= position['sl']:
                    pnl = ((position['sl'] - position['entry']) / position['entry']) * 100
                    trades.append({
                        "outcome": "LOSS",
                        "return": round(pnl, 2),
                        "days": (df.index[i] - position['entry_date']).days
                    })
                    position = None
        
        if not trades:
            return None
        
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round((len(wins) / len(trades)) * 100, 1),
            "avg_win": round(sum(t['return'] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t['return'] for t in losses) / len(losses), 2) if losses else 0,
            "avg_days": round(sum(t['days'] for t in trades) / len(trades), 1),
            "total_return": round(sum(t['return'] for t in trades), 2)
        }
    except:
        return None

def get_backtest_score(ticker, memory):
    """Get historical performance score for ticker"""
    if ticker in memory.get('backtest_results', {}):
        cached = memory['backtest_results'][ticker]
        cache_age = (datetime.now() - datetime.fromisoformat(cached.get('date', '2020-01-01'))).days
        if cache_age < 30:
            return cached.get('data')
    
    result = backtest_ticker(ticker)
    if result:
        memory['backtest_results'][ticker] = {
            "date": datetime.now().isoformat(),
            "data": result
        }
    return result

# ============================================================
# 🎯 IMPROVEMENT #3: SECTOR ROTATION INTELLIGENCE
# ============================================================

NIFTY_SECTORS = {
    "IT": "^CNXIT",
    "Banking": "^NSEBANK",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "Metal": "^CNXMETAL",
    "Realty": "^CNXREALTY",
    "Energy": "^CNXENERGY",
    "PSU Bank": "^CNXPSUBANK",
    "Finance": "^CNXFIN"
}

def analyze_sector_rotation():
    """Advanced sector rotation analysis"""
    sector_scores = {}
    
    for sector, symbol in NIFTY_SECTORS.items():
        try:
            df = yf.download(symbol, period="3mo", progress=False, timeout=10)
            if len(df) < 50:
                continue
            
            close = df['Close']
            
            # Multi-period returns
            ret_5d = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100
            ret_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            ret_60d = ((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]) * 100
            
            # Trend strength
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()
            
            trend_score = 0
            if close.iloc[-1] > ema20.iloc[-1]: trend_score += 1
            if ema20.iloc[-1] > ema50.iloc[-1]: trend_score += 1
            
            # Momentum acceleration
            momentum_recent = ret_5d
            momentum_older = ((close.iloc[-10] - close.iloc[-15]) / close.iloc[-15]) * 100 if len(close) > 15 else 0
            accelerating = momentum_recent > momentum_older
            
            # Combined score
            momentum_score = (ret_5d * 0.4) + (ret_20d * 0.35) + (ret_60d * 0.25)
            total_score = momentum_score + (trend_score * 3) + (5 if accelerating else 0)
            
            sector_scores[sector] = {
                "score": round(total_score, 2),
                "ret_5d": round(ret_5d, 2),
                "ret_20d": round(ret_20d, 2),
                "ret_60d": round(ret_60d, 2),
                "trend_strength": trend_score,
                "accelerating": accelerating,
                "phase": get_sector_phase(ret_5d, ret_20d, ret_60d)
            }
        except:
            continue
    
    ranked = sorted(sector_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return {
        "hot_sectors": [s[0] for s in ranked[:3]],
        "warm_sectors": [s[0] for s in ranked[3:5]],
        "cold_sectors": [s[0] for s in ranked[-3:]],
        "all_scores": sector_scores,
        "top_3_details": ranked[:3]
    }

def get_sector_phase(ret_5d, ret_20d, ret_60d):
    """Identify sector phase"""
    if ret_5d > 3 and ret_20d > 5:
        return "ACCELERATING"
    elif ret_5d > 1 and ret_20d > 3:
        return "GROWING"
    elif abs(ret_5d) < 1 and abs(ret_20d) < 2:
        return "CONSOLIDATING"
    elif ret_5d < -2:
        return "DECLINING"
    else:
        return "NEUTRAL"

def get_sector_bonus(stock_sector, sector_data):
    """Get bonus based on sector strength"""
    hot_sectors = sector_data.get('hot_sectors', [])
    warm_sectors = sector_data.get('warm_sectors', [])
    cold_sectors = sector_data.get('cold_sectors', [])
    
    if stock_sector in hot_sectors[:1]:  # #1 sector
        return {"bonus": 20, "label": "TOP SECTOR"}
    elif stock_sector in hot_sectors:
        return {"bonus": 15, "label": "HOT SECTOR"}
    elif stock_sector in warm_sectors:
        return {"bonus": 8, "label": "WARM SECTOR"}
    elif stock_sector in cold_sectors:
        return {"bonus": -20, "label": "COLD SECTOR - AVOID"}
    else:
        return {"bonus": 0, "label": "NEUTRAL"}

# ============================================================
# 🎯 IMPROVEMENT #4: ENHANCED ML WITH MORE FEATURES
# ============================================================

def extract_enhanced_features(df, sector_data, stock_sector, memory):
    """Extract 15+ features for ML"""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        c = len(df) - 1
        
        # Technical
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        rsi = get_rsi(close, 14)
        atr = get_atr(high, low, close, 14)
        
        # Basic features
        vol_ratio = vol.iloc[c] / vol.rolling(20).mean().iloc[c]
        volatility = close.pct_change().rolling(20).std().iloc[c] * 100
        
        # Price position
        price_position = (close.iloc[c] - close.rolling(20).min().iloc[c]) / (close.rolling(20).max().iloc[c] - close.rolling(20).min().iloc[c])
        
        # Momentum
        roc_5 = ((close.iloc[c] - close.iloc[c-5]) / close.iloc[c-5]) * 100
        roc_20 = ((close.iloc[c] - close.iloc[c-20]) / close.iloc[c-20]) * 100
        
        # EMA distances
        dist_ema20 = ((close.iloc[c] - ema20.iloc[c]) / ema20.iloc[c]) * 100
        dist_ema50 = ((close.iloc[c] - ema50.iloc[c]) / ema50.iloc[c]) * 100
        
        atr_pct = (atr.iloc[c] / close.iloc[c]) * 100
        
        # Sector features
        sector_score = 0
        sector_phase_score = 0
        if stock_sector in sector_data.get('all_scores', {}):
            sec = sector_data['all_scores'][stock_sector]
            sector_score = sec['score']
            phase = sec.get('phase', 'NEUTRAL')
            phase_scores = {"ACCELERATING": 10, "GROWING": 7, "CONSOLIDATING": 3, "NEUTRAL": 0, "DECLINING": -5}
            sector_phase_score = phase_scores.get(phase, 0)
        
        # Historical sector performance
        sector_win_rate = 0.5
        if stock_sector in memory.get('sector_performance', {}):
            sp = memory['sector_performance'][stock_sector]
            if sp.get('count', 0) >= 3:
                sector_win_rate = sp.get('win_rate', 50) / 100
        
        # Volume trend
        vol_ma5 = vol.rolling(5).mean().iloc[c]
        vol_ma20 = vol.rolling(20).mean().iloc[c]
        volume_trend = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
        
        # Price stability (lower is better for consolidation)
        price_stability = close.iloc[-10:].std() / close.iloc[c]
        
        return {
            "rsi": float(rsi.iloc[c]) if not pd.isna(rsi.iloc[c]) else 50,
            "volatility": float(volatility) if not pd.isna(volatility) else 2,
            "vol_ratio": float(vol_ratio) if not pd.isna(vol_ratio) else 1,
            "price_position": float(price_position) if not pd.isna(price_position) else 0.5,
            "roc_5": float(roc_5) if not pd.isna(roc_5) else 0,
            "roc_20": float(roc_20) if not pd.isna(roc_20) else 0,
            "dist_ema20": float(dist_ema20) if not pd.isna(dist_ema20) else 0,
            "dist_ema50": float(dist_ema50) if not pd.isna(dist_ema50) else 0,
            "atr_pct": float(atr_pct) if not pd.isna(atr_pct) else 2,
            "sector_score": float(sector_score),
            "sector_phase_score": float(sector_phase_score),
            "sector_win_rate": float(sector_win_rate),
            "volume_trend": float(volume_trend) if not pd.isna(volume_trend) else 1,
            "price_stability": float(price_stability) if not pd.isna(price_stability) else 0.02,
            "market_hour": float(datetime.now().hour)
        }
    except Exception as e:
        return None

def train_enhanced_ml(memory):
    """Train enhanced ML model"""
    if not ML_AVAILABLE:
        return None
    
    features_data = memory.get('ml_features', [])
    completed = memory.get('completed_trades', [])
    
    if len(completed) < 30:
        return None
    
    try:
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
                    trade_features.get('atr_pct', 2),
                    trade_features.get('sector_score', 0),
                    trade_features.get('sector_phase_score', 0),
                    trade_features.get('sector_win_rate', 0.5),
                    trade_features.get('volume_trend', 1),
                    trade_features.get('price_stability', 0.02),
                    trade_features.get('market_hour', 10)
                ])
                y.append(1 if trade['outcome'] == 'WIN' else 0)
        
        if len(X) < 20:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Use Random Forest for interpretability
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Get feature importance
        feature_names = ['rsi', 'volatility', 'vol_ratio', 'price_position',
                        'roc_5', 'roc_20', 'dist_ema20', 'dist_ema50', 'atr_pct',
                        'sector_score', 'sector_phase_score', 'sector_win_rate',
                        'volume_trend', 'price_stability', 'market_hour']
        
        importance = dict(zip(feature_names, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "model": model,
            "sample_size": len(X),
            "top_features": top_features
        }
    except Exception as e:
        print(f"ML training error: {e}")
        return None

def predict_win_probability(model_data, features):
    """Predict win probability using enhanced model"""
    if model_data is None or features is None:
        return 50
    
    try:
        model = model_data['model']
        X = np.array([[
            features.get('rsi', 50),
            features.get('volatility', 2),
            features.get('vol_ratio', 1),
            features.get('price_position', 0.5),
            features.get('roc_5', 0),
            features.get('roc_20', 0),
            features.get('dist_ema20', 0),
            features.get('dist_ema50', 0),
            features.get('atr_pct', 2),
            features.get('sector_score', 0),
            features.get('sector_phase_score', 0),
            features.get('sector_win_rate', 0.5),
            features.get('volume_trend', 1),
            features.get('price_stability', 0.02),
            features.get('market_hour', 10)
        ]])
        
        prob = model.predict_proba(X)[0][1] * 100
        return round(prob, 1)
    except:
        return 50

# ============================================================
# 🎯 IMPROVEMENT #5: ADVANCED RISK MANAGEMENT
# ============================================================

def calculate_kelly_size(win_rate, avg_win, avg_loss, capital, aggressive=False):
    """Enhanced Kelly Criterion"""
    try:
        if avg_loss == 0 or win_rate == 0:
            return capital * 0.01
        
        win_rate_dec = win_rate / 100
        loss_rate = 1 - win_rate_dec
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        
        kelly_fraction = (b * win_rate_dec - loss_rate) / b
        
        # Use quarter Kelly for conservative, half Kelly for aggressive
        multiplier = 0.5 if aggressive else 0.25
        kelly_fraction = max(0.005, min(0.03, kelly_fraction * multiplier))
        
        return capital * kelly_fraction
    except:
        return capital * 0.01

def calculate_advanced_position_size(entry_price, atr, capital, regime, memory):
    """Advanced position sizing with regime and volatility"""
    try:
        # Regime-based risk
        regime_risk = {
            "STRONG_BULL": 2.0,
            "BULL": 1.5,
            "SIDEWAYS": 0.75,
            "BEAR": 0.5,
            "STRONG_BEAR": 0.25
        }
        
        risk_percent = regime_risk.get(regime, 1.0)
        
        # Volatility adjustment
        volatility_pct = (atr / entry_price) * 100
        if volatility_pct > 5:  # High volatility
            risk_percent *= 0.7
        elif volatility_pct < 1.5:  # Low volatility
            risk_percent *= 1.2
        
        # Recent performance adjustment
        total_trades = memory.get('total_wins', 0) + memory.get('total_losses', 0)
        if total_trades >= 20:
            recent_trades = memory.get('completed_trades', [])[-10:]
            recent_wins = sum(1 for t in recent_trades if t.get('outcome') == 'WIN')
            recent_wr = (recent_wins / len(recent_trades)) if recent_trades else 0.5
            
            # Boost if hot streak, reduce if cold streak
            if recent_wr > 0.7:
                risk_percent *= 1.2
            elif recent_wr < 0.3:
                risk_percent *= 0.6
        
        # Cap max risk
        risk_percent = min(risk_percent, 2.5)
        
        # Calculate position
        risk_amount = capital * (risk_percent / 100)
        stop_distance = atr * 1.5
        
        if stop_distance <= 0:
            return {"shares": 0, "value": 0, "risk": 0, "risk_pct": 0}
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * entry_price
        actual_risk = shares * stop_distance
        
        # Max 15% per position
        max_position = capital * 0.15
        if position_value > max_position:
            shares = int(max_position / entry_price)
            position_value = shares * entry_price
            actual_risk = shares * stop_distance
        
        return {
            "shares": shares,
            "value": round(position_value, 2),
            "risk": round(actual_risk, 2),
            "risk_pct": round((actual_risk / capital) * 100, 2)
        }
    except:
        return {"shares": 0, "value": 0, "risk": 0, "risk_pct": 0}

def check_portfolio_heat(memory, new_risk):
    """Check total portfolio risk"""
    active = memory.get('pending_evaluations', {})
    current_risk = sum(p.get('risk_percent', 0) for p in active.values())
    total_risk = current_risk + new_risk
    
    return {
        "current_heat": round(current_risk, 2),
        "total_heat": round(total_risk, 2),
        "allowed": total_risk <= MAX_PORTFOLIO_HEAT,
        "remaining_capacity": max(0, MAX_PORTFOLIO_HEAT - current_risk)
    }

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
        
        # Additional metrics
        rsi = get_rsi(close, 14).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100
        
        if curr > ema20 > ema50 > ema200 and change_20d > 3 and rsi > 60:
            regime = "STRONG_BULL"
            trade_allowed = True
        elif curr > ema20 > ema50 and change_20d > 0:
            regime = "BULL"
            trade_allowed = True
        elif curr < ema50 and change_20d < -3:
            regime = "STRONG_BEAR"
            trade_allowed = False
        elif curr < ema50:
            regime = "BEAR"
            trade_allowed = False
        elif volatility > 2:
            regime = "VOLATILE"
            trade_allowed = False
        else:
            regime = "SIDEWAYS"
            trade_allowed = False
        
        return {
            "regime": regime,
            "trade_allowed": trade_allowed,
            "change_20d": round(change_20d, 2),
            "price": round(curr, 2),
            "rsi": round(rsi, 1),
            "volatility": round(volatility, 2)
        }
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
# 📰 SENTIMENT ANALYSIS (Simplified)
# ============================================================

POSITIVE_WORDS = ['growth', 'profit', 'surge', 'gain', 'strong', 'bullish', 'beat', 'record', 'upgrade', 'buy']
NEGATIVE_WORDS = ['loss', 'fall', 'decline', 'weak', 'bearish', 'miss', 'downgrade', 'sell', 'concern', 'risk']

def get_sentiment(ticker_name):
    try:
        query = quote(f"{ticker_name} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, 'xml')
        items = soup.find_all('item')[:10]
        
        if not items:
            return {"sentiment": "NEUTRAL", "score": 0}
        
        total_score = 0
        count = 0
        for item in items:
            title = item.find('title').text.lower() if item.find('title') else ""
            if ticker_name.lower() in title:
                pos = sum(1 for w in POSITIVE_WORDS if w in title)
                neg = sum(1 for w in NEGATIVE_WORDS if w in title)
                total_score += (pos - neg)
                count += 1
        
        if count == 0:
            return {"sentiment": "NEUTRAL", "score": 0}
        
        avg = total_score / count
        if avg >= 1: sentiment = "BULLISH"
        elif avg <= -1: sentiment = "BEARISH"
        else: sentiment = "NEUTRAL"
        
        return {"sentiment": sentiment, "score": round(avg, 2), "count": count}
    except:
        return {"sentiment": "NEUTRAL", "score": 0}

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
# 📤 TELEGRAM
# ============================================================

def send_telegram(msg, buttons=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        parts = [msg[i:i+3800] for i in range(0, len(msg), 3800)]
        
        for i, part in enumerate(parts):
            payload = {"chat_id": CHAT_ID, "text": part}
            if i == len(parts) - 1 and buttons:
                payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
            
            requests.post(url, json=payload, timeout=15)
            time.sleep(0.8)
    except Exception as e:
        print(f"Telegram error: {e}")

def create_pick_buttons(ticker):
    return [[
        {"text": "✅ Win", "callback_data": f"win_{ticker}"},
        {"text": "❌ Loss", "callback_data": f"loss_{ticker}"}
    ]]

def create_menu_buttons():
    return [
        [{"text": "📊 Memory", "callback_data": "memory_report"}],
        [{"text": "📈 Performance", "callback_data": "performance"}],
        [{"text": "🎯 Sector Analysis", "callback_data": "sectors"}]
    ]

# ============================================================
# 🎯 ULTIMATE SCANNER (v11.0)
# ============================================================

def ultimate_scan(ticker, nifty_close, memory, ml_model_data, sector_data, regime):
    try:
        df = yf.download(ticker, period="6mo", progress=False, timeout=10)
        if len(df) < 100: return None
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        vol = df['Volume'].squeeze()
        
        if close.iloc[-1] < 30 or vol.mean() < 50000: return None
        
        # Indicators
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
        
        # Entry type
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
        
        # Sector info
        ticker_clean = ticker.replace(".NS", "")
        sector = detect_sector(ticker_clean)
        
        # SCORING (max 200 points now)
        score = 0
        signals = []
        
        # 1. Trend (20 pts)
        score += 20 * memory["weights"]["trend"]
        signals.append("Trend OK")
        
        # 2. Volume (15 pts)
        if vol_ratio >= 3:
            score += 15
            signals.append("High Volume")
        elif vol_ratio >= 2:
            score += 10
        elif vol_ratio >= 1.5:
            score += 5
        
        # 3. RSI (15 pts)
        if 60 <= rsi.iloc[c] <= 65:
            score += 15
            signals.append("Optimal RSI")
        elif 55 <= rsi.iloc[c] <= 70:
            score += 10
        
        # 4. Entry type (20 pts)
        if is_early_breakout:
            score += 20 * memory["weights"]["early_breakout"]
        if is_pullback:
            score += 20 * memory["weights"]["pullback"]
        
        # 5. SECTOR ROTATION (25 pts) - NEW
        sector_bonus = get_sector_bonus(sector, sector_data)
        score += sector_bonus['bonus'] * memory["weights"]["sector_rotation"]
        if sector_bonus['bonus'] > 0:
            signals.append(sector_bonus['label'])
        elif sector_bonus['bonus'] < 0:
            return None  # Skip cold sectors
        
        # 6. Historical sector (10 pts)
        if sector in memory.get("sector_performance", {}):
            sec_data = memory["sector_performance"][sector]
            if sec_data.get('count', 0) >= 3:
                if sec_data.get('win_rate', 0) >= 65:
                    score += 10
                    signals.append(f"Sector WR: {sec_data['win_rate']}%")
                elif sec_data.get('win_rate', 0) <= 35:
                    return None
        
        # 7. BACKTEST SCORE (20 pts) - NEW
        backtest = get_backtest_score(ticker_clean, memory)
        if backtest:
            if backtest['win_rate'] >= 60:
                score += 20
                signals.append(f"Backtest WR: {backtest['win_rate']}%")
            elif backtest['win_rate'] >= 50:
                score += 10
            elif backtest['win_rate'] < 40:
                return None  # Skip poor historical performers
        
        # 8. Sentiment (15 pts)
        sentiment = get_sentiment(ticker_clean)
        if sentiment['sentiment'] == "BULLISH":
            score += 15
            signals.append("Bullish News")
        elif sentiment['sentiment'] == "BEARISH":
            return None
        
        # 9. RS Rating (10 pts)
        rs = 100
        if nifty_close is not None and len(nifty_close) > 63:
            stock_perf = close.iloc[-1] / close.iloc[-63]
            nifty_perf = nifty_close.iloc[-1] / nifty_close.iloc[-63]
            rs = (stock_perf / nifty_perf) * 100
            if rs > 110:
                score += 10
                signals.append(f"Strong RS")
        
        # 10. ML PREDICTION (30 pts) - Enhanced
        ml_features = extract_enhanced_features(df, sector_data, sector, memory)
        ml_probability = 50
        
        if ml_model_data and ml_features:
            ml_probability = predict_win_probability(ml_model_data, ml_features)
            
            if ml_probability < 45:
                return None
            
            if ml_probability >= 75:
                score += 30
                signals.append(f"ML: {ml_probability}%")
            elif ml_probability >= 65:
                score += 20
                signals.append(f"ML: {ml_probability}%")
            elif ml_probability >= 55:
                score += 10
        
        # Only accept high scores
        if score < 90: return None
        
        # ADVANCED POSITION SIZING - NEW
        position_data = calculate_advanced_position_size(curr, atr_val, TOTAL_CAPITAL, regime['regime'], memory)
        
        # PORTFOLIO HEAT CHECK - NEW
        heat_check = check_portfolio_heat(memory, position_data['risk_pct'])
        if not heat_check['allowed']:
            return None  # Skip if portfolio too hot
        
        # Targets
        recent_low = low.iloc[-5:].min()
        sl = max(recent_low * 0.99, ema20.iloc[c] * 0.98)
        risk = curr - sl
        if risk <= 0: return None
        
        risk_pct = (risk / curr) * 100
        if risk_pct > 4: return None
        
        tgt1 = curr + (risk * 1.5)
        tgt2 = curr + (risk * 2.5)
        tgt3 = curr + (risk * 4)  # New: Extended target
        
        return {
            "ticker": ticker_clean,
            "score": round(score, 1),
            "price": round(curr, 2),
            "rsi": round(rsi.iloc[c], 1),
            "vol_ratio": round(vol_ratio, 1),
            "entry_type": entry_type,
            "sector": sector,
            "sector_phase": sector_data.get('all_scores', {}).get(sector, {}).get('phase', 'N/A'),
            "sl": round(sl, 2),
            "tgt1": round(tgt1, 2),
            "tgt2": round(tgt2, 2),
            "tgt3": round(tgt3, 2),
            "risk_pct": round(risk_pct, 2),
            "rr_ratio": round((tgt1 - curr) / risk, 2),
            "rs_rating": round(rs, 0),
            "atr": round(atr_val, 2),
            "recommended_shares": position_data['shares'],
            "recommended_value": position_data['value'],
            "sentiment": sentiment['sentiment'],
            "ml_probability": ml_probability,
            "backtest_wr": backtest['win_rate'] if backtest else None,
            "backtest_trades": backtest['total_trades'] if backtest else 0,
            "portfolio_heat": heat_check['total_heat'],
            "features": ml_features,
            "signals": signals[:6]
        }
    except Exception as e:
        return None

# ============================================================
# 🎓 EVALUATE PICKS
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
                pct = ((current_price - entry_price) / entry_price) * 100
                outcome = "WIN" if pct > 2 else "LOSS" if pct < -1 else "NEUTRAL"
                actual_return = pct
            
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
# 🚀 MAIN
# ============================================================

def main():
    print("=" * 60)
    print(f"ARTHA {BOT_VERSION} - {BOT_TAGLINE}")
    print(datetime.now().strftime("%d %b %Y %I:%M %p"))
    print("=" * 60)
    
    memory = load_memory()
    print(f"\n[MEMORY] {memory['total_picks']} picks | {memory['total_wins']}W/{memory['total_losses']}L")
    
    send_telegram(f"⚡ ARTHA {BOT_VERSION} starting ultimate scan...")
    
    # Evaluate past picks
    print("\n[EVAL] Evaluating past picks...")
    memory, new_completed = evaluate_past_picks(memory)
    memory = learn_from_trades(memory)
    
    # Check confirmations - NEW
    print("\n[CONFIRM] Checking confirmations...")
    memory, confirmed_picks = check_confirmations(memory)
    print(f"  Confirmed: {len(confirmed_picks)} picks")
    
    # Train ML - Enhanced
    print("\n[ML] Training enhanced model...")
    ml_model_data = train_enhanced_ml(memory) if ML_AVAILABLE else None
    if ml_model_data:
        print(f"  ML trained on {ml_model_data['sample_size']} samples")
    
    # Market regime
    regime = detect_market_regime()
    print(f"\n[REGIME] {regime['regime']}")
    
    if not regime['trade_allowed']:
        msg = f"⚡ ARTHA {BOT_VERSION}\n"
        msg += f"📅 {datetime.now().strftime('%A, %d %b %Y')}\n\n"
        msg += "[⚠️ NO TRADES TODAY]\n"
        msg += f"Regime: {regime['regime']}\n"
        msg += f"Nifty: {regime['price']} ({regime['change_20d']:+.2f}% 20D)\n"
        msg += f"Volatility: {regime.get('volatility', 0):.2f}%\n\n"
        msg += "🧠 MEMORY STATUS:\n"
        wr = (memory['total_wins']/(memory['total_wins']+memory['total_losses'])*100) if (memory['total_wins']+memory['total_losses'])>0 else 0
        msg += f"WR: {wr:.1f}%\n"
        
        save_memory(memory)
        send_telegram(msg, create_menu_buttons())
        return
    
    # Get benchmark
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        nifty_close = nifty['Close'].squeeze()
    except:
        nifty_close = None
    
    # SECTOR ROTATION ANALYSIS - NEW
    print("\n[SECTOR] Analyzing sector rotation...")
    sector_data = analyze_sector_rotation()
    print(f"  Hot: {sector_data['hot_sectors']}")
    print(f"  Cold: {sector_data['cold_sectors']}")
    
    # Scan stocks
    print("\n[SCAN] Ultimate scanning...")
    tickers = get_all_tickers()
    results = []
    
    for i, t in enumerate(tickers):
        if (i+1) % 300 == 0:
            print(f"  {i+1}/{len(tickers)} | Found: {len(results)}")
        r = ultimate_scan(t, nifty_close, memory, ml_model_data, sector_data, regime)
        if r: results.append(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # ADD TO CONFIRMATION QUEUE (Instead of immediate picks) - NEW
    for pick in results[:5]:
        memory = add_to_confirmation_queue(
            memory, pick['ticker'], pick['price'],
            pick['sl'], pick['tgt1'], pick['tgt2'],
            pick['score'], pick['sector'], pick['entry_type']
        )
    
    # Save confirmed picks to pending evaluations
    for cp in confirmed_picks:
        pick_id = f"{cp['ticker']}_{datetime.now().strftime('%Y%m%d')}"
        memory["pending_evaluations"][pick_id] = {
            "ticker": cp['ticker'],
            "date": datetime.now().isoformat(),
            "entry_price": cp['confirmed_price'],
            "sl": cp['detected_sl'],
            "tgt1": cp['detected_tgt1'],
            "tgt2": cp['detected_tgt2'],
            "score": cp['score'],
            "sector": cp['sector'],
            "entry_type": cp['entry_type'],
            "rsi": 0
        }
        memory["total_picks"] += 1
    
    save_memory(memory)
    
    # Build comprehensive report
    today = datetime.now().strftime("%A, %d %b %Y")
    msg = "=" * 40 + "\n"
    msg += f"⚡ ARTHA {BOT_VERSION}\n"
    msg += f"{BOT_TAGLINE}\n"
    msg += f"📅 {today}\n"
    msg += "=" * 40 + "\n\n"
    
    # Brain status
    total_eval = memory['total_wins'] + memory['total_losses']
    wr = (memory['total_wins'] / total_eval * 100) if total_eval > 0 else 0
    
    msg += "🧠 [BRAIN STATUS v11.0]\n"
    msg += f"Picks: {memory['total_picks']} | WR: {wr:.1f}%\n"
    msg += f"ML Model: {'✅ Active' if ml_model_data else '⏳ Learning'}\n"
    if ml_model_data:
        msg += f"ML Sample: {ml_model_data['sample_size']}\n"
        top_feats = ml_model_data.get('top_features', [])
        if top_feats:
            msg += f"Top Feature: {top_feats[0][0]}\n"
    msg += "\n"
    
    # Market
    msg += f"[🌍 MARKET REGIME]\n"
    msg += f"Regime: {regime['regime']}\n"
    msg += f"Nifty: {regime['price']} ({regime['change_20d']:+.2f}%)\n"
    msg += f"Volatility: {regime.get('volatility', 0):.2f}%\n\n"
    
    # Sector Rotation - NEW
    msg += f"[🔥 SECTOR ROTATION]\n"
    msg += f"Hot: {', '.join(sector_data['hot_sectors'])}\n"
    msg += f"Warm: {', '.join(sector_data['warm_sectors'])}\n"
    msg += f"Cold: {', '.join(sector_data['cold_sectors'])}\n\n"
    
    # Confirmed picks (from previous scans)
    if confirmed_picks:
        msg += "=" * 40 + "\n"
        msg += "[✅ CONFIRMED PICKS (Ready to Trade)]\n"
        msg += "=" * 40 + "\n\n"
        
        for i, s in enumerate(confirmed_picks[:3]):
            grade = "A+" if s['score'] >= 150 else "A" if s['score'] >= 120 else "B+"
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/200)\n"
            msg += f"Sector: {s['sector']} | {s['entry_type']}\n"
            msg += f"Confirmations: {s.get('confirmations_passed', 2)}/3 ✅\n\n"
            
            msg += "💰 TRADE READY:\n"
            msg += f"  Entry: Rs.{s['confirmed_price']}\n"
            msg += f"  SL: Rs.{s['detected_sl']}\n"
            msg += f"  T1: Rs.{s['detected_tgt1']} | T2: Rs.{s['detected_tgt2']}\n\n"
            msg += "-" * 40 + "\n\n"
    
    # New picks in confirmation queue
    if results:
        msg += "=" * 40 + "\n"
        msg += "[⏳ NEW PICKS (In 2-Day Confirmation)]\n"
        msg += "=" * 40 + "\n\n"
        msg += "These are being validated for 2 days\n"
        msg += "Will appear as CONFIRMED if they hold\n\n"
        
        for i, s in enumerate(results[:5]):
            grade = "A+" if s['score'] >= 150 else "A" if s['score'] >= 120 else "B+"
            msg += f"#{i+1} {s['ticker']} | {grade} ({s['score']}/200)\n"
            msg += f"🏢 {s['sector']} ({s['sector_phase']})\n"
            msg += f"📊 RSI: {s['rsi']} | Vol: {s['vol_ratio']}x\n"
            msg += f"🧠 ML: {s['ml_probability']}%\n"
            if s['backtest_wr']:
                msg += f"📈 Backtest WR: {s['backtest_wr']}% ({s['backtest_trades']} trades)\n"
            msg += f"💰 Price: Rs.{s['price']}\n"
            msg += f"📉 Sentiment: {s['sentiment']}\n"
            msg += f"💼 Position: {s['recommended_shares']} shares (Rs.{s['recommended_value']:,})\n"
            msg += f"🔥 Portfolio Heat: {s['portfolio_heat']}%\n"
            if s['signals']:
                msg += f"✨ {', '.join(s['signals'])}\n"
            msg += "-" * 40 + "\n\n"
    else:
        msg += "[❌ NO PICKS TODAY]\n"
        msg += "All stocks filtered by strict v11.0 criteria\n\n"
    
    # Portfolio status
    heat_check = check_portfolio_heat(memory, 0)
    msg += f"[🎯 PORTFOLIO STATUS]\n"
    msg += f"Current Heat: {heat_check['current_heat']}%\n"
    msg += f"Max Allowed: {MAX_PORTFOLIO_HEAT}%\n"
    msg += f"Remaining: {heat_check['remaining_capacity']}%\n\n"
    
    msg += "=" * 40 + "\n"
    msg += f"{BOT_NAME} {BOT_VERSION}"
    
    # Send with buttons
    buttons = []
    if confirmed_picks:
        for pick in confirmed_picks[:3]:
            buttons.extend(create_pick_buttons(pick['ticker']))
    buttons.extend(create_menu_buttons())
    
    send_telegram(msg, buttons)
    print("\n[DONE] v11.0 scan complete!")

if __name__ == "__main__":
    main()
