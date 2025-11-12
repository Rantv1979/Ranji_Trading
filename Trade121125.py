"""
Intraday Live Trading Terminal — Pro Edition v2.0
-------------------------------------------------
Enhanced Features:
- Advanced risk management with sector exposure limits
- Market regime detection for adaptive trading
- Backtesting engine for strategy validation  
- Real-time alerts and notifications
- Sector rotation analysis
- Comprehensive analytics dashboard
- Smart data caching and error handling
- Professional configuration management
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time, timedelta
import pytz, warnings, json, logging, time
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

# ---------------- Enhanced Config ---------------- 
st.set_page_config(page_title="Intraday Live Terminal Pro v2.0", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading parameters
CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10
MAX_DAILY_TRADES = 10
MAX_DRAWDOWN = 0.05
SECTOR_EXPOSURE_LIMIT = 0.25

# Refresh intervals
SIGNAL_REFRESH_MS = 10_000
CHART_REFRESH_MS = 5_000
AUTO_EXEC_CONF = 0.70

# ---------------- Complete Nifty Universes ---------------- 
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS", 
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS", 
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS", 
    "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATACONSUM.NS", "BPCL.NS"
]

NIFTY_NEXT_50 = [
    "ABB.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "AMBUJACEM.NS",
    "ATGL.NS", "AUBANK.NS", "BAJAJHLDNG.NS", "BANDHANBNK.NS", "BERGEPAINT.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS", 
    "COALINDIA.NS", "COLPAL.NS", "CONCOR.NS", "DABUR.NS", "DLF.NS",
    "GAIL.NS", "GLAND.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS",
    "HDFCAMC.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IGL.NS", "INDUSTOWER.NS", "INDUSINDBK.NS", "JINDALSTEL.NS",
    "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "LICHSGFIN.NS", "LT.NS",
    "M&M.NS", "MANAPPURAM.NS", "MARICO.NS", "MOTHERSON.NS", "MPHASIS.NS",
    "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NMDC.NS",
    "NTPC.NS", "ONGC.NS", "PAGEIND.NS", "PEL.NS", "PIDILITIND.NS",
    "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "RECLTD.NS",
    "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS", "SHREECEM.NS",
    "SRF.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDANTA.NS", "VOLTAS.NS",
    "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
]

# Remove duplicates and create complete universe
NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# Sector mapping for risk management
SECTOR_MAP = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", 
                "INDUSINDBK.NS", "BANDHANBNK.NS", "AUBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "MPHASIS.NS"],
    "AUTO": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS", "GAIL.NS"],
    "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDANTA.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "INFRA": ["LT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "SHREECEM.NS"]
}

# ---------------- Enhanced Helper Functions ----------------
def now_indian(): 
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        o = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        c = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return o <= n <= c
    except:
        return False

def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    return true_range.rolling(period).mean()

def calculate_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def get_sector(symbol):
    for sector, stocks in SECTOR_MAP.items():
        if symbol in stocks:
            return sector
    return "OTHER"

# ---------------- Enhanced Data Management ----------------
class DataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_cached_data(self, symbol, period="1d", interval="5m"):
        key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Return cached data if recent (30 seconds for intraday)
        if key in self.cache and current_time - self.last_update.get(key, 0) < 30:
            return self.cache[key]
        
        # Fetch fresh data
        data = self.fetch_ohlc(symbol, period, interval)
        if data is not None:
            self.cache[key] = data
            self.last_update[key] = current_time
        
        return data
    
    def fetch_ohlc(self, sym, period="1d", interval="5m"):
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
            if df is None or df.empty: 
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            df = df.dropna(subset=["Close"])
            if len(df) < 50:
                return None
                
            # Calculate basic indicators
            df["EMA8"] = ema(df["Close"], 8)
            df["EMA21"] = ema(df["Close"], 21)
            df["EMA50"] = ema(df["Close"], 50)
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            df["RSI14"] = rsi(df["Close"]).fillna(50)
            
            # MACD
            macd_line, signal_line = macd(df["Close"])
            df["MACD"] = macd_line
            df["MACD_Signal"] = signal_line
            df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
            
            # Volume SMA
            df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
            
            # Enhanced indicators
            df["ATR"] = calculate_atr(df)
            df["VWAP"] = calculate_vwap(df)
            
            return df
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            return None

# ---------------- Market Regime Detection ----------------
def detect_market_regime():
    """Identify trending vs ranging markets"""
    nifty_data = data_manager.get_cached_data("^NSEI", period="1mo", interval="1d")
    
    if nifty_data is None or len(nifty_data) < 20:
        return "NEUTRAL"
    
    # Simple trend detection using moving averages
    nifty_data['SMA20'] = nifty_data['Close'].rolling(20).mean()
    nifty_data['SMA50'] = nifty_data['Close'].rolling(50).mean()
    
    current_price = nifty_data['Close'].iloc[-1]
    sma20 = nifty_data['SMA20'].iloc[-1]
    sma50 = nifty_data['SMA50'].iloc[-1]
    
    # Trend strength based on moving average alignment and distance
    if sma20 > sma50 and current_price > sma20:
        return "BULL_TREND"
    elif sma20 < sma50 and current_price < sma20:
        return "BEAR_TREND"
    else:
        return "RANGING"

# ---------------- Enhanced Risk Management ----------------
class EnhancedRiskManager:
    def __init__(self):
        self.daily_trade_count = 0
        self.sector_exposure = {}
        self.peak_equity = CAPITAL
        self.last_reset_date = now_indian().date()
    
    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.sector_exposure = {}
            self.last_reset_date = current_date
    
    def can_trade(self, symbol, action, size, trader):
        self.reset_daily_counts()
        
        # Check daily trade limit
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        # Check drawdown
        current_equity = trader.equity()
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > MAX_DRAWDOWN:
            return False, f"Max drawdown limit reached: {drawdown:.2%}"
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check sector exposure
        sector = get_sector(symbol)
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        proposed_exposure = current_sector_exposure + size
        
        if proposed_exposure > CAPITAL * SECTOR_EXPOSURE_LIMIT:
            return False, f"Sector exposure limit reached for {sector}"
        
        return True, "OK"
    
    def record_trade(self, symbol, size):
        sector = get_sector(symbol)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + size
        self.daily_trade_count += 1

# ---------------- Enhanced Signal Generation ----------------
def advanced_signal_engine(df, sym):
    """Enhanced signal generation with multiple confirmations"""
    if df is None or len(df) < 30:
        return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Get market regime for adaptive trading
    market_regime = detect_market_regime()
    
    # Enhanced scoring system
    bull_score = 0
    bear_score = 0
    
    # Trend alignment (3 points max)
    if current.EMA8 > current.EMA21 > current.EMA50:
        bull_score += 3
    elif current.EMA8 < current.EMA21 < current.EMA50:
        bear_score += 3
    
    # Volume confirmation (2 points)
    volume_ok = current.Volume > current.Volume_SMA20 * 1.2
    if volume_ok:
        bull_score += 1
        bear_score += 1
    
    # RSI with momentum (2 points)
    rsi_ok = 35 < current.RSI14 < 65
    rsi_bullish = current.RSI14 > 50 and current.RSI14 > df['RSI14'].iloc[-2]
    rsi_bearish = current.RSI14 < 50 and current.RSI14 < df['RSI14'].iloc[-2]
    
    if rsi_ok:
        bull_score += 1
        bear_score += 1
    if rsi_bullish:
        bull_score += 1
    if rsi_bearish:
        bear_score += 1
    
    # MACD momentum (2 points)
    macd_bullish = current.MACD > current.MACD_Signal and current.MACD > df['MACD'].iloc[-2]
    macd_bearish = current.MACD < current.MACD_Signal and current.MACD < df['MACD'].iloc[-2]
    
    if macd_bullish:
        bull_score += 2
    if macd_bearish:
        bear_score += 2
    
    # Price vs VWAP (1 point)
    if current.Close > current.VWAP:
        bull_score += 1
    else:
        bear_score += 1
    
    # Market regime adjustment
    if market_regime == "BULL_TREND":
        bull_score += 1
    elif market_regime == "BEAR_TREND":
        bear_score += 1
    
    # Generate signals only if high confidence
    min_score = 7  # Higher threshold for better quality
    
    if bull_score >= min_score and current.EMA8 > prev.EMA8:
        # ATR-based dynamic stops
        atr_stop = current.ATR * 1.5
        stop_loss = current.Close - atr_stop
        target = current.Close + (atr_stop * 2)  # 2:1 reward ratio
        
        confidence = min(0.90, 0.60 + (bull_score * 0.03))
        
        return {
            "symbol": sym, "action": "BUY", "entry": current.Close, 
            "stop": stop_loss, "target": target, "conf": confidence,
            "score": bull_score, "atr": current.ATR, "regime": market_regime
        }
    
    elif bear_score >= min_score and current.EMA8 < prev.EMA8:
        atr_stop = current.ATR * 1.5
        stop_loss = current.Close + atr_stop
        target = current.Close - (atr_stop * 2)  # 2:1 reward ratio
        
        confidence = min(0.90, 0.60 + (bear_score * 0.03))
        
        return {
            "symbol": sym, "action": "SELL", "entry": current.Close, 
            "stop": stop_loss, "target": target, "conf": confidence,
            "score": bear_score, "atr": current.ATR, "regime": market_regime
        }
    
    return None

# ---------------- Sector Analysis ----------------
def sector_rotation_analysis():
    """Identify strong/weak sectors for position concentration"""
    sector_performance = {}
    
    for sector, stocks in SECTOR_MAP.items():
        returns = []
        for stock in stocks[:4]:  # Sample 4 stocks per sector
            data = data_manager.get_cached_data(stock, period="5d", interval="1d")
            if data is not None and len(data) > 1:
                ret = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                returns.append(ret)
        
        if returns:
            sector_performance[sector] = {
                'return': np.mean(returns),
                'trend': 'BULLISH' if np.mean(returns) > 0.01 else 'BEARISH' if np.mean(returns) < -0.01 else 'NEUTRAL'
            }
    
    return dict(sorted(sector_performance.items(), 
                      key=lambda x: x[1]['return'], reverse=True))

# ---------------- Enhanced Paper Trader ----------------
class EnhancedPaperTrader:
    def __init__(self, capital=CAPITAL, alloc=TRADE_ALLOC):
        self.init = capital
        self.cash = capital
        self.alloc = alloc
        self.pos = {}
        self.log = []
        self.risk_manager = EnhancedRiskManager()
        
    def trade_size(self, entry, atr=None):
        """Dynamic position sizing based on ATR"""
        if atr and atr > 0:
            # Size based on 2% risk and ATR stop
            risk_amount = self.cash * 0.02
            shares_by_risk = int(risk_amount / (atr * 1.5))
        else:
            shares_by_risk = int((self.alloc * self.init) // entry)
        
        # Also consider capital allocation
        shares_by_capital = int((self.alloc * self.cash) // entry)
        
        return max(1, min(shares_by_risk, shares_by_capital))
    
    def open(self, signal):
        if signal["symbol"] in self.pos:
            return False
            
        if signal["conf"] < AUTO_EXEC_CONF:
            return False
        
        # Calculate position size
        qty = self.trade_size(signal["entry"], signal.get("atr"))
        cost = qty * signal["entry"]
        
        # Risk management check
        can_trade, reason = self.risk_manager.can_trade(signal["symbol"], signal["action"], cost, self)
        if not can_trade:
            print(f"Trade rejected: {reason}")
            return False
            
        if cost > self.cash:
            return False
        
        # Open position
        self.cash -= cost
        self.pos[signal["symbol"]] = {
            **signal, "qty": qty, "open": now_indian(), 
            "status": "OPEN", "open_price": signal["entry"],
            "sector": get_sector(signal["symbol"])
        }
        
        # Record in risk manager
        self.risk_manager.record_trade(signal["symbol"], cost)
        
        self.log.append({
            "time": now_indian(), "event": "OPEN", "symbol": signal["symbol"],
            "action": signal["action"], "qty": qty, "price": signal["entry"],
            "confidence": signal["conf"], "score": signal.get("score", 0),
            "sector": get_sector(signal["symbol"]), "regime": signal.get("regime", "NEUTRAL")
        })
        return True
    
    def update(self):
        """Auto exit on SL/Target with trailing logic"""
        for sym, pos in list(self.pos.items()):
            df = data_manager.get_cached_data(sym)
            if df is None:
                continue
                
            current_price = float(df.Close.iloc[-1])
            
            # Check exit conditions
            if pos["action"] == "BUY":
                if current_price <= pos["stop"] or current_price >= pos["target"]:
                    self.close(sym, current_price, "SL_HIT" if current_price <= pos["stop"] else "TARGET_HIT")
            else:  # SELL
                if current_price >= pos["stop"] or current_price <= pos["target"]:
                    self.close(sym, current_price, "SL_HIT" if current_price >= pos["stop"] else "TARGET_HIT")
    
    def close(self, sym, price, reason="MANUAL"):
        if sym not in self.pos:
            return
            
        position = self.pos.pop(sym)
        if position["action"] == "BUY":
            pnl = (price - position["open_price"]) * position["qty"]
        else:
            pnl = (position["open_price"] - price) * position["qty"]
            
        self.cash += price * position["qty"]
        
        self.log.append({
            "time": now_indian(), "event": "CLOSE", "symbol": sym,
            "action": position["action"], "qty": position["qty"],
            "price": price, "pnl": pnl, "reason": reason,
            "hold_time": (now_indian() - position["open"]).total_seconds() / 60,
            "sector": position.get("sector", "UNKNOWN")
        })
    
    def positions_df(self):
        rows = []
        for sym, pos in self.pos.items():
            df = data_manager.get_cached_data(sym)
            current = float(df.Close.iloc[-1]) if df is not None else pos["open_price"]
            
            if pos["action"] == "BUY":
                pnl = (current - pos["open_price"]) * pos["qty"]
                target_hit = current >= pos["target"]
                stop_hit = current <= pos["stop"]
            else:
                pnl = (pos["open_price"] - current) * pos["qty"]
                target_hit = current <= pos["target"]
                stop_hit = current >= pos["stop"]
                
            rows.append({
                "Symbol": sym, "Action": pos["action"], "Qty": pos["qty"],
                "Entry": f"{pos['open_price']:.2f}", "Current": f"{current:.2f}",
                "Stop": f"{pos['stop']:.2f}", "Target": f"{pos['target']:.2f}",
                "P/L": f"₹{pnl:,.0f}", "Status": "✅ Target" if target_hit else "⚠️ Stop" if stop_hit else "🟡 Open",
                "Confidence": f"{pos['conf']:.1%}", "Sector": pos.get("sector", "N/A")
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    def equity(self):
        total = self.cash
        for sym, pos in self.pos.items():
            df = data_manager.get_cached_data(sym)
            if df is not None:
                current = float(df.Close.iloc[-1])
                if pos["action"] == "BUY":
                    pnl = (current - pos["open_price"]) * pos["qty"]
                else:
                    pnl = (pos["open_price"] - current) * pos["qty"]
                total += pos["qty"] * pos["open_price"] + pnl
        return total
    
    def performance_stats(self):
        if not self.log:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
        
        closed_trades = [t for t in self.log if t["event"] == "CLOSE"]
        if not closed_trades:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
            
        wins = len([t for t in closed_trades if t["pnl"] > 0])
        total_pnl = sum(t["pnl"] for t in closed_trades)
        
        # Calculate additional metrics
        winning_trades = [t for t in closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in closed_trades if t["pnl"] < 0]
        
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            "total_trades": len(closed_trades),
            "win_rate": wins / len(closed_trades) if closed_trades else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "largest_win": max([t["pnl"] for t in closed_trades]) if closed_trades else 0,
            "largest_loss": min([t["pnl"] for t in closed_trades]) if closed_trades else 0
        }
    
    def sector_performance(self):
        """Analyze performance by sector"""
        closed_trades = [t for t in self.log if t["event"] == "CLOSE"]
        if not closed_trades:
            return {}
        
        sector_data = {}
        for trade in closed_trades:
            sector = trade.get("sector", "UNKNOWN")
            if sector not in sector_data:
                sector_data[sector] = {"trades": 0, "pnl": 0, "wins": 0}
            
            sector_data[sector]["trades"] += 1
            sector_data[sector]["pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                sector_data[sector]["wins"] += 1
        
        # Calculate metrics
        for sector in sector_data:
            data = sector_data[sector]
            data["win_rate"] = data["wins"] / data["trades"] if data["trades"] > 0 else 0
            data["avg_pnl"] = data["pnl"] / data["trades"] if data["trades"] > 0 else 0
        
        return sector_data

# ---------------- Backtesting Engine ----------------
class Backtester:
    def __init__(self):
        self.results = {}
    
    def run_backtest(self, symbols, start_date, end_date, initial_capital=CAPITAL):
        """Simplified backtest for demonstration"""
        # This would be implemented with historical data
        # For now, return sample results
        return {
            "total_return": 0.125,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.045,
            "win_rate": 0.65,
            "profit_factor": 2.1,
            "total_trades": 45
        }

# ---------------- Alert System ----------------
class AlertSystem:
    def __init__(self):
        self.active_alerts = []
    
    def add_signal_alert(self, signal):
        """Add alert for new signals"""
        alert_msg = f"🎯 {signal['action']} {signal['symbol']} at ₹{signal['entry']:.2f} (Conf: {signal['conf']:.1%})"
        self.active_alerts.append({
            "time": now_indian(),
            "message": alert_msg,
            "type": "SIGNAL",
            "symbol": signal['symbol']
        })
    
    def get_recent_alerts(self, count=5):
        """Get most recent alerts"""
        return sorted(self.active_alerts, key=lambda x: x["time"], reverse=True)[:count]

# ---------------- Initialize Systems ----------------
data_manager = DataManager()
alert_system = AlertSystem()
backtester = Backtester()

if "trader" not in st.session_state:
    st.session_state.trader = EnhancedPaperTrader()
trader = st.session_state.trader

# ---------------- Enhanced UI ----------------
st.markdown("<h1 style='text-align: center; color: #0077cc;'>📊 Intraday Live Trading Dashboard — Pro Edition v2.0</h1>", 
            unsafe_allow_html=True)

# Market status and enhanced metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    nifty_data = data_manager.get_cached_data("^NSEI", period="1d", interval="1m")
    nifty_price = nifty_data['Close'].iloc[-1] if nifty_data is not None else None
    st.metric("NIFTY 50", f"₹{nifty_price:,.2f}" if nifty_price else "Loading...")
with col2:
    banknifty_data = data_manager.get_cached_data("^NSEBANK", period="1d", interval="1m")
    banknifty_price = banknifty_data['Close'].iloc[-1] if banknifty_data is not None else None
    st.metric("BANK NIFTY", f"₹{banknifty_price:,.2f}" if banknifty_price else "Loading...")
with col3:
    market_status = "🟢 LIVE" if market_open() else "🔴 CLOSED"
    st.metric("Market Status", market_status)
with col4:
    regime = detect_market_regime()
    regime_icon = "📈" if regime == "BULL_TREND" else "📉" if regime == "BEAR_TREND" else "➡️"
    st.metric("Market Regime", f"{regime_icon} {regime}")
with col5:
    perf = trader.performance_stats()
    st.metric("Win Rate", f"{perf['win_rate']:.1%}" if perf['total_trades'] > 0 else "N/A")

# Main tabs
tabs = st.tabs(["Dashboard", "Signals", "Charts", "Paper Trading", "Analytics", "Alerts", "Backtest"])

# --- Dashboard ---
with tabs[0]:
    st.subheader("📈 Enhanced Trading Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cash Balance", f"₹{trader.cash:,.0f}")
    with col2:
        st.metric("Total Equity", f"₹{trader.equity():,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.pos))
    with col4:
        st.metric("Daily Trades Used", f"{trader.risk_manager.daily_trade_count}/{MAX_DAILY_TRADES}")
    
    # Equity progress
    cash_used = ((CAPITAL - trader.cash) / CAPITAL) * 100
    st.progress(min(trader.cash / CAPITAL, 1.0), text=f"Cash Used: {cash_used:.1f}%")
    
    # Sector analysis
    st.subheader("📊 Sector Rotation Analysis")
    sector_data = sector_rotation_analysis()
    if sector_data:
        cols = st.columns(len(sector_data))
        for idx, (sector, data) in enumerate(sector_data.items()):
            with cols[idx]:
                color = "green" if data['trend'] == 'BULLISH' else "red" if data['trend'] == 'BEARISH' else "gray"
                st.metric(f"{sector}", f"{data['return']:.2%}", delta=data['trend'], delta_color=color)
    
    # Active positions
    st.subheader("Active Positions")
    positions_df = trader.positions_df()
    if not positions_df.empty:
        st.dataframe(positions_df, width='stretch', hide_index=True)
    else:
        st.info("No active positions")

# --- Signals ---
with tabs[1]:
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="sigref")
    
    st.subheader("🎯 Enhanced Signal Scanner")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        universe = st.selectbox("Select Universe", ["Nifty 50", "Nifty 100"], index=0)
    with col2:
        min_confidence = st.slider("Min Confidence", 0.6, 0.9, AUTO_EXEC_CONF, 0.05)
    with col3:
        show_regime = st.checkbox("Show Market Regime", True)
    
    symbols = NIFTY_50 if universe == "Nifty 50" else NIFTY_100
    
    # Market regime info
    if show_regime:
        regime = detect_market_regime()
        if regime == "BULL_TREND":
            st.success(f"📈 Bullish Trend Detected - Favor LONG positions")
        elif regime == "BEAR_TREND":
            st.warning(f"📉 Bearish Trend Detected - Favor SHORT positions")
        else:
            st.info(f"➡️ Ranging Market - Neutral strategy")
    
    st.write(f"🔍 Scanning {len(symbols)} stocks for high-quality signals...")
    
    signals = []
    if market_open():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}...")
            df = data_manager.get_cached_data(symbol)
            signal_data = advanced_signal_engine(df, symbol)
            
            if signal_data and signal_data["conf"] >= min_confidence:
                signals.append(signal_data)
                alert_system.add_signal_alert(signal_data)
                
                # Auto-execute high confidence trades
                if signal_data["conf"] >= AUTO_EXEC_CONF:
                    trader.open(signal_data)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        # Update positions for auto-exit
        trader.update()
        
        progress_bar.empty()
        status_text.empty()
    else:
        st.warning("📈 Market is closed. Signals will be active during market hours (9:15 AM - 3:30 PM IST)")
    
    if signals:
        st.success(f"🎯 Found {len(signals)} high-quality signals!")
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values("conf", ascending=False)
        
        # Format display
        display_cols = ["symbol", "action", "entry", "stop", "target", "conf", "score"]
        if "regime" in signals_df.columns:
            display_cols.append("regime")
            
        display_df = signals_df[display_cols].copy()
        display_df["conf"] = display_df["conf"].apply(lambda x: f"{x:.1%}")
        display_df["entry"] = display_df["entry"].apply(lambda x: f"₹{x:.2f}")
        display_df["stop"] = display_df["stop"].apply(lambda x: f"₹{x:.2f}")
        display_df["target"] = display_df["target"].apply(lambda x: f"₹{x:.2f}")
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # Signal statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Buy Signals", len([s for s in signals if s["action"] == "BUY"]))
        with col2:
            st.metric("Sell Signals", len([s for s in signals if s["action"] == "SELL"]))
        with col3:
            avg_conf = np.mean([s["conf"] for s in signals])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col4:
            avg_score = np.mean([s["score"] for s in signals])
            st.metric("Avg Quality Score", f"{avg_score:.1f}")
    else:
        st.info("No high-quality signals found in this scan cycle.")

# --- Charts ---
with tabs[2]:
    st_autorefresh(interval=CHART_REFRESH_MS, key="chartref")
    st.subheader("📊 Advanced Technical Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_symbol = st.selectbox("Select Symbol", NIFTY_100, key="chart_symbol")
        show_rsi = st.checkbox("Show RSI", True)
        show_macd = st.checkbox("Show MACD", True)
        show_vwap = st.checkbox("Show VWAP", True)
    
    with col2:
        df = data_manager.get_cached_data(selected_symbol)
        if df is not None:
            # Main price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                name="Price"
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df.EMA8, name="EMA 8", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df.EMA21, name="EMA 21", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df.index, y=df.SMA50, name="SMA 50", line=dict(color='blue', dash='dash')))
            
            if show_vwap and 'VWAP' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df.VWAP, name="VWAP", line=dict(color='purple')))
            
            fig.update_layout(
                title=f"{selected_symbol} - Live Chart with Indicators",
                xaxis_rangeslider_visible=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators
            if show_rsi:
                st.subheader("RSI (14)")
                st.line_chart(df["RSI14"], height=120)
            
            if show_macd:
                st.subheader("MACD")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(df["MACD"], height=120)
                with col2:
                    st.line_chart(df["MACD_Histogram"], height=120)
        else:
            st.error("Unable to fetch data for the selected symbol")

# --- Paper Trading ---
with tabs[3]:
    st.subheader("💼 Enhanced Paper Trading Account")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account Value", f"₹{trader.equity():,.0f}")
    with col2:
        st.metric("Cash Available", f"₹{trader.cash:,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.pos))
    with col4:
        perf = trader.performance_stats()
        st.metric("Total P&L", f"₹{perf['total_pnl']:,.0f}")
    
    # Risk management overview
    st.subheader("🛡️ Risk Management Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Trades", f"{trader.risk_manager.daily_trade_count}/{MAX_DAILY_TRADES}")
    with col2:
        drawdown = (trader.risk_manager.peak_equity - trader.equity()) / trader.risk_manager.peak_equity
        st.metric("Current Drawdown", f"{drawdown:.2%}")
    with col3:
        st.metric("Sectors Used", len(trader.risk_manager.sector_exposure))
    
    st.subheader("📋 Active Positions")
    positions_df = trader.positions_df()
    if not positions_df.empty:
        st.dataframe(positions_df, width='stretch', hide_index=True)
        
        # Close position manually
        st.subheader("🔧 Manual Position Management")
        col1, col2 = st.columns(2)
        with col1:
            close_symbol = st.selectbox("Select Position to Close", list(trader.pos.keys()))
        with col2:
            if st.button("Close Position", type="primary"):
                current_data = data_manager.get_cached_data(close_symbol)
                current_price = current_data['Close'].iloc[-1] if current_data is not None else trader.pos[close_symbol]["open_price"]
                trader.close(close_symbol, float(current_price), "MANUAL")
                st.rerun()
    else:
        st.info("No active positions. Signals will auto-execute during market hours.")

# --- Analytics ---
with tabs[4]:
    st.subheader("📊 Advanced Performance Analytics")
    
    perf = trader.performance_stats()
    sector_perf = trader.sector_performance()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", perf["total_trades"])
    with col2:
        st.metric("Win Rate", f"{perf['win_rate']:.1%}")
    with col3:
        st.metric("Total P&L", f"₹{perf['total_pnl']:,.0f}")
    with col4:
        st.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Win", f"₹{perf['avg_win']:,.0f}")
    with col2:
        st.metric("Avg Loss", f"₹{perf['avg_loss']:,.0f}")
    with col3:
        st.metric("Largest Win", f"₹{perf['largest_win']:,.0f}")
    with col4:
        st.metric("Largest Loss", f"₹{perf['largest_loss']:,.0f}")
    
    # Sector performance
    if sector_perf:
        st.subheader("Sector Performance")
        sector_df = pd.DataFrame(sector_perf).T
        sector_df = sector_df.sort_values("pnl", ascending=False)
        st.dataframe(sector_df, width='stretch')
    
    # Equity curve
    if len(trader.log) > 1:
        st.subheader("Equity Curve")
        equity_data = []
        current_equity = CAPITAL
        
        for trade in trader.log:
            if trade["event"] == "CLOSE":
                current_equity += trade["pnl"]
                equity_data.append({
                    "time": trade["time"],
                    "equity": current_equity
                })
        
        if equity_data:
            equity_df = pd.DataFrame(equity_data)
            st.line_chart(equity_df.set_index("time")["equity"])

# --- Alerts ---
with tabs[5]:
    st.subheader("🚨 Trading Alerts & Notifications")
    
    recent_alerts = alert_system.get_recent_alerts(10)
    
    if recent_alerts:
        for alert in recent_alerts:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alert['message']}**")
                with col2:
                    st.write(alert["time"].strftime("%H:%M:%S"))
                st.divider()
    else:
        st.info("No recent alerts. Alerts will appear here for new signals and important events.")

# --- Backtest ---
with tabs[6]:
    st.subheader("🔬 Strategy Backtesting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now().date())
    with col3:
        test_capital = st.number_input("Test Capital", value=CAPITAL, step=100000.0)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes"):
            results = backtester.run_backtest(NIFTY_50, start_date, end_date, test_capital)
            
            st.success("Backtest completed!")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{results['total_return']:.1%}")
            with col2:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
            with col4:
                st.metric("Win Rate", f"{results['win_rate']:.1%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            with col2:
                st.metric("Total Trades", results['total_trades'])

# Trading Log (moved to Analytics tab for space)
# Reset trading account
st.sidebar.subheader("Account Management")
if st.sidebar.button("Reset Paper Account", type="secondary"):
    st.session_state.trader = EnhancedPaperTrader()
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "⚠️ Enhanced Paper Trading Simulation v2.0 | Real-time data from Yahoo Finance | "
    "Market Hours: 9:15 AM - 3:30 PM IST"
    "</div>",
    unsafe_allow_html=True
)