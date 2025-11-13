"""
Intraday Live Trading Terminal — Ultimate Pro Edition v6.0
----------------------------------------------------------
Enhanced Features:
- Fixed indices loading in Dashboard
- Improved data fetching reliability
- Added trending stocks analysis
- Fibonacci Retracement strategy (Golden Zone)
- Enhanced dashboard with better visualization
- Improved signal quality with multiple confirmations
- Better error handling and fallback mechanisms
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time, timedelta
import pytz, warnings, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
st.set_page_config(page_title="Intraday Terminal Pro v6.0", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading parameters
CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10
MAX_DAILY_TRADES = 8
MAX_DRAWDOWN = 0.05
SECTOR_EXPOSURE_LIMIT = 0.25

# Refresh intervals
SIGNAL_REFRESH_MS = 25_000
CHART_REFRESH_MS = 5_000
AUTO_EXEC_CONF = 0.70

# Data intervals fallback
CHART_INTERVALS = [("1d", "5m"), ("1d", "15m"), ("5d", "15m"), ("1d", "1m")]

# ---------------- Nifty Universe ----------------
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS", 
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS", 
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
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
    "SRF.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAPOWER.NS",
    "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "UPL.NS", "VOLTAS.NS",
    "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# Sector mapping
SECTOR_MAP = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "AUTO": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS"],
    "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "INFRA": ["LT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS"]
}

# ---------------- Core Functions ----------------
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        market_open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        market_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return market_open_time <= n <= market_close_time
    except:
        return False

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

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

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }

def fibonacci_golden_zone_signal(df):
    """Generate Fibonacci Golden Zone signals (0.5 - 0.618)"""
    if len(df) < 20:
        return None
    
    recent_high = df['High'].max()
    recent_low = df['Low'].min()
    
    fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
    current_price = df['Close'].iloc[-1]
    
    golden_zone_low = fib_levels['0.618']
    golden_zone_high = fib_levels['0.5']
    
    # Check if price is in golden zone
    if golden_zone_low <= current_price <= golden_zone_high:
        # Check for bullish reversal patterns
        if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and  # Green candle
            df['Close'].iloc[-2] < df['Open'].iloc[-2] and  # Previous red candle
            df['RSI14'].iloc[-1] < 60):  # RSI not overbought
            return "BUY"
        
        # Check for bearish reversal patterns
        elif (df['Close'].iloc[-1] < df['Open'].iloc[-1] and  # Red candle
              df['Close'].iloc[-2] > df['Open'].iloc[-2] and  # Previous green candle
              df['RSI14'].iloc[-1] > 40):  # RSI not oversold
            return "SELL"
    
    return None

# ---------------- Enhanced Data Manager ----------------
class DataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_cached_data(self, symbol, period="1d", interval="5m"):
        key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Return cached data if recent
        if key in self.cache and current_time - self.last_update.get(key, 0) < 30:
            return self.cache[key]
        
        # Fetch fresh data
        data = self.fetch_ohlc(symbol, period, interval)
        if data is not None:
            self.cache[key] = data
            self.last_update[key] = current_time
        
        return data
    
    def fetch_ohlc(self, symbol, period="1d", interval="5m"):
        """Robust data fetching with multiple fallback strategies"""
        for retry in range(3):  # Increased retries
            try:
                # Handle index symbols
                if symbol.startswith('^'):
                    ticker_symbol = symbol
                else:
                    ticker_symbol = symbol
                
                df = yf.download(ticker_symbol, period=period, interval=interval, progress=False, threads=False)
                if df is None or df.empty:
                    continue
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0) if df.columns.nlevels > 1 else df.columns
                
                df.columns = [str(col).upper() for col in df.columns]
                
                # Ensure required columns exist
                required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                df = df.rename(columns={
                    'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 
                    'CLOSE': 'Close', 'VOLUME': 'Volume'
                })
                
                df = df.dropna(subset=['Close']).copy()
                if len(df) < 10:  # Reduced minimum length requirement
                    continue
                
                # Calculate indicators
                df['EMA8'] = ema(df['Close'], 8)
                df['EMA21'] = ema(df['Close'], 21)
                df['EMA50'] = ema(df['Close'], 50)
                df['RSI14'] = rsi(df['Close'], 14).fillna(50)
                
                macd_line, signal_line = macd(df['Close'])
                df['MACD'] = macd_line
                df['MACD_Signal'] = signal_line
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
                # Volume and VWAP
                if 'Volume' in df.columns:
                    df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
                    df['VWAP'] = calculate_vwap(df)
                else:
                    df['VWAP'] = df['Close'].rolling(20).mean()
                
                # ATR for risk management
                df['ATR'] = calculate_atr(df)
                
                return df
                
            except Exception as e:
                print(f"Error fetching {symbol} (attempt {retry+1}): {e}")
                time.sleep(1)  # Add delay between retries
                continue
        
        return None

    def get_index_data(self, symbol):
        """Special method for index data with fallback"""
        try:
            # Try multiple period/interval combinations for indices
            intervals_to_try = [("1d", "5m"), ("1d", "15m"), ("2d", "30m")]
            
            for period, interval in intervals_to_try:
                data = self.get_cached_data(symbol, period, interval)
                if data is not None and len(data) > 0:
                    return data
            
            # Final fallback - try daily data
            return self.get_cached_data(symbol, "5d", "1d")
            
        except Exception as e:
            print(f"Error fetching index data for {symbol}: {e}")
            return None

# ---------------- Market Analysis ----------------
def detect_market_regime():
    """Detect current market trend"""
    nifty_data = data_manager.get_index_data("^NSEI")
    
    if nifty_data is None or len(nifty_data) < 10:
        return "NEUTRAL"
    
    nifty_data['SMA20'] = nifty_data['Close'].rolling(20).mean()
    nifty_data['SMA50'] = nifty_data['Close'].rolling(50).mean()
    
    current_price = nifty_data['Close'].iloc[-1]
    sma20 = nifty_data['SMA20'].iloc[-1] if not pd.isna(nifty_data['SMA20'].iloc[-1]) else current_price
    sma50 = nifty_data['SMA50'].iloc[-1] if not pd.isna(nifty_data['SMA50'].iloc[-1]) else current_price
    
    if sma20 > sma50 and current_price > sma20:
        return "BULL_TREND"
    elif sma20 < sma50 and current_price < sma20:
        return "BEAR_TREND"
    else:
        return "RANGING"

def get_trending_stocks():
    """Identify trending stocks based on volume and price action"""
    trending_stocks = []
    
    for symbol in NIFTY_50[:20]:  # Check top 20 for performance
        data = data_manager.get_cached_data(symbol, period="1d", interval="15m")
        if data is not None and len(data) > 10:
            # Calculate trend strength
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            volume_spike = False
            
            if 'Volume' in data.columns:
                avg_volume = data['Volume'].mean()
                current_volume = data['Volume'].iloc[-1]
                volume_spike = current_volume > avg_volume * 1.5
            
            trend_strength = abs(price_change) * 100
            
            if trend_strength > 1.0 or volume_spike:  # Minimum 1% move or volume spike
                trending_stocks.append({
                    'symbol': symbol,
                    'price_change': price_change,
                    'trend_strength': trend_strength,
                    'volume_spike': volume_spike,
                    'current_price': data['Close'].iloc[-1]
                })
    
    # Sort by trend strength
    trending_stocks.sort(key=lambda x: x['trend_strength'], reverse=True)
    return trending_stocks[:10]  # Return top 10 trending stocks

def sector_rotation_analysis():
    """Analyze sector performance for rotation strategy"""
    sector_performance = {}
    
    for sector, stocks in SECTOR_MAP.items():
        returns = []
        for stock in stocks[:3]:  # Sample 3 stocks per sector
            data = data_manager.get_cached_data(stock, period="5d", interval="1d")
            if data is not None and len(data) > 1:
                ret = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            sector_performance[sector] = {
                'return': avg_return,
                'trend': 'BULLISH' if avg_return > 0.01 else 'BEARISH' if avg_return < -0.01 else 'NEUTRAL'
            }
    
    return dict(sorted(sector_performance.items(), key=lambda x: x[1]['return'], reverse=True))

# ---------------- Enhanced Signal Engine ----------------
def generate_signal(df, symbol, market_regime="NEUTRAL"):
    """
    Generate trading signals with multiple confirmations including Fibonacci
    Returns: dict with signal details or None
    """
    if df is None or len(df) < 20:
        return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Initialize scoring
    bull_score = 0
    bear_score = 0
    reasons = []
    
    # Trend Analysis (30% weight)
    if current['EMA8'] > current['EMA21']:
        bull_score += 2
        reasons.append("EMA8 > EMA21")
    if current['EMA21'] > current['EMA50']:
        bull_score += 1
        reasons.append("EMA21 > EMA50")
    
    if current['EMA8'] < current['EMA21']:
        bear_score += 2
        reasons.append("EMA8 < EMA21")
    if current['EMA21'] < current['EMA50']:
        bear_score += 1
        reasons.append("EMA21 < EMA50")
    
    # Momentum Indicators (25% weight)
    if 30 < current['RSI14'] < 65:
        bull_score += 1
        reasons.append("RSI Optimal")
    elif current['RSI14'] > 70:
        bear_score += 1
        reasons.append("RSI Overbought")
    elif current['RSI14'] < 35:
        bull_score += 1
        reasons.append("RSI Oversold")
    
    if current['MACD'] > current['MACD_Signal']:
        bull_score += 1
        reasons.append("MACD Bullish")
    else:
        bear_score += 1
        reasons.append("MACD Bearish")
    
    # Volume & Price Action (20% weight)
    if 'Volume_SMA20' in current and current['Volume'] > current['Volume_SMA20'] * 1.2:
        if current['Close'] > current['Open']:  # Green candle with high volume
            bull_score += 1
            reasons.append("Bullish Volume")
        else:  # Red candle with high volume
            bear_score += 1
            reasons.append("Bearish Volume")
    
    if current['Close'] > current['VWAP']:
        bull_score += 1
        reasons.append("Above VWAP")
    else:
        bear_score += 1
        reasons.append("Below VWAP")
    
    # Fibonacci Analysis (15% weight)
    fib_signal = fibonacci_golden_zone_signal(df)
    if fib_signal == "BUY":
        bull_score += 2
        reasons.append("Fibonacci Golden Zone BUY")
    elif fib_signal == "SELL":
        bear_score += 2
        reasons.append("Fibonacci Golden Zone SELL")
    
    # Market Regime Adjustment (10% weight)
    if market_regime == "BULL_TREND":
        bull_score += 1
    elif market_regime == "BEAR_TREND":
        bear_score += 1
    
    # Signal Generation
    entry_price = float(current['Close'])
    atr_value = current['ATR'] if 'ATR' in current and not pd.isna(current['ATR']) else entry_price * 0.01
    
    # BUY Signal (Score >= 7 with positive momentum)
    if bull_score >= 7 and current['EMA8'] > prev['EMA8']:
        stop_loss = entry_price - (atr_value * 1.5)
        target = entry_price + (atr_value * 3)  # 2:1 Reward Ratio
        confidence = min(0.95, 0.5 + (bull_score * 0.05))
        
        return {
            "symbol": symbol,
            "action": "BUY",
            "entry": entry_price,
            "stop": stop_loss,
            "target": target,
            "conf": confidence,
            "score": bull_score,
            "reason": " | ".join(reasons),
            "regime": market_regime,
            "fib_signal": fib_signal
        }
    
    # SELL Signal (Score >= 7 with negative momentum)
    elif bear_score >= 7 and current['EMA8'] < prev['EMA8']:
        stop_loss = entry_price + (atr_value * 1.5)
        target = entry_price - (atr_value * 3)  # 2:1 Reward Ratio
        confidence = min(0.95, 0.5 + (bear_score * 0.05))
        
        return {
            "symbol": symbol,
            "action": "SELL", 
            "entry": entry_price,
            "stop": stop_loss,
            "target": target,
            "conf": confidence,
            "score": bear_score,
            "reason": " | ".join(reasons),
            "regime": market_regime,
            "fib_signal": fib_signal
        }
    
    return None

# ---------------- Risk Management ----------------
class RiskManager:
    def __init__(self):
        self.daily_trade_count = 0
        self.sector_exposure = {}
        self.peak_equity = CAPITAL
        self.last_reset = now_indian().date()
    
    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trade_count = 0
            self.sector_exposure = {}
            self.last_reset = current_date
    
    def can_trade(self, symbol, size, trader):
        self.reset_daily_counts()
        
        # Daily trade limit
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        # Drawdown protection
        current_equity = trader.equity()
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > MAX_DRAWDOWN:
            return False, f"Max drawdown reached: {drawdown:.2%}"
        
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Sector exposure
        sector = get_sector(symbol)
        current_exposure = self.sector_exposure.get(sector, 0)
        if current_exposure + size > CAPITAL * SECTOR_EXPOSURE_LIMIT:
            return False, f"Sector exposure limit for {sector}"
        
        return True, "OK"
    
    def record_trade(self, symbol, size):
        sector = get_sector(symbol)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + size
        self.daily_trade_count += 1

# ---------------- Paper Trading System ----------------
class PaperTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.trade_log = []
        self.risk_manager = RiskManager()
    
    def calculate_position_size(self, entry_price, atr=None):
        """Calculate position size with risk management"""
        if atr and atr > 0:
            risk_amount = self.cash * 0.02  # 2% risk per trade
            shares_by_risk = int(risk_amount / (atr * 1.5))
        else:
            shares_by_risk = int((TRADE_ALLOC * self.initial_capital) // entry_price)
        
        shares_by_capital = int((TRADE_ALLOC * self.cash) // entry_price)
        return max(1, min(shares_by_risk, shares_by_capital))
    
    def execute_trade(self, signal):
        """Execute a trade based on signal"""
        if signal is None:
            return False
        
        if signal["conf"] < AUTO_EXEC_CONF:
            return False
        
        symbol = signal["symbol"]
        if symbol in self.positions:
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(signal["entry"], signal.get("atr"))
        trade_value = quantity * signal["entry"]
        
        # Risk check
        can_trade, reason = self.risk_manager.can_trade(symbol, trade_value, self)
        if not can_trade:
            return False
        
        if trade_value > self.cash:
            return False
        
        # Execute trade
        self.cash -= trade_value
        self.positions[symbol] = {
            **signal,
            "quantity": quantity,
            "open_time": now_indian(),
            "status": "OPEN"
        }
        
        self.risk_manager.record_trade(symbol, trade_value)
        
        # Log trade
        self.trade_log.append({
            "time": now_indian(),
            "event": "OPEN",
            "symbol": symbol,
            "action": signal["action"],
            "quantity": quantity,
            "price": signal["entry"],
            "stop": signal["stop"],
            "target": signal["target"],
            "confidence": signal["conf"]
        })
        
        return True
    
    def update_positions(self):
        """Check for exit conditions on open positions"""
        closed_positions = []
        
        for symbol, position in list(self.positions.items()):
            data = data_manager.get_cached_data(symbol)
            if data is None:
                continue
            
            current_price = float(data['Close'].iloc[-1])
            
            if position["action"] == "BUY":
                if current_price <= position["stop"] or current_price >= position["target"]:
                    reason = "SL" if current_price <= position["stop"] else "TARGET"
                    pnl = (current_price - position["entry"]) * position["quantity"]
                    self.cash += position["quantity"] * current_price
                    
                    self.trade_log.append({
                        "time": now_indian(),
                        "event": "CLOSE",
                        "symbol": symbol,
                        "action": position["action"],
                        "quantity": position["quantity"],
                        "price": current_price,
                        "pnl": pnl,
                        "reason": reason
                    })
                    
                    closed_positions.append(symbol)
            
            else:  # SELL position
                if current_price >= position["stop"] or current_price <= position["target"]:
                    reason = "SL" if current_price >= position["stop"] else "TARGET"
                    pnl = (position["entry"] - current_price) * position["quantity"]
                    self.cash += position["quantity"] * current_price
                    
                    self.trade_log.append({
                        "time": now_indian(),
                        "event": "CLOSE", 
                        "symbol": symbol,
                        "action": position["action"],
                        "quantity": position["quantity"],
                        "price": current_price,
                        "pnl": pnl,
                        "reason": reason
                    })
                    
                    closed_positions.append(symbol)
        
        # Remove closed positions
        for symbol in closed_positions:
            del self.positions[symbol]
    
    def get_positions_dataframe(self):
        """Get current positions as DataFrame"""
        rows = []
        for symbol, position in self.positions.items():
            data = data_manager.get_cached_data(symbol)
            current_price = float(data['Close'].iloc[-1]) if data is not None else position["entry"]
            
            if position["action"] == "BUY":
                pnl = (current_price - position["entry"]) * position["quantity"]
                status = "✅ TARGET" if current_price >= position["target"] else "⚠️ STOP" if current_price <= position["stop"] else "🟡 OPEN"
            else:
                pnl = (position["entry"] - current_price) * position["quantity"]
                status = "✅ TARGET" if current_price <= position["target"] else "⚠️ STOP" if current_price >= position["stop"] else "🟡 OPEN"
            
            rows.append({
                "Symbol": symbol,
                "Action": position["action"],
                "Quantity": position["quantity"],
                "Entry": f"{position['entry']:.2f}",
                "Current": f"{current_price:.2f}",
                "Stop": f"{position['stop']:.2f}",
                "Target": f"{position['target']:.2f}",
                "P/L": f"₹{pnl:,.0f}",
                "Status": status,
                "Confidence": f"{position['conf']:.1%}"
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    def get_trade_log_dataframe(self):
        """Get trade log as DataFrame"""
        if not self.trade_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_log)
        df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df
    
    def equity(self):
        """Calculate total account equity"""
        total_equity = self.cash
        
        for symbol, position in self.positions.items():
            data = data_manager.get_cached_data(symbol)
            if data is not None:
                current_price = float(data['Close'].iloc[-1])
                if position["action"] == "BUY":
                    pnl = (current_price - position["entry"]) * position["quantity"]
                else:
                    pnl = (position["entry"] - current_price) * position["quantity"]
                
                total_equity += position["quantity"] * position["entry"] + pnl
        
        return total_equity
    
    def get_performance_stats(self):
        """Calculate performance statistics"""
        closed_trades = [t for t in self.trade_log if t["event"] == "CLOSE"]
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0
            }
        
        winning_trades = [t for t in closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in closed_trades if t["pnl"] < 0]
        
        total_pnl = sum(t["pnl"] for t in closed_trades)
        win_rate = len(winning_trades) / len(closed_trades)
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
        else:
            profit_factor = float('inf') if avg_win > 0 else 0
        
        return {
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }

# ---------------- Initialize Systems ----------------
data_manager = DataManager()

if "trader" not in st.session_state:
    st.session_state.trader = PaperTrader()
trader = st.session_state.trader

# ---------------- Streamlit UI ----------------
st.markdown("<h1 style='text-align: center; color: #0077cc;'>📈 Intraday Trading Terminal Pro v6.0</h1>", unsafe_allow_html=True)

# Market Overview - FIXED INDICES LOADING
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    try:
        nifty_data = data_manager.get_index_data("^NSEI")
        nifty_price = nifty_data['Close'].iloc[-1] if nifty_data is not None and len(nifty_data) > 0 else None
        nifty_change = ((nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[0]) / nifty_data['Close'].iloc[0] * 100) if nifty_data is not None and len(nifty_data) > 1 else 0
        st.metric("NIFTY 50", 
                 f"₹{nifty_price:,.2f}" if nifty_price else "Loading...",
                 delta=f"{nifty_change:+.2f}%" if nifty_price else None)
    except:
        st.metric("NIFTY 50", "Loading...")

with col2:
    try:
        bank_nifty_data = data_manager.get_index_data("^NSEBANK")
        bank_nifty_price = bank_nifty_data['Close'].iloc[-1] if bank_nifty_data is not None and len(bank_nifty_data) > 0 else None
        bank_nifty_change = ((bank_nifty_data['Close'].iloc[-1] - bank_nifty_data['Close'].iloc[0]) / bank_nifty_data['Close'].iloc[0] * 100) if bank_nifty_data is not None and len(bank_nifty_data) > 1 else 0
        st.metric("BANK NIFTY", 
                 f"₹{bank_nifty_price:,.2f}" if bank_nifty_price else "Loading...",
                 delta=f"{bank_nifty_change:+.2f}%" if bank_nifty_price else None)
    except:
        st.metric("BANK NIFTY", "Loading...")

with col3:
    market_status = "🟢 LIVE" if market_open() else "🔴 CLOSED"
    st.metric("Market Status", market_status)

with col4:
    market_regime = detect_market_regime()
    regime_icon = "📈" if market_regime == "BULL_TREND" else "📉" if market_regime == "BEAR_TREND" else "➡️"
    st.metric("Market Regime", f"{regime_icon} {market_regime}")

with col5:
    performance = trader.get_performance_stats()
    st.metric("Win Rate", f"{performance['win_rate']:.1%}" if performance['total_trades'] > 0 else "N/A")

# Main Tabs
tabs = st.tabs(["📊 Dashboard", "🎯 Signals", "📈 Live Charts", "💼 Trading", "📋 Analytics"])

# Dashboard Tab - ENHANCED
with tabs[0]:
    st.subheader("Trading Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cash Balance", f"₹{trader.cash:,.0f}")
    with col2:
        st.metric("Total Equity", f"₹{trader.equity():,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.positions))
    with col4:
        st.metric("Daily Trades", f"{trader.risk_manager.daily_trade_count}/{MAX_DAILY_TRADES}")
    
    # Trending Stocks
    st.subheader("🔥 Trending Stocks")
    trending_stocks = get_trending_stocks()
    if trending_stocks:
        trending_cols = st.columns(5)
        for idx, stock in enumerate(trending_stocks[:5]):
            with trending_cols[idx % 5]:
                change_color = "green" if stock['price_change'] > 0 else "red"
                st.metric(
                    stock['symbol'].replace('.NS', ''),
                    f"₹{stock['current_price']:.1f}",
                    delta=f"{stock['price_change']:+.2%}",
                    delta_color=change_color
                )
    else:
        st.info("No trending stocks detected")
    
    # Sector Analysis
    st.subheader("Sector Rotation")
    sector_data = sector_rotation_analysis()
    if sector_data:
        sector_cols = st.columns(len(sector_data))
        for idx, (sector, data) in enumerate(sector_data.items()):
            with sector_cols[idx]:
                color = "green" if data['trend'] == 'BULLISH' else "red" if data['trend'] == 'BEARISH' else "gray"
                st.metric(sector, f"{data['return']:.2%}", delta=data['trend'], delta_color=color)
    
    # Active Positions
    st.subheader("Active Positions")
    positions_df = trader.get_positions_dataframe()
    if not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No active positions")

# Signals Tab - ENHANCED
with tabs[1]:
    st.subheader("Live Signal Scanner")
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="signal_refresh")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_universe = st.selectbox("Select Universe", ["Nifty 50", "Nifty 100"], index=0)
    with col2:
        min_confidence = st.slider("Min Confidence", 0.5, 0.9, AUTO_EXEC_CONF, 0.05)
    with col3:
        enable_fib = st.checkbox("Fibonacci Strategy", True)
    
    symbols_to_scan = NIFTY_50 if selected_universe == "Nifty 50" else NIFTY_100
    
    if st.button("Scan for Signals", type="primary") or market_open():
        signals_found = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols_to_scan):
            status_text.text(f"Analyzing {symbol}...")
            data = data_manager.get_cached_data(symbol)
            signal = generate_signal(data, symbol, market_regime)
            
            if signal and signal["conf"] >= min_confidence:
                # Filter by Fibonacci strategy if enabled
                if not enable_fib or signal.get("fib_signal"):
                    signals_found.append(signal)
                
                # Auto-execute high confidence signals
                if signal["conf"] >= AUTO_EXEC_CONF:
                    trader.execute_trade(signal)
            
            progress_bar.progress((i + 1) / len(symbols_to_scan))
        
        progress_bar.empty()
        status_text.empty()
        
        # Update positions for exit conditions
        trader.update_positions()
        
        if signals_found:
            st.success(f"🎯 Found {len(signals_found)} trading signals!")
            signals_df = pd.DataFrame(signals_found)
            signals_df = signals_df.sort_values("conf", ascending=False)
            
            # Format display
            display_df = signals_df[['symbol', 'action', 'entry', 'stop', 'target', 'conf', 'score', 'reason']].copy()
            display_df['conf'] = display_df['conf'].apply(lambda x: f"{x:.1%}")
            display_df['entry'] = display_df['entry'].apply(lambda x: f"₹{x:.2f}")
            display_df['stop'] = display_df['stop'].apply(lambda x: f"₹{x:.2f}")
            display_df['target'] = display_df['target'].apply(lambda x: f"₹{x:.2f}")
            
            # Color code actions
            def color_action(action):
                return 'color: green' if action == 'BUY' else 'color: red'
            
            styled_df = display_df.style.applymap(color_action, subset=['action'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No trading signals found in this scan.")
    
    else:
        st.warning("Click 'Scan for Signals' or wait for market hours for auto-scanning")

# Live Charts Tab - ENHANCED WITH FIBONACCI
with tabs[2]:
    st.subheader("Live Technical Analysis")
    st_autorefresh(interval=CHART_REFRESH_MS, key="chart_refresh")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_symbol = st.selectbox("Select Symbol", NIFTY_100, key="chart_symbol")
        show_rsi = st.checkbox("Show RSI", True)
        show_macd = st.checkbox("Show MACD", True)
        show_vwap = st.checkbox("Show VWAP", True)
        show_fib = st.checkbox("Show Fibonacci", True)
        show_volume = st.checkbox("Show Volume", False)
    
    with col2:
        # Try multiple intervals for data
        chart_data = None
        used_interval = ""
        for period, interval in CHART_INTERVALS:
            chart_data = data_manager.get_cached_data(selected_symbol, period, interval)
            if chart_data is not None and len(chart_data) > 10:
                used_interval = f"{period} - {interval}"
                break
        
        if chart_data is not None and len(chart_data) > 10:
            st.write(f"Data: {used_interval} | Last Updated: {now_indian().strftime('%H:%M:%S')}")
            
            # Create subplots
            if show_rsi or show_macd:
                rows = 1
                if show_rsi and show_macd:
                    rows = 3
                elif show_rsi or show_macd:
                    rows = 2
                
                fig = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price', 'RSI', 'MACD')[:rows],
                    row_heights=[0.6, 0.2, 0.2][:rows]
                )
            else:
                fig = go.Figure()
            
            # Candlesticks
            row_idx = 1
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Price"
            ), row=row_idx, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(
                x=chart_data.index, y=chart_data['EMA8'],
                name="EMA 8", line=dict(color='orange', width=1)
            ), row=row_idx, col=1)
            
            fig.add_trace(go.Scatter(
                x=chart_data.index, y=chart_data['EMA21'],
                name="EMA 21", line=dict(color='red', width=1)
            ), row=row_idx, col=1)
            
            # VWAP
            if show_vwap and 'VWAP' in chart_data.columns:
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['VWAP'],
                    name="VWAP", line=dict(color='purple', width=1, dash='dash')
                ), row=row_idx, col=1)
            
            # Fibonacci Retracement
            if show_fib and len(chart_data) >= 20:
                recent_high = chart_data['High'].max()
                recent_low = chart_data['Low'].min()
                fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
                
                # Add Fibonacci levels
                for level, price in fib_levels.items():
                    color = 'gold' if level in ['0.5', '0.618'] else 'gray'
                    width = 2 if level in ['0.5', '0.618'] else 1
                    dash = 'dash' if level in ['0.5', '0.618'] else 'dot'
                    
                    fig.add_hline(
                        y=price,
                        line_dash=dash,
                        line_color=color,
                        line_width=width,
                        annotation_text=f"Fib {level}",
                        row=row_idx, col=1
                    )
            
            # RSI
            if show_rsi:
                row_idx += 1
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['RSI14'],
                    name="RSI", line=dict(color='blue', width=1)
                ), row=row_idx, col=1)
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=row_idx, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=row_idx, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=row_idx, col=1)
            
            # MACD
            if show_macd:
                row_idx += 1
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['MACD'],
                    name="MACD", line=dict(color='blue', width=1)
                ), row=row_idx, col=1)
                
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['MACD_Signal'],
                    name="Signal", line=dict(color='red', width=1)
                ), row=row_idx, col=1)
            
            fig.update_layout(
                title=f"{selected_symbol} - Live Chart",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume
            if show_volume and 'Volume' in chart_data.columns:
                st.subheader("Volume")
                st.bar_chart(chart_data["Volume"], height=120)
        
        else:
            st.error("Unable to fetch data for the selected symbol")

# Trading Tab (unchanged, but using fixed equity method)
with tabs[3]:
    st.subheader("Paper Trading Account")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account Value", f"₹{trader.equity():,.0f}")
    with col2:
        st.metric("Available Cash", f"₹{trader.cash:,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.positions))
    with col4:
        performance = trader.get_performance_stats()
        st.metric("Total P&L", f"₹{performance['total_pnl']:,.0f}")
    
    # Position Management
    st.subheader("Position Management")
    if trader.positions:
        positions_df = trader.get_positions_dataframe()
        st.dataframe(positions_df, use_container_width=True)
        
        # Manual close
        st.subheader("Manual Close")
        col1, col2 = st.columns(2)
        with col1:
            position_to_close = st.selectbox("Select Position", list(trader.positions.keys()))
        with col2:
            if st.button("Close Position", type="primary"):
                data = data_manager.get_cached_data(position_to_close)
                if data is not None:
                    current_price = float(data['Close'].iloc[-1])
                    position = trader.positions[position_to_close]
                    if position["action"] == "BUY":
                        position["stop"] = current_price
                    else:
                        position["stop"] = current_price
                    trader.update_positions()
                    st.success(f"Closed {position_to_close} at ₹{current_price:.2f}")
                    st.rerun()
    else:
        st.info("No active positions")
    
    # Trade History
    st.subheader("Trade History")
    trade_log_df = trader.get_trade_log_dataframe()
    if not trade_log_df.empty:
        st.dataframe(trade_log_df, use_container_width=True)
        
        # Export capability
        csv_data = trade_log_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log",
            data=csv_data,
            file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history available")

# Analytics Tab (unchanged)
with tabs[4]:
    st.subheader("Performance Analytics")
    
    performance = trader.get_performance_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", performance["total_trades"])
    with col2:
        st.metric("Win Rate", f"{performance['win_rate']:.1%}")
    with col3:
        st.metric("Total P&L", f"₹{performance['total_pnl']:,.0f}")
    with col4:
        pf_value = performance['profit_factor']
        pf_display = "∞" if pf_value == float('inf') else f"{pf_value:.2f}"
        st.metric("Profit Factor", pf_display)
    
    # Additional metrics
    if performance['total_trades'] > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Win", f"₹{performance['avg_win']:,.0f}")
        with col2:
            st.metric("Avg Loss", f"₹{performance['avg_loss']:,.0f}")
        with col3:
            avg_trade = performance['total_pnl'] / performance['total_trades']
            st.metric("Avg Trade", f"₹{avg_trade:,.0f}")
        with col4:
            expectancy = (performance['win_rate'] * performance['avg_win'] + 
                         (1-performance['win_rate']) * performance['avg_loss'])
            st.metric("Expectancy", f"₹{expectancy:,.0f}")
    
    # Equity Curve
    if len(trader.trade_log) > 1:
        st.subheader("Equity Curve")
        equity_data = []
        running_equity = CAPITAL
        
        for trade in trader.trade_log:
            if trade["event"] == "CLOSE":
                running_equity += trade["pnl"]
                equity_data.append({
                    "time": trade["time"],
                    "equity": running_equity
                })
        
        if equity_data:
            equity_df = pd.DataFrame(equity_data)
            st.line_chart(equity_df.set_index("time")["equity"])

# Account Management
st.sidebar.header("Account Management")
if st.sidebar.button("Reset Paper Account", type="secondary"):
    st.session_state.trader = PaperTrader()
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "⚠️ Paper Trading Simulation | Real-time data from Yahoo Finance | "
    "Market Hours: 9:15 AM - 3:30 PM IST | v6.0 with Fibonacci & Trending Stocks"
    "</div>",
    unsafe_allow_html=True
)