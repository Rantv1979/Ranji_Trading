"""
Intraday Live Trading Terminal — Pure Intraday Pro Edition v7.0
---------------------------------------------------------------
Pure Intraday Features:
- Fixed indices loading with direct API calls
- Enhanced intraday data reliability
- Trending stocks detection
- High-quality intraday signals
- Fibonacci intraday strategy
- Real-time 5min/15min data focus
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time, timedelta
import pytz, warnings, time, requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
st.set_page_config(page_title="Intraday Terminal Pro v7.0", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading parameters
CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10
MAX_DAILY_TRADES = 8
MAX_DRAWDOWN = 0.05
SECTOR_EXPOSURE_LIMIT = 0.25

# Refresh intervals
SIGNAL_REFRESH_MS = 30000  # 30 seconds
CHART_REFRESH_MS = 10000   # 10 seconds
AUTO_EXEC_CONF = 0.70

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

# Sector mapping for intraday
SECTOR_MAP = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "AUTO": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS"],
    "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"]
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
    """Calculate Fibonacci retracement levels for intraday"""
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
    """Generate Fibonacci Golden Zone signals (0.5 - 0.618) for intraday"""
    if len(df) < 10:
        return None
    
    # Use last 2 hours data for Fibonacci levels
    recent_data = df.tail(20)
    recent_high = recent_data['High'].max()
    recent_low = recent_data['Low'].min()
    
    fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
    current_price = df['Close'].iloc[-1]
    
    golden_zone_low = fib_levels['0.618']
    golden_zone_high = fib_levels['0.5']
    
    # Check if price is in golden zone
    if golden_zone_low <= current_price <= golden_zone_high:
        # Bullish reversal pattern
        if (df['Close'].iloc[-1] > df['Open'].iloc[-1] and  # Green candle
            df['Close'].iloc[-1] > df['Close'].iloc[-2] and  # Higher close
            df['RSI14'].iloc[-1] < 65):  # RSI not overbought
            return "BUY"
        
        # Bearish reversal pattern  
        elif (df['Close'].iloc[-1] < df['Open'].iloc[-1] and  # Red candle
              df['Close'].iloc[-1] < df['Close'].iloc[-2] and  # Lower close
              df['RSI14'].iloc[-1] > 35):  # RSI not oversold
            return "SELL"
    
    return None

# ---------------- Enhanced Data Manager for Intraday ----------------
class IntradayDataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_intraday_data(self, symbol, interval="15m"):
        """Get intraday data with focus on 5min/15min intervals"""
        key = f"{symbol}_{interval}"
        current_time = time.time()
        
        # Return cached data if recent (2 minutes for intraday)
        if key in self.cache and current_time - self.last_update.get(key, 0) < 120:
            return self.cache[key]
        
        # Fetch fresh intraday data
        data = self.fetch_intraday_ohlc(symbol, interval)
        if data is not None and len(data) > 5:
            self.cache[key] = data
            self.last_update[key] = current_time
        
        return data
    
    def fetch_intraday_ohlc(self, symbol, interval="15m"):
        """Fetch intraday data with multiple fallback strategies"""
        periods = {
            "5m": "1d",
            "15m": "1d", 
            "30m": "2d",
            "1h": "5d"
        }
        
        period = periods.get(interval, "1d")
        
        for retry in range(3):
            try:
                # For indices, use proper Yahoo Finance symbols
                if symbol == "NIFTY_50":
                    ticker_symbol = "^NSEI"
                elif symbol == "BANK_NIFTY":
                    ticker_symbol = "^NSEBANK"
                else:
                    ticker_symbol = symbol
                
                df = yf.download(ticker_symbol, period=period, interval=interval, progress=False, threads=False)
                
                if df is None or df.empty or len(df) < 5:
                    continue
                
                # Clean column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0)
                
                df.columns = [str(col).upper() for col in df.columns]
                
                # Standardize column names
                column_map = {
                    'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 
                    'CLOSE': 'Close', 'VOLUME': 'Volume',
                    'ADJ CLOSE': 'Close'
                }
                
                df = df.rename(columns=column_map)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']] if 'Volume' in df.columns else df[['Open', 'High', 'Low', 'Close']]
                
                df = df.dropna().copy()
                
                if len(df) < 5:
                    continue
                
                # Calculate intraday indicators
                df['EMA8'] = ema(df['Close'], 8)
                df['EMA21'] = ema(df['Close'], 21)
                df['RSI14'] = rsi(df['Close'], 14).fillna(50)
                
                macd_line, signal_line = macd(df['Close'])
                df['MACD'] = macd_line
                df['MACD_Signal'] = signal_line
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
                # Volume and VWAP
                if 'Volume' in df.columns:
                    df['Volume_SMA20'] = df['Volume'].rolling(10).mean()
                    df['VWAP'] = calculate_vwap(df)
                
                # ATR for risk management
                df['ATR'] = calculate_atr(df)
                
                return df
                
            except Exception as e:
                print(f"Error fetching {symbol} (attempt {retry+1}): {e}")
                time.sleep(1)
                continue
        
        return None

    def get_index_value(self, index_name):
        """Get current index value with direct API fallback"""
        try:
            if index_name == "NIFTY_50":
                symbol = "^NSEI"
            elif index_name == "BANK_NIFTY":
                symbol = "^NSEBANK"
            else:
                return None
            
            # Try Yahoo Finance first
            data = self.get_intraday_data(symbol, "5m")
            if data is not None and len(data) > 0:
                return data['Close'].iloc[-1]
            
            # Fallback to direct API
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="5m")
            if not hist.empty:
                return hist['Close'].iloc[-1]
                
        except Exception as e:
            print(f"Error getting index value for {index_name}: {e}")
        
        return None

# ---------------- Market Analysis ----------------
def detect_market_regime():
    """Detect current market trend using intraday data"""
    nifty_data = data_manager.get_intraday_data("NIFTY_50", "15m")
    
    if nifty_data is None or len(nifty_data) < 10:
        return "NEUTRAL"
    
    current_price = nifty_data['Close'].iloc[-1]
    ema21 = nifty_data['EMA21'].iloc[-1]
    ema8 = nifty_data['EMA8'].iloc[-1]
    
    if ema8 > ema21 and current_price > ema8:
        return "BULL_TREND"
    elif ema8 < ema21 and current_price < ema8:
        return "BEAR_TREND"
    else:
        return "RANGING"

def get_trending_stocks():
    """Identify trending stocks based on intraday momentum"""
    trending_stocks = []
    
    # Check only 15 stocks for performance
    for symbol in NIFTY_50[:15]:
        data = data_manager.get_intraday_data(symbol, "15m")
        if data is not None and len(data) > 10:
            current_rsi = data['RSI14'].iloc[-1]
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
            
            # Trending criteria
            if (abs(price_change) > 0.8 or  # Minimum 0.8% move
                (current_rsi > 65 and price_change > 0) or  # Strong bullish
                (current_rsi < 35 and price_change < 0)):   # Strong bearish
                
                trending_stocks.append({
                    'symbol': symbol,
                    'price_change': price_change,
                    'rsi': current_rsi,
                    'current_price': data['Close'].iloc[-1],
                    'trend': 'BULLISH' if price_change > 0 else 'BEARISH'
                })
    
    # Sort by absolute price change
    trending_stocks.sort(key=lambda x: abs(x['price_change']), reverse=True)
    return trending_stocks[:8]  # Return top 8 trending stocks

# ---------------- Enhanced Intraday Signal Engine ----------------
def generate_intraday_signal(df, symbol, market_regime="NEUTRAL"):
    """
    Generate intraday trading signals with multiple confirmations
    """
    if df is None or len(df) < 10:
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
    else:
        bear_score += 1
        reasons.append("EMA8 < EMA21")
    
    # Momentum Indicators (30% weight)
    rsi_value = current['RSI14']
    if 40 < rsi_value < 60:
        bull_score += 1
        reasons.append("RSI Neutral-Bullish")
    elif rsi_value > 70:
        bear_score += 1
        reasons.append("RSI Overbought")
    elif rsi_value < 30:
        bull_score += 1
        reasons.append("RSI Oversold")
    
    if current['MACD'] > current['MACD_Signal']:
        bull_score += 1
        reasons.append("MACD Bullish")
    else:
        bear_score += 1
        reasons.append("MACD Bearish")
    
    # Price Action (20% weight)
    if current['Close'] > current['Open']:  # Green candle
        bull_score += 1
        reasons.append("Bullish Candle")
    else:
        bear_score += 1
        reasons.append("Bearish Candle")
    
    # Volume Confirmation (10% weight)
    if 'Volume_SMA20' in current and current['Volume'] > current['Volume_SMA20'] * 1.5:
        if current['Close'] > current['Open']:
            bull_score += 1
            reasons.append("High Bullish Volume")
        else:
            bear_score += 1
            reasons.append("High Bearish Volume")
    
    # Fibonacci Strategy (10% weight)
    fib_signal = fibonacci_golden_zone_signal(df)
    if fib_signal == "BUY":
        bull_score += 2
        reasons.append("Fibonacci BUY")
    elif fib_signal == "SELL":
        bear_score += 2
        reasons.append("Fibonacci SELL")
    
    # Signal Generation
    entry_price = float(current['Close'])
    atr_value = current['ATR'] if 'ATR' in current and not pd.isna(current['ATR']) else entry_price * 0.005
    
    # BUY Signal (Strong bullish conditions)
    if bull_score >= 5 and current['Close'] > prev['Close']:
        stop_loss = entry_price - (atr_value * 1.5)
        target = entry_price + (atr_value * 2.5)  # 1.67:1 Reward Ratio
        confidence = min(0.95, 0.4 + (bull_score * 0.08))
        
        return {
            "symbol": symbol,
            "action": "BUY",
            "entry": entry_price,
            "stop": stop_loss,
            "target": target,
            "conf": confidence,
            "score": bull_score,
            "reason": " | ".join(reasons),
            "timestamp": now_indian()
        }
    
    # SELL Signal (Strong bearish conditions)
    elif bear_score >= 5 and current['Close'] < prev['Close']:
        stop_loss = entry_price + (atr_value * 1.5)
        target = entry_price - (atr_value * 2.5)  # 1.67:1 Reward Ratio
        confidence = min(0.95, 0.4 + (bear_score * 0.08))
        
        return {
            "symbol": symbol,
            "action": "SELL", 
            "entry": entry_price,
            "stop": stop_loss,
            "target": target,
            "conf": confidence,
            "score": bear_score,
            "reason": " | ".join(reasons),
            "timestamp": now_indian()
        }
    
    return None

# ---------------- Trading System ----------------
class IntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.last_reset = now_indian().date()
    
    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.last_reset = current_date
    
    def calculate_position_size(self, entry_price, atr=None):
        """Calculate position size for intraday"""
        if atr and atr > 0:
            risk_amount = self.cash * 0.015  # 1.5% risk per trade for intraday
            shares_by_risk = int(risk_amount / (atr * 1.5))
        else:
            shares_by_risk = int((TRADE_ALLOC * self.initial_capital) // entry_price)
        
        shares_by_capital = int((TRADE_ALLOC * self.cash) // entry_price)
        return max(1, min(shares_by_risk, shares_by_capital, 1000))  # Max 1000 shares
    
    def execute_trade(self, signal):
        """Execute intraday trade"""
        if signal is None:
            return False
        
        self.reset_daily_counts()
        
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False
        
        symbol = signal["symbol"]
        if symbol in self.positions:
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(signal["entry"], signal.get("atr"))
        trade_value = quantity * signal["entry"]
        
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
        
        self.daily_trades += 1
        
        # Log trade
        self.trade_log.append({
            "time": now_indian(),
            "event": "OPEN",
            "symbol": symbol,
            "action": signal["action"],
            "quantity": quantity,
            "price": signal["entry"],
            "value": trade_value,
            "stop": signal["stop"],
            "target": signal["target"],
            "confidence": signal["conf"]
        })
        
        return True
    
    def update_positions(self):
        """Check for exit conditions"""
        closed_positions = []
        
        for symbol, position in list(self.positions.items()):
            data = data_manager.get_intraday_data(symbol, "5m")
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
            data = data_manager.get_intraday_data(symbol, "5m")
            current_price = float(data['Close'].iloc[-1]) if data is not None else position["entry"]
            
            if position["action"] == "BUY":
                pnl = (current_price - position["entry"]) * position["quantity"]
                pnl_percent = (current_price - position["entry"]) / position["entry"] * 100
            else:
                pnl = (position["entry"] - current_price) * position["quantity"]
                pnl_percent = (position["entry"] - current_price) / position["entry"] * 100
            
            rows.append({
                "Symbol": symbol.replace('.NS', ''),
                "Action": position["action"],
                "Qty": position["quantity"],
                "Entry": position['entry'],
                "Current": current_price,
                "Stop": position['stop'],
                "Target": position['target'],
                "P/L": pnl,
                "P/L %": pnl_percent,
                "Confidence": f"{position['conf']:.1%}"
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    def equity(self):
        """Calculate total equity"""
        total_equity = self.cash
        for symbol, position in self.positions.items():
            data = data_manager.get_intraday_data(symbol, "5m")
            if data is not None:
                current_price = float(data['Close'].iloc[-1])
                total_equity += position["quantity"] * current_price
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
                "avg_loss": 0
            }
        
        winning_trades = [t for t in closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in closed_trades if t["pnl"] < 0]
        
        total_pnl = sum(t["pnl"] for t in closed_trades)
        win_rate = len(winning_trades) / len(closed_trades)
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        
        return {
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }

# ---------------- Initialize Systems ----------------
data_manager = IntradayDataManager()

if "trader" not in st.session_state:
    st.session_state.trader = IntradayTrader()
trader = st.session_state.trader

# ---------------- Streamlit UI ----------------
st.markdown("<h1 style='text-align: center; color: #0077cc;'>🎯 Pure Intraday Trading Terminal v7.0</h1>", unsafe_allow_html=True)

# Market Overview - FIXED
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    nifty_price = data_manager.get_index_value("NIFTY_50")
    st.metric("NIFTY 50", 
             f"₹{nifty_price:,.2f}" if nifty_price else "Loading...",
             delta="Live" if nifty_price else None)

with col2:
    bank_nifty_price = data_manager.get_index_value("BANK_NIFTY")
    st.metric("BANK NIFTY", 
             f"₹{bank_nifty_price:,.2f}" if bank_nifty_price else "Loading...",
             delta="Live" if bank_nifty_price else None)

with col3:
    market_status = "🟢 LIVE" if market_open() else "🔴 CLOSED"
    st.metric("Market Status", market_status)

with col4:
    market_regime = detect_market_regime()
    regime_color = "green" if market_regime == "BULL_TREND" else "red" if market_regime == "BEAR_TREND" else "gray"
    st.metric("Market Regime", market_regime)

with col5:
    performance = trader.get_performance_stats()
    st.metric("Win Rate", f"{performance['win_rate']:.1%}" if performance['total_trades'] > 0 else "N/A")

# Main Tabs
tabs = st.tabs(["📊 Dashboard", "🎯 Signals", "📈 Charts", "💼 Trading", "📋 Performance"])

# Dashboard Tab
with tabs[0]:
    st.subheader("Intraday Dashboard")
    
    # Account Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account Value", f"₹{trader.equity():,.0f}")
    with col2:
        st.metric("Available Cash", f"₹{trader.cash:,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.positions))
    with col4:
        st.metric("Daily Trades", f"{trader.daily_trades}/{MAX_DAILY_TRADES}")
    
    # Trending Stocks
    st.subheader("🔥 Trending Stocks (Intraday)")
    trending_stocks = get_trending_stocks()
    if trending_stocks:
        cols = st.columns(4)
        for idx, stock in enumerate(trending_stocks):
            with cols[idx % 4]:
                emoji = "📈" if stock['trend'] == 'BULLISH' else "📉"
                color = "green" if stock['trend'] == 'BULLISH' else "red"
                st.metric(
                    f"{emoji} {stock['symbol'].replace('.NS', '')}",
                    f"₹{stock['current_price']:.1f}",
                    delta=f"{stock['price_change']:+.2f}%",
                    delta_color=color
                )
    else:
        st.info("Scanning for trending stocks...")
    
    # Active Positions
    st.subheader("Active Positions")
    positions_df = trader.get_positions_dataframe()
    if not positions_df.empty:
        # Format display
        display_df = positions_df.copy()
        display_df['Entry'] = display_df['Entry'].apply(lambda x: f"₹{x:.2f}")
        display_df['Current'] = display_df['Current'].apply(lambda x: f"₹{x:.2f}")
        display_df['Stop'] = display_df['Stop'].apply(lambda x: f"₹{x:.2f}")
        display_df['Target'] = display_df['Target'].apply(lambda x: f"₹{x:.2f}")
        display_df['P/L'] = display_df['P/L'].apply(lambda x: f"₹{x:,.0f}")
        display_df['P/L %'] = display_df['P/L %'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No active positions")

# Signals Tab
with tabs[1]:
    st.subheader("Intraday Signal Scanner")
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="signal_refresh")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        min_confidence = st.slider("Minimum Confidence", 0.5, 0.9, 0.65, 0.05)
    with col2:
        st.write(f"Auto-execute: {AUTO_EXEC_CONF:.0%}+")
    
    if st.button("🔍 Scan for Intraday Signals", type="primary") or market_open():
        signals_found = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan Nifty 50 stocks
        for i, symbol in enumerate(NIFTY_50):
            status_text.text(f"Scanning {symbol.replace('.NS', '')}...")
            data = data_manager.get_intraday_data(symbol, "15m")
            signal = generate_intraday_signal(data, symbol, market_regime)
            
            if signal and signal["conf"] >= min_confidence:
                signals_found.append(signal)
                
                # Auto-execute high confidence signals
                if signal["conf"] >= AUTO_EXEC_CONF and market_open():
                    trader.execute_trade(signal)
            
            progress_bar.progress((i + 1) / len(NIFTY_50))
        
        progress_bar.empty()
        status_text.empty()
        
        # Update positions
        trader.update_positions()
        
        if signals_found:
            st.success(f"🎯 Found {len(signals_found)} intraday signals!")
            signals_df = pd.DataFrame(signals_found)
            signals_df = signals_df.sort_values("conf", ascending=False)
            
            # Format display
            display_df = signals_df[['symbol', 'action', 'entry', 'stop', 'target', 'conf', 'score', 'reason']].copy()
            display_df['symbol'] = display_df['symbol'].str.replace('.NS', '')
            display_df['conf'] = display_df['conf'].apply(lambda x: f"{x:.1%}")
            display_df['entry'] = display_df['entry'].apply(lambda x: f"₹{x:.2f}")
            display_df['stop'] = display_df['stop'].apply(lambda x: f"₹{x:.2f}")
            display_df['target'] = display_df['target'].apply(lambda x: f"₹{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No intraday signals found. Market may be ranging.")
    
    else:
        st.warning("Click scan button or wait for market hours")

# Charts Tab - FIXED
with tabs[2]:
    st.subheader("Live Intraday Charts")
    st_autorefresh(interval=CHART_REFRESH_MS, key="chart_refresh")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_symbol = st.selectbox("Select Stock", NIFTY_50, key="chart_symbol")
        chart_interval = st.selectbox("Interval", ["5m", "15m", "30m"], index=1)
        show_indicators = st.checkbox("Show Indicators", True)
        show_fib = st.checkbox("Show Fibonacci", True)
    
    with col2:
        chart_data = data_manager.get_intraday_data(selected_symbol, chart_interval)
        
        if chart_data is not None and len(chart_data) > 10:
            st.write(f"**{selected_symbol.replace('.NS', '')}** - {chart_interval} Chart | Last: ₹{chart_data['Close'].iloc[-1]:.2f}")
            
            # Create chart
            fig = go.Figure()
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Price"
            ))
            
            if show_indicators:
                # EMAs
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['EMA8'],
                    name="EMA 8", line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['EMA21'],
                    name="EMA 21", line=dict(color='red', width=1)
                ))
            
            # Fibonacci Levels
            if show_fib and len(chart_data) >= 10:
                recent_high = chart_data['High'].max()
                recent_low = chart_data['Low'].min()
                fib_levels = calculate_fibonacci_levels(recent_high, recent_low)
                
                for level, price in fib_levels.items():
                    color = 'gold' if level in ['0.5', '0.618'] else 'gray'
                    fig.add_hline(
                        y=price,
                        line_dash="dash",
                        line_color=color,
                        annotation_text=f"Fib {level}",
                        annotation_position="right"
                    )
            
            fig.update_layout(
                title=f"Intraday Chart - {selected_symbol.replace('.NS', '')}",
                xaxis_rangeslider_visible=False,
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current Signal
            current_signal = generate_intraday_signal(chart_data, selected_symbol, market_regime)
            if current_signal:
                action_color = "green" if current_signal["action"] == "BUY" else "red"
                st.markdown(f"**Current Signal:** <span style='color: {action_color}; font-weight: bold;'>{current_signal['action']}</span> | "
                           f"Confidence: {current_signal['conf']:.1%} | Score: {current_signal['score']}", 
                           unsafe_allow_html=True)
                st.write(f"**Reason:** {current_signal['reason']}")
            else:
                st.info("No clear signal at current levels")
            
        else:
            st.error("Loading chart data... Please wait or try different symbol/interval.")

# Trading Tab
with tabs[3]:
    st.subheader("Intraday Trading")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account Value", f"₹{trader.equity():,.0f}")
    with col2:
        st.metric("Available Cash", f"₹{trader.cash:,.0f}")
    with col3:
        st.metric("Open Positions", len(trader.positions))
    with col4:
        st.metric("Today's P&L", f"₹{trader.get_performance_stats()['total_pnl']:,.0f}")
    
    # Manual Trading
    st.subheader("Manual Trade")
    col1, col2, col3 = st.columns(3)
    with col1:
        manual_symbol = st.selectbox("Symbol", NIFTY_50, key="manual_trade")
    with col2:
        manual_action = st.selectbox("Action", ["BUY", "SELL"])
    with col3:
        if st.button("Execute Manual Trade", type="primary"):
            data = data_manager.get_intraday_data(manual_symbol, "5m")
            if data is not None:
                entry_price = data['Close'].iloc[-1]
                atr_value = data['ATR'].iloc[-1] if 'ATR' in data else entry_price * 0.005
                
                signal = {
                    "symbol": manual_symbol,
                    "action": manual_action,
                    "entry": entry_price,
                    "stop": entry_price - (atr_value * 1.5) if manual_action == "BUY" else entry_price + (atr_value * 1.5),
                    "target": entry_price + (atr_value * 2.5) if manual_action == "BUY" else entry_price - (atr_value * 2.5),
                    "conf": 0.8,
                    "score": 6,
                    "reason": "Manual Trade"
                }
                
                if trader.execute_trade(signal):
                    st.success(f"Manual {manual_action} executed for {manual_symbol} at ₹{entry_price:.2f}")
                else:
                    st.error("Trade execution failed")
    
    # Positions Management
    st.subheader("Position Management")
    positions_df = trader.get_positions_dataframe()
    if not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No active positions")

# Performance Tab
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
        if performance['avg_loss'] != 0:
            profit_factor = abs(performance['avg_win'] / performance['avg_loss'])
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        else:
            st.metric("Profit Factor", "∞")
    
    # Trade History
    st.subheader("Trade History")
    if trader.trade_log:
        trade_df = pd.DataFrame(trader.trade_log)
        trade_df['time'] = trade_df['time'].dt.strftime("%H:%M:%S")
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.info("No trade history yet")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "⚡ Pure Intraday Trading System | Live Market Data | "
    "Market Hours: 9:15 AM - 3:30 PM IST | v7.0"
    "</div>",
    unsafe_allow_html=True
)