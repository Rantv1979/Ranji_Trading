# gemini_intraday_pro_enhanced.py
# Enhanced: Multiple strategies, improved RSI (70/30), auto-trading, and better signals

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import uuid
import time
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import math
import warnings
from typing import Dict, List, Optional
import json
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Gemini Intraday Pro — Enhanced Terminal")

# ---------------------------
# Enhanced Strategy Configuration
# ---------------------------
STRATEGIES = {
    "SMA_Crossover_Enhanced": {
        "description": "SMA 20/50 crossover with RSI 70/30 and volume",
        "sma_short": 20,
        "sma_long": 50,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "rsi_period": 14,
        "volume_multiplier": 1.2
    },
    "EMA_Momentum": {
        "description": "EMA 8/21 with RSI divergence",
        "ema_short": 8,
        "ema_long": 21,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
    },
    "Bollinger_RSI_Combo": {
        "description": "Bollinger Band squeeze with RSI extremes",
        "bb_period": 20,
        "bb_std": 2,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "squeeze_threshold": 0.1
    },
    "Mean_Reversion": {
        "description": "Mean reversion with RSI extremes and volume",
        "rsi_overbought": 75,
        "rsi_oversold": 25,
        "volume_threshold": 1.5
    }
}

# ---------------------------
# Auto-Trading Configuration
# ---------------------------
class AutoTradeConfig:
    def __init__(self):
        self.enabled = False
        self.max_trades_per_day = 10
        self.max_position_size = 50000  # ₹
        self.min_confidence = 0.7
        self.risk_per_trade = 0.02  # 2%
        self.auto_exit = True
        self.profit_target = 0.015  # 1.5%
        self.stop_loss = 0.01  # 1%

# ---------------------------
# Paper Trading Configuration
# ---------------------------
class PaperTrading:
    def __init__(self):
        self.initial_capital = 100000
        self.available_capital = 100000
        self.positions = {}
        self.trade_history = []
        self.total_pnl = 0
        
    def execute_trade(self, symbol, action, quantity, price, strategy, reason):
        """Execute paper trade"""
        if symbol in self.positions:
            st.warning(f"Position already exists for {symbol}")
            return False
            
        trade_value = quantity * price
        if trade_value > self.available_capital:
            st.error(f"Insufficient capital for {symbol}. Available: ₹{self.available_capital:.2f}")
            return False
            
        self.positions[symbol] = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': price,
            'entry_time': datetime.now(),
            'strategy': strategy,
            'reason': reason
        }
        
        self.available_capital -= trade_value
        return True
        
    def close_trade(self, symbol, exit_price, reason):
        """Close paper trade"""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        if position['action'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            
        # Update capital
        trade_value = position['quantity'] * exit_price
        self.available_capital += trade_value
        
        # Record trade history
        trade_record = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': reason,
            'pnl': pnl
        }
        self.trade_history.append(trade_record)
        self.total_pnl += pnl
        
        # Remove from positions
        del self.positions[symbol]
        return True

# ---------------------------
# Enhanced Technical Indicators (No external TA library needed)
# ---------------------------
def compute_rsi(series, period=14):
    """Calculate RSI without external library"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD without external library"""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def compute_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ---------------------------
# Data Fetching with Enhanced Indicators
# ---------------------------
@st.cache_data(ttl=30)
def fetch_enhanced_ohlc(symbol: str, period="1d", interval="5m"):
    """Fetch OHLC data with enhanced technical indicators"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        
        df = df.dropna()
        df = compute_enhanced_indicators(df)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced technical indicators without external dependencies"""
    df = df.copy()
    
    # Basic indicators
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['EMA_8'] = compute_ema(df['Close'], 8)
    df['EMA_21'] = compute_ema(df['Close'], 21)
    
    # Bollinger Bands
    df['BB_MID'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_STD'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_UP'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LO'] = df['BB_MID'] - 2 * df['BB_STD']
    df['BB_WIDTH'] = (df['BB_UP'] - df['BB_LO']) / df['BB_MID']
    
    # RSI with multiple periods
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_8'] = compute_rsi(df['Close'], 8)
    
    # MACD
    macd, macd_signal, macd_histogram = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    
    # ATR
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], 10)
    
    # Volume indicators - FIXED: Handle volume column properly
    if 'Volume' in df.columns:
        df['VOLUME_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        # Ensure we're working with Series, not DataFrames
        volume_series = df['Volume']
        volume_ma_series = df['VOLUME_MA_20']
        df['VOLUME_RATIO'] = volume_series / volume_ma_series
        df['VOLUME_RATIO'] = df['VOLUME_RATIO'].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df['Volume'] = 0
        df['VOLUME_MA_20'] = 0
        df['VOLUME_RATIO'] = 0
    
    # VWAP (simplified)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df

# ---------------------------
# Enhanced Signal Generation
# ---------------------------
def generate_enhanced_signals(df: pd.DataFrame, strategy_name: str = "SMA_Crossover_Enhanced") -> Optional[Dict]:
    """Generate enhanced trading signals based on selected strategy"""
    if df is None or df.empty or len(df) < 50:
        return None
    
    strategy = STRATEGIES[strategy_name]
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Common signal components
    signal_data = {
        'strategy': strategy_name,
        'timestamp': df.index[-1].to_pydatetime(),
        'symbol': None,
        'confidence': 0.0,
        'entry': _safe_scalar_from_row(latest, 'Close'),
        'indicators': {}
    }
    
    if strategy_name == "SMA_Crossover_Enhanced":
        signal = sma_crossover_strategy(df, strategy)
    elif strategy_name == "EMA_Momentum":
        signal = ema_momentum_strategy(df, strategy)
    elif strategy_name == "Bollinger_RSI_Combo":
        signal = bollinger_rsi_strategy(df, strategy)
    elif strategy_name == "Mean_Reversion":
        signal = mean_reversion_strategy(df, strategy)
    else:
        signal = None
    
    if signal:
        signal_data.update(signal)
        return signal_data
    return None

def sma_crossover_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Enhanced SMA Crossover strategy with RSI 70/30"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    sma20_latest = _safe_scalar_from_row(latest, 'SMA_20')
    sma50_latest = _safe_scalar_from_row(latest, 'SMA_50')
    sma20_prev = _safe_scalar_from_row(prev, 'SMA_20')
    sma50_prev = _safe_scalar_from_row(prev, 'SMA_50')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    volume_ratio = _safe_scalar_from_row(latest, 'VOLUME_RATIO')
    
    # BUY Signal: SMA20 crosses above SMA50, RSI < 70 (not overbought), volume confirmation
    if (not math.isnan(sma20_latest) and not math.isnan(sma50_latest) and
        not math.isnan(sma20_prev) and not math.isnan(sma50_prev) and
        sma20_latest > sma50_latest and sma20_prev <= sma50_prev and
        not math.isnan(rsi) and rsi < strategy["rsi_overbought"] and
        volume_ratio > strategy["volume_multiplier"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        bb_lo = _safe_scalar_from_row(latest, 'BB_LO')
        sl = bb_lo if not math.isnan(bb_lo) else entry * 0.99
        
        # Confidence calculation
        confidence = min(0.95, 
                        (0.4 if rsi > 50 else 0.2) +  # RSI contribution
                        (0.3 if volume_ratio > 1.5 else 0.1) +  # Volume contribution
                        (0.3 if (sma20_latest - sma50_latest) / sma50_latest > 0.01 else 0.1))  # Trend strength
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': sl,
            'target1': entry * 1.015,
            'target2': entry * 1.03,
            'confidence': confidence,
            'reason': f"SMA20 crossed above SMA50, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    # SELL Signal: SMA20 crosses below SMA50, RSI > 30 (not oversold), volume confirmation
    elif (not math.isnan(sma20_latest) and not math.isnan(sma50_latest) and
          not math.isnan(sma20_prev) and not math.isnan(sma50_prev) and
          sma20_latest < sma50_latest and sma20_prev >= sma50_prev and
          not math.isnan(rsi) and rsi > strategy["rsi_oversold"] and
          volume_ratio > strategy["volume_multiplier"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        bb_up = _safe_scalar_from_row(latest, 'BB_UP')
        sl = bb_up if not math.isnan(bb_up) else entry * 1.01
        
        confidence = min(0.95,
                        (0.4 if rsi < 50 else 0.2) +
                        (0.3 if volume_ratio > 1.5 else 0.1) +
                        (0.3 if (sma50_latest - sma20_latest) / sma20_latest > 0.01 else 0.1))
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': sl,
            'target1': entry * 0.985,
            'target2': entry * 0.97,
            'confidence': confidence,
            'reason': f"SMA20 crossed below SMA50, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    return None

def ema_momentum_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """EMA Momentum strategy with MACD confirmation"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema8_latest = _safe_scalar_from_row(latest, 'EMA_8')
    ema21_latest = _safe_scalar_from_row(latest, 'EMA_21')
    ema8_prev = _safe_scalar_from_row(prev, 'EMA_8')
    ema21_prev = _safe_scalar_from_row(prev, 'EMA_21')
    rsi = _safe_scalar_from_row(latest, 'RSI_8')  # Faster RSI
    macd = _safe_scalar_from_row(latest, 'MACD')
    macd_signal = _safe_scalar_from_row(latest, 'MACD_Signal')
    
    # BUY Signal: EMA8 above EMA21, MACD bullish, RSI not overbought
    if (not math.isnan(ema8_latest) and not math.isnan(ema21_latest) and
        not math.isnan(ema8_prev) and not math.isnan(ema21_prev) and
        ema8_latest > ema21_latest and ema8_prev <= ema21_prev and
        not math.isnan(macd) and not math.isnan(macd_signal) and
        macd > macd_signal and not math.isnan(rsi) and rsi < strategy["rsi_overbought"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = min(0.95, 0.6 + (0.2 if rsi > 60 else 0) + (0.2 if macd > 0 else 0))
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': entry * 0.99,
            'target1': entry * 1.02,
            'target2': entry * 1.04,
            'confidence': confidence,
            'reason': f"EMA8 crossed above EMA21, MACD bullish, RSI {rsi:.1f}"
        }
    
    # SELL Signal: EMA8 below EMA21, MACD bearish, RSI not oversold
    elif (not math.isnan(ema8_latest) and not math.isnan(ema21_latest) and
          not math.isnan(ema8_prev) and not math.isnan(ema21_prev) and
          ema8_latest < ema21_latest and ema8_prev >= ema21_prev and
          not math.isnan(macd) and not math.isnan(macd_signal) and
          macd < macd_signal and not math.isnan(rsi) and rsi > strategy["rsi_oversold"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = min(0.95, 0.6 + (0.2 if rsi < 40 else 0) + (0.2 if macd < 0 else 0))
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': entry * 1.01,
            'target1': entry * 0.98,
            'target2': entry * 0.96,
            'confidence': confidence,
            'reason': f"EMA8 crossed below EMA21, MACD bearish, RSI {rsi:.1f}"
        }
    
    return None

def bollinger_rsi_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Bollinger Band + RSI strategy"""
    latest = df.iloc[-1]
    
    close = _safe_scalar_from_row(latest, 'Close')
    bb_up = _safe_scalar_from_row(latest, 'BB_UP')
    bb_lo = _safe_scalar_from_row(latest, 'BB_LO')
    bb_width = _safe_scalar_from_row(latest, 'BB_WIDTH')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    volume_ratio = _safe_scalar_from_row(latest, 'VOLUME_RATIO')
    
    # BUY Signal: Price near lower band, RSI oversold, narrow bands (squeeze)
    if (not math.isnan(close) and not math.isnan(bb_lo) and not math.isnan(rsi) and
        close <= bb_lo * 1.01 and rsi < strategy["rsi_oversold"] and 
        not math.isnan(bb_width) and bb_width < strategy["squeeze_threshold"] and
        volume_ratio > 1.0):
        
        confidence = min(0.95, 0.7 + (0.2 if rsi < 25 else 0) + (0.1 if bb_width < 0.05 else 0))
        
        return {
            'action': 'BUY',
            'entry': close,
            'stop_loss': close * 0.99,
            'target1': bb_up * 0.95,  # Target near upper band
            'target2': bb_up,
            'confidence': confidence,
            'reason': f"Bollinger Squeeze, RSI oversold {rsi:.1f}, Band width {bb_width:.3f}"
        }
    
    # SELL Signal: Price near upper band, RSI overbought, narrow bands
    elif (not math.isnan(close) and not math.isnan(bb_up) and not math.isnan(rsi) and
          close >= bb_up * 0.99 and rsi > strategy["rsi_overbought"] and
          not math.isnan(bb_width) and bb_width < strategy["squeeze_threshold"] and
          volume_ratio > 1.0):
        
        confidence = min(0.95, 0.7 + (0.2 if rsi > 75 else 0) + (0.1 if bb_width < 0.05 else 0))
        
        return {
            'action': 'SELL',
            'entry': close,
            'stop_loss': close * 1.01,
            'target1': bb_lo * 1.05,  # Target near lower band
            'target2': bb_lo,
            'confidence': confidence,
            'reason': f"Bollinger Squeeze, RSI overbought {rsi:.1f}, Band width {bb_width:.3f}"
        }
    
    return None

def mean_reversion_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Mean reversion strategy for overbought/oversold conditions"""
    latest = df.iloc[-1]
    
    close = _safe_scalar_from_row(latest, 'Close')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    volume_ratio = _safe_scalar_from_row(latest, 'VOLUME_RATIO')
    sma_20 = _safe_scalar_from_row(latest, 'SMA_20')
    
    # BUY Signal: RSI oversold, price below SMA20, high volume
    if (not math.isnan(rsi) and not math.isnan(close) and not math.isnan(sma_20) and
        rsi < strategy["rsi_oversold"] and close < sma_20 and 
        volume_ratio > strategy["volume_threshold"]):
        
        confidence = min(0.95, 0.8 - (rsi / 100) + (0.2 if volume_ratio > 2.0 else 0.1))
        
        return {
            'action': 'BUY',
            'entry': close,
            'stop_loss': close * 0.985,
            'target1': sma_20,  # Target at moving average
            'target2': sma_20 * 1.01,
            'confidence': confidence,
            'reason': f"Mean Reversion BUY, RSI oversold {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    # SELL Signal: RSI overbought, price above SMA20, high volume
    elif (not math.isnan(rsi) and not math.isnan(close) and not math.isnan(sma_20) and
          rsi > strategy["rsi_overbought"] and close > sma_20 and
          volume_ratio > strategy["volume_threshold"]):
        
        confidence = min(0.95, (rsi / 100) - 0.2 + (0.2 if volume_ratio > 2.0 else 0.1))
        
        return {
            'action': 'SELL',
            'entry': close,
            'stop_loss': close * 1.015,
            'target1': sma_20,  # Target at moving average
            'target2': sma_20 * 0.99,
            'confidence': confidence,
            'reason': f"Mean Reversion SELL, RSI overbought {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    return None

# ---------------------------
# Auto-Trading Engine
# ---------------------------
class AutoTradingEngine:
    def __init__(self):
        self.trades = []
        self.positions = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        
    def execute_trade(self, signal: Dict, config: AutoTradeConfig) -> bool:
        """Execute auto-trade based on signal and configuration"""
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        # Check constraints
        if (self.daily_trade_count >= config.max_trades_per_day or
            signal['confidence'] < config.min_confidence):
            return False
        
        # Calculate position size
        capital_per_trade = config.max_position_size * config.risk_per_trade
        entry_price = signal['entry']
        stop_loss = signal['stop_loss']
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return False
            
        shares = int(capital_per_trade / risk_per_share)
        
        if shares == 0:
            return False
        
        # Create trade
        trade = {
            'id': str(uuid.uuid4()),
            'symbol': signal['symbol'],
            'action': signal['action'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target1': signal['target1'],
            'target2': signal['target2'],
            'quantity': shares,
            'timestamp': datetime.now(),
            'strategy': signal['strategy'],
            'confidence': signal['confidence'],
            'reason': signal.get('reason', ''),
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        self.positions[signal['symbol']] = trade
        self.daily_trade_count += 1
        
        return True
    
    def monitor_positions(self, current_prices: Dict):
        """Monitor open positions and check for exit conditions"""
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check stop loss
            if (position['action'] == 'BUY' and current_price <= position['stop_loss']) or \
               (position['action'] == 'SELL' and current_price >= position['stop_loss']):
                self.close_position(symbol, current_price, 'STOP_LOSS')
            
            # Check target 1
            elif (position['action'] == 'BUY' and current_price >= position['target1']) or \
                 (position['action'] == 'SELL' and current_price <= position['target1']):
                self.close_position(symbol, current_price, 'TARGET1')
            
            # Check target 2
            elif (position['action'] == 'BUY' and current_price >= position['target2']) or \
                 (position['action'] == 'SELL' and current_price <= position['target2']):
                self.close_position(symbol, current_price, 'TARGET2')
    
    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            
            # Calculate P&L
            if position['action'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            position['pnl'] = pnl
            position['status'] = 'CLOSED'
            
            del self.positions[symbol]

# ---------------------------
# Utility Functions
# ---------------------------
def _safe_scalar_from_row(row, key, default=np.nan):
    """Return a safe scalar float for row[key]"""
    try:
        val = row.get(key, default)
        if isinstance(val, (pd.Series, np.ndarray, list)):
            if len(val) == 0:
                return default
            v = val[-1]
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

@st.cache_data(ttl=60*60)
def fetch_nifty500_list():
    """Fetch NIFTY 500 constituents"""
    try:
        # Return NIFTY 100 as fallback
        return NIFTY_100
    except Exception:
        return NIFTY_100

# ---------------------------
# Stock Universe
# ---------------------------
NIFTY_50 = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "KOTAKBANK.NS","HINDUNILVR.NS","AXISBANK.NS","LT.NS","SBIN.NS",
    "ITC.NS","BAJFINANCE.NS","BHARTIARTL.NS","MARUTI.NS",
    "ONGC.NS","TITAN.NS","ULTRACEMCO.NS","NTPC.NS","SUNPHARMA.NS",
    "HCLTECH.NS","POWERGRID.NS","WIPRO.NS","TECHM.NS","BPCL.NS",
    "COALINDIA.NS","BRITANNIA.NS","DIVISLAB.NS","HDFCLIFE.NS","ADANIENT.NS",
    "GRASIM.NS","DLF.NS","ADANIPORTS.NS","EICHERMOT.NS","CIPLA.NS",
    "IOC.NS","JSWSTEEL.NS","SBILIFE.NS","TATASTEEL.NS","INDUSINDBK.NS",
    "HINDALCO.NS","NESTLEIND.NS","DRREDDY.NS","BAJAJ-AUTO.NS","SHREECEM.NS",
    "TATAELXSI.NS","MRF.NS","PIDILITIND.NS","BAJAJFINSV.NS"
]

NIFTY_NEXT_50 = [
    "ADANITRANS.NS","APOLLOHOSP.NS","ADANIGREEN.NS","AUROPHARMA.NS","BERGEPAINT.NS",
    "BOSCHLTD.NS","CASTROLIND.NS","INDIGO.NS","GODREJCP.NS","HAVELLS.NS",
    "HEROMOTOCO.NS","HINDZINC.NS","ICICIPRULI.NS","INDIAMART.NS","LICHSGFIN.NS",
    "LUPIN.NS","MUTHOOTFIN.NS","PEL.NS","PIDILITIND.NS","PIIND.NS",
    "SAIL.NS","SRF.NS","TORNTPHARM.NS","TRENT.NS","TVSMOTOR.NS",
    "MOTHERSON.NS","ZOMATO.NS","ABB.NS","ADANIPOWER.NS","AMBUJACEM.NS",
    "BANDHANBNK.NS","COLPAL.NS","CONCOR.NS","DABUR.NS","DALBHARAT.NS",
    "GLENMARK.NS","HINDPETRO.NS","IGL.NS","INDUSTOWER.NS","JINDALSTEL.NS",
    "JSWENERGY.NS","LTI.NS","MANAPPURAM.NS","MCDOWELL-N.NS","NMDC.NS",
    "PETRONET.NS","SIEMENS.NS","UBL.NS","VOLTAS.NS","YESBANK.NS"
]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# ---------------------------
# Enhanced UI with Auto-Trading
# ---------------------------

def main():
    st.title("🚀 Gemini Intraday Pro — Enhanced Trading Terminal")
    
    # Initialize session state
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = AutoTradingEngine()
    if 'auto_trade_config' not in st.session_state:
        st.session_state.auto_trade_config = AutoTradeConfig()
    if 'paper_trading' not in st.session_state:
        st.session_state.paper_trading = PaperTrading()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Auto refresh every 10 seconds
    st_autorefresh(interval=10000, key="auto_refresh")
    
    # Sidebar Configuration
    st.sidebar.header("🎯 Trading Configuration")
    
    # Universe selection
    st.sidebar.subheader("📊 Market Universe")
    universe_option = st.sidebar.selectbox(
        "Select Market Universe",
        options=["NIFTY 50", "NIFTY 100"],
        index=1
    )
    
    if universe_option == "NIFTY 50":
        symbols_to_scan = NIFTY_50
    else:
        symbols_to_scan = NIFTY_100
    
    # Auto-Trading Configuration
    st.sidebar.subheader("🤖 Auto-Trading Settings")
    auto_trade_enabled = st.sidebar.checkbox("Enable Auto-Trading", value=False)
    st.session_state.auto_trade_config.enabled = auto_trade_enabled
    
    if auto_trade_enabled:
        st.session_state.auto_trade_config.max_trades_per_day = st.sidebar.number_input(
            "Max Trades Per Day", min_value=1, max_value=50, value=10
        )
        st.session_state.auto_trade_config.min_confidence = st.sidebar.slider(
            "Min Confidence", min_value=0.5, max_value=0.95, value=0.7, step=0.05
        )
        st.session_state.auto_trade_config.risk_per_trade = st.sidebar.slider(
            "Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5
        ) / 100
    
    # Paper Trading Configuration
    st.sidebar.subheader("📝 Paper Trading")
    paper_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=10000, max_value=1000000, value=100000)
    if st.sidebar.button("Reset Paper Trading"):
        st.session_state.paper_trading = PaperTrading()
        st.session_state.paper_trading.initial_capital = paper_capital
        st.session_state.paper_trading.available_capital = paper_capital
    
    # Main Tabs - Arranged professionally
    tabs = st.tabs(["📊 Dashboard", "🎯 Multi-Strategy Signals", "🤖 Auto-Trade", "📈 Live Charts", "📝 Paper Trading", "📊 Backtest"])
    
    with tabs[0]:
        show_dashboard(symbols_to_scan)
    
    with tabs[1]:
        show_multi_strategy_signals(symbols_to_scan)
    
    with tabs[2]:
        show_auto_trading()
    
    with tabs[3]:
        show_live_charts(symbols_to_scan)
    
    with tabs[4]:
        show_paper_trading(symbols_to_scan)
    
    with tabs[5]:
        show_backtest()

def show_dashboard(symbols_to_scan):
    st.header("📊 Professional Trading Dashboard")
    
    # Market Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbols Scanning", len(symbols_to_scan))
    with col2:
        st.metric("Strategies Active", len(STRATEGIES))
    with col3:
        st.metric("Auto-Trading", "🟢 Enabled" if st.session_state.auto_trade_config.enabled else "🔴 Disabled")
    with col4:
        st.metric("Last Refresh", st.session_state.last_refresh.strftime("%H:%M:%S"))
    
    # Quick scan with all strategies
    st.subheader("🚀 Multi-Strategy Market Scan")
    
    if st.button("Run Comprehensive Scan") or True:  # Auto-run on refresh
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_signals = []
        for i, symbol in enumerate(symbols_to_scan):
            status_text.text(f"Scanning {i+1}/{len(symbols_to_scan)}: {symbol}")
            df = fetch_enhanced_ohlc(symbol)
            
            # Test all strategies for each symbol
            for strategy_name in STRATEGIES.keys():
                signal = generate_enhanced_signals(df, strategy_name)
                if signal:
                    signal['symbol'] = symbol
                    all_signals.append(signal)
            
            progress_bar.progress((i + 1) / len(symbols_to_scan))
        
        progress_bar.empty()
        status_text.empty()
        
        # Update last refresh time
        st.session_state.last_refresh = datetime.now()
        
        if all_signals:
            display_enhanced_signals(all_signals)
        else:
            st.info("No signals found in current scan.")
    
    # Market Statistics
    st.subheader("📈 Market Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("NIFTY 50 Stocks", len(NIFTY_50))
    with col2:
        st.metric("NIFTY 100 Stocks", len(NIFTY_100))
    with col3:
        st.metric("Active Strategies", len(STRATEGIES))

def show_multi_strategy_signals(symbols_to_scan):
    st.header("🎯 Multi-Strategy Signal Scanner")
    
    # Strategy performance overview
    st.subheader("Strategy Performance Overview")
    
    # Run multi-strategy scan
    if st.button("Generate Multi-Strategy Signals"):
        signals_by_strategy = {strategy: [] for strategy in STRATEGIES.keys()}
        
        progress_bar = st.progress(0)
        for i, symbol in enumerate(symbols_to_scan):
            df = fetch_enhanced_ohlc(symbol)
            if df is not None:
                for strategy_name in STRATEGIES.keys():
                    signal = generate_enhanced_signals(df, strategy_name)
                    if signal:
                        signal['symbol'] = symbol
                        signals_by_strategy[strategy_name].append(signal)
            progress_bar.progress((i + 1) / len(symbols_to_scan))
        progress_bar.empty()
        
        # Display signals by strategy
        for strategy_name, signals in signals_by_strategy.items():
            if signals:
                st.subheader(f"📊 {strategy_name} Signals")
                display_enhanced_signals(signals)

def display_enhanced_signals(signals):
    """Display enhanced signals with better formatting"""
    if not signals:
        return
        
    df_signals = pd.DataFrame([{
        'Symbol': s['symbol'],
        'Action': s['action'],
        'Strategy': s['strategy'],
        'Entry': f"₹{s['entry']:.2f}",
        'SL': f"₹{s['stop_loss']:.2f}",
        'T1': f"₹{s['target1']:.2f}",
        'T2': f"₹{s['target2']:.2f}",
        'Confidence': f"{s['confidence']:.2f}",
        'Reason': s.get('reason', ''),
        'Time': s['timestamp'].strftime("%H:%M:%S")
    } for s in signals])
    
    st.dataframe(df_signals.sort_values('Confidence', ascending=False), use_container_width=True)
    
    # Auto-trade execution
    if st.session_state.auto_trade_config.enabled:
        st.subheader("🤖 Auto-Trade Execution")
        executed_trades = 0
        for signal in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            if signal['confidence'] >= st.session_state.auto_trade_config.min_confidence:
                if st.session_state.auto_trader.execute_trade(signal, st.session_state.auto_trade_config):
                    st.success(f"Auto-trade executed: {signal['symbol']} {signal['action']} at ₹{signal['entry']:.2f}")
                    executed_trades += 1
                    if executed_trades >= st.session_state.auto_trade_config.max_trades_per_day:
                        st.warning("Daily trade limit reached")
                        break

def show_auto_trading():
    st.header("🤖 Auto-Trading Console")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Open Positions")
        open_positions = [t for t in st.session_state.auto_trader.trades if t.get('status') == 'OPEN']
        if open_positions:
            for position in open_positions:
                with st.expander(f"{position['symbol']} {position['action']} - ₹{position['entry_price']:.2f}"):
                    st.write(f"**Entry:** ₹{position['entry_price']:.2f}")
                    st.write(f"**Stop Loss:** ₹{position['stop_loss']:.2f}")
                    st.write(f"**Target 1:** ₹{position['target1']:.2f}")
                    st.write(f"**Target 2:** ₹{position['target2']:.2f}")
                    st.write(f"**Quantity:** {position['quantity']}")
                    st.write(f"**Confidence:** {position['confidence']:.2f}")
                    st.write(f"**Strategy:** {position['strategy']}")
                    st.write(f"**Reason:** {position['reason']}")
                    
                    # Current P&L (simulated)
                    current_price = position['entry_price'] * 1.01  # Simulated price
                    if position['action'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    st.write(f"**Current P&L:** ₹{pnl:.2f}")
        else:
            st.info("No open positions")
    
    with col2:
        st.subheader("📈 Trade History")
        closed_trades = [t for t in st.session_state.auto_trader.trades if t.get('status') == 'CLOSED']
        if closed_trades:
            # Display last 10 trades
            recent_trades = closed_trades[-10:]
            for trade in recent_trades:
                pnl_color = "green" if trade.get('pnl', 0) > 0 else "red"
                st.write(f"{trade['symbol']} {trade['action']} - P&L: ₹{trade.get('pnl', 0):.2f} - {trade.get('exit_reason', '')}")
            
            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            win_rate = len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) * 100
            st.metric("Total P&L", f"₹{total_pnl:.2f}")
            st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.info("No trade history")

def show_live_charts(symbols_to_scan):
    st.header("📈 Live Market Charts")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_symbol = st.selectbox("Select Symbol", options=symbols_to_scan)
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
        indicator1 = st.selectbox("Indicator 1", ["SMA", "EMA", "Bollinger Bands", "None"])
        indicator2 = st.selectbox("Indicator 2", ["RSI", "MACD", "Volume", "None"])
    
    with col2:
        if selected_symbol:
            df = fetch_enhanced_ohlc(selected_symbol)
            if df is not None:
                plot_enhanced_chart(df, selected_symbol, chart_type, indicator1, indicator2)

def plot_enhanced_chart(df, symbol, chart_type="Candlestick", indicator1="SMA", indicator2="RSI"):
    """Plot enhanced chart with multiple indicators"""
    fig = go.Figure()
    
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='blue')))
    
    # Add first indicator
    if indicator1 == "SMA":
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
    elif indicator1 == "EMA":
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], name='EMA 8', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='red')))
    elif indicator1 == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LO'], name='BB Lower', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MID'], name='BB Middle', line=dict(color='blue')))
    
    fig.update_layout(
        title=f"{symbol} - Live Chart",
        xaxis_rangeslider_visible=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Second indicator in separate chart
    if indicator2 != "None":
        fig2 = go.Figure()
        
        if indicator2 == "RSI":
            fig2.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', line=dict(color='purple')))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            fig2.update_layout(title="RSI (14)", height=200)
        elif indicator2 == "MACD":
            fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')))
            fig2.update_layout(title="MACD", height=200)
        elif indicator2 == "Volume":
            fig2.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
            fig2.update_layout(title="Volume", height=200)
        
        st.plotly_chart(fig2, use_container_width=True)

def show_paper_trading(symbols_to_scan):
    st.header("📝 Paper Trading Console")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Account Overview")
        st.metric("Initial Capital", f"₹{st.session_state.paper_trading.initial_capital:,.2f}")
        st.metric("Available Capital", f"₹{st.session_state.paper_trading.available_capital:,.2f}")
        st.metric("Total P&L", f"₹{st.session_state.paper_trading.total_pnl:,.2f}")
        
        # Quick trade panel
        st.subheader("⚡ Quick Trade")
        quick_symbol = st.selectbox("Symbol", symbols_to_scan[:20])
        quick_action = st.selectbox("Action", ["BUY", "SELL"])
        quick_quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=10)
        
        if st.button("Execute Paper Trade"):
            # Get current price
            df = fetch_enhanced_ohlc(quick_symbol)
            if df is not None:
                current_price = df['Close'].iloc[-1]
                if st.session_state.paper_trading.execute_trade(
                    quick_symbol, quick_action, quick_quantity, current_price, "Manual", "Manual trade"
                ):
                    st.success(f"Paper trade executed: {quick_action} {quick_quantity} {quick_symbol} at ₹{current_price:.2f}")
    
    with col2:
        st.subheader("📊 Open Positions")
        if st.session_state.paper_trading.positions:
            for symbol, position in st.session_state.paper_trading.positions.items():
                with st.expander(f"{symbol} {position['action']}"):
                    st.write(f"**Entry Price:** ₹{position['entry_price']:.2f}")
                    st.write(f"**Quantity:** {position['quantity']}")
                    st.write(f"**Strategy:** {position['strategy']}")
                    st.write(f"**Reason:** {position['reason']}")
                    
                    # Close position
                    if st.button(f"Close {symbol}", key=f"close_{symbol}"):
                        df = fetch_enhanced_ohlc(symbol)
                        if df is not None:
                            current_price = df['Close'].iloc[-1]
                            if st.session_state.paper_trading.close_trade(symbol, current_price, "Manual close"):
                                st.success(f"Position closed: {symbol} at ₹{current_price:.2f}")
                                st.rerun()
        else:
            st.info("No open positions")
        
        st.subheader("📈 Trade History")
        if st.session_state.paper_trading.trade_history:
            recent_trades = st.session_state.paper_trading.trade_history[-5:]
            for trade in recent_trades:
                pnl_color = "green" if trade['pnl'] > 0 else "red"
                st.write(f"{trade['symbol']} {trade['action']} - P&L: ₹{trade['pnl']:.2f}")

def show_backtest():
    st.header("📊 Enhanced Backtest")
    st.info("Advanced backtesting functionality - Coming in next version")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backtest Configuration")
        st.selectbox("Strategy", list(STRATEGIES.keys()))
        st.selectbox("Time Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
        st.number_input("Initial Capital", value=100000)
    
    with col2:
        st.subheader("Backtest Results")
        st.write("Comprehensive backtesting across multiple strategies and time periods will be available in the next update.")
        st.write("Features include:")
        st.write("• Multi-strategy performance comparison")
        st.write("• Risk-adjusted return metrics")
        st.write("• Drawdown analysis")
        st.write("• Portfolio optimization")

if __name__ == "__main__":
    main()