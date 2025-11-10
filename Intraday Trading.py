# gemini_intraday_pro_final.py
# Final Version: Fixed errors, email notifications, server deployment ready

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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Gemini Intraday Pro — Trading Terminal")

# ---------------------------
# Email Configuration
# ---------------------------
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your.email@gmail.com",  # Change this to your email
    "sender_password": "your_app_password",  # Use App Password, not regular password
    "receiver_email": "rantv2002@gmail.com"
}

def send_email_notification(subject, message):
    """Send email notification for auto-trades"""
    try:
        # Check if email is configured
        if EMAIL_CONFIG["sender_email"] == "your.email@gmail.com":
            print("Email not configured. Please update EMAIL_CONFIG with your details.")
            return False
            
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender_email"]
        msg['To'] = EMAIL_CONFIG["receiver_email"]
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["receiver_email"], text)
        server.quit()
        
        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

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
    },
    "MACD_Momentum": {
        "description": "MACD crossover with volume confirmation",
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "volume_threshold": 1.2
    },
    "RSI_Divergence": {
        "description": "RSI divergence with price action",
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "divergence_lookback": 5
    },
    "Golden_Cross": {
        "description": "50/200 EMA Golden Cross strategy",
        "ema_short": 50,
        "ema_long": 200,
        "rsi_period": 14,
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
        self.min_confidence = 0.6  # Reduced for more signals
        self.risk_per_trade = 0.02  # 2%
        self.auto_exit = True
        self.profit_target = 0.015  # 1.5%
        self.stop_loss = 0.01  # 1%
        self.email_notifications = True

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
            return False
            
        trade_value = quantity * price
        if trade_value > self.available_capital:
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
            
        trade_value = position['quantity'] * exit_price
        self.available_capital += trade_value
        
        trade_record = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': reason,
            'pnl': pnl
        }
        self.trade_history.append(trade_record)
        self.total_pnl += pnl
        
        del self.positions[symbol]
        return True

# ---------------------------
# Enhanced Technical Indicators (Fixed Version)
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

def compute_bollinger_bands(close, period=20, std=2):
    """Calculate Bollinger Bands"""
    middle = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower

# ---------------------------
# Data Fetching with Enhanced Indicators (FIXED)
# ---------------------------
@st.cache_data(ttl=30)
def fetch_enhanced_ohlc(symbol: str, period="1d", interval="5m"):
    """Fetch OHLC data with enhanced technical indicators"""
    try:
        ticker = symbol.replace('.NS', '') + '.NS' if not symbol.endswith('.NS') else symbol
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        
        if df is None or df.empty:
            return None
        
        df = df.dropna()
        if len(df) < 50:  # Ensure enough data points
            return None
            
        df = compute_enhanced_indicators(df)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced technical indicators - FIXED VERSION"""
    df = df.copy()
    
    # Basic indicators
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['EMA_8'] = compute_ema(df['Close'], 8)
    df['EMA_21'] = compute_ema(df['Close'], 21)
    df['EMA_50'] = compute_ema(df['Close'], 50)
    df['EMA_200'] = compute_ema(df['Close'], 200)
    
    # Bollinger Bands - FIXED: Proper calculation
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df['Close'], 20, 2)
    df['BB_UP'] = bb_upper
    df['BB_MID'] = bb_middle
    df['BB_LO'] = bb_lower
    df['BB_WIDTH'] = (df['BB_UP'] - df['BB_LO']) / df['BB_MID']
    
    # FIXED: Safe BB_POSITION calculation
    bb_range = df['BB_UP'] - df['BB_LO']
    df['BB_POSITION'] = (df['Close'] - df['BB_LO']) / bb_range.replace(0, 1)  # Avoid division by zero
    
    # RSI with multiple periods
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_8'] = compute_rsi(df['Close'], 8)
    
    # MACD
    macd, macd_signal, macd_histogram = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    
    # ATR
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], 14)
    
    # Volume indicators - FIXED: Proper Series handling
    if 'Volume' in df.columns:
        volume_series = df['Volume']
        df['VOLUME_MA_20'] = volume_series.rolling(window=20, min_periods=1).mean()
        
        # Safe volume ratio calculation
        volume_ma = df['VOLUME_MA_20']
        volume_ratio = volume_series / volume_ma
        volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
        df['VOLUME_RATIO'] = volume_ratio
        
    else:
        df['Volume'] = 0
        df['VOLUME_MA_20'] = 0
        df['VOLUME_RATIO'] = 1.0
    
    # VWAP (simplified)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Price momentum
    df['MOMENTUM_5'] = df['Close'].pct_change(5) * 100
    
    return df

# ---------------------------
# Enhanced Signal Generation (Relaxed Conditions)
# ---------------------------
def generate_enhanced_signals(df: pd.DataFrame, strategy_name: str = "SMA_Crossover_Enhanced") -> Optional[Dict]:
    """Generate enhanced trading signals based on selected strategy"""
    if df is None or df.empty or len(df) < 50:
        return None
    
    strategy = STRATEGIES[strategy_name]
    
    # Strategy-specific signal generation
    if strategy_name == "SMA_Crossover_Enhanced":
        signal = sma_crossover_strategy(df, strategy)
    elif strategy_name == "EMA_Momentum":
        signal = ema_momentum_strategy(df, strategy)
    elif strategy_name == "Bollinger_RSI_Combo":
        signal = bollinger_rsi_strategy(df, strategy)
    elif strategy_name == "Mean_Reversion":
        signal = mean_reversion_strategy(df, strategy)
    elif strategy_name == "MACD_Momentum":
        signal = macd_momentum_strategy(df, strategy)
    elif strategy_name == "RSI_Divergence":
        signal = rsi_divergence_strategy(df, strategy)
    elif strategy_name == "Golden_Cross":
        signal = golden_cross_strategy(df, strategy)
    else:
        signal = None
    
    if signal:
        signal.update({
            'strategy': strategy_name,
            'timestamp': datetime.now(),
            'confidence': signal.get('confidence', 0.5)
        })
        return signal
    
    return None

def sma_crossover_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Enhanced SMA Crossover strategy with relaxed conditions"""
    if len(df) < 3:
        return None
        
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    sma20_latest = _safe_scalar_from_row(latest, 'SMA_20')
    sma50_latest = _safe_scalar_from_row(latest, 'SMA_50')
    sma20_prev = _safe_scalar_from_row(prev, 'SMA_20')
    sma50_prev = _safe_scalar_from_row(prev, 'SMA_50')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    volume_ratio = _safe_scalar_from_row(latest, 'VOLUME_RATIO')
    
    # Relaxed BUY Signal
    if (sma20_latest > sma50_latest and sma20_prev <= sma50_prev and
        rsi < strategy["rsi_overbought"] and volume_ratio > 1.0):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.6 + min(0.3, (rsi - 30) / 40)  # Higher confidence when RSI > 30
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': entry * 0.99,
            'target1': entry * 1.015,
            'target2': entry * 1.03,
            'confidence': confidence,
            'reason': f"SMA20 crossed above SMA50, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    # Relaxed SELL Signal
    elif (sma20_latest < sma50_latest and sma20_prev >= sma50_prev and
          rsi > strategy["rsi_oversold"] and volume_ratio > 1.0):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.6 + min(0.3, (70 - rsi) / 40)  # Higher confidence when RSI < 70
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': entry * 1.01,
            'target1': entry * 0.985,
            'target2': entry * 0.97,
            'confidence': confidence,
            'reason': f"SMA20 crossed below SMA50, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    return None

def ema_momentum_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """EMA Momentum strategy with relaxed conditions"""
    if len(df) < 3:
        return None
        
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema8_latest = _safe_scalar_from_row(latest, 'EMA_8')
    ema21_latest = _safe_scalar_from_row(latest, 'EMA_21')
    ema8_prev = _safe_scalar_from_row(prev, 'EMA_8')
    ema21_prev = _safe_scalar_from_row(prev, 'EMA_21')
    rsi = _safe_scalar_from_row(latest, 'RSI_8')
    
    # Relaxed BUY Signal
    if (ema8_latest > ema21_latest and ema8_prev <= ema21_prev and
        rsi < strategy["rsi_overbought"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.65
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': entry * 0.99,
            'target1': entry * 1.02,
            'target2': entry * 1.04,
            'confidence': confidence,
            'reason': f"EMA8 crossed above EMA21, RSI {rsi:.1f}"
        }
    
    # Relaxed SELL Signal
    elif (ema8_latest < ema21_latest and ema8_prev >= ema21_prev and
          rsi > strategy["rsi_oversold"]):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.65
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': entry * 1.01,
            'target1': entry * 0.98,
            'target2': entry * 0.96,
            'confidence': confidence,
            'reason': f"EMA8 crossed below EMA21, RSI {rsi:.1f}"
        }
    
    return None

def bollinger_rsi_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Bollinger Band + RSI strategy with relaxed conditions"""
    latest = df.iloc[-1]
    
    close = _safe_scalar_from_row(latest, 'Close')
    bb_up = _safe_scalar_from_row(latest, 'BB_UP')
    bb_lo = _safe_scalar_from_row(latest, 'BB_LO')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    
    # Relaxed BUY Signal
    if (close <= bb_lo * 1.02 and rsi < strategy["rsi_oversold"] + 10):  # More tolerant
        
        confidence = 0.7 - (rsi / 100)  # Higher confidence when RSI is lower
        
        return {
            'action': 'BUY',
            'entry': close,
            'stop_loss': close * 0.99,
            'target1': bb_up * 0.95,
            'target2': bb_up,
            'confidence': confidence,
            'reason': f"Near lower Bollinger Band, RSI oversold {rsi:.1f}"
        }
    
    # Relaxed SELL Signal
    elif (close >= bb_up * 0.98 and rsi > strategy["rsi_overbought"] - 10):  # More tolerant
        
        confidence = (rsi / 100) - 0.3  # Higher confidence when RSI is higher
        
        return {
            'action': 'SELL',
            'entry': close,
            'stop_loss': close * 1.01,
            'target1': bb_lo * 1.05,
            'target2': bb_lo,
            'confidence': confidence,
            'reason': f"Near upper Bollinger Band, RSI overbought {rsi:.1f}"
        }
    
    return None

def mean_reversion_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Mean reversion strategy with relaxed conditions"""
    latest = df.iloc[-1]
    
    close = _safe_scalar_from_row(latest, 'Close')
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    volume_ratio = _safe_scalar_from_row(latest, 'VOLUME_RATIO')
    sma_20 = _safe_scalar_from_row(latest, 'SMA_20')
    
    # Relaxed BUY Signal
    if (rsi < strategy["rsi_oversold"] + 15 and  # More tolerant
        close < sma_20 * 1.02 and  # Price can be slightly above SMA
        volume_ratio > 1.0):  # Lower volume threshold
        
        confidence = 0.7 - (rsi / 100)
        
        return {
            'action': 'BUY',
            'entry': close,
            'stop_loss': close * 0.985,
            'target1': sma_20,
            'target2': sma_20 * 1.02,
            'confidence': confidence,
            'reason': f"Mean Reversion BUY, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    # Relaxed SELL Signal
    elif (rsi > strategy["rsi_overbought"] - 15 and  # More tolerant
          close > sma_20 * 0.98 and  # Price can be slightly below SMA
          volume_ratio > 1.0):  # Lower volume threshold
        
        confidence = (rsi / 100) - 0.3
        
        return {
            'action': 'SELL',
            'entry': close,
            'stop_loss': close * 1.015,
            'target1': sma_20,
            'target2': sma_20 * 0.98,
            'confidence': confidence,
            'reason': f"Mean Reversion SELL, RSI {rsi:.1f}, Volume {volume_ratio:.1f}x"
        }
    
    return None

def macd_momentum_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """MACD Momentum strategy with relaxed conditions"""
    if len(df) < 3:
        return None
        
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    macd = _safe_scalar_from_row(latest, 'MACD')
    macd_signal = _safe_scalar_from_row(latest, 'MACD_Signal')
    macd_prev = _safe_scalar_from_row(prev, 'MACD')
    macd_signal_prev = _safe_scalar_from_row(prev, 'MACD_Signal')
    
    # Relaxed BUY Signal
    if (macd > macd_signal and macd_prev <= macd_signal_prev):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.6 + (0.2 if macd > 0 else 0)
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': entry * 0.99,
            'target1': entry * 1.02,
            'target2': entry * 1.04,
            'confidence': confidence,
            'reason': f"MACD bullish crossover"
        }
    
    # Relaxed SELL Signal
    elif (macd < macd_signal and macd_prev >= macd_signal_prev):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.6 + (0.2 if macd < 0 else 0)
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': entry * 1.01,
            'target1': entry * 0.98,
            'target2': entry * 0.96,
            'confidence': confidence,
            'reason': f"MACD bearish crossover"
        }
    
    return None

def rsi_divergence_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """RSI Divergence strategy"""
    if len(df) < 10:
        return None
    
    latest = df.iloc[-1]
    rsi = _safe_scalar_from_row(latest, 'RSI_14')
    
    # Simple RSI based signals
    if rsi < strategy["rsi_oversold"]:
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.7 - (rsi / 100)
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': entry * 0.985,
            'target1': entry * 1.02,
            'target2': entry * 1.04,
            'confidence': confidence,
            'reason': f"RSI oversold {rsi:.1f}"
        }
    
    elif rsi > strategy["rsi_overbought"]:
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = (rsi / 100) - 0.3
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': entry * 1.015,
            'target1': entry * 0.98,
            'target2': entry * 0.96,
            'confidence': confidence,
            'reason': f"RSI overbought {rsi:.1f}"
        }
    
    return None

def golden_cross_strategy(df: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
    """Golden Cross strategy with relaxed conditions"""
    if len(df) < 3:
        return None
        
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema50 = _safe_scalar_from_row(latest, 'EMA_50')
    ema200 = _safe_scalar_from_row(latest, 'EMA_200')
    ema50_prev = _safe_scalar_from_row(prev, 'EMA_50')
    ema200_prev = _safe_scalar_from_row(prev, 'EMA_200')
    
    # Relaxed Golden Cross
    if (ema50 > ema200 and ema50_prev <= ema200_prev):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.7
        
        return {
            'action': 'BUY',
            'entry': entry,
            'stop_loss': ema200,
            'target1': entry * 1.03,
            'target2': entry * 1.06,
            'confidence': confidence,
            'reason': f"Golden Cross (EMA50 > EMA200)"
        }
    
    # Relaxed Death Cross
    elif (ema50 < ema200 and ema50_prev >= ema200_prev):
        
        entry = _safe_scalar_from_row(latest, 'Close')
        confidence = 0.7
        
        return {
            'action': 'SELL',
            'entry': entry,
            'stop_loss': ema200,
            'target1': entry * 0.97,
            'target2': entry * 0.94,
            'confidence': confidence,
            'reason': f"Death Cross (EMA50 < EMA200)"
        }
    
    return None

# ---------------------------
# Auto-Trading Engine with Email Notifications
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
        
        # Send email notification
        if config.email_notifications:
            subject = f"Auto-Trade Executed: {signal['symbol']} {signal['action']}"
            message = f"""
            Auto-Trade Execution Details:
            
            Symbol: {signal['symbol']}
            Action: {signal['action']}
            Entry Price: ₹{entry_price:.2f}
            Quantity: {shares}
            Stop Loss: ₹{stop_loss:.2f}
            Target 1: ₹{signal['target1']:.2f}
            Target 2: ₹{signal['target2']:.2f}
            Strategy: {signal['strategy']}
            Confidence: {signal['confidence']:.2f}
            Reason: {signal.get('reason', 'N/A')}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Trade Value: ₹{entry_price * shares:.2f}
            """
            send_email_notification(subject, message)
        
        return True

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
            v = val.iloc[0] if hasattr(val, 'iloc') else val[0]
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

# ---------------------------
# Stock Universe
# ---------------------------
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ADANIENT.NS",
    "ONGC.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS",
    "SBILIFE.NS", "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS"
]

NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "DMART.NS", "BAJAJHLDNG.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "BIOCON.NS",
    "BOSCHLTD.NS", "CADILAHC.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
    "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HINDPETRO.NS", "ICICIPRULI.NS",
    "IGL.NS", "INFRATEL.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LUPIN.NS",
    "MANAPPURAM.NS", "MARICO.NS", "MOTHERSUMI.NS", "NHPC.NS", "OFSS.NS",
    "PETRONET.NS", "PIDILITIND.NS", "PEL.NS", "PGHH.NS", "PNB.NS",
    "RBLBANK.NS", "SAIL.NS", "SRF.NS", "SIEMENS.NS", "TORNTPHARM.NS",
    "TORNTPOWER.NS", "TRENT.NS", "UBL.NS", "MCDOWELL-N.NS", "VEDL.NS",
    "IDEA.NS", "YESBANK.NS", "ZEEL.NS"
]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# ---------------------------
# Main Application
# ---------------------------

def main():
    st.title("🚀 Gemini Intraday Pro — Trading Terminal")
    
    # Initialize session state
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = AutoTradingEngine()
    if 'auto_trade_config' not in st.session_state:
        st.session_state.auto_trade_config = AutoTradeConfig()
    if 'paper_trading' not in st.session_state:
        st.session_state.paper_trading = PaperTrading()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'total_scans' not in st.session_state:
        st.session_state.total_scans = 0
    if 'total_signals' not in st.session_state:
        st.session_state.total_signals = 0
    
    # Auto refresh every 10 seconds
    st_autorefresh(interval=10000, key="auto_refresh")
    
    # Sidebar Configuration
    st.sidebar.header("🎯 Trading Configuration")
    
    # Universe selection
    st.sidebar.subheader("📊 Market Universe")
    universe_option = st.sidebar.selectbox(
        "Select Market Universe",
        options=["NIFTY 50", "NIFTY 100"],
        index=0
    )
    
    if universe_option == "NIFTY 50":
        symbols_to_scan = NIFTY_50
    else:
        symbols_to_scan = NIFTY_100
    
    # Strategy selection
    st.sidebar.subheader("🎯 Strategy Selection")
    focus_strategies = st.sidebar.multiselect(
        "Select Strategies to Run",
        options=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        help="Select strategies for scanning"
    )
    
    # Auto-Trading Configuration
    st.sidebar.subheader("🤖 Auto-Trading Settings")
    auto_trade_enabled = st.sidebar.checkbox("Enable Auto-Trading", value=False)
    st.session_state.auto_trade_config.enabled = auto_trade_enabled
    
    if auto_trade_enabled:
        st.session_state.auto_trade_config.min_confidence = st.sidebar.slider(
            "Min Confidence", min_value=0.3, max_value=0.9, value=0.6, step=0.05
        )
        st.session_state.auto_trade_config.email_notifications = st.sidebar.checkbox(
            "Email Notifications", value=True
        )
        
        # Email configuration
        st.sidebar.subheader("📧 Email Configuration")
        st.sidebar.info("Configure email settings in the code (EMAIL_CONFIG dictionary)")
    
    # Paper Trading
    st.sidebar.subheader("📝 Paper Trading")
    if st.sidebar.button("Reset Paper Trading"):
        st.session_state.paper_trading = PaperTrading()
    
    # Main Tabs
    tabs = st.tabs(["📊 Dashboard", "🎯 Live Signals", "🤖 Auto-Trade", "📈 Charts", "📝 Paper Trading"])
    
    with tabs[0]:
        show_dashboard(symbols_to_scan, focus_strategies)
    
    with tabs[1]:
        show_live_signals(symbols_to_scan, focus_strategies)
    
    with tabs[2]:
        show_auto_trading()
    
    with tabs[3]:
        show_charts(symbols_to_scan)
    
    with tabs[4]:
        show_paper_trading(symbols_to_scan)

def show_dashboard(symbols_to_scan, focus_strategies):
    st.header("📊 Trading Dashboard")
    
    # Market Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbols Scanning", len(symbols_to_scan))
    with col2:
        st.metric("Active Strategies", len(focus_strategies))
    with col3:
        st.metric("Auto-Trading", "🟢 ON" if st.session_state.auto_trade_config.enabled else "🔴 OFF")
    with col4:
        st.metric("Last Scan", st.session_state.last_refresh.strftime("%H:%M:%S"))
    
    # Auto-scan on dashboard load
    st.subheader("🚀 Real-time Market Scan")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_signals = []
    total_scanned = 0
    signals_found = 0
    
    for i, symbol in enumerate(symbols_to_scan):
        status_text.text(f"Scanning {i+1}/{len(symbols_to_scan)}: {symbol}")
        df = fetch_enhanced_ohlc(symbol)
        total_scanned += 1
        
        if df is not None:
            for strategy_name in focus_strategies:
                signal = generate_enhanced_signals(df, strategy_name)
                if signal:
                    signal['symbol'] = symbol
                    all_signals.append(signal)
                    signals_found += 1
        
        progress_bar.progress((i + 1) / len(symbols_to_scan))
    
    progress_bar.empty()
    status_text.empty()
    
    # Update statistics
    st.session_state.total_scans += total_scanned
    st.session_state.total_signals += signals_found
    st.session_state.last_refresh = datetime.now()
    
    # Display results
    if all_signals:
        st.success(f"🎯 Found {len(all_signals)} trading signals!")
        
        # Display signals
        display_signals_table(all_signals)
        
        # Strategy distribution
        st.subheader("📈 Strategy Distribution")
        strategy_counts = {}
        for signal in all_signals:
            strategy = signal['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        for strategy, count in strategy_counts.items():
            st.write(f"**{strategy}**: {count} signals")
            
    else:
        st.warning("⚠️ No trading signals found in current scan.")
        st.info("""
        **Why no signals might be generated:**
        - Market conditions don't meet strategy criteria
        - Strategies might be too strict
        - Try adjusting confidence levels or using more strategies
        - Check if market is open and data is available
        """)
    
    # Performance statistics
    st.subheader("📊 Scan Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scans", st.session_state.total_scans)
    with col2:
        st.metric("Total Signals", st.session_state.total_signals)
    with col3:
        hit_rate = (st.session_state.total_signals / st.session_state.total_scans * 100) if st.session_state.total_scans > 0 else 0
        st.metric("Signal Hit Rate", f"{hit_rate:.1f}%")

def show_live_signals(symbols_to_scan, focus_strategies):
    st.header("🎯 Live Trading Signals")
    
    if st.button("Run Signal Scan"):
        with st.spinner("Scanning for signals..."):
            all_signals = []
            
            for symbol in symbols_to_scan:
                df = fetch_enhanced_ohlc(symbol)
                if df is not None:
                    for strategy_name in focus_strategies:
                        signal = generate_enhanced_signals(df, strategy_name)
                        if signal:
                            signal['symbol'] = symbol
                            all_signals.append(signal)
            
            if all_signals:
                st.success(f"Found {len(all_signals)} signals!")
                display_signals_table(all_signals)
                
                # Auto-trade execution
                if st.session_state.auto_trade_config.enabled:
                    st.subheader("🤖 Auto-Trade Execution")
                    executed_trades = 0
                    for signal in sorted(all_signals, key=lambda x: x['confidence'], reverse=True):
                        if signal['confidence'] >= st.session_state.auto_trade_config.min_confidence:
                            if st.session_state.auto_trader.execute_trade(signal, st.session_state.auto_trade_config):
                                st.success(f"Auto-trade executed: {signal['symbol']} {signal['action']}")
                                executed_trades += 1
                                if executed_trades >= st.session_state.auto_trade_config.max_trades_per_day:
                                    st.warning("Daily trade limit reached")
                                    break
            else:
                st.warning("No signals found. Try adjusting strategy parameters.")

def display_signals_table(signals):
    """Display signals in a formatted table"""
    if not signals:
        return
        
    # Sort by confidence
    signals_sorted = sorted(signals, key=lambda x: x['confidence'], reverse=True)
    
    # Create dataframe
    data = []
    for signal in signals_sorted:
        data.append({
            'Symbol': signal['symbol'],
            'Action': '🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL',
            'Strategy': signal['strategy'],
            'Entry': f"₹{signal['entry']:.2f}",
            'SL': f"₹{signal['stop_loss']:.2f}",
            'T1': f"₹{signal['target1']:.2f}",
            'Confidence': f"{signal['confidence']:.2f}",
            'Reason': signal.get('reason', 'N/A')
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=400)

def show_auto_trading():
    st.header("🤖 Auto-Trading Console")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Open Positions")
        open_positions = [t for t in st.session_state.auto_trader.trades if t.get('status') == 'OPEN']
        if open_positions:
            for position in open_positions:
                with st.expander(f"{position['symbol']} {position['action']}", expanded=True):
                    st.write(f"**Entry:** ₹{position['entry_price']:.2f}")
                    st.write(f"**Stop Loss:** ₹{position['stop_loss']:.2f}")
                    st.write(f"**Target 1:** ₹{position['target1']:.2f}")
                    st.write(f"**Target 2:** ₹{position['target2']:.2f}")
                    st.write(f"**Quantity:** {position['quantity']}")
                    st.write(f"**Confidence:** {position['confidence']:.2f}")
                    st.write(f"**Strategy:** {position['strategy']}")
        else:
            st.info("No open positions")
    
    with col2:
        st.subheader("📈 Trade History")
        closed_trades = [t for t in st.session_state.auto_trader.trades if t.get('status') == 'CLOSED']
        if closed_trades:
            for trade in closed_trades[-10:]:
                pnl_color = "green" if trade.get('pnl', 0) > 0 else "red"
                st.write(f"{trade['symbol']} {trade['action']} - P&L: ₹{trade.get('pnl', 0):.2f}")
        else:
            st.info("No trade history")

def show_charts(symbols_to_scan):
    st.header("📈 Live Charts")
    
    selected_symbol = st.selectbox("Select Symbol", symbols_to_scan)
    
    if selected_symbol:
        df = fetch_enhanced_ohlc(selected_symbol)
        if df is not None:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
            fig.update_layout(title=f"{selected_symbol} Price Chart", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators
            col1, col2 = st.columns(2)
            with col1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI (14)", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

def show_paper_trading(symbols_to_scan):
    st.header("📝 Paper Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Account Overview")
        st.metric("Initial Capital", f"₹{st.session_state.paper_trading.initial_capital:,.2f}")
        st.metric("Available Capital", f"₹{st.session_state.paper_trading.available_capital:,.2f}")
        st.metric("Total P&L", f"₹{st.session_state.paper_trading.total_pnl:,.2f}")
    
    with col2:
        st.subheader("📊 Open Positions")
        if st.session_state.paper_trading.positions:
            for symbol, position in st.session_state.paper_trading.positions.items():
                st.write(f"{symbol} {position['action']} - ₹{position['entry_price']:.2f}")
        else:
            st.info("No open positions")

# ---------------------------
# Server Deployment Functions
# ---------------------------
def run_headless_scan():
    """Function to run scans without browser for server deployment"""
    print("Starting headless market scan...")
    
    symbols_to_scan = NIFTY_100
    focus_strategies = list(STRATEGIES.keys())
    
    all_signals = []
    
    for symbol in symbols_to_scan:
        df = fetch_enhanced_ohlc(symbol)
        if df is not None:
            for strategy_name in focus_strategies:
                signal = generate_enhanced_signals(df, strategy_name)
                if signal:
                    signal['symbol'] = symbol
                    all_signals.append(signal)
    
    print(f"Scan completed. Found {len(all_signals)} signals.")
    
    # Execute auto-trades if enabled
    if st.session_state.auto_trade_config.enabled and all_signals:
        for signal in sorted(all_signals, key=lambda x: x['confidence'], reverse=True):
            if signal['confidence'] >= st.session_state.auto_trade_config.min_confidence:
                if st.session_state.auto_trader.execute_trade(signal, st.session_state.auto_trade_config):
                    print(f"Auto-trade executed: {signal['symbol']} {signal['action']}")
    
    return all_signals

if __name__ == "__main__":
    main()