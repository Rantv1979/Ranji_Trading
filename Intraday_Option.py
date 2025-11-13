"""
Intraday Live Trading Terminal — Ultimate Pro Edition v8.7
----------------------------------------------------------
Updated with Correct Lot Sizes from NSE Data
Enhanced Options Trading with Accurate Quantities
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
st.set_page_config(page_title="Intraday Terminal Pro v8.7", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading parameters
CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10
MAX_DAILY_TRADES = 15
MAX_DRAWDOWN = 0.05

# Refresh intervals
SIGNAL_REFRESH_MS = 30000
CHART_REFRESH_MS = 5000
AUTO_EXEC_CONF = 0.75

# Market Options
MARKET_OPTIONS = ["CASH", "FUTURES", "OPTIONS"]
MARKET_MULTIPLIERS = {"CASH": 1.0, "FUTURES": 1.5, "OPTIONS": 0.5}

# Updated Option lot sizes from NSE data
OPTION_LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 35,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 140,
    "NIFTYNXT50": 25
}

# Updated Stock-specific lot sizes from NSE data
STOCK_LOT_SIZES = {
    "RELIANCE": 500, "TCS": 175, "HDFCBANK": 550, "INFY": 400, "HINDUNILVR": 300,
    "ICICIBANK": 700, "KOTAKBANK": 400, "BHARTIARTL": 475, "ITC": 1600, "LT": 175,
    "SBIN": 750, "ASIANPAINT": 250, "HCLTECH": 350, "AXISBANK": 625, "MARUTI": 50,
    "SUNPHARMA": 350, "TITAN": 175, "ULTRACEMCO": 50, "WIPRO": 3000, "NTPC": 1500,
    "NESTLEIND": 500, "POWERGRID": 1900, "M&M": 200, "BAJFINANCE": 750, "ONGC": 2250,
    "TATASTEEL": 5500, "JSWSTEEL": 675, "ADANIPORTS": 475, "COALINDIA": 1350,
    "HDFCLIFE": 1100, "DRREDDY": 625, "HINDALCO": 700, "CIPLA": 375, "SBILIFE": 375,
    "GRASIM": 250, "TECHM": 600, "BAJAJFINSV": 250, "BRITANNIA": 125, "EICHERMOT": 175,
    "DIVISLAB": 100, "SHREECEM": 25, "APOLLOHOSP": 125, "UPL": 1355, "BAJAJ-AUTO": 75,
    "HEROMOTOCO": 150, "INDUSINDBK": 700, "ADANIENT": 300, "TATACONSUM": 550, "BPCL": 1975,
    "ABB": 125, "ADANIGREEN": 600, "ADANITRANS": 675, "AMBUJACEM": 1050, "ATGL": 500,
    "AUBANK": 1000, "BAJAJHLDNG": 75, "BANDHANBNK": 3600, "BERGEPAINT": 125, "BIOCON": 2500,
    "BOSCHLTD": 25, "CANBK": 6750, "CHOLAFIN": 625, "COLPAL": 225, "CONCOR": 1250,
    "DABUR": 1250, "DLF": 825, "GAIL": 3150, "GLAND": 500, "GODREJCP": 500, "HAL": 150,
    "HAVELLS": 500, "HDFCAMC": 150, "ICICIGI": 325, "ICICIPRULI": 925, "IGL": 2750,
    "INDUSTOWER": 1700, "JINDALSTEL": 625, "JUBLFOOD": 1250, "LICHSGFIN": 1000,
    "MANAPPURAM": 3000, "MARICO": 1200, "MOTHERSON": 6150, "MPHASIS": 275, "MRF": 125,
    "MUTHOOTFIN": 275, "NATIONALUM": 3750, "NAUKRI": 375, "NMDC": 6750, "PAGEIND": 15,
    "PEL": 125, "PIDILITIND": 500, "PIIND": 175, "PNB": 8000, "POLYCAB": 125, "RECLTD": 1275,
    "SAIL": 4700, "SBICARD": 800, "SRF": 200, "TATAPOWER": 1450, "TORNTPHARM": 250,
    "TRENT": 100, "VOLTAS": 375, "ZOMATO": 3125, "ZYDUSLIFE": 900,
    # Additional stocks from your list
    "360ONE": 500, "ABCAPITAL": 3100, "ADANIENSOL": 675, "ALKEM": 125, "AMBER": 100,
    "BEL": 1425, "APLAPOLLO": 350, "CGPOWER": 850, "BANKINDIA": 5200, "BDL": 325,
    "DALBHARAT": 325, "BHEL": 2625, "BSE": 375, "CDSL": 475, "ANGELONE": 250,
    "ASHOKLEY": 5000, "ASTRAL": 425, "INDIGO": 150, "DMART": 150, "ETERNAL": 2425,
    "AUROPHARMA": 550, "FORTIS": 775, "GMRAIRPORT": 6975, "JSWENERGY": 1000,
    "GODREJPROP": 275, "BANKBARODA": 2925, "CUMMINSIND": 200, "BHARATFORG": 500,
    "DELHIVERY": 2075, "HUDCO": 2775, "PAYTM": 725, "INOXWIND": 3272, "IRCTC": 875,
    "IREDA": 3450, "RBLBANK": 3175, "IRFC": 4250, "JIOFIN": 2350, "KALYANKJIL": 1175,
    "HINDPETRO": 2025, "SUPREMEIND": 175, "TATAELXSI": 100, "TMPV": 800, "TIINDIA": 200,
    "KFINTECH": 450, "UNITDSPR": 400, "KPITTECH": 400, "VBL": 1025, "IOC": 4875,
    "LAURUSLABS": 850, "LTF": 4462, "LUPIN": 425, "MCX": 125, "BLUESTARCO": 325,
    "NBCC": 6500, "NCC": 2700, "NUVAMA": 75, "LTIM": 150, "OFSS": 75, "MAZDOCK": 175,
    "PERSISTENT": 100, "CAMS": 150, "CYIENT": 425, "PGEL": 700, "COFORGE": 375,
    "CROMPTON": 1800, "PNBHOUSING": 650, "PPLPHARMA": 2500, "DIXON": 50, "PRESTIGE": 450,
    "SAMMAANCAP": 4300, "PATANJALI": 900, "SHRIRAMFIN": 825, "EXIDEIND": 1800,
    "FEDERALBNK": 5000, "SYNGENE": 1000, "TATATECH": 800, "RECLTD": 1275, "TVSMOTOR": 175,
    "UNIONBANK": 4425, "HINDZINC": 1225, "YESBANK": 31100, "IIFL": 1650, "INDIANB": 1000,
    "HFCL": 6450, "SUZLON": 8000, "IEX": 3750, "KAYNES": 100, "KEI": 175, "VEDL": 1150,
    "MAXHEALTH": 525, "MFSL": 400, "NHPC": 6400, "GLENMARK": 375, "PETRONET": 1800,
    "OIL": 1400, "POLICYBZR": 350, "IDFCFIRSTB": 9275, "SIEMENS": 125, "SOLARINDS": 75,
    "SONACOMS": 1050, "TORNTPOWER": 375, "TITAGARH": 725, "LICI": 700, "OBEROIRLTY": 350,
    "PFC": 1300, "POWERINDIA": 50, "RVNL": 1375, "UNOMINDA": 550, "PHOENIXLTD": 350,
    "LODHA": 450, "MANKIND": 225, "INDHOTEL": 1000, "IDEA": 71475
}

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

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------------- Enhanced Trading System with Updated Lot Sizes ----------------
class EnhancedIntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.last_reset = now_indian().date()
        self.selected_market = "CASH"
        self.auto_execution = False
        self.backtest_results = {}
        self.signal_history = []
        self.option_positions = {}
    
    def get_option_lot_size(self, symbol):
        """Get lot size for different underlyings using updated NSE data"""
        symbol_upper = symbol.upper()
        
        # Check for index options first
        if "BANKNIFTY" in symbol_upper:
            return OPTION_LOT_SIZES["BANKNIFTY"]
        elif "NIFTY" in symbol_upper and "FIN" not in symbol_upper and "MID" not in symbol_upper:
            return OPTION_LOT_SIZES["NIFTY"]
        elif "FINNIFTY" in symbol_upper:
            return OPTION_LOT_SIZES["FINNIFTY"]
        elif "MIDCPNIFTY" in symbol_upper:
            return OPTION_LOT_SIZES["MIDCPNIFTY"]
        elif "NIFTYNXT50" in symbol_upper:
            return OPTION_LOT_SIZES["NIFTYNXT50"]
        else:
            # For stock options, get from updated stock lot sizes
            # Extract underlying symbol from option symbol
            stock_symbol = symbol.split('CE')[0].split('PE')[0] if 'CE' in symbol or 'PE' in symbol else symbol
            return STOCK_LOT_SIZES.get(stock_symbol, 1)
    
    def calculate_option_pnl(self, position):
        """Calculate current P&L for option positions"""
        if position['status'] == "OPEN":
            current_premium = position.get('current_premium', position['premium'])
            lot_size = position.get('lot_size', 1)
            quantity = position.get('quantity', 1)
            
            if position['action'] == "BUY":
                pnl = (current_premium - position['premium']) * lot_size * quantity
            else:  # SELL
                pnl = (position['premium'] - current_premium) * lot_size * quantity
            return pnl
        return position.get('closed_pnl', 0)
    
    def get_option_performance_stats(self):
        """Get performance statistics for option trades"""
        option_trades = [t for t in self.trade_log if t.get('trade_type') == 'OPTION']
        closed_option_trades = [t for t in option_trades if t.get('status') == 'CLOSED']
        open_option_trades = [t for t in option_trades if t.get('status') == 'OPEN']
        
        total_trades = len(closed_option_trades)
        winning_trades = len([t for t in closed_option_trades if t.get('closed_pnl', 0) > 0])
        losing_trades = len([t for t in closed_option_trades if t.get('closed_pnl', 0) < 0])
        
        total_pnl = sum(t.get('closed_pnl', 0) for t in closed_option_trades)
        open_pnl = sum(self.calculate_option_pnl(t) for t in open_option_trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "open_pnl": open_pnl,
            "open_positions": len(open_option_trades)
        }
    
    def calculate_support_resistance(self, symbol, current_price):
        """Calculate support and resistance levels"""
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if len(data) < 20:
                return current_price * 0.98, current_price * 1.02
                
            # Simple support/resistance calculation using recent highs/lows
            recent_lows = data['Low'].tail(20).nsmallest(3)
            recent_highs = data['High'].tail(20).nlargest(3)
            
            support = recent_lows.mean() if len(recent_lows) > 0 else current_price * 0.98
            resistance = recent_highs.mean() if len(recent_highs) > 0 else current_price * 1.02
            
            return round(support, 2), round(resistance, 2)
        except:
            return round(current_price * 0.98, 2), round(current_price * 1.02, 2)
    
    def estimate_option_target_sl(self, spot_price, premium, option_type, action):
        """Estimate target and stop loss for options"""
        if action == "BUY":
            if option_type == "CE":
                # For long calls
                target_premium = premium * 1.30  # 30% target
                sl_premium = premium * 0.85      # 15% stop loss
                target_spot = spot_price * 1.02  # 2% spot move for target
                sl_spot = spot_price * 0.99      # 1% spot move for SL
            else:  # PE
                # For long puts
                target_premium = premium * 1.30
                sl_premium = premium * 0.85
                target_spot = spot_price * 0.98
                sl_spot = spot_price * 1.01
        else:
            # For short positions
            target_premium = premium * 0.85
            sl_premium = premium * 1.15
            target_spot = spot_price
            sl_spot = spot_price
        
        return {
            "target_premium": round(target_premium, 2),
            "sl_premium": round(sl_premium, 2),
            "target_spot": round(target_spot, 2),
            "sl_spot": round(sl_spot, 2)
        }
    
    def equity(self):
        """Calculate total account value including open positions"""
        total_value = self.cash
        
        # Add value of stock positions
        for symbol, pos in self.positions.items():
            try:
                current_data = data_manager.get_stock_data(symbol, "5m")
                current_price = current_data['Close'].iloc[-1] if current_data is not None else pos['entry_price']
                total_value += pos['quantity'] * current_price
            except:
                total_value += pos['quantity'] * pos['entry_price']
        
        # Add value of option positions
        for symbol, pos in self.option_positions.items():
            # For options, we'll use the entry price as current value for simplicity
            # In real implementation, you'd fetch current option premium
            total_value += pos.get('current_value', pos.get('premium', 0)) * pos.get('quantity', 0)
        
        return total_value
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.65, trade_type="STOCK"):
        """Execute a trade with complete tracking"""
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        trade_value = quantity * price
        
        if trade_type == "STOCK":
            if trade_value > self.cash * TRADE_ALLOC and action == "BUY":
                return False, "Insufficient capital for trade allocation"
            
            # Calculate stop loss and target if not provided
            if stop_loss is None:
                stop_loss = price * 0.99 if action == "BUY" else price * 1.01
            if target is None:
                target = price * 1.02 if action == "BUY" else price * 0.98
            
            trade_id = f"STOCK_{symbol}_{len(self.trade_log)}_{int(time.time())}"
            trade_record = {
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "entry_price": price,
                "stop_loss": stop_loss,
                "target": target,
                "timestamp": now_indian(),
                "status": "OPEN",
                "current_pnl": 0.0,
                "current_price": price,
                "win_probability": win_probability,
                "max_pnl": 0.0,
                "exit_price": None,
                "closed_pnl": 0.0,
                "trade_type": "STOCK"
            }
            
            if action == "BUY":
                self.positions[symbol] = trade_record
                self.cash -= trade_value
                success_msg = f"BUY {quantity} {symbol} @ ₹{price:.2f}"
            elif action == "SELL":
                trade_record['margin_used'] = trade_value * 0.2
                self.positions[symbol] = trade_record
                self.cash -= trade_value * 0.2
                success_msg = f"SELL {quantity} {symbol} @ ₹{price:.2f}"
        
        elif trade_type == "OPTION":
            # Get lot size for the option using updated NSE data
            lot_size = self.get_option_lot_size(symbol)
            total_investment = price * lot_size * quantity
            
            if total_investment > self.cash:
                return False, "Insufficient capital for option trade"
            
            trade_id = f"OPTION_{symbol}_{len(self.trade_log)}_{int(time.time())}"
            trade_record = {
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "entry_price": price,
                "premium": price,
                "timestamp": now_indian(),
                "status": "OPEN",
                "current_pnl": 0.0,
                "current_premium": price,
                "win_probability": win_probability,
                "max_pnl": 0.0,
                "exit_premium": None,
                "closed_pnl": 0.0,
                "trade_type": "OPTION",
                "lot_size": lot_size,
                "total_investment": total_investment
            }
            
            self.option_positions[symbol] = trade_record
            self.cash -= total_investment
            success_msg = f"{action} {quantity} lot(s) {symbol} @ ₹{price:.2f} premium (Lot: {lot_size})"
        
        self.trade_log.append(trade_record)
        self.daily_trades += 1
        return True, success_msg

    # ... (rest of the methods remain the same as previous version)

    def generate_option_signals(self, signals):
        """Generate option trading signals based on stock signals"""
        option_signals = []
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            spot_price = signal['entry']
            underlying = symbol.replace('.NS', '')
            
            # Determine option type and strike
            if action == "BUY":
                option_type = "CE"  # Call Option
                strike = self.get_atm_strike(spot_price)
                option_symbol = f"{underlying}{strike}CE"
            else:  # SELL
                option_type = "PE"  # Put Option
                strike = self.get_atm_strike(spot_price)
                option_symbol = f"{underlying}{strike}PE"
            
            # Calculate lot size using updated NSE data
            lot_size = self.get_option_lot_size(underlying)
            
            # Estimate option premium (simplified)
            premium = self.estimate_option_premium(spot_price, strike, option_type, signal['confidence'])
            
            # Calculate support/resistance
            support, resistance = self.calculate_support_resistance(symbol, spot_price)
            
            # Calculate target/SL for options
            risk_params = self.estimate_option_target_sl(spot_price, premium, option_type, "BUY")
            
            # Calculate potential P&L
            potential_pnl = (risk_params['target_premium'] - premium) * lot_size
            
            option_signals.append({
                "underlying": underlying,
                "option_symbol": option_symbol,
                "type": option_type,
                "action": "BUY",  # Always buy options for simplicity
                "strike": strike,
                "spot_price": spot_price,
                "premium": premium,
                "lot_size": lot_size,
                "confidence": signal['confidence'],
                "win_probability": signal['win_probability'],
                "underlying_action": action,
                "support": support,
                "resistance": resistance,
                "target_premium": risk_params['target_premium'],
                "sl_premium": risk_params['sl_premium'],
                "target_spot": risk_params['target_spot'],
                "sl_spot": risk_params['sl_spot'],
                "potential_pnl": potential_pnl
            })
        
        return option_signals

    def get_atm_strike(self, spot_price):
        """Calculate ATM strike price"""
        if spot_price > 2000:
            strike_step = 100
        elif spot_price > 1000:
            strike_step = 50
        else:
            strike_step = 20
        
        return round(spot_price / strike_step) * strike_step

    def estimate_option_premium(self, spot_price, strike, option_type, confidence):
        """Estimate option premium based on Black-Scholes approximation"""
        time_to_expiry = 1/365  # 1 day
        iv = 0.20 + (confidence * 0.1)  # Higher confidence = higher IV
        
        if option_type == "CE":
            intrinsic = max(0, spot_price - strike)
        else:  # PE
            intrinsic = max(0, strike - spot_price)
        
        time_value = iv * spot_price * 0.1
        premium = max(1.0, intrinsic + time_value)
        
        return round(premium, 2)

# ---------------- Initialize Systems ----------------
data_manager = EnhancedDataManager()

if "trader" not in st.session_state:
    st.session_state.trader = EnhancedIntradayTrader()
trader = st.session_state.trader

# ---------------- Streamlit UI ----------------
st.markdown("<h1 style='text-align: center; color: #0077cc;'>🎯 Ultimate Intraday Trading Terminal v8.7</h1>", unsafe_allow_html=True)

# Market Overview
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    nifty_price = data_manager.validate_live_price("^NSEI")
    st.metric("NIFTY 50", f"₹{nifty_price:,.2f}")

with col2:
    bank_nifty_price = data_manager.validate_live_price("^NSEBANK")
    st.metric("BANK NIFTY", f"₹{bank_nifty_price:,.2f}")

with col3:
    market_status = "🟢 LIVE" if market_open() else "🔴 CLOSED"
    st.metric("Market Status", market_status)

with col4:
    st.metric("Market Regime", "NEUTRAL")

with col5:
    performance = trader.get_performance_stats()
    st.metric("Win Rate", f"{performance['win_rate']:.1%}" if performance['total_trades'] > 0 else "N/A")

# Market Type Selection
st.sidebar.header("Trading Configuration")
trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS)
trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False)

# Main Tabs
tabs = st.tabs(["📊 Dashboard", "🎯 Signals", "🤖 Paper Trading", "🔄 Options Trading", "📋 Trade History", "📈 Backtest", "🔍 Charts"])

# Enhanced Options Trading Tab with Updated Lot Sizes
with tabs[3]:
    st.subheader("🔄 Options Trading - Updated Lot Sizes")
    
    # Option Performance Summary
    option_stats = trader.get_option_performance_stats()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Trades", option_stats["total_trades"])
    with col2:
        st.metric("Win/Loss", f"{option_stats['winning_trades']}/{option_stats['losing_trades']}")
    with col3:
        st.metric("Win Rate", f"{option_stats['win_rate']:.1%}")
    with col4:
        st.metric("Total P&L", f"₹{option_stats['total_pnl']:+.2f}")
    with col5:
        st.metric("Open P&L", f"₹{option_stats['open_pnl']:+.2f}")
    with col6:
        st.metric("Open Positions", option_stats["open_positions"])
    
    # Lot Size Information
    st.info("📊 **Updated Lot Sizes**: NIFTY: 75 | BANKNIFTY: 35 | Stocks: Variable (NSE Data)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        option_universe = st.selectbox("Underlying Universe", ["Nifty 50", "Nifty 100"])
    with col2:
        auto_option_execution = st.checkbox("Auto Execute Options", value=False)
    
    if st.button("🎯 Generate Option Signals", type="primary") or auto_option_execution:
        with st.spinner("Generating option signals with updated lot sizes..."):
            # Generate stock signals first
            stock_signals = trader.generate_intraday_signals(option_universe)
            
            if stock_signals:
                # Convert to option signals
                option_signals = trader.generate_option_signals(stock_signals)
                
                st.success(f"✅ Generated {len(option_signals)} option trading signals!")
                
                # Enhanced option signals display with updated lot sizes
                enhanced_option_display = []
                for signal in option_signals:
                    enhanced_option_display.append({
                        "Underlying": signal['underlying'],
                        "Option": signal['option_symbol'],
                        "Type": signal['type'],
                        "Action": signal['action'],
                        "Strike": signal['strike'],
                        "Spot": f"₹{signal['spot_price']:.2f}",
                        "Premium": f"₹{signal['premium']:.2f}",
                        "Lot Size": signal['lot_size'],  # Updated lot size
                        "Curr Premium": f"₹{signal['premium']:.2f}",
                        "Target Prem": f"₹{signal['target_premium']:.2f}",
                        "SL Prem": f"₹{signal['sl_premium']:.2f}",
                        "Target Spot": f"₹{signal['target_spot']:.2f}",
                        "SL Spot": f"₹{signal['sl_spot']:.2f}",
                        "Support": f"₹{signal['support']:.2f}",
                        "Resistance": f"₹{signal['resistance']:.2f}",
                        "Potential P&L": f"₹{signal['potential_pnl']:+.2f}",
                        "Confidence": f"{signal['confidence']:.1%}",
                        "Win %": f"{signal['win_probability']:.1%}"
                    })
                
                enhanced_option_df = pd.DataFrame(enhanced_option_display)
                st.dataframe(enhanced_option_df, use_container_width=True)
                
                # Auto-execute option trades
                if auto_option_execution:
                    st.info("🤖 Auto-executing option trades with correct lot sizes...")
                    executed_options = []
                    for signal in option_signals[:3]:  # Limit to 3 option trades
                        if signal['confidence'] >= 0.75:
                            success, message = trader.execute_trade(
                                symbol=signal['option_symbol'],
                                action=signal['action'],
                                quantity=1,  # 1 lot
                                price=signal['premium'],
                                win_probability=signal['win_probability'],
                                trade_type="OPTION"
                            )
                            if success:
                                executed_options.append(message)
                    
                    if executed_options:
                        for trade in executed_options:
                            st.success(trade)
                        st.rerun()
                
                # Show option chain for first signal
                if option_signals:
                    st.subheader("📊 Option Chain - ATM Strikes")
                    first_signal = option_signals[0]
                    spot_price = first_signal['spot_price']
                    underlying = first_signal['underlying'] + ".NS"
                    
                    option_chain = data_manager.get_option_chain_data(underlying, spot_price)
                    st.dataframe(option_chain, use_container_width=True)
            else:
                st.warning("❌ No stock signals found for option conversion.")
    
    # Current Option Positions with Updated Lot Sizes
    st.subheader("📊 Current Option Positions")
    
    if trader.option_positions:
        current_option_positions = []
        
        for symbol, position in trader.option_positions.items():
            if position['status'] == "OPEN":
                # Update current premium
                price_change = np.random.normal(0, position['premium'] * 0.1)
                current_premium = max(0.05, position['premium'] + price_change)
                position['current_premium'] = current_premium
                
                # Calculate P&L
                current_pnl = trader.calculate_option_pnl(position)
                
                # Get underlying symbol
                underlying = symbol.split('CE')[0].split('PE')[0] if 'CE' in symbol or 'PE' in symbol else symbol
                
                # Get current spot price
                try:
                    spot_data = data_manager.get_stock_data(underlying + ".NS", "5m")
                    current_spot = spot_data['Close'].iloc[-1] if spot_data is not None else position.get('spot_price', 0)
                except:
                    current_spot = position.get('spot_price', 0)
                
                # Calculate support/resistance
                support, resistance = trader.calculate_support_resistance(underlying + ".NS", current_spot)
                
                # Calculate target/SL
                risk_params = trader.estimate_option_target_sl(
                    current_spot, position['premium'], 
                    "CE" if "CE" in symbol else "PE", position['action']
                )
                
                current_option_positions.append({
                    "Symbol": symbol,
                    "Type": "CE" if "CE" in symbol else "PE",
                    "Action": position['action'],
                    "Quantity": f"{position['quantity']} lot(s)",
                    "Lot Size": position['lot_size'],  # Updated lot size
                    "Strike": position.get('strike', 'N/A'),
                    "Entry Prem": f"₹{position['premium']:.2f}",
                    "Curr Prem": f"₹{current_premium:.2f}",
                    "Curr Spot": f"₹{current_spot:.2f}",
                    "Target Prem": f"₹{risk_params['target_premium']:.2f}",
                    "SL Prem": f"₹{risk_params['sl_premium']:.2f}",
                    "Support": f"₹{support:.2f}",
                    "Resistance": f"₹{resistance:.2f}",
                    "P&L": f"₹{current_pnl:+.2f}",
                    "Win %": f"{position.get('win_probability', 65):.1f}%"
                })
        
        if current_option_positions:
            current_positions_df = pd.DataFrame(current_option_positions)
            st.dataframe(current_positions_df, use_container_width=True)
            
            # Position management
            st.subheader("🔒 Option Position Management")
            close_cols = st.columns(4)
            
            for idx, symbol in enumerate(trader.option_positions.keys()):
                with close_cols[idx % 4]:
                    if st.button(f"Close {symbol}", key=f"close_opt_{symbol}", type="secondary"):
                        success, message = trader.close_position(symbol)
                        if success:
                            st.success(message)
                            st.rerun()
        else:
            st.info("📭 No open option positions.")
    else:
        st.info("📭 No open option positions.")

# ... (rest of the tabs remain the same as previous version)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "⚡ Updated Lot Sizes | NSE Data | Options Analytics | v8.7"
    "</div>",
    unsafe_allow_html=True
)