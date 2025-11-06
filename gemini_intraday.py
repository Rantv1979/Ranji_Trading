# final_gemini_intraday.py - ADVANCED INTRADAY OPTIONS TRADING TERMINAL
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
import uuid
warnings.filterwarnings('ignore')
import yfinance as yf
import math
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# 1. SESSION STATE & DATA CLASSES
# ==============================================================================

# Initialize session state with proper widget key management
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.refresh_count = 0
    st.session_state.last_refresh = datetime.now()
    st.session_state.generated_signals = []
    st.session_state.signal_history = []
    st.session_state.executed_trades = []
    st.session_state.options_data = {}
    st.session_state.auto_refresh = True
    st.session_state.auto_execute = False  # Set to False by default for safety
    st.session_state.refresh_interval = 5  # Reduced for live charts
    st.session_state.tracked_symbols = set()
    st.session_state.last_candle_time = {}
    st.session_state.pending_execution = []
    st.session_state.support_resistance_data = {}
    st.session_state.execution_logs = []
    st.session_state.capital = 100000.00
    st.session_state.pnl = 0.00
    st.session_state.unrealized_pnl = 0.00
    st.session_state.portfolio = {} # {'symbol': {'quantity': X, 'entry_price': Y, 'type': 'CE'|'PE'}}
    st.session_state.signals_by_cohort = {} # NEW: Store signals categorized by index

# Data Classes
@dataclass
class Signal:
    id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY or SELL
    strategy: str
    confidence: float # 0.0 to 1.0
    entry: float
    stop_loss: float
    target1: float
    target2: float
    option_type: Optional[str] = None # CE or PE
    executed: bool = False

@dataclass
class Trade:
    id: str
    timestamp: datetime
    symbol: str
    action: str
    option_type: str
    strategy: str
    entry_price: float
    stop_loss: float
    target1: float
    target2: float
    quantity: int
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: float = 0.0

# ==============================================================================
# 2. CONFIGURATION & SYMBOL LISTS
# ==============================================================================

# Note: These lists are illustrative. For the full official lists, please refer to NSE website
# and use the corresponding Yahoo Finance symbols (.NS for NSE).

# --- NIFTY 50 ---
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", 
    "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "LT.NS", "SBIN.NS"
]

# --- NIFTY NEXT 50 ---
NIFTY_NEXT_50_SYMBOLS = [
    "ADANIENT.NS", "PIDILITIND.NS", "IOC.NS", "HINDALCO.NS", "WIPRO.NS",
    "GODREJCP.NS", "BERGEPAINT.NS", "INDIGO.NS", "DLF.NS", "ALKEM.NS"
]

# --- NIFTY 100 (Union of Nifty 50 and Nifty Next 50) ---
NIFTY_100_SYMBOLS = list(set(NIFTY_50_SYMBOLS + NIFTY_NEXT_50_SYMBOLS))

# MASTER LIST OF ALL SYMBOLS TO TRACK (Used for data fetching and signal generation)
ALL_TRACKED_SYMBOLS = NIFTY_100_SYMBOLS 

# Initial Symbol for Chart Display
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = NIFTY_50_SYMBOLS[0] if NIFTY_50_SYMBOLS else ""

# Set all symbols to be tracked
st.session_state.tracked_symbols = set(ALL_TRACKED_SYMBOLS)


# ==============================================================================
# 3. CORE LOGIC FUNCTIONS
# ==============================================================================

# Caching the expensive data fetching and processing
@st.cache_data(ttl=30) # Cache data for 30 seconds
def fetch_data_yf(symbol, period="1d", interval="5m"):
    """Fetches historical OHLCV data from Yahoo Finance."""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        # Clean up columns and ensure correct types
        data.columns = [col.capitalize() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df):
    """Calculates simple indicators: SMA, Bollinger Bands, RSI."""
    if df is None or df.empty:
        return None
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Bollinger Bands
    df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * 2)

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def generate_signals_for_symbol(symbol):
    """Generates a sample trading signal based on indicators."""
    data = fetch_data_yf(symbol)
    data = calculate_indicators(data)
    
    if data is None or data.empty:
        return None
    
    # Get the latest data point
    latest = data.iloc[-1]
    
    # Simple Crossover Strategy (SMA 20 vs SMA 50)
    if latest['SMA_20'] > latest['SMA_50'] and data.iloc[-2]['SMA_20'] <= data.iloc[-2]['SMA_50']:
        action = "BUY"
        strategy = "SMA Crossover"
        confidence = latest['RSI'] / 100.0 if latest['RSI'] else 0.5
        entry = latest['Close']
        stop_loss = latest['Bollinger_Lower'] if 'Bollinger_Lower' in latest else entry * 0.99
        target1 = entry * 1.015
        target2 = entry * 1.03
        option_type = 'CE'
        
        # Check if a similar signal was already generated and not executed
        if any(s.symbol == symbol and s.action == action and not s.executed for s in st.session_state.generated_signals):
            return None

        return Signal(
            id=str(uuid.uuid4()),
            timestamp=data.index[-1].to_pydatetime(),
            symbol=symbol,
            action=action,
            strategy=strategy,
            confidence=min(0.95, confidence),
            entry=entry,
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            option_type=option_type
        )

    elif latest['SMA_20'] < latest['SMA_50'] and data.iloc[-2]['SMA_20'] >= data.iloc[-2]['SMA_50']:
        action = "SELL"
        strategy = "SMA Crossover"
        confidence = (100.0 - latest['RSI']) / 100.0 if latest['RSI'] else 0.5
        entry = latest['Close']
        stop_loss = latest['Bollinger_Upper'] if 'Bollinger_Upper' in latest else entry * 1.01
        target1 = entry * 0.985
        target2 = entry * 0.97
        option_type = 'PE'

        if any(s.symbol == symbol and s.action == action and not s.executed for s in st.session_state.generated_signals):
            return None

        return Signal(
            id=str(uuid.uuid4()),
            timestamp=data.index[-1].to_pydatetime(),
            symbol=symbol,
            action=action,
            strategy=strategy,
            confidence=min(0.95, confidence),
            entry=entry,
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            option_type=option_type
        )

    return None

def generate_all_signals():
    """Generates signals for all configured cohorts."""
    # Note: Only clear the signals if we're not planning to use them for ongoing trade checks.
    # For a clean display of *new* signals per refresh, we clear and regenerate.
    # The execution logic should handle checking for signals to execute regardless of the list being cleared.
    
    # Define Cohorts for signal generation
    cohorts = {
        "Nifty 50": NIFTY_50_SYMBOLS,
        "Nifty Next 50": NIFTY_NEXT_50_SYMBOLS,
        "Nifty 100": NIFTY_100_SYMBOLS
    }

    st.session_state.signals_by_cohort = {}
    all_generated_signals = []
    
    for cohort_name, symbols in cohorts.items():
        cohort_signals = []
        for symbol in symbols:
            signal = generate_signals_for_symbol(symbol)
            if signal:
                cohort_signals.append(signal)
                # Ensure the execution logic can find the signal
                if not any(s.id == signal.id for s in st.session_state.signal_history):
                    st.session_state.signal_history.append(signal)
            
        st.session_state.signals_by_cohort[cohort_name] = cohort_signals
        all_generated_signals.extend(cohort_signals)
        
    # The main generated_signals list is used by the execution logic
    st.session_state.generated_signals = all_generated_signals
    
    # Update refresh stats
    st.session_state.refresh_count += 1
    st.session_state.last_refresh = datetime.now()


# ... (rest of the existing logic functions: update_pnl, execute_signal, execute_all_signals, etc. - assumed to be present and unchanged)
# The execution functions should iterate over st.session_state.generated_signals which is the combined list.

def update_pnl():
    """Calculates realized and unrealized PNL."""
    # ... (Implementation remains unchanged, but relies on st.session_state.portfolio)
    st.session_state.pnl = sum(t.pnl for t in st.session_state.executed_trades)
    st.session_state.unrealized_pnl = 0.0
    # For each open position, calculate unrealized PNL
    for symbol, pos in st.session_state.portfolio.items():
        # Quick price check (using the latest close from a quick fetch)
        try:
            latest_price = yf.Ticker(symbol).history(period="1m", interval="1m")['Close'].iloc[-1]
            if pos['type'] == 'CE': # BUY signal
                pnl_per_unit = latest_price - pos['entry_price']
            else: # SELL signal (or PE option)
                # Assuming PE (PUT option): Profit if price falls
                pnl_per_unit = pos['entry_price'] - latest_price
            
            st.session_state.unrealized_pnl += pnl_per_unit * pos['quantity']
        except:
            pass # Ignore if data fetch fails

def execute_signal(signal: Signal):
    """Simulates paper trading execution."""
    # Simple check: Ensure we have enough capital and not already executed
    if signal.executed or st.session_state.capital < signal.entry: # Simple capital check
        return

    # Determine quantity (Fixed lot size for simplicity)
    quantity = 100 
    
    # Paper execution
    st.session_state.capital -= signal.entry * quantity # Simulating option premium cost
    signal.executed = True
    
    # Log the trade in portfolio
    st.session_state.portfolio[signal.symbol] = {
        'quantity': quantity, 
        'entry_price': signal.entry, 
        'type': signal.option_type
    }
    
    # Log execution
    log_entry = f"✅ EXECUTED {signal.action} {signal.symbol} ({signal.option_type}) @ ₹{signal.entry:.2f}. Qty: {quantity}. SL: ₹{signal.stop_loss:.2f}."
    st.session_state.execution_logs.insert(0, log_entry)
    

def execute_all_signals():
    """Tries to execute all pending signals."""
    for signal in st.session_state.generated_signals:
        if not signal.executed and signal.confidence >= 0.7: # Example confidence filter
            execute_signal(signal)
            
# ==============================================================================
# 4. UI COMPONENTS (Streamlit)
# ==============================================================================

def display_custom_css():
    """Injects custom CSS for styling."""
    st.markdown("""
    <style>
    .main-header {color: #1E90FF; text-align: center;}
    .sidebar-header {color: #FFD700;}
    .stSpinner > div {
        border-top-color: #1E90FF;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #1f1f1f; 
    }
    .signal-buy { background-color: #d4edda; color: #155724; padding: 5px; margin-bottom: 5px; border-radius: 5px; border-left: 5px solid #28a745; font-size: 0.85rem;}
    .signal-sell { background-color: #f8d7da; color: #721c24; padding: 5px; margin-bottom: 5px; border-radius: 5px; border-left: 5px solid #dc3545; font-size: 0.85rem;}
    .metric-value {font-size: 2rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

def display_candlestick_chart(symbol, data):
    """Displays a candlestick chart with indicators."""
    if data is None or data.empty:
        st.warning(f"Chart data not available for {symbol}.")
        return

    # Candlestick trace
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    # Moving Average Traces
    if 'SMA_20' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)))
    if 'SMA_50' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', width=1)))

    # Bollinger Band Traces
    if 'Bollinger_Upper' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Upper'], mode='lines', name='Upper BB', line=dict(color='gray', width=0.5, dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Lower'], mode='lines', name='Lower BB', line=dict(color='gray', width=0.5, dash='dash')))

    fig.update_layout(
        title=f'{symbol} Intraday Chart (5m) with Indicators',
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_sidebar_analysis():
    """Displays controls and chart analysis in the sidebar."""
    st.sidebar.markdown("<h2 class='sidebar-header'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # Auto-refresh control
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto Refresh Data", 
        value=st.session_state.auto_refresh,
        help="Automatically refresh data, generate signals, and execute trades at the set interval."
    )
    
    # Auto-execute control
    st.session_state.auto_execute = st.sidebar.checkbox(
        "Auto Execute Signals (Paper)",
        value=st.session_state.auto_execute,
        help="Automatically executes generated signals into the paper portfolio."
    )

    # Refresh Interval
    st.session_state.refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=5, max_value=60, value=st.session_state.refresh_interval, step=5
    )

    st.sidebar.info(f"Last Refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')} (Count: {st.session_state.refresh_count})")
    
    st.sidebar.markdown("<h2 class='sidebar-header'>📊 Chart Analysis</h2>", unsafe_allow_html=True)

    # All symbols across all cohorts for selection
    all_symbols_list = sorted(list(set(ALL_TRACKED_SYMBOLS)))
    
    # Symbol selector
    st.session_state.selected_symbol = st.sidebar.selectbox(
        "Select Stock for Chart",
        options=all_symbols_list,
        index=all_symbols_list.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in all_symbols_list else 0
    )

    # Display chart for the selected symbol
    if st.session_state.selected_symbol:
        chart_data = fetch_data_yf(st.session_state.selected_symbol)
        chart_data = calculate_indicators(chart_data)
        st.sidebar.subheader(f"Intraday Chart: {st.session_state.selected_symbol}")
        display_candlestick_chart(st.session_state.selected_symbol, chart_data)


def display_trade_execution_panel():
    """Displays quick actions and manual execution."""
    st.subheader("⚡ Quick Actions & Manual Execute")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            generate_all_signals()
            execute_all_signals()
            update_pnl()
            st.rerun()
    
    with action_col2:
        if st.button("📊 Generate Signals", use_container_width=True):
            generate_all_signals()
            st.rerun()
            
    with action_col3:
        if st.button("🚀 Execute Pending", use_container_width=True):
            execute_all_signals()
            update_pnl()
            st.rerun()
            
    # Manual Trade Input (Simplified)
    # manual_symbol = st.selectbox("Manual Trade Symbol", options=ALL_TRACKED_SYMBOLS)
    # ... (Manual input logic omitted for brevity, focusing on requested features)

def display_portfolio_summary():
    """Displays the current portfolio summary and PNL."""
    update_pnl()
    st.subheader("💰 Portfolio Summary (Paper)")
    
    col_cap, col_pnl_real, col_pnl_unreal = st.columns(3)
    col_cap.metric("Capital", f"₹{st.session_state.capital:,.2f}")
    
    pnl_real_str = f"₹{st.session_state.pnl:,.2f}"
    pnl_unreal_str = f"₹{st.session_state.unrealized_pnl:,.2f}"
    
    col_pnl_real.metric("Realized PNL", pnl_real_str, delta=st.session_state.pnl)
    col_pnl_unreal.metric("Unrealized PNL", pnl_unreal_str, delta=st.session_state.unrealized_pnl)

    st.markdown("---")
    st.subheader("Current Positions")
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame([
            {
                'Symbol': s,
                'Quantity': data['quantity'],
                'Entry Price': f"₹{data['entry_price']:.2f}",
                'Type': data['type']
            }
            for s, data in st.session_state.portfolio.items()
        ])
        st.dataframe(portfolio_df, hide_index=True, use_container_width=True)
    else:
        st.info("No open positions in the paper portfolio.")

def display_execution_logs():
    """Displays the execution history."""
    st.subheader("📜 Execution Logs")
    if st.session_state.execution_logs:
        log_text = "\n".join([f"[{datetime.now().strftime('%H:%M:%S')}] {log}" for log in st.session_state.execution_logs])
        st.code(log_text, language='text')
    else:
        st.info("No execution history yet.")

def display_cohort_signals(cohort_name: str):
    """Displays signals filtered by the given cohort name."""
    signals = st.session_state.signals_by_cohort.get(cohort_name, [])
    
    if signals:
        st.markdown(f"**Generated Signals: {len(signals)}**")
        
        # Display signals in reverse chronological order
        for signal in reversed(signals):
            # Check if signal is executed to change color/style slightly (optional)
            if signal.executed:
                signal_class = "signal-buy" if signal.action == "BUY" else "signal-sell"
                executed_style = "opacity: 0.6; text-decoration: line-through;"
            else:
                signal_class = "signal-buy" if signal.action == "BUY" else "signal-sell"
                executed_style = ""
                
            st.markdown(f"""
            <div class="{signal_class}" style="{executed_style}">
                <strong>{signal.symbol}</strong> | {signal.action} {signal.option_type or ''} | 
                {signal.strategy} | Confidence: {signal.confidence:.1%} | 
                Entry: ₹{signal.entry:.2f} | SL: ₹{signal.stop_loss:.2f} | 
                T1: ₹{signal.target1:.2f} | Executed: {'✅' if signal.executed else '❌'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No active signals generated for **{cohort_name}** yet.")

# ==============================================================================
# 5. MAIN APP FUNCTION
# ==============================================================================

def main_app():
    display_custom_css()
    
    st.markdown("<h1 class='main-header'>📈 Intraday Options Trading Terminal (Paper)</h1>", unsafe_allow_html=True)

    # Autorefresh logic (runs the function every 'refresh_interval' seconds)
    if st.session_state.auto_refresh:
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="data_refresher")
        
        # Auto-generate and auto-execute on refresh tick
        with st.spinner(f"Refreshing data and running engine... (Every {st.session_state.refresh_interval}s)"):
            generate_all_signals()
            if st.session_state.auto_execute:
                execute_all_signals()
            update_pnl() # Update PNL after execution attempt

    # --- UI Layout ---
    display_sidebar_analysis()
    
    # Row 1: Execution Panel and Portfolio Summary
    col_exec, col_port = st.columns([1, 2])
    with col_exec:
        display_trade_execution_panel()
    
    with col_port:
        display_portfolio_summary()

    st.markdown("---")

    # --- TABS for Stock Cohorts and Signals ---
    tab_n50, tab_nn50, tab_n100 = st.tabs(["Nifty 50 Signals", "Nifty Next 50 Signals", "Nifty 100 Signals"])
    
    with tab_n50:
        display_cohort_signals("Nifty 50")
        
    with tab_nn50:
        display_cohort_signals("Nifty Next 50")
        
    with tab_n100:
        display_cohort_signals("Nifty 100")
        
    # Execution Logs at the bottom
    st.markdown("---")
    display_execution_logs()

if __name__ == '__main__':
    main_app()