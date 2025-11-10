# gemini_intraday_updated.py
# Advanced Intraday Options Trading Terminal (Updated with Paper Trading, Trade Log, Live Chart tabs)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings
import uuid
warnings.filterwarnings('ignore')
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import math

# -------------------------
# 1. SESSION STATE INIT
# -------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.refresh_count = 0
    st.session_state.last_refresh = datetime.now()
    st.session_state.generated_signals = []
    st.session_state.signal_history = []
    st.session_state.executed_trades = []  # list of Trade dataclass instances
    st.session_state.options_data = {}
    st.session_state.auto_refresh = True
    st.session_state.auto_execute = False
    st.session_state.refresh_interval = 10
    st.session_state.tracked_symbols = set()
    st.session_state.pending_execution = []
    st.session_state.execution_logs = []
    st.session_state.capital = 100000.00
    st.session_state.pnl = 0.00
    st.session_state.unrealized_pnl = 0.00
    st.session_state.portfolio = {}  # {'symbol': {'quantity': X, 'entry_price': Y, 'type': 'CE'|'PE'}}
    st.session_state.signals_by_cohort = {}
    st.session_state.selected_symbol = None
    st.session_state.live_chart_ref = 0

# -------------------------
# 2. DATACLASSES
# -------------------------
@dataclass
class Signal:
    id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY or SELL
    strategy: str
    confidence: float
    entry: float
    stop_loss: float
    target1: float
    target2: float
    option_type: Optional[str] = None
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

# -------------------------
# 3. SYMBOL CONFIG
# -------------------------
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "LT.NS", "SBIN.NS"
]
NIFTY_NEXT_50_SYMBOLS = [
    "ADANIENT.NS", "PIDILITIND.NS", "IOC.NS", "HINDALCO.NS", "WIPRO.NS",
    "GODREJCP.NS", "BERGEPAINT.NS", "INDIGO.NS", "DLF.NS", "ALKEM.NS"
]
NIFTY_100_SYMBOLS = list(set(NIFTY_50_SYMBOLS + NIFTY_NEXT_50_SYMBOLS))
ALL_TRACKED_SYMBOLS = sorted(list(set(NIFTY_100_SYMBOLS)))

if not st.session_state.selected_symbol:
    st.session_state.selected_symbol = ALL_TRACKED_SYMBOLS[0]

st.session_state.tracked_symbols = set(ALL_TRACKED_SYMBOLS)

# -------------------------
# 4. DATA FETCH & INDICATORS
# -------------------------
@st.cache_data(ttl=30)
def fetch_data_yf(symbol: str, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df.columns = [col.capitalize() for col in df.columns]
        return df
    except Exception:
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['Bollinger_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * 2)
    # RSI simple
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
    return df

# -------------------------
# 5. SIGNALS (SIMPLE SMA CROSS)
# -------------------------
def generate_signals_for_symbol(symbol: str):
    df = fetch_data_yf(symbol)
    df = calculate_indicators(df)
    if df is None or df.empty or len(df) < 3:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    # SMA crossover
    try:
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            action = "BUY"
            entry = latest['Close']
            stop_loss = float(latest.get('Bollinger_Lower', entry * 0.99))
            target1 = entry * 1.015
            target2 = entry * 1.03
            confidence = float(min(0.95, (latest['RSI'] / 100) if not np.isnan(latest['RSI']) else 0.6))
            return Signal(str(uuid.uuid4()), df.index[-1].to_pydatetime(), symbol, action, "SMA Crossover",
                          confidence, entry, stop_loss, target1, target2, option_type='CE')
        if latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            action = "SELL"
            entry = latest['Close']
            stop_loss = float(latest.get('Bollinger_Upper', entry * 1.01))
            target1 = entry * 0.985
            target2 = entry * 0.97
            confidence = float(min(0.95, (100 - latest['RSI']) / 100 if not np.isnan(latest['RSI']) else 0.6))
            return Signal(str(uuid.uuid4()), df.index[-1].to_pydatetime(), symbol, action, "SMA Crossover",
                          confidence, entry, stop_loss, target1, target2, option_type='PE')
    except Exception:
        return None
    return None

def generate_all_signals():
    cohorts = {
        "Nifty 50": NIFTY_50_SYMBOLS,
        "Nifty Next 50": NIFTY_NEXT_50_SYMBOLS,
        "Nifty 100": NIFTY_100_SYMBOLS
    }
    st.session_state.signals_by_cohort = {}
    all_signals = []
    for cname, symbols in cohorts.items():
        cohort_signals = []
        for s in symbols:
            sig = generate_signals_for_symbol(s)
            if sig:
                cohort_signals.append(sig)
                st.session_state.signal_history.append(sig)
        st.session_state.signals_by_cohort[cname] = cohort_signals
        all_signals.extend(cohort_signals)
    st.session_state.generated_signals = all_signals
    st.session_state.refresh_count += 1
    st.session_state.last_refresh = datetime.now()

# -------------------------
# 6. PAPER TRADE EXECUTION + PNL
# -------------------------
def execute_signal_paper(signal: Signal, quantity: int = 100):
    if signal.executed:
        return None
    # Basic capital check: require capital >= entry * quantity
    cost = signal.entry * quantity
    if st.session_state.capital < cost:
        st.warning("Insufficient capital for simulated execution.")
        return None
    st.session_state.capital -= cost
    signal.executed = True
    st.session_state.portfolio[signal.symbol] = {
        'quantity': quantity,
        'entry_price': signal.entry,
        'type': signal.option_type or 'CE'
    }
    trade = Trade(str(uuid.uuid4()), datetime.now(), signal.symbol, signal.action, signal.option_type or '',
                  signal.strategy, signal.entry, signal.stop_loss, signal.target1, signal.target2, quantity)
    st.session_state.executed_trades.append(trade)
    log = f"EXECUTED {trade.action} {trade.symbol} @ {trade.entry_price:.2f} x{trade.quantity}"
    st.session_state.execution_logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {log}")
    return trade

def manual_place_order(symbol: str, action: str, option_type: str, entry_price: float, quantity: int, stop_loss: float, target1: float, target2: float):
    # Create a fake signal and execute it
    sig = Signal(str(uuid.uuid4()), datetime.now(), symbol, action, "Manual", 1.0, entry_price, stop_loss, target1, target2, option_type=option_type)
    return execute_signal_paper(sig, quantity=quantity)

def update_pnl():
    # Realized pnl from trades that have exit_price set (none here unless exit is implemented)
    realized = sum(t.pnl for t in st.session_state.executed_trades if t.exit_price is not None)
    unrealized = 0.0
    for symbol, pos in st.session_state.portfolio.items():
        try:
            hist = yf.Ticker(symbol).history(period="1d", interval="1m")
            latest_price = float(hist['Close'].iloc[-1])
            if pos['type'] == 'CE':
                pnl_per_unit = latest_price - pos['entry_price']
            else:
                pnl_per_unit = pos['entry_price'] - latest_price
            unrealized += pnl_per_unit * pos['quantity']
        except Exception:
            pass
    st.session_state.pnl = realized
    st.session_state.unrealized_pnl = unrealized

# -------------------------
# 7. UI HELPERS: CHARTS, STYLES
# -------------------------
def inject_css():
    st.markdown("""
    <style>
    .main-header {color: #1E90FF; text-align: center;}
    div[data-testid="stSidebar"] { background-color: #0b0f14; color: white; }
    .signal-box { padding:8px; border-radius:6px; margin-bottom:6px; }
    .buy { background-color:#e6fff0; border-left:6px solid #2ecc71; }
    .sell { background-color:#fff0f0; border-left:6px solid #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

def plot_candlestick_with_indicators(symbol: str, df: pd.DataFrame, height=500):
    if df is None or df.empty:
        st.warning("No chart data")
        return
    df = df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if 'SMA_20' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
    if 'SMA_50' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
    if 'Bollinger_Upper' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], mode='lines', name='BB Upper', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], mode='lines', name='BB Lower', line=dict(dash='dash')), row=1, col=1)
    # RSI
    if 'RSI' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_layout(title=f"{symbol} Intraday", xaxis_rangeslider_visible=False, height=height, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 8. UI: SIDEBAR
# -------------------------
def sidebar_controls():
    st.sidebar.title("⚙️ Controls")
    st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_execute = st.sidebar.checkbox("Auto Execute (Paper)", value=st.session_state.auto_execute)
    st.session_state.refresh_interval = st.sidebar.slider("Refresh Interval (s)", min_value=5, max_value=60, value=st.session_state.refresh_interval, step=5)
    st.sidebar.markdown(f"**Last refresh:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.markdown("### Chart / Symbol")
    sym = st.sidebar.selectbox("Selected Symbol", options=ALL_TRACKED_SYMBOLS, index=ALL_TRACKED_SYMBOLS.index(st.session_state.selected_symbol))
    st.session_state.selected_symbol = sym

# -------------------------
# 9. UI: TABS CONTENT
# -------------------------
def tab_dashboard():
    st.header("📈 Dashboard - Signals & Quick Actions")
    col1, col2 = st.columns([2, 1])
    with col2:
        st.metric("Capital", f"₹{st.session_state.capital:,.2f}")
        update_pnl()
        st.metric("Realized PNL", f"₹{st.session_state.pnl:,.2f}")
        st.metric("Unrealized PNL", f"₹{st.session_state.unrealized_pnl:,.2f}")
    with col1:
        st.subheader("Quick Actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Refresh Now"):
                generate_all_signals()
                if st.session_state.auto_execute:
                    for s in st.session_state.generated_signals:
                        if s.confidence >= 0.7:
                            execute_signal_paper(s)
                update_pnl()
                st.experimental_rerun()
        with c2:
            if st.button("📊 Generate Signals"):
                generate_all_signals()
                st.success("Signals generated")
        with c3:
            if st.button("🚀 Execute High Confidence (>=0.7)"):
                for s in st.session_state.generated_signals:
                    if not s.executed and s.confidence >= 0.7:
                        execute_signal_paper(s)
                update_pnl()
                st.success("Executed eligible signals")
    st.markdown("---")
    st.subheader("Signals by Cohort")
    tabs = st.tabs(["Nifty 50", "Nifty Next 50", "Nifty 100"])
    cohort_keys = ["Nifty 50", "Nifty Next 50", "Nifty 100"]
    for t, key in zip(tabs, cohort_keys):
        with t:
            signals = st.session_state.signals_by_cohort.get(key, [])
            if not signals:
                st.info("No signals. Generate signals to populate.")
            else:
                for sig in signals:
                    cls = "buy" if sig.action == "BUY" else "sell"
                    st.markdown(f"<div class='signal-box {cls}'>"
                                f"<strong>{sig.symbol}</strong> — {sig.action} | {sig.strategy} | Confidence: {sig.confidence:.0%}<br>"
                                f"Entry: ₹{sig.entry:.2f} | SL: ₹{sig.stop_loss:.2f} | T1: ₹{sig.target1:.2f}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Execution Logs")
    if st.session_state.execution_logs:
        st.code("\n".join(st.session_state.execution_logs), language='text')
    else:
        st.info("No execution logs yet.")

def tab_paper_trading():
    st.header("🏦 Paper Trading")
    # Manual order entry
    st.subheader("Manual Order Entry")
    with st.form("manual_order_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            symbol = st.selectbox("Symbol", ALL_TRACKED_SYMBOLS, index=ALL_TRACKED_SYMBOLS.index(st.session_state.selected_symbol))
            action = st.selectbox("Action", ["BUY", "SELL"])
        with col2:
            option_type = st.selectbox("Option Type", ["CE", "PE"])
            quantity = st.number_input("Quantity", min_value=1, value=100, step=1)
        with col3:
            price = st.number_input("Entry Price (₹)", min_value=0.0, value=0.0, format="%.2f")
            sl = st.number_input("Stop Loss (₹)", min_value=0.0, value=0.0, format="%.2f")
        t1 = st.number_input("Target 1 (₹)", min_value=0.0, value=0.0, format="%.2f")
        t2 = st.number_input("Target 2 (₹)", min_value=0.0, value=0.0, format="%.2f")
        submitted = st.form_submit_button("Place Manual Paper Order")
    if submitted:
        if price <= 0:
            st.error("Enter a valid entry price.")
        else:
            trade = manual_place_order(symbol, action, option_type, price, int(quantity), sl or 0.0, t1 or 0.0, t2 or 0.0)
            if trade:
                st.success(f"Order placed: {action} {symbol} x{quantity} @ ₹{price:.2f}")
            else:
                st.error("Order failed (insufficient capital or other error).")

    # Portfolio summary
    st.markdown("---")
    st.subheader("Portfolio (Paper)")
    update_pnl()
    st.metric("Available Capital", f"₹{st.session_state.capital:,.2f}")
    st.metric("Unrealized PNL", f"₹{st.session_state.unrealized_pnl:,.2f}")
    if st.session_state.portfolio:
        df_port = pd.DataFrame([
            {"Symbol": s, "Quantity": d['quantity'], "Entry Price": d['entry_price'], "Type": d['type']}
            for s, d in st.session_state.portfolio.items()
        ])
        st.dataframe(df_port, use_container_width=True)
    else:
        st.info("No open positions in the paper portfolio.")

    # Quick actions on portfolio
    st.markdown("---")
    st.subheader("Portfolio Actions")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if st.button("Close All Positions (Simulated)"):
            # Simple close: compute pnl using last price and add back capital
            closed = []
            for s, pos in list(st.session_state.portfolio.items()):
                try:
                    hist = yf.Ticker(s).history(period="1d", interval="1m")
                    last_price = float(hist['Close'].iloc[-1])
                    if pos['type'] == 'CE':
                        pnl_per_unit = last_price - pos['entry_price']
                    else:
                        pnl_per_unit = pos['entry_price'] - last_price
                    pnl_total = pnl_per_unit * pos['quantity']
                except Exception:
                    pnl_total = 0.0
                    last_price = pos['entry_price']
                # create trade exit
                # find matching trade and set exit
                for t in st.session_state.executed_trades:
                    if t.symbol == s and t.exit_price is None:
                        t.exit_price = last_price
                        t.exit_timestamp = datetime.now()
                        t.pnl = pnl_total
                st.session_state.capital += last_price * pos['quantity']
                closed.append((s, pnl_total))
                del st.session_state.portfolio[s]
            update_pnl()
            st.success(f"Closed {len(closed)} positions.")
    with pcol2:
        if st.button("Reset Paper Account"):
            st.session_state.capital = 100000.0
            st.session_state.portfolio = {}
            st.session_state.executed_trades = []
            st.session_state.execution_logs = []
            st.session_state.pnl = 0.0
            st.session_state.unrealized_pnl = 0.0
            st.success("Paper account reset.")

def tab_trade_log():
    st.header("📜 Trade Log")
    if st.session_state.executed_trades:
        # Build dataframe
        rows = []
        for t in st.session_state.executed_trades:
            rows.append({
                "ID": t.id,
                "Timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M:%S") if t.timestamp else "",
                "Symbol": t.symbol,
                "Action": t.action,
                "Qty": t.quantity,
                "Entry": t.entry_price,
                "Exit": t.exit_price if t.exit_price is not None else "",
                "Realized PNL": t.pnl
            })
        df_log = pd.DataFrame(rows)
        st.dataframe(df_log.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, file_name="trade_log.csv", mime="text/csv")
        if st.button("Clear Trade Log"):
            st.session_state.executed_trades = []
            st.success("Trade log cleared.")
    else:
        st.info("No trades in log yet.")

def tab_live_chart():
    st.header("📊 Live Chart")
    st.markdown("Live chart updates independently. You can adjust the chart refresh interval in the sidebar.")
    sym = st.session_state.selected_symbol
    if st.session_state.auto_refresh:
        # ensure an independent ref to force rerender
        st.session_state.live_chart_ref += 1
        # autorefresh for chart specifically - use smaller interval if desired
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key=f"live_chart_{st.session_state.live_chart_ref}")
    df = fetch_data_yf(sym, period="1d", interval="5m")
    df = calculate_indicators(df)
    plot_candlestick_with_indicators(sym, df, height=650)

# -------------------------
# 10. MAIN
# -------------------------
def main():
    inject_css()
    st.markdown("<h1 class='main-header'>📈 Intraday Options Trading Terminal (Paper)</h1>", unsafe_allow_html=True)
    sidebar_controls()

    # Auto refresh engine (generates signals + optional auto-execute)
    if st.session_state.auto_refresh:
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="global_refresher")
        generate_all_signals()
        if st.session_state.auto_execute:
            for s in st.session_state.generated_signals:
                if not s.executed and s.confidence >= 0.7:
                    execute_signal_paper(s)
        update_pnl()

    # Tabs for primary views
    main_tabs = st.tabs(["Dashboard", "Paper Trading", "Trade Log", "Live Chart"])
    with main_tabs[0]:
        tab_dashboard()
    with main_tabs[1]:
        tab_paper_trading()
    with main_tabs[2]:
        tab_trade_log()
    with main_tabs[3]:
        tab_live_chart()

    # Footer quick info
    st.markdown("---")
    st.markdown(f"Last engine refresh: **{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}**  |  Refresh Count: **{st.session_state.refresh_count}**")

if __name__ == "__main__":
    main()
