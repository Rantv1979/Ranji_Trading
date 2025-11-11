# intraday_live_final.py
"""
Live Intraday Trading Terminal (Streamlit)
- Signals auto-scan every 10s (Live Mode)
- Charts refresh every 5s
- Paper trading capital: ₹500,000
Notes:
- Live intraday 5m data works only during NSE market hours (09:15 - 15:30 IST).
- For testing outside market hours, run with period='5d', interval='15m' in fetch_enhanced_ohlc.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
import pytz
import uuid
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Intraday Live Terminal", layout="wide", page_icon="📈")
INDIAN_TZ = pytz.timezone("Asia/Kolkata")

# Paper trading capital (5 Lakh INR)
PAPER_INITIAL_CAPITAL = 500_000.0

# Auto refresh intervals in ms
SIGNAL_REFRESH_MS = 10_000   # 10 seconds
CHART_REFRESH_MS = 5_000     # 5 seconds

# Basic universe (Nifty 50 sample) - use tickers with .NS suffix for yfinance
NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","KOTAKBANK.NS","BHARTIARTL.NS","ITC.NS","LT.NS",
    "SBIN.NS","ASIANPAINT.NS","HCLTECH.NS","AXISBANK.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","NTPC.NS",
    "NESTLEIND.NS","POWERGRID.NS","M&M.NS","BAJFINANCE.NS","ONGC.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIPORTS.NS","COALINDIA.NS"
]

# ---------------------------
# Utilities
# ---------------------------
def now_indian():
    return datetime.now(INDIAN_TZ)

def format_time(dt):
    return dt.astimezone(INDIAN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

def market_is_open(now=None):
    """Return True if current Indian time is within NSE intraday hours."""
    if now is None:
        now = now_indian()
    today = now.date()
    open_dt = INDIAN_TZ.localize(datetime.combine(today, dt_time(hour=9, minute=15)))
    close_dt = INDIAN_TZ.localize(datetime.combine(today, dt_time(hour=15, minute=30)))
    return open_dt <= now <= close_dt

# ---------------------------
# Indicators
# ---------------------------
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    return macd, macd_signal, macd - macd_signal

def bollinger_bands(close, period=20, std=2):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    up = mid + sd * std
    lo = mid - sd * std
    return up, mid, lo

# ---------------------------
# Data fetching (defensive)
# ---------------------------
@st.cache_data(ttl=15)
def fetch_enhanced_ohlc(symbol: str, period="1d", interval="5m"):
    """Fetch OHLC and compute indicators. Returns None on failure."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        # Ensure columns are standard
        # yfinance can sometimes return multi-index columns in rare cases; force expected columns
        expected = ["Open","High","Low","Close","Volume"]
        df = df.copy()
        df = df.rename_axis(index=None)
        for col in expected:
            if col not in df.columns:
                # try to find close-like column
                if "Close" in df.columns:
                    continue
                else:
                    return None
        # Drop rows with NaN close
        df = df[df['Close'].notna()]
        if df.shape[0] < 10:
            return None

        # Indicators
        df['SMA20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['SMA50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['EMA8'] = compute_ema(df['Close'], 8)
        df['EMA21'] = compute_ema(df['Close'], 21)
        df['RSI14'] = compute_rsi(df['Close'], 14)
        macd, macd_signal, macd_hist = compute_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_HIST'] = macd_hist
        bb_up, bb_mid, bb_lo = bollinger_bands(df['Close'], 20, 2)
        df['BB_UP'] = bb_up
        df['BB_MID'] = bb_mid
        df['BB_LO'] = bb_lo
        # Safe BB position (0..1)
        bb_range = (df['BB_UP'] - df['BB_LO']).replace(0, np.nan)
        df['BB_POSITION'] = ((df['Close'] - df['BB_LO']) / bb_range).fillna(0.5).clip(0,1)
        # Volume ratio
        if 'Volume' in df.columns:
            df['VOL_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
            df['VOL_RATIO'] = (df['Volume'] / df['VOL_MA20']).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        else:
            df['VOL_RATIO']=1.0
        return df
    except Exception as e:
        # don't crash app; return None
        st.session_state.setdefault("_fetch_errors", {})
        st.session_state["_fetch_errors"][symbol] = str(e)
        return None

# ---------------------------
# Signal generation (live-focused)
# ---------------------------
def generate_signals_from_df(df, symbol):
    """Return a dict signal if conditions met on latest bar else None."""
    if df is None or df.empty or len(df) < 3:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest['Close'])
    sma20 = float(latest['SMA20'])
    sma50 = float(latest['SMA50'])
    sma20_prev = float(prev['SMA20'])
    sma50_prev = float(prev['SMA50'])
    ema8 = float(latest['EMA8'])
    ema21 = float(latest['EMA21'])
    rsi = float(latest['RSI14'])
    vol_ratio = float(latest.get('VOL_RATIO', 1.0))
    macd = float(latest['MACD'])
    macd_signal = float(latest['MACD_Signal'])

    # Conservative SMA crossover BUY
    if sma20 > sma50 and sma20_prev <= sma50_prev and rsi < 80 and vol_ratio >= 0.8:
        confidence = 0.6 + min(0.3, (sma20 - sma50) / max(0.01, sma50))
        return {
            "symbol": symbol,
            "action": "BUY",
            "entry": close,
            "stop_loss": close * 0.995,
            "target1": close * 1.01,
            "confidence": float(min(0.99, confidence)),
            "reason": f"SMA20 crossed above SMA50 (rsi {rsi:.1f}, vol {vol_ratio:.2f})"
        }

    # Conservative SMA crossover SELL
    if sma20 < sma50 and sma20_prev >= sma50_prev and rsi > 20 and vol_ratio >= 0.8:
        confidence = 0.6 + min(0.3, (sma50 - sma20) / max(0.01, sma50))
        return {
            "symbol": symbol,
            "action": "SELL",
            "entry": close,
            "stop_loss": close * 1.005,
            "target1": close * 0.99,
            "confidence": float(min(0.99, confidence)),
            "reason": f"SMA20 crossed below SMA50 (rsi {rsi:.1f}, vol {vol_ratio:.2f})"
        }

    # EMA momentum (faster intraday signals)
    if ema8 > ema21 and float(prev['EMA8']) <= float(prev['EMA21']) and rsi < 85:
        return {
            "symbol": symbol,
            "action": "BUY",
            "entry": close,
            "stop_loss": close * 0.995,
            "target1": close * 1.015,
            "confidence": 0.58,
            "reason": "EMA8 crossed above EMA21"
        }

    if ema8 < ema21 and float(prev['EMA8']) >= float(prev['EMA21']) and rsi > 15:
        return {
            "symbol": symbol,
            "action": "SELL",
            "entry": close,
            "stop_loss": close * 1.005,
            "target1": close * 0.985,
            "confidence": 0.58,
            "reason": "EMA8 crossed below EMA21"
        }

    # MACD quick cross
    if macd > macd_signal and float(prev['MACD']) <= float(prev['MACD_Signal']):
        return {
            "symbol": symbol,
            "action": "BUY",
            "entry": close,
            "stop_loss": close * 0.995,
            "target1": close * 1.012,
            "confidence": 0.55,
            "reason": "MACD bullish cross"
        }
    if macd < macd_signal and float(prev['MACD']) >= float(prev['MACD_Signal']):
        return {
            "symbol": symbol,
            "action": "SELL",
            "entry": close,
            "stop_loss": close * 1.005,
            "target1": close * 0.988,
            "confidence": 0.55,
            "reason": "MACD bearish cross"
        }

    return None

# ---------------------------
# Paper trading simple engine
# ---------------------------
class PaperTrader:
    def __init__(self, capital=PAPER_INITIAL_CAPITAL):
        self.initial = float(capital)
        self.available = float(capital)
        self.positions = {}   # symbol -> position dict
        self.history = []

    def buy(self, signal):
        symbol = signal['symbol']
        if symbol in self.positions:
            return False
        # determine quantity limited by e.g., 1% of capital per trade
        risk_per_trade = 0.01 * self.initial
        entry = float(signal['entry'])
        stop = float(signal['stop_loss'])
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return False
        qty = int(max(1, risk_per_trade // (risk_per_share)))
        cost = qty * entry
        if cost > self.available:
            # restrict qty if not enough
            qty = int(self.available // entry)
            if qty <= 0:
                return False
            cost = qty * entry
        # record
        pos = {
            "symbol": symbol,
            "action": signal['action'],
            "qty": qty,
            "entry": entry,
            "stop": stop,
            "target1": signal.get('target1'),
            "open_time": now_indian(),
            "reason": signal.get('reason'),
        }
        self.positions[symbol] = pos
        self.available -= cost
        return True

    def close(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return False
        pos = self.positions.pop(symbol)
        if exit_price is None:
            # try get latest market price
            df = fetch_enhanced_ohlc(symbol)
            exit_price = float(df['Close'].iloc[-1]) if df is not None else pos['entry']
        pnl = (exit_price - pos['entry']) * pos['qty'] if pos['action'] == "BUY" else (pos['entry'] - exit_price) * pos['qty']
        self.available += exit_price * pos['qty']
        rec = {**pos, "exit": exit_price, "pnl": pnl, "close_time": now_indian()}
        self.history.append(rec)
        return rec

paper = PaperTrader()

# ---------------------------
# Streamlit app UI
# ---------------------------
def main():
    st.title("📈 Intraday Live Terminal — Live Mode")
    st.markdown(f"**Local time (India):** {format_time(now_indian())}")
    st.markdown("Signals auto-scan every **10s** (only when app is running). Live charts auto-refresh every **5s**.")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        universe = st.selectbox("Universe", options=["NIFTY 50", "Custom"], index=0)
        if universe == "NIFTY 50":
            symbols = NIFTY_50
        else:
            custom = st.text_area("Enter tickers (one per line)", value="RELIANCE.NS\nTCS.NS")
            symbols = [s.strip() for s in custom.splitlines() if s.strip()]
        min_confidence = st.slider("Min confidence to show", min_value=0.3, max_value=0.9, value=0.55, step=0.05)
        st.markdown("---")
        st.markdown(f"Paper trading capital: **₹{paper.initial:,.0f}**")
        if st.button("Reset Paper Trading"):
            global paper
            paper = PaperTrader()
            st.success("Paper trading reset (₹500,000).")

    # Show market status
    is_open = market_is_open()
    if is_open:
        st.success("🟢 Market is OPEN (NSE)")
    else:
        st.info("🔴 Market appears CLOSED. Live intraday 5m data will be limited/outdated until 09:15 IST.")

    # Auto-refresh for signals (10s)
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="sig_refresh")

    # Signal scanning area
    st.subheader("🎯 Live Signals (auto-run every 10s)")
    placeholder = st.empty()

    # We'll run a scan only if market open; otherwise optionally show message
    signals = []
    scan_time = now_indian()
    if is_open:
        with placeholder.container():
            st.info(f"Running live scan at {format_time(scan_time)} across {len(symbols)} symbols...")
            progress = st.progress(0)
            for i, sym in enumerate(symbols):
                df = fetch_enhanced_ohlc(sym, period="1d", interval="5m")
                if df is not None:
                    sig = generate_signals_from_df(df, sym)
                    if sig and sig.get("confidence", 0) >= min_confidence:
                        signals.append(sig)
                progress.progress((i + 1) / len(symbols))
            progress.empty()
    else:
        with placeholder.container():
            st.info("Market closed — live 5m intraday signals will not be produced. To test, run during market hours.")
    # display signals
    if signals:
        st.success(f"Found {len(signals)} signal(s) at {format_time(now_indian())}")
        df_signals = pd.DataFrame(signals)
        df_signals_display = df_signals[["symbol","action","entry","stop_loss","target1","confidence","reason"]]
        df_signals_display['entry'] = df_signals_display['entry'].map(lambda x: f"₹{x:.2f}")
        df_signals_display['stop_loss'] = df_signals_display['stop_loss'].map(lambda x: f"₹{x:.2f}")
        df_signals_display['target1'] = df_signals_display['target1'].map(lambda x: f"₹{x:.2f}")
        st.dataframe(df_signals_display, use_container_width=True)
        # quick buttons to paper-buy the top signal(s)
        for sig in signals:
            cols = st.columns([1,3,3,2])
            with cols[0]:
                st.write(f"**{sig['symbol']}**")
            with cols[1]:
                st.write(f"{sig['action']} @ ₹{sig['entry']:.2f}")
            with cols[2]:
                st.write(sig['reason'])
            with cols[3]:
                if st.button(f"Paper {sig['action']} {sig['symbol']}", key=f"paper_{sig['symbol']}"):
                    ok = paper.buy(sig)
                    if ok:
                        st.success(f"Paper {sig['action']} executed: {sig['symbol']} x{paper.positions[sig['symbol']]['qty']}")
                    else:
                        st.warning("Could not execute paper trade (maybe position exists or insufficient capital).")
    else:
        st.info("No signals found in this scan.")

    # Auto-refresh charts independently
    st.markdown("---")
    st.subheader("📊 Live Chart (refresh every 5s)")
    chart_col1, chart_col2 = st.columns([3,1])
    with chart_col2:
        selected_symbol = st.selectbox("Select Symbol to chart", options=symbols, index=0)
        st_autorefresh(interval=CHART_REFRESH_MS, key="chart_refresh_" + selected_symbol)

    with chart_col1:
        df_chart = fetch_enhanced_ohlc(selected_symbol, period="1d", interval="5m")
        if df_chart is None:
            st.warning("Couldn't fetch chart data for selected symbol (market closed or data missing).")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                low=df_chart['Low'], close=df_chart['Close'], name="Price"
            ))
            # overlay EMA/SMAs
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA20'], name="SMA20", opacity=0.8))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], name="SMA50", opacity=0.8))
            fig.update_layout(title=f"{selected_symbol} — Live (5m)", xaxis_rangeslider_visible=False, height=520)
            st.plotly_chart(fig, use_container_width=True)

            # small indicators below
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI14'], name="RSI14"))
            rsi_fig.update_layout(title="RSI (14)", height=220)
            st.plotly_chart(rsi_fig, use_container_width=True)

    # Paper trading UI
    st.markdown("---")
    st.subheader("📝 Paper Trading Account")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Initial Capital", f"₹{paper.initial:,.0f}")
        st.metric("Available", f"₹{paper.available:,.0f}")
        st.metric("Open Positions", len(paper.positions))
    with col_b:
        if paper.positions:
            for sym, pos in paper.positions.items():
                st.write(f"**{sym}** — {pos['action']} x{pos['qty']} @ ₹{pos['entry']:.2f}")
                if st.button(f"Close {sym}", key=f"close_{sym}"):
                    rec = paper.close(sym)
                    if rec:
                        st.success(f"Closed {sym} — P&L ₹{rec['pnl']:.2f}")

    if paper.history:
        st.markdown("**Recent closed trades (paper):**")
        hist_df = pd.DataFrame(paper.history[-10:])[["symbol","action","qty","entry","exit","pnl","open_time","close_time"]]
        st.dataframe(hist_df, use_container_width=True)

    # Show fetch errors (if any) to help debugging
    if "_fetch_errors" in st.session_state and st.session_state["_fetch_errors"]:
        with st.expander("Fetch errors (debug)"):
            for k,v in st.session_state["_fetch_errors"].items():
                st.write(f"{k}: {v}")

    st.markdown("---")
    st.caption("Live Mode — run during NSE hours (09:15–15:30 IST) for real intraday signals. This is paper-trading only.")

if __name__ == "__main__":
    main()
