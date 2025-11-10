# gemini_intraday_final_all_symbols.py
# Final: Intraday terminal with signals from Nifty 50 / Nifty Next 50 / Nifty 100 / Nifty 500,
# separate Backtest tab, and Live Chart for selected signal (auto-refresh 10s).
#
# Notes:
# - At runtime the app will attempt to fetch the NIFTY 500 constituents from Yahoo Finance.
#   If that fails it falls back to bundled NIFTY 100 symbols.
# - Requires: streamlit, yfinance, pandas, numpy, requests, beautifulsoup4, plotly, streamlit-autorefresh
#
# Run:
#    streamlit run gemini_intraday_final_all_symbols.py

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
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Gemini Intraday Pro — All Symbols")

# ---------------------------
# Helper: fetch NIFTY 500 constituents from Yahoo Finance (runtime)
# ---------------------------
@st.cache_data(ttl=60*60)  # cache 1 hour
def fetch_nifty500_list():
    """Try to scrape Yahoo Finance NIFTY 500 components page for tickers (.NS).
    If scraping fails, return None to allow fallback."""
    try:
        url = "https://finance.yahoo.com/quote/%5ECRSLDX/components/"  # Yahoo NIFTY 500 components
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Yahoo renders component table rows with 'data-test' attributes; find tickers in table
        tickers = set()
        # Find table rows and extract first <a> or text that is ticker
        table = soup.find("table")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 1:
                    txt = cells[0].get_text(strip=True)
                    # Sometimes tickers are like "RELIANCE.NS" or "RELIANCE.NS (Reliance Ind...)"
                    # Only keep if looks like .NS or uppercase letters/digits.
                    if txt:
                        # normalize
                        t = txt.split()[0]
                        if t.endswith(".NS") or t.isupper():
                            tickers.add(t if t.endswith(".NS") else f"{t}.NS")
        # Fallback: attempt to extract from JSON LD or scripts (less reliable)
        if not tickers:
            # search for "components" in scripts
            scripts = soup.find_all("script")
            for s in scripts:
                txt = s.string
                if not txt:
                    continue
                if "components" in txt and ".NS" in txt:
                    # crude extraction
                    for token in txt.split('"'):
                        if token.endswith(".NS"):
                            tickers.add(token)
        return sorted(list(tickers))
    except Exception:
        return None

# ---------------------------
# Built-in symbol lists (fallbacks)
# ---------------------------
NIFTY_50 = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "KOTAKBANK.NS","HINDUNILVR.NS","AXISBANK.NS","LT.NS","SBIN.NS",
    "ITC.NS","BAJFINANCE.NS","HDFC.NS","BHARTIARTL.NS","MARUTI.NS",
    "ONGC.NS","TITAN.NS","ULTRACEMCO.NS","NTPC.NS","SUNPHARMA.NS",
    "HCLTECH.NS","POWERGRID.NS","WIPRO.NS","TECHM.NS","BPCL.NS",
    "COALINDIA.NS","BRITANNIA.NS","DIVISLAB.NS","HDFCLIFE.NS","ADANIENT.NS",
    "GRASIM.NS","DLF.NS","ADANIPORTS.NS","EICHERMOT.NS","CIPLA.NS",
    "IOC.NS","JSWSTEEL.NS","SBILIFE.NS","TATASTEEL.NS","INDUSINDBK.NS",
    "HINDALCO.NS","NESTLEIND.NS","DRREDDY.NS","BAJAJ-AUTO.NS","SHREECEM.NS",
    "TATAELXSI.NS","MRF.NS","PIDILITIND.NS","BAJAJFINSV.NS","ULTRACEMCO.NS"  # duplicates trimmed later
]
NIFTY_NEXT_50 = [
    "ADANITRANS.NS","APOLLOHOSP.NS","ADANIGREEN.NS","AUROPHARMA.NS","BERGEPAINT.NS",
    "BOSCHLTD.NS","CASTROLIND.NS","INDIGO.NS","GODREJCP.NS","GRSE.NS",
    "HAVELLS.NS","HEROMOTOCO.NS","HINDZINC.NS","ICICIPRULI.NS","INDIAMART.NS",
    "LICHSGFIN.NS","LUPIN.NS","MUTHOOTFIN.NS","PEL.NS","PEL.NS"
]

# We'll form NIFTY_100 as union of above
NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# ---------------------------
# Utility: indicator calculations
# ---------------------------
def compute_indicators(df: pd.DataFrame):
    """Add SMA20, SMA50, Bollinger bands, RSI, and Volume MA to df."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['BB_MID'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_STD'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_UP'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LO'] = df['BB_MID'] - 2 * df['BB_STD']
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
    # Volume MA
    if 'Volume' in df.columns:
        df['VOL_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    else:
        df['Volume'] = 0
        df['VOL_MA_20'] = 0
    return df

# ---------------------------
# Data fetch wrapper with caching
# ---------------------------
@st.cache_data(ttl=30)
def fetch_ohlc(symbol: str, period="1d", interval="5m"):
    """Fetch OHLCV from yfinance and compute indicators."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        df = compute_indicators(df)
        return df
    except Exception:
        return None

# ---------------------------
# Signal generation per symbol (enhanced confirmation)
# ---------------------------
def generate_signals_for_df(df):
    """Return latest signal (if any) for a dataframe. Uses SMA crossover + RSI + Volume filter."""
    if df is None or df.empty or len(df) < 3:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    # Volume confirmation: latest volume > 20MA
    vol_ok = latest['Volume'] > latest.get('VOL_MA_20', 0)
    # RSI zones
    rsi = latest.get('RSI', np.nan)
    # SMA crossover
    try:
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50'] and (rsi > 55) and vol_ok:
            # BUY signal
            entry = latest['Close']
            sl = latest['BB_LO'] if not math.isnan(latest['BB_LO']) else entry * 0.99
            t1 = entry * 1.015
            t2 = entry * 1.03
            confidence = min(0.98, (rsi / 100.0))
            return {
                'id': str(uuid.uuid4()), 'timestamp': df.index[-1].to_pydatetime(), 'symbol': None,
                'action': 'BUY', 'entry': entry, 'stop_loss': sl, 'target1': t1, 'target2': t2, 'confidence': confidence
            }
        elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50'] and (rsi < 45) and vol_ok:
            # SELL signal
            entry = latest['Close']
            sl = latest['BB_UP'] if not math.isnan(latest['BB_UP']) else entry * 1.01
            t1 = entry * 0.985
            t2 = entry * 0.97
            confidence = min(0.98, ((100 - rsi) / 100.0))
            return {
                'id': str(uuid.uuid4()), 'timestamp': df.index[-1].to_pydatetime(), 'symbol': None,
                'action': 'SELL', 'entry': entry, 'stop_loss': sl, 'target1': t1, 'target2': t2, 'confidence': confidence
            }
    except Exception:
        return None
    return None

# ---------------------------
# Generate signals across a list of symbols (may take time)
# ---------------------------
def generate_signals_for_symbols(symbols, period="1d", interval="5m", progress_callback=None):
    signals = []
    for i, sym in enumerate(symbols):
        df = fetch_ohlc(sym, period=period, interval=interval)
        s = generate_signals_for_df(df)
        if s:
            s['symbol'] = sym
            signals.append(s)
        if progress_callback:
            progress_callback(i + 1)
    return signals

# ---------------------------
# Backtest logic (separate tab)
# ---------------------------
def backtest_on_symbol(symbol, period="30d", qty=100, capital=100000):
    """Backtest enhanced strategy on a single symbol over period.
    Uses SMA crossover + RSI + Volume confirmations, targets 1.5% and stop 1%."""
    df = fetch_ohlc(symbol, period=period, interval="5m")
    if df is None or df.empty:
        return None
    trades = []
    balance = capital
    equity_curve = [balance]
    wins = 0
    losses = 0

    # Iterate through df and check signals using earlier enhanced rule
    for i in range(51, len(df)-1):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        vol_ok = row['Volume'] > row.get('VOL_MA_20', 0)
        rsi = row.get('RSI', np.nan)
        if row['SMA_20'] > row['SMA_50'] and prev['SMA_20'] <= prev['SMA_50'] and (rsi > 55) and vol_ok:
            action = 'BUY'
            entry = row['Close']
            sl = row['BB_LO'] if not math.isnan(row['BB_LO']) else entry * 0.99
            tgt = entry * 1.015
        elif row['SMA_20'] < row['SMA_50'] and prev['SMA_20'] >= prev['SMA_50'] and (rsi < 45) and vol_ok:
            action = 'SELL'
            entry = row['Close']
            sl = row['BB_UP'] if not math.isnan(row['BB_UP']) else entry * 1.01
            tgt = entry * 0.985
        else:
            continue

        # Look forward for exit (same session)
        exit_price = None
        for j in range(i+1, len(df)):
            p = df.iloc[j]['Close']
            if action == 'BUY':
                if p <= sl:
                    exit_price = sl
                    break
                if p >= tgt:
                    exit_price = tgt
                    break
            else:
                if p >= sl:
                    exit_price = sl
                    break
                if p <= tgt:
                    exit_price = tgt
                    break
        if exit_price is None:
            # exit at last candle close of dataset
            exit_price = df.iloc[-1]['Close']

        pnl = (exit_price - entry) * qty if action == 'BUY' else (entry - exit_price) * qty
        balance += pnl
        equity_curve.append(balance)
        trades.append({'symbol': symbol, 'action': action, 'entry': entry, 'exit': exit_price, 'pnl': pnl, 'time': row.name})
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    # Stats
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = -sum(t['pnl'] for t in trades if t['pnl'] < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit>0 else 0.0

    stats = {
        'symbol': symbol,
        'period': period,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate_pct': round(win_rate, 2),
        'net_pnl': round(balance - capital, 2),
        'final_balance': round(balance, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else np.nan
    }
    return stats, trades, equity_curve

# ---------------------------
# Rendering: candlestick chart with indicators for a symbol's dataframe
# ---------------------------
def plot_candles(df, symbol, height=520):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if 'SMA_20' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', mode='lines'))
    if 'SMA_50' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', mode='lines'))
    if 'BB_UP' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Up', line=dict(dash='dash'), mode='lines'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LO'], name='BB Lo', line=dict(dash='dash'), mode='lines'))
    fig.update_layout(title=f"{symbol} — Intraday (5m)", xaxis_rangeslider_visible=False, height=height, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# App UI
# ---------------------------

st.title("📈 Gemini Intraday Pro — All Symbols (Nifty 50/Next50/100/500)")

# Try to fetch Nifty 500 list (cached)
with st.spinner("Loading NIFTY 500 constituents (if available)..."):
    nifty500 = fetch_nifty500_list()

if nifty500 and len(nifty500) >= 400:
    st.success(f"Loaded NIFTY 500 list ({len(nifty500)} tickers).")
else:
    if nifty500 is None:
        st.warning("Could not fetch full NIFTY 500 list automatically — falling back to NIFTY 100 sample list. You can place a local 'nifty500.txt' (one ticker per line) in the app folder to override.")
    else:
        st.info(f"Fetched {len(nifty500)} tickers (partial). Using what was retrieved.")
    nifty500 = None

# If local override file exists, try to load it
try:
    local_file = "nifty500.txt"
    if st.experimental_get_query_params().get("use_local_nifty500"):
        # user opted query param to force
        pass
    try:
        with open(local_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                nifty500 = lines
                st.info(f"Loaded {len(lines)} tickers from {local_file}")
    except Exception:
        pass
except Exception:
    pass

# Build final symbol universe
universe = sorted(set(NIFTY_100 + (nifty500 if nifty500 else [])))
if not universe:
    universe = NIFTY_100  # ultimate fallback

# Sidebar controls
st.sidebar.header("Controls")
refresh_now = st.sidebar.button("Refresh Signals Now")
auto_refresh = st.sidebar.checkbox("Auto Refresh Signals (engine)", value=True)
engine_interval = st.sidebar.slider("Engine refresh (s)", min_value=5, max_value=60, value=30, step=5)
live_chart_refresh = st.sidebar.slider("Live chart refresh (s)", min_value=5, max_value=30, value=10, step=1)
symbols_to_scan = st.sidebar.multiselect("Symbols to scan (empty = full universe)", options=universe, default=[])
if not symbols_to_scan:
    symbols_to_scan = universe

# AUTORELOAD engine
if auto_refresh:
    st_autorefresh(interval=engine_interval * 1000, key="engine_refresher")

# Tabs: Dashboard / Signals / Live Chart / Backtest / Trade Log
tabs = st.tabs(["Dashboard", "Signals", "Live Chart (Signal)", "Backtest", "Trade Log"])

# --- Dashboard tab: quick stats and top signals ---
with tabs[0]:
    st.header("Dashboard — Quick Scan")
    st.write(f"Universe: {len(universe)} symbols. Scanning: {len(symbols_to_scan)} symbols.")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Last refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.metric("Symbols scanned", len(symbols_to_scan))
    with col2:
        st.info("Generating signals across the selected universe (this may take a while for NIFTY 500).")
        progress_text = st.empty()
        prog_bar = st.progress(0)
        signals = []
        total = len(symbols_to_scan)
        for idx, sym in enumerate(symbols_to_scan):
            progress_text.text(f"Scanning {idx+1}/{total}: {sym}")
            df = fetch_ohlc(sym, period="1d", interval="5m")
            s = generate_signals_for_df(df)
            if s:
                s['symbol'] = sym
                signals.append(s)
            prog_bar.progress((idx+1)/total)
        prog_bar.empty()
        progress_text.empty()
        st.write(f"Signals found: {len(signals)}")
        if len(signals) > 0:
            df_signals = pd.DataFrame([{
                'Symbol': s['symbol'], 'Action': s['action'], 'Entry': round(s['entry'],2),
                'SL': round(s['stop_loss'],2), 'T1': round(s['target1'],2), 'Conf': round(s['confidence'],2),
                'Time': s['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            } for s in signals])
            st.dataframe(df_signals.sort_values(by='Conf', ascending=False), use_container_width=True)
        else:
            st.info("No signals generated on this scan.")

# --- Signals tab: full signals list + ability to view details / place paper trades ---
with tabs[1]:
    st.header("Signals — Full Universe")
    st.write("Signals are generated using SMA20/50 crossover with RSI (>55 buy, <45 sell) and volume confirmation (volume > 20MA).")
    # Option to refresh signals
    if refresh_now:
        st.experimental_rerun()
    # Re-use signals variable if present else generate
    try:
        signals  # from previous tab
    except NameError:
        # generate quickly (no progress UI)
        signals = generate_signals_for_symbols(symbols_to_scan, period="1d", interval="5m")
    st.write(f"Total signals: {len(signals)}")
    if signals:
        # Show as expandable list
        for s in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            with st.expander(f"{s['symbol']} — {s['action']} — Conf {s['confidence']:.2f} — Entry {s['entry']:.2f}"):
                st.write(f"Time: {s['timestamp']}")
                st.write(f"Entry: ₹{s['entry']:.2f} | SL: ₹{s['stop_loss']:.2f} | T1: ₹{s['target1']:.2f} | T2: ₹{s['target2']:.2f}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(f"Place Paper Order {s['symbol']}", key=f"paper_{s['id']}"):
                        # store paper trade in session state
                        if 'paper_trades' not in st.session_state:
                            st.session_state.paper_trades = []
                        st.session_state.paper_trades.insert(0, {
                            'id': s['id'], 'symbol': s['symbol'], 'action': s['action'],
                            'entry': s['entry'], 'sl': s['stop_loss'], 't1': s['target1'],
                            'qty': 100, 'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success("Paper order placed.")
                with c2:
                    if st.button(f"View Live Chart {s['symbol']}", key=f"chart_{s['id']}"):
                        # set selected signal for live chart tab via session_state
                        st.session_state.selected_signal = s
                        # navigate to Live Chart tab by re-render (can't programmatically switch tabs)
                        st.experimental_rerun()
                with c3:
                    st.write("")

# --- Live Chart (Signal-specific) tab ---
with tabs[2]:
    st.header("Live Chart for Selected Signal")
    # pick signal from session_state or choose symbol manually
    sel_signal = st.session_state.get('selected_signal', None)
    if sel_signal:
        st.subheader(f"{sel_signal['symbol']} — {sel_signal['action']} (signal time {sel_signal['timestamp']})")
        chart_sym = sel_signal['symbol']
    else:
        chart_sym = st.selectbox("Select symbol to view live chart", options=symbols_to_scan, index=0)
    # set autorefresh for chart at live_chart_refresh seconds
    st_autorefresh(interval=live_chart_refresh * 1000, key=f"live_chart_{chart_sym}")
    df_chart = fetch_ohlc(chart_sym, period="1d", interval="5m")
    if df_chart is None or df_chart.empty:
        st.warning(f"No chart data for {chart_sym}")
    else:
        plot_candles(df_chart, chart_sym)
        # If there is an active signal for this symbol, overlay markers
        if sel_signal and sel_signal['symbol'] == chart_sym:
            st.markdown(f"**Signal** — {sel_signal['action']} @ ₹{sel_signal['entry']:.2f}  |  SL: ₹{sel_signal['stop_loss']:.2f}  |  T1: ₹{sel_signal['target1']:.2f}")

# --- Backtest tab (separate) ---
with tabs[3]:
    st.header("Backtest — Enhanced Strategy (SMA20/50 + RSI + Volume)")
    st.write("Backtest runs per-symbol over historical 5m bars. Select symbol and period.")
    col1, col2 = st.columns([2,1])
    with col1:
        b_symbol = st.selectbox("Symbol for backtest", options=universe, index=0)
        b_period = st.selectbox("Backtest period", options=["5d","10d","30d","60d","90d"], index=1)
        b_qty = st.number_input("Quantity per trade", min_value=1, value=100, step=1)
        b_capital = st.number_input("Starting capital (₹)", min_value=1000.0, value=100000.0, step=1000.0)
    with col2:
        if st.button("Run Backtest"):
            with st.spinner("Running backtest (this may take some time)..."):
                bt_result = backtest_on_symbol(b_symbol, period=b_period, qty=int(b_qty), capital=float(b_capital))
                if bt_result is None:
                    st.error("No data / no trades for this symbol/period.")
                else:
                    stats, trades, eq = bt_result
                    st.subheader("Backtest Summary")
                    st.json(stats)
                    st.subheader("Equity curve")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(eq, linewidth=2)
                    ax.set_xlabel("Trade #")
                    ax.set_ylabel("Equity (₹)")
                    st.pyplot(fig)
                    if trades:
                        st.subheader("Trades (most recent first)")
                        df_trades = pd.DataFrame(trades)[::-1]
                        st.dataframe(df_trades, use_container_width=True)
                        csv = df_trades.to_csv(index=False).encode('utf-8')
                        st.download_button("Download trades CSV", csv, file_name=f"backtest_{b_symbol}.csv", mime="text/csv")
                    else:
                        st.info("No trades occurred in backtest period.")

# --- Trade Log tab ---
with tabs[4]:
    st.header("Trade Log / Paper Trades")
    if 'paper_trades' not in st.session_state or not st.session_state.paper_trades:
        st.info("No paper trades yet. Place paper orders from the Signals tab.")
    else:
        df_log = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("Download Paper Trades CSV", csv, file_name="paper_trades.csv", mime="text/csv")
        if st.button("Clear Paper Trades"):
            st.session_state.paper_trades = []
            st.success("Cleared paper trades.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Strategy: SMA20/50 crossover with RSI + Volume confirmations. SL uses Bollinger band. Targets: ~1.5% / 3%. "
           "Backtest is illustrative — please validate with your own data and slippage assumptions before using for live trading.")
