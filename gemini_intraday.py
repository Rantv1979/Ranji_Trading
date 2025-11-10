# ===============================================================
# ENHANCED GEMINI INTRADAY TERMINAL WITH BACKTEST MODULE
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gemini Intraday Pro", layout="wide")

# ===============================================================
# 1. CONFIGURATION
# ===============================================================
REFRESH_INTERVAL = 30  # seconds
ALL_TRACKED_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "AXISBANK.NS", "ITC.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "NTPC.NS", "ASIANPAINT.NS", "ONGC.NS"
]

# ===============================================================
# 2. DATA FETCHING
# ===============================================================
@st.cache_data(ttl=60)
def fetch_data_yf(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ===============================================================
# 3. INDICATORS
# ===============================================================
def calculate_indicators(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["Upper"], df["Lower"] = bollinger_bands(df["Close"], 20)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window, num_std=2):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return mean + num_std * std, mean - num_std * std

# ===============================================================
# 4. SIGNAL GENERATION
# ===============================================================
def generate_signals(df):
    signals = []
    for i in range(51, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        avg_vol = df["Volume"].iloc[i - 20:i].mean() if i >= 20 else df["Volume"].iloc[:i].mean()
        vol_ok = row["Volume"] > avg_vol
        rsi_buy_ok = row["RSI"] > 55
        rsi_sell_ok = row["RSI"] < 45
        if row["SMA_20"] > row["SMA_50"] and prev["SMA_20"] <= prev["SMA_50"] and rsi_buy_ok and vol_ok:
            signals.append(("BUY", row.name, row["Close"]))
        elif row["SMA_20"] < row["SMA_50"] and prev["SMA_20"] >= prev["SMA_50"] and rsi_sell_ok and vol_ok:
            signals.append(("SELL", row.name, row["Close"]))
    return signals

# ===============================================================
# 5. LIVE CHART
# ===============================================================
def tab_live_chart():
    st.header("📈 Live Chart")
    symbol = st.selectbox("Select Symbol", ALL_TRACKED_SYMBOLS)
    df = fetch_data_yf(symbol)
    df = calculate_indicators(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Candles"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], line=dict(width=1.5), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], line=dict(width=1.5), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], line=dict(width=1, dash="dot"), name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], line=dict(width=1, dash="dot"), name="Lower Band"))
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 6. PAPER TRADING
# ===============================================================
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

def tab_paper_trading():
    st.header("💰 Paper Trading Simulator")
    symbol = st.selectbox("Select Symbol for Paper Trade", ALL_TRACKED_SYMBOLS)
    action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
    qty = st.number_input("Quantity", 1, 1000, 100)
    price = st.number_input("Entry Price", 0.0, 100000.0, 100.0)
    if st.button("Execute Paper Trade"):
        st.session_state.paper_trades.append({
            "Symbol": symbol,
            "Action": action,
            "Qty": qty,
            "Price": price,
            "Time": datetime.datetime.now().strftime("%H:%M:%S")
        })
        st.success(f"{action} {qty} {symbol} @ {price}")

    if st.session_state.paper_trades:
        st.subheader("Open Paper Trades")
        df = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(df, use_container_width=True)

# ===============================================================
# 7. TRADE LOG
# ===============================================================
def tab_trade_log():
    st.header("📜 Trade Log")
    if st.session_state.paper_trades:
        df = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "trade_log.csv")
    else:
        st.info("No trades recorded yet.")

# ===============================================================
# 8. BACKTEST MODULE
# ===============================================================
def enhanced_signal_conditions(df):
    signals = []
    for i in range(51, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        avg_vol = df['Volume'].iloc[i - 20:i].mean() if i >= 20 else df['Volume'].iloc[:i].mean()
        vol_ok = row['Volume'] > avg_vol
        rsi_buy_ok = row['RSI'] > 55
        rsi_sell_ok = row['RSI'] < 45
        if row['SMA_20'] > row['SMA_50'] and prev['SMA_20'] <= prev['SMA_50'] and rsi_buy_ok and vol_ok:
            signals.append({'index': df.index[i], 'action': 'BUY', 'price': row['Close']})
        elif row['SMA_20'] < row['SMA_50'] and prev['SMA_20'] >= prev['SMA_50'] and rsi_sell_ok and vol_ok:
            signals.append({'index': df.index[i], 'action': 'SELL', 'price': row['Close']})
    return signals

def backtest_strategy(symbol="RELIANCE.NS", period="5d", capital=100000, qty=100):
    df = fetch_data_yf(symbol, period=period)
    df = calculate_indicators(df)
    if df.empty: return None, None
    signals = enhanced_signal_conditions(df)
    trades, balance, pnl_curve = [], capital, [capital]
    wins, losses = 0, 0
    for sig in signals:
        entry = sig['price']
        sl = entry * (0.99 if sig['action']=="BUY" else 1.01)
        tgt = entry * (1.015 if sig['action']=="BUY" else 0.985)
        exitp = None
        for j in range(df.index.get_loc(sig['index'])+1, len(df)):
            p = df.iloc[j]['Close']
            if sig['action']=="BUY":
                if p<=sl: exitp=sl; break
                elif p>=tgt: exitp=tgt; break
            else:
                if p>=sl: exitp=sl; break
                elif p<=tgt: exitp=tgt; break
        if not exitp: exitp=df.iloc[-1]['Close']
        pnl = (exitp-entry)*qty if sig['action']=="BUY" else (entry-exitp)*qty
        balance += pnl
        pnl_curve.append(balance)
        trades.append({"Action":sig['action'], "Entry":entry, "Exit":exitp, "PNL":pnl})
        if pnl>0: wins+=1
        else: losses+=1
    df_trades = pd.DataFrame(trades)
    stats = {
        "Total Trades": len(df_trades),
        "Winning Trades": wins,
        "Losing Trades": losses,
        "Win Rate %": round((wins/len(df_trades)*100) if len(df_trades)>0 else 0,2),
        "Net PNL (₹)": round(balance-capital,2),
        "Final Balance (₹)": round(balance,2),
        "Profit Factor": round(abs(df_trades[df_trades['PNL']>0]['PNL'].sum()/df_trades[df_trades['PNL']<0]['PNL'].sum()) if not df_trades[df_trades['PNL']<0].empty else np.nan,2)
    }
    return stats, pnl_curve

def tab_backtest():
    st.header("📊 Backtest Module (Enhanced Strategy)")
    symbol = st.selectbox("Select Stock for Backtest", ALL_TRACKED_SYMBOLS)
    period = st.selectbox("Data Period", ["2d", "5d", "10d", "1mo", "3mo"], index=1)
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            stats, pnl_curve = backtest_strategy(symbol, period)
            if stats:
                st.success("✅ Backtest Completed")
                st.subheader("Performance Summary")
                st.write(stats)
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(pnl_curve, label="Equity Curve", linewidth=2)
                ax.set_xlabel("Trade #"); ax.set_ylabel("Equity (₹)")
                ax.legend(); st.pyplot(fig)
            else:
                st.warning("No trades generated.")

# ===============================================================
# 9. DASHBOARD
# ===============================================================
def tab_dashboard():
    st.header("📡 Intraday Signal Dashboard")
    st.info("Auto-refreshing every 30 seconds.")
    for sym in ALL_TRACKED_SYMBOLS[:5]:
        df = fetch_data_yf(sym)
        df = calculate_indicators(df)
        signals = generate_signals(df)
        if signals:
            action, t, p = signals[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("Symbol", sym)
            col2.metric("Signal", action)
            col3.metric("Price", round(p,2))
        else:
            st.text(f"{sym}: No Signal")

# ===============================================================
# 10. MAIN LAYOUT
# ===============================================================
tabs = st.tabs(["Dashboard", "Paper Trading", "Trade Log", "Live Chart", "Backtest"])
with tabs[0]: tab_dashboard()
with tabs[1]: tab_paper_trading()
with tabs[2]: tab_trade_log()
with tabs[3]: tab_live_chart()
with tabs[4]: tab_backtest()
