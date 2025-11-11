# intraday_live_final_auto_v2.py
"""
Final Live Intraday Terminal — auto paper trading, full UI, equity & P/L display.
- Signals auto-scan every 10s (live mode)
- Charts refresh every 5s
- Paper capital ₹500,000
- Auto-exec only if confidence >= 0.60
- Shows Entry / Current / Unrealized PnL / Target Achieved
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time
import pytz, warnings
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
st.set_page_config(page_title="Intraday Live Terminal v2", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")
PAPER_CAPITAL = 500_000.0
SIGNAL_REFRESH_MS = 10_000
CHART_REFRESH_MS = 5_000
MIN_CONF_DEFAULT = 0.60

# ---------------- Universe lists (expanded) ----------------
NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS","ICICIBANK.NS","KOTAKBANK.NS",
    "BHARTIARTL.NS","ITC.NS","LT.NS","SBIN.NS","ASIANPAINT.NS","HCLTECH.NS","AXISBANK.NS",
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","NTPC.NS","NESTLEIND.NS",
    "POWERGRID.NS","M&M.NS","BAJFINANCE.NS","ONGC.NS","TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS",
    "ADANIPORTS.NS","COALINDIA.NS","HDFCLIFE.NS","DRREDDY.NS","HINDALCO.NS","CIPLA.NS","SBILIFE.NS",
    "GRASIM.NS","TECHM.NS","BAJAJFINSV.NS","BRITANNIA.NS","EICHERMOT.NS","DIVISLAB.NS","SHREECEM.NS",
    "APOLLOHOSP.NS","UPL.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","INDUSINDBK.NS","ADANIENT.NS","HDFC.NS"
]

# Nifty Next 50 (sample subset, you can replace with full official list)
NIFTY_NEXT_50 = [
    "ABB.NS","ADANIGREEN.NS","BANKBARODA.NS","BEL.NS","CANBK.NS","CHOLAFIN.NS","DABUR.NS",
    "GAIL.NS","HAL.NS","IOC.NS","JINDALSTEL.NS","PIDILITIND.NS","PNB.NS","SHREECEM.NS","TORNTPOWER.NS"
]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))
NIFTY_500 = NIFTY_100 + ["ONGC.NS","TATAMOTORS.NS","ADANIPORTS.NS","TATASTEEL.NS","JSWSTEEL.NS","COALINDIA.NS"]

# ---------------- Helpers ----------------
def now_indian(): return datetime.now(IND_TZ)
def market_open(now=None):
    if now is None: now = now_indian()
    today = now.date()
    open_dt = IND_TZ.localize(datetime.combine(today, dt_time(9,15)))
    close_dt = IND_TZ.localize(datetime.combine(today, dt_time(15,30)))
    return open_dt <= now <= close_dt

def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

# ---------------- Robust yfinance fetch ----------------
@st.cache_data(ttl=15)
def fetch_ohlc(symbol, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty: return None
        # Flatten multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        required = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in required): return None
        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 10: return None
        df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["SMA50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        return df
    except Exception as e:
        st.session_state.setdefault("_fetch_errors", {})[symbol] = str(e)
        return None

# ---------------- Strategy (returns dict or None) ----------------
def generate_signal(df, symbol):
    if df is None or len(df) < 3: return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    sma20, sma50 = latest["SMA20"], latest["SMA50"]
    sma20_prev, sma50_prev = prev["SMA20"], prev["SMA50"]
    ema8, ema21 = latest["EMA8"], latest["EMA21"]
    ema8_prev, ema21_prev = prev["EMA8"], prev["EMA21"]
    rsi_val = latest["RSI14"]
    close = float(latest["Close"])

    # Conservative SMA crossover
    if (sma20 > sma50 and sma20_prev <= sma50_prev and rsi_val < 85):
        conf = 0.60 + min(0.25, (sma20 - sma50) / max(0.01, sma50))
        return {"symbol": symbol, "action": "BUY", "entry": close, "stop": close*0.995,
                "target": close*1.01, "confidence": float(min(0.99, conf)), "reason": "SMA20>SMA50"}

    if (sma20 < sma50 and sma20_prev >= sma50_prev and rsi_val > 15):
        conf = 0.60 + min(0.25, (sma50 - sma20) / max(0.01, sma50))
        return {"symbol": symbol, "action": "SELL", "entry": close, "stop": close*1.005,
                "target": close*0.99, "confidence": float(min(0.99, conf)), "reason": "SMA20<SMA50"}

    # EMA momentum
    if (ema8 > ema21 and ema8_prev <= ema21_prev):
        return {"symbol": symbol, "action": "BUY", "entry": close, "stop": close*0.995,
                "target": close*1.015, "confidence": 0.58, "reason": "EMA8>EMA21"}

    if (ema8 < ema21 and ema8_prev >= ema21_prev):
        return {"symbol": symbol, "action": "SELL", "entry": close, "stop": close*1.005,
                "target": close*0.985, "confidence": 0.58, "reason": "EMA8<EMA21"}

    return None

# ---------------- Paper Trading engine (robust) ----------------
class PaperTrader:
    def __init__(self, capital=PAPER_CAPITAL):
        self.initial = float(capital)
        self.cash = float(capital)
        self.positions = {}   # symbol -> dict
        self.history = []     # closed trades

    def can_buy(self, entry_price, qty):
        return (qty * entry_price) <= self.cash

    def calc_trade_qty(self, entry_price, stop_price):
        # risk per trade = 1% initial capital
        risk_amount = 0.01 * self.initial
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        qty = int(risk_amount // risk_per_share)
        # ensure qty not to exceed cash
        qty = min(qty, int(self.cash // entry_price))
        return max(0, qty)

    def open_position(self, signal):
        sym = signal["symbol"]
        if sym in self.positions:
            return False
        entry = float(signal["entry"])
        stop = float(signal["stop"])
        qty = self.calc_trade_qty(entry, stop)
        if qty <= 0:
            return False
        cost = qty * entry
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[sym] = {
            "symbol": sym,
            "action": signal["action"],
            "qty": qty,
            "entry": entry,
            "stop": stop,
            "target": signal.get("target"),
            "confidence": signal.get("confidence", 0),
            "reason": signal.get("reason"),
            "open_time": now_indian()
        }
        return True

    def close_position(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return None
        pos = self.positions.pop(symbol)
        if exit_price is None:
            df = fetch_ohlc(symbol)
            exit_price = float(df["Close"].iloc[-1]) if df is not None else pos["entry"]
        pnl = ((exit_price - pos["entry"]) if pos["action"] == "BUY" else (pos["entry"] - exit_price)) * pos["qty"]
        self.cash += exit_price * pos["qty"]
        rec = {**pos, "exit": exit_price, "pnl": pnl, "close_time": now_indian()}
        self.history.append(rec)
        return rec

    def unrealized_pnl(self, market_prices: dict):
        """market_prices: dict symbol -> current_price"""
        total = 0.0
        for sym, pos in self.positions.items():
            cur = market_prices.get(sym, pos["entry"])
            pnl = ((cur - pos["entry"]) if pos["action"] == "BUY" else (pos["entry"] - cur)) * pos["qty"]
            total += pnl
        return total

    def equity(self, market_prices: dict):
        return self.cash + self.unrealized_pnl(market_prices)

# Keep paper trader in session state
if "paper_trader" not in st.session_state:
    st.session_state.paper_trader = PaperTrader(PAPER_CAPITAL)
paper = st.session_state.paper_trader

# ---------------- Streamlit UI ----------------
def main():
    st.title("🚀 Intraday Live Terminal (Final v2)")
    st.sidebar.header("Settings")
    universe_choice = st.sidebar.selectbox("Universe to scan", ["Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty 500", "Custom list"])
    if universe_choice == "Nifty 50":
        universe = NIFTY_50
    elif universe_choice == "Nifty Next 50":
        universe = NIFTY_NEXT_50
    elif universe_choice == "Nifty 100":
        universe = NIFTY_100
    elif universe_choice == "Nifty 500":
        universe = NIFTY_500
    else:
        # allow user to paste their list in the sidebar
        custom_text = st.sidebar.text_area("Paste tickers (one per line), e.g. RELIANCE.NS", height=120)
        universe = [s.strip() for s in custom_text.splitlines() if s.strip()] if custom_text.strip() else NIFTY_50

    min_conf = st.sidebar.slider("Min confidence to auto-exec / show", 0.50, 0.95, MIN_CONF_DEFAULT, 0.01)
    st.sidebar.write("Auto-exec requires confidence >= 0.60 by system policy; UI min_conf must be >= 0.60 for auto-exec.")
    # enforce minimum 0.60 for actual auto-exec
    enforce_auto_exec_threshold = 0.60

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Live Signals", "Live Charts", "Paper Trading", "Backtest"])

    # Dashboard
    with tab1:
        st.subheader("Market & Account Overview")
        col1, col2, col3 = st.columns(3)
        # show Nifty indices using yfinance tickers
        nifty_ticker = "^NSEI"
        banknifty_ticker = "^NSEBANK"
        try:
            idx1 = yf.Ticker(nifty_ticker).history(period="1d", interval="1d")
            idx1_val = idx1["Close"].iloc[-1] if not idx1.empty else None
        except:
            idx1_val = None
        try:
            idx2 = yf.Ticker(banknifty_ticker).history(period="1d", interval="1d")
            idx2_val = idx2["Close"].iloc[-1] if not idx2.empty else None
        except:
            idx2_val = None

        col1.metric("NIFTY 50", f"{idx1_val:.2f}" if idx1_val is not None else "n/a")
        col2.metric("BANK NIFTY", f"{idx2_val:.2f}" if idx2_val is not None else "n/a")

        # Market status & times
        now = now_indian()
        market_status = "OPEN" if market_open() else "CLOSED"
        col3.metric("Market Status", f"{market_status} — {now.strftime('%H:%M:%S %Z')}")

        # Account snapshot (compute unrealized PnL using latest market prices)
        # Gather current prices for open positions
        market_prices = {}
        for sym in list(paper.positions.keys()):
            df = fetch_ohlc(sym)
            if df is not None:
                market_prices[sym] = float(df["Close"].iloc[-1])
            else:
                market_prices[sym] = paper.positions[sym]["entry"]
        unreal = paper.unrealized_pnl(market_prices)
        equity = paper.equity(market_prices)
        st.write("")
        st.markdown("**Account Summary**")
        st.write(f"- Cash: ₹{paper.cash:,.2f}")
        st.write(f"- Unrealized P&L: ₹{unreal:,.2f}")
        st.write(f"- **Equity (Cash + Unrealized): ₹{equity:,.2f}**")
        st.write("")

    # Live Signals
    with tab2:
        st.subheader("Live Signals (auto-scan every 10s)")
        st.write(f"Universe: {len(universe)} symbols · Min confidence in UI: {min_conf:.2f} · Auto-exec threshold: {enforce_auto_exec_threshold:.2f}")
        st_autorefresh(interval=SIGNAL_REFRESH_MS, key="signal_refresh_v2")

        found_signals = []
        auto_executed = []
        if market_open():
            progress = st.progress(0)
            for i, sym in enumerate(universe):
                df = fetch_ohlc(sym)
                sig = generate_signal(df, sym)
                if sig and sig.get("confidence", 0) >= min_conf:
                    # include only those meeting UI filter; actual auto-exec only if >= enforce_auto_exec_threshold
                    found_signals.append(sig)
                    if sig["confidence"] >= enforce_auto_exec_threshold:
                        # attempt auto-exec
                        ok = paper.open_position(sig)
                        if ok:
                            auto_executed.append(sig)
                progress.progress((i+1)/len(universe))
            progress.empty()
        else:
            st.info("Market is closed — live 5m signals will not be produced. Run during market hours (09:15–15:30 IST).")

        if found_signals:
            st.success(f"Found {len(found_signals)} signals ({len(auto_executed)} auto-executed).")
            df_sig = pd.DataFrame(found_signals)
            df_sig_display = df_sig[["symbol","action","entry","stop","target","confidence","reason"]].copy()
            df_sig_display["entry"] = df_sig_display["entry"].map(lambda x: f"₹{x:.2f}")
            df_sig_display["stop"] = df_sig_display["stop"].map(lambda x: f"₹{x:.2f}")
            df_sig_display["target"] = df_sig_display["target"].map(lambda x: f"₹{x:.2f}")
            df_sig_display["confidence"] = df_sig_display["confidence"].map(lambda x: f"{x:.2f}")
            st.dataframe(df_sig_display, use_container_width=True)
        else:
            st.info("No signals found in this scan.")

    # Live Charts
    with tab3:
        st.subheader("Live Chart (refreshes every 5s)")
        st_autorefresh(interval=CHART_REFRESH_MS, key="chart_refresh_v2")
        universe_for_chart = st.selectbox("Chart universe", ["Nifty 50","Nifty Next 50","Nifty 100","Nifty 500"], index=0)
        if universe_for_chart == "Nifty 50":
            chart_symbols = NIFTY_50
        elif universe_for_chart == "Nifty Next 50":
            chart_symbols = NIFTY_NEXT_50
        elif universe_for_chart == "Nifty 100":
            chart_symbols = NIFTY_100
        else:
            chart_symbols = NIFTY_500
        selected_sym = st.selectbox("Select symbol", chart_symbols)
        df_chart = fetch_ohlc(selected_sym)
        if df_chart is None:
            st.warning("Could not fetch data for selected symbol (market closed / no intraday data).")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
                                         low=df_chart["Low"], close=df_chart["Close"], name="Price"))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["SMA20"], name="SMA20"))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["SMA50"], name="SMA50"))
            fig.update_layout(title=f"{selected_sym} — 5m", xaxis_rangeslider_visible=False, height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.line_chart(df_chart["RSI14"], height=180)

    # Paper Trading
    with tab4:
        st.subheader("Paper Trading — Open Positions")
        # update market prices for display
        market_prices = {}
        for sym in list(paper.positions.keys()):
            df = fetch_ohlc(sym)
            market_prices[sym] = float(df["Close"].iloc[-1]) if df is not None else paper.positions[sym]["entry"]

        rows = []
        for sym, pos in paper.positions.items():
            entry = pos["entry"]
            cur = market_prices.get(sym, entry)
            qty = pos["qty"]
            action = pos["action"]
            pnl = ((cur - entry) if action == "BUY" else (entry - cur)) * qty
            target_achieved = False
            target = pos.get("target")
            if target is not None:
                if action == "BUY" and cur >= target:
                    target_achieved = True
                if action == "SELL" and cur <= target:
                    target_achieved = True
            rows.append({
                "Symbol": sym,
                "Action": action,
                "Qty": qty,
                "Entry (₹)": f"{entry:.2f}",
                "Current (₹)": f"{cur:.2f}",
                "Unrealized P/L (₹)": f"{pnl:.2f}",
                "Target": f"{target:.2f}" if target is not None else "n/a",
                "Target Achieved": "YES" if target_achieved else "NO",
                "Open Time": pos.get("open_time")
            })
        if rows:
            df_open = pd.DataFrame(rows)
            st.dataframe(df_open, use_container_width=True)
        else:
            st.info("No open positions.")

        # Summary metrics
        unreal = paper.unrealized_pnl(market_prices)
        equity = paper.equity(market_prices)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cash Balance", f"₹{paper.cash:,.2f}")
        col_b.metric("Unrealized P/L", f"₹{unreal:,.2f}")
        col_c.metric("Equity", f"₹{equity:,.2f}")

        # Controls to close positions manually
        if paper.positions:
            st.markdown("**Close Position (manual)**")
            sym_to_close = st.selectbox("Choose position to close", list(paper.positions.keys()))
            if st.button("Close Selected Position"):
                df_close = fetch_ohlc(sym_to_close)
                px = float(df_close["Close"].iloc[-1]) if df_close is not None else paper.positions[sym_to_close]["entry"]
                rec = paper.close_position(sym_to_close, px)
                if rec:
                    st.success(f"Closed {sym_to_close}. P/L ₹{rec['pnl']:.2f}")

        if paper.history:
            st.markdown("**Recent Closed Trades**")
            hist_df = pd.DataFrame(paper.history)[["symbol","action","qty","entry","exit","pnl","open_time","close_time"]]
            st.dataframe(hist_df.tail(20), use_container_width=True)

    # Backtest placeholder
    with tab5:
        st.info("Backtest module can be added — currently disabled. Live Mode active.")

    # Debug: fetch errors
    if "_fetch_errors" in st.session_state and st.session_state["_fetch_errors"]:
        with st.expander("Fetch errors (debug)"):
            for k, v in st.session_state["_fetch_errors"].items():
                st.write(f"{k}: {v}")

if __name__ == "__main__":
    main()
