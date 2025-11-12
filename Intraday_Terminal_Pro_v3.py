# Intraday Terminal — Final (v4.0)
# - Full Nifty50 + Next50 scan (100 symbols)
# - Robust yfinance fallback fetch (tries multiple interval/period combos)
# - Relaxed EMA signal logic + RSI & VWAP confirmation + confidence score
# - Auto-execution (paper) with Entry/Target/Stop table
# - Live chart tab with interval fallback and status
# - Trending stocks table
#
# Notes:
# - This is PAPER trading only. Replace PaperTrader.open() internals with broker API calls for real orders.
# - yfinance may still be rate-limited; consider a paid data feed for production.
# Dependencies: streamlit, yfinance, pandas, numpy, pytz, streamlit-autorefresh, plotly
# Install: pip install streamlit yfinance pandas numpy pytz streamlit-autorefresh plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz, time, math
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
st.set_page_config(page_title="Intraday Terminal — Final v4.0", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

CAPITAL = 1_000_000
TRADE_ALLOC = 0.10           # fraction of capital to allocate per trade
MAX_CONCURRENT_TRADES = 10
AUTO_EXEC_CONF = 0.65        # lower to allow more auto-exec but keep reasonable
SIGNAL_REFRESH_MS = 25_000   # 25 sec
CHART_DEFAULT_INTERVALS = [("5d","5m"),("1d","1m"),("5d","15m")]  # fallback order

# ---------------- Universe ----------------
NIFTY_50 = [
"ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS",
"BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
"EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
"ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
"NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHREECEM.NS","SUNPHARMA.NS",
"TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS"
]

NIFTY_NEXT_50 = [
"ABB.NS","ACC.NS","ADANIGREEN.NS","ADANIPOWER.NS","AUBANK.NS","BANKBARODA.NS","BERGEPAINT.NS","BIOCON.NS","BOSCHLTD.NS",
"CANBK.NS","CHOLAFIN.NS","COLPAL.NS","DALBHARAT.NS","DABUR.NS","DLF.NS","GAIL.NS","GLAND.NS","GODREJCP.NS","HAVELLS.NS",
"HDFCAMC.NS","HINDPETRO.NS","ICICIGI.NS","ICICIPRULI.NS","IGL.NS","INDHOTEL.NS","INDIGO.NS","IRCTC.NS","JINDALSTEL.NS",
"LUPIN.NS","MARICO.NS","MCDOWELL-N.NS","MOTHERSON.NS","MPHASIS.NS","MUTHOOTFIN.NS","NMDC.NS","PEL.NS","PIIND.NS","PNB.NS",
"POLYCAB.NS","RECLTD.NS","SBICARD.NS","SRF.NS","TATAPOWER.NS","TATACHEM.NS","TORNTPHARM.NS","TRENT.NS","TVSMOTOR.NS",
"VEDL.NS","VOLTAS.NS","ZOMATO.NS"
]

ALL_SYMBOLS = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# ---------------- Utilities ----------------
def now_indian():
    return datetime.now(IND_TZ)

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = (delta.where(delta>0,0)).rolling(n).mean()
    loss = (-delta.where(delta<0,0)).rolling(n).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ef = ema(close, fast); es = ema(close, slow)
    ml = ef - es
    sl = ema(ml, signal)
    return ml, sl

# Robust fetch with interval fallback + simple caching
_fetch_cache = {}
def fetch_with_fallback(symbol, tries=CHART_DEFAULT_INTERVALS):
    # caching small-window to reduce yfinance hits
    cache_key = f"{symbol}"
    now_ts = time.time()
    if cache_key in _fetch_cache and now_ts - _fetch_cache[cache_key]['ts'] < 12:  # 12 sec cache
        return _fetch_cache[cache_key]['df'], _fetch_cache[cache_key]['used']
    for period, interval in tries:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if df is None or df.empty:
                continue
            # Normalize MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.droplevel(0)
                except:
                    df.columns = [c[1] if len(c)>1 else c[0] for c in df.columns]
            df.columns = [str(c) for c in df.columns]
            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df = df.rename(columns={'Adj Close':'Close'})
            # require OHLC
            if not all(k in df.columns for k in ['Open','High','Low','Close']):
                continue
            df = df.dropna(subset=['Close']).copy()
            # add indicators
            df['EMA8'] = ema(df['Close'], 8)
            df['EMA21'] = ema(df['Close'], 21)
            df['EMA50'] = ema(df['Close'], 50)
            df['RSI14'] = rsi(df['Close'], 14).fillna(50)
            ml, sl = macd(df['Close'])
            df['MACD'] = ml; df['MACD_Signal'] = sl; df['MACD_HIST'] = df['MACD'] - df['MACD_Signal']
            # VWAP (simple) - only if Volume exists
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
            else:
                df['VWAP'] = df['Close'].rolling(20).mean()
            _fetch_cache[cache_key] = {'df': df, 'used': (period, interval), 'ts': now_ts}
            return df, (period, interval)
        except Exception:
            continue
    return None, (None, None)

# ---------------- Signal Engine (relaxed & scoring) ----------------
def signal_engine(df, symbol, market_bias='NEUTRAL'):
    """
    Returns dict:
    {symbol, action, entry, stop, target, conf, score, reason}
    """
    if df is None or len(df) < 20:
        return None
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    # EMA bias (relaxed)
    if cur['EMA8'] > cur['EMA21']:
        score += 2; reasons.append('EMA8>EMA21')
    if cur['EMA21'] > cur['EMA50']:
        score += 1; reasons.append('EMA21>EMA50')
    if cur['Close'] > cur['EMA50']:
        score += 1; reasons.append('Price>EMA50')

    # VWAP
    try:
        if cur['Close'] > cur['VWAP']:
            score += 1; reasons.append('AboveVWAP')
        else:
            reasons.append('BelowVWAP')
    except:
        pass

    # RSI moderate momentum
    if 40 <= cur['RSI14'] <= 75:
        score += 1; reasons.append('RSI OK')

    # MACD positive
    if cur['MACD'] > cur['MACD_Signal']:
        score += 1; reasons.append('MACD bullish')

    # Market bias: if overall index trending up, favor BUYs
    if market_bias == 'BULL' and score >= 2:
        score += 1
    if market_bias == 'BEAR' and score <= 2:
        score -= 1

    # Decide action:
    # BUY if short-term momentum positive and price > EMA21
    if cur['EMA8'] > cur['EMA21'] and cur['Close'] > cur['EMA21'] and score >= 4:
        entry = float(cur['Close'])
        atr = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1] if 'High' in df else max(0.01, entry*0.005))
        stop = entry - atr * 1.5
        target = entry + atr * 3
        conf = min(0.99, 0.5 + score * 0.08)
        return {"symbol": symbol, "action": "BUY", "entry": entry, "stop": float(stop), "target": float(target), "conf": conf, "score": score, "reason": ";".join(reasons)}

    # SELL if short-term momentum negative
    if cur['EMA8'] < cur['EMA21'] and cur['Close'] < cur['EMA21'] and score <= 2:
        entry = float(cur['Close'])
        atr = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1] if 'High' in df else max(0.01, entry*0.005))
        stop = entry + atr * 1.5
        target = entry - atr * 3
        conf = min(0.99, 0.5 + (6 - score) * 0.08)
        return {"symbol": symbol, "action": "SELL", "entry": entry, "stop": float(stop), "target": float(target), "conf": conf, "score": score, "reason": ";".join(reasons)}

    return None

# ---------------- Paper Trader ----------------
class PaperTrader:
    def __init__(self, capital=CAPITAL):
        self.initial = capital
        self.cash = capital
        self.positions = {}   # symbol -> dict
        self.tradelog = []    # list of events (open/close)
    def trade_size(self, entry):
        # simple: allocate TRADE_ALLOC * capital
        alloc = TRADE_ALLOC * self.initial
        qty = max(1, int(alloc // entry))
        return qty
    def open(self, sig):
        # auto-exec if confidence threshold met and capacity available
        if sig is None: return False
        if sig['conf'] < AUTO_EXEC_CONF: return False
        sym = sig['symbol']
        if sym in self.positions: return False
        if len(self.positions) >= MAX_CONCURRENT_TRADES: return False
        qty = self.trade_size(sig['entry'])
        cost = qty * sig['entry']
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[sym] = {**sig, 'qty': qty, 'open_time': now_indian(), 'status': 'OPEN'}
        self.tradelog.append({'time': now_indian(), 'event': 'OPEN', 'symbol': sym, 'action': sig['action'], 'qty': qty, 'price': sig['entry'], 'stop': sig['stop'], 'target': sig['target'], 'conf': sig['conf']})
        return True
    def update_positions(self):
        closed = []
        for sym, p in list(self.positions.items()):
            df, used = fetch_with_fallback(sym)
            if df is None: continue
            cur = float(df['Close'].iloc[-1])
            if p['action']=='BUY':
                if cur <= p['stop'] or cur >= p['target']:
                    reason = 'SL' if cur<=p['stop'] else 'TG'
                    pnl = (cur - p['entry']) * p['qty']
                    self.cash += p['qty'] * cur
                    self.tradelog.append({'time': now_indian(), 'event': 'CLOSE', 'symbol': sym, 'action': p['action'], 'qty': p['qty'], 'price': cur, 'pnl': pnl, 'reason': reason})
                    del self.positions[sym]
            else:
                if cur >= p['stop'] or cur <= p['target']:
                    reason = 'SL' if cur>=p['stop'] else 'TG'
                    pnl = (p['entry'] - cur) * p['qty']
                    self.cash += p['qty'] * cur
                    self.tradelog.append({'time': now_indian(), 'event': 'CLOSE', 'symbol': sym, 'action': p['action'], 'qty': p['qty'], 'price': cur, 'pnl': pnl, 'reason': reason})
                    del self.positions[sym]
    def positions_df(self):
        rows = []
        for sym,p in self.positions.items():
            df, used = fetch_with_fallback(sym)
            cur = float(df['Close'].iloc[-1]) if df is not None else p['entry']
            rows.append({'Symbol': sym, 'Action': p['action'], 'Qty': p['qty'], 'Entry': p['entry'], 'Current': cur, 'Stop': p['stop'], 'Target': p['target'], 'Conf': f"{p['conf']:.0%}", 'Status': p['status']})
        return pd.DataFrame(rows)
    def trades_df(self):
        if not self.tradelog:
            return pd.DataFrame()
        df = pd.DataFrame(self.tradelog)
        # format time
        df['time'] = df['time'].apply(lambda t: t.strftime("%Y-%m-%d %H:%M:%S"))
        return df

if 'trader' not in st.session_state:
    st.session_state.trader = PaperTrader()
trader = st.session_state.trader

# ---------------- Market Regime helper ----------------
def market_bias():
    # simple: check 20/50 SMA on NSEI 60d 1d bars
    df, used = fetch_with_fallback("^NSEI", tries=[("90d","1d")])
    if df is None or len(df)<50: return 'NEUTRAL'
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    cur = df['Close'].iloc[-1]
    if sma20 > sma50 and cur > sma20: return 'BULL'
    if sma20 < sma50 and cur < sma20: return 'BEAR'
    return 'NEUTRAL'

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;color:#0b486b;'>📈 Intraday Terminal — Final v4.0</h1>", unsafe_allow_html=True)

tabs = st.tabs(["Dashboard","Signals & Trades","Live Chart","Backtest & Analytics","Settings"])

# Dashboard
with tabs[0]:
    st.subheader("Overview")
    bias = market_bias()
    st.metric("Market Bias (NSEI)", bias)
    st.write(f"Account cash: ₹{trader.cash:,.0f} — Open positions: {len(trader.positions)}")
    st.divider()
    st.subheader("Trending (Top 10 % Movers from Nifty50 intraday)")
    movers = []
    progress = st.progress(0)
    for i, s in enumerate(NIFTY_50):
        df, used = fetch_with_fallback(s, tries=[("1d","1m"),("5d","5m")])
        if df is None or len(df) < 2:
            continue
        cur = df['Close'].iloc[-1]; prv = df['Close'].iloc[0]
        movers.append((s, (cur-prv)/prv))
        progress.progress((i+1)/len(NIFTY_50))
    progress.empty()
    if movers:
        top = pd.DataFrame(sorted(movers, key=lambda x: x[1], reverse=True)[:10], columns=['Symbol','%Change'])
        top['%Change'] = top['%Change'].apply(lambda x: f"{x:.2%}")
        st.table(top)
    else:
        st.info("No trending data available (data fetch issues).")

# Signals & Trades
with tabs[1]:
    st.subheader("Signal Scanner & Auto Execution")
    st.write(f"Scanning {len(ALL_SYMBOLS)} symbols - refresh every {SIGNAL_REFRESH_MS/1000:.0f}s")
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key="sig_refresh")
    # progress and status
    scan_progress = st.progress(0)
    status_area = st.empty()
    found_signals = []
    bias = market_bias()
    for idx, sym in enumerate(ALL_SYMBOLS):
        status_area.text(f"Fetching {sym} ({idx+1}/{len(ALL_SYMBOLS)})")
        df, used = fetch_with_fallback(sym)
        sig = signal_engine(df, sym, market_bias=bias)
        if sig:
            sig['data_interval'] = used
            found_signals.append(sig)
            # auto execute paper trade
            executed = trader.open(sig)
            if executed:
                status_area.text(f"Executed PAPER order for {sym} @ {sig['entry']:.2f}")
        scan_progress.progress((idx+1)/len(ALL_SYMBOLS))
    scan_progress.empty()
    status_area.empty()

    if found_signals:
        df_sig = pd.DataFrame(found_signals)
        df_sig = df_sig.sort_values('conf', ascending=False)
        df_sig['conf'] = df_sig['conf'].apply(lambda x: f"{x:.0%}")
        st.subheader("Signals (recent scan)")
        st.dataframe(df_sig[['symbol','action','entry','stop','target','conf','score','data_interval','reason']])
    else:
        st.info("No signals found in this scan.")

    st.divider()
    st.subheader("Open Positions (Paper)")
    posdf = trader.positions_df()
    if not posdf.empty:
        st.dataframe(posdf)
        # allow manual close for testing
        to_close = st.selectbox("Select position to close manually", options=["-"] + list(trader.positions.keys()))
        if st.button("Close Selected") and to_close != "-":
            p = trader.positions.get(to_close)
            if p:
                df, used = fetch_with_fallback(to_close)
                price = float(df['Close'].iloc[-1]) if df is not None else p['entry']
                # simulate close
                if p['action']=='BUY':
                    pnl = (price - p['entry'])*p['qty']
                else:
                    pnl = (p['entry'] - price)*p['qty']
                trader.cash += p['qty'] * price
                trader.tradelog.append({'time': now_indian(), 'event': 'CLOSE', 'symbol': to_close, 'action': p['action'], 'qty': p['qty'], 'price': price, 'pnl': pnl, 'reason':'MANUAL'})
                del trader.positions[to_close]
                st.success(f"Closed {to_close} @ {price:.2f}")
    else:
        st.info("No open positions.")

    # update positions (sl/tg)
    trader.update_positions()
    st.divider()
    st.subheader("Trade Log")
    tlog = trader.trades_df()
    if not tlog.empty:
        st.dataframe(tlog)
    else:
        st.info("No trades yet.")

# Live Chart
with tabs[2]:
    st.subheader("Live Chart (auto fallback intervals)")
    chart_sym = st.selectbox("Symbol", ALL_SYMBOLS, index=0)
    df, used = fetch_with_fallback(chart_sym)
    if df is None:
        st.error("No data available for selected symbol with available intervals.")
    else:
        st.write(f"Data interval used: {used}")
        # plot candlestick + indicators
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='price'))
        if 'EMA8' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA8'], name='EMA8', line=dict(width=1)))
        if 'EMA21' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(width=1)))
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(width=1, dash='dot')))
        fig.update_layout(xaxis_rangeslider_visible=False, height=520)
        st.plotly_chart(fig, use_container_width=True)

# Backtest & Analytics (light)
with tabs[3]:
    st.subheader("Backtest (light single-run)")
    st.info("This backtester runs a simple simulate-on-historical-5m sample for a few symbols. Use sparingly.")
    symbols_input = st.text_input("Symbols (comma separated)", value="RELIANCE.NS,TCS.NS,INFY.NS")
    days = st.number_input("Lookback days", min_value=3, max_value=90, value=30)
    if st.button("Run quick backtest"):
        syms = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        results = []
        st.info("Running - this may take some time.")
        pb = st.progress(0)
        for i,sym in enumerate(syms):
            df, used = fetch_with_fallback(sym, tries=[(f"{days}d","5m"),(f"{days}d","15m")])
            if df is None or len(df)<100:
                results.append({'symbol': sym, 'trades': 0, 'pnl': 0, 'win_rate': 0})
                pb.progress((i+1)/len(syms)); continue
            # naive walk-forward: check signals point-in-time and next bar entry and scanning forward for exit
            trades = 0; wins = 0; pnl = 0.0
            for idx in range(60, len(df)-10):
                window = df.iloc[:idx+1]
                sig = signal_engine(window, sym, market_bias=bias)
                if not sig: continue
                entry = float(df['Open'].iloc[idx+1])
                qty = max(1, int((TRADE_ALLOC*CAPITAL)//entry))
                # walk forward
                exit_price = None; reason=None
                for j in range(idx+2, min(len(df), idx+200)):
                    price = float(df['Close'].iloc[j])
                    if sig['action']=='BUY':
                        if price <= sig['stop']:
                            exit_price = sig['stop']; reason='SL'; break
                        if price >= sig['target']:
                            exit_price = sig['target']; reason='TG'; break
                    else:
                        if price >= sig['stop']:
                            exit_price = sig['stop']; reason='SL'; break
                        if price <= sig['target']:
                            exit_price = sig['target']; reason='TG'; break
                if exit_price is None:
                    # close at last available
                    exit_price = float(df['Close'].iloc[min(len(df)-1, idx+199)])
                    reason='END'
                if sig['action']=='BUY':
                    p = (exit_price - entry) * qty
                else:
                    p = (entry - exit_price) * qty
                trades += 1; pnl += p
                if p > 0: wins += 1
            results.append({'symbol': sym, 'trades': trades, 'pnl': pnl, 'win_rate': (wins/trades if trades>0 else 0)})
            pb.progress((i+1)/len(syms))
        st.write(pd.DataFrame(results))

# Settings
with tabs[4]:
    st.subheader("Settings & Tuning")
    st.write("Tweak thresholds (restart app for safety after large changes).")
    st.write(f"Auto-exec confidence (currently {AUTO_EXEC_CONF:.2f}) - to change, edit the file or request a UI control.")
    st.write("Notes: For production, integrate a reliable intraday data feed and a broker API.")

# Footer / housekeeping
st.markdown("---")
st.markdown("<div style='text-align:center;color:#777;'>Paper Trading Terminal v4.0 — For strategy development only. Replace paper logic with broker connector for live trading.</div>", unsafe_allow_html=True)
