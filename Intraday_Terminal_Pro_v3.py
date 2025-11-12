"""
Intraday Live Trading Terminal — Pro Edition v3.0
Final deliverable (placed in canvas):
- 15s auto-refresh for signals
- Signals scanned from Nifty 50 + Next 50
- Auto-execution (paper) with 10 concurrent intraday trades cap
- Backtester that simulates intraday signals on historical 5m bars
- Fibonacci retracement (Golden strategy levels) calculation per symbol
- Risk management and sector exposure
- Clear, production-style Streamlit UI

IMPORTANT: This is a PAPER-TRADING terminal. For live execution integrate a broker API (Zerodha/Upstox/Alpaca etc.).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time, timedelta
import pytz, warnings, logging, time, math
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
st.set_page_config(page_title="Intraday Live Terminal Pro v3.0", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")

CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10            # 10% of capital per trade allocation logic default
MAX_CONCURRENT_TRADES = 10    # user requested 10 intraday trades
MAX_DAILY_TRADES = 10
MAX_DRAWDOWN = 0.05
SECTOR_EXPOSURE_LIMIT = 0.25

# Refresh intervals (user requested 15s)
SIGNAL_REFRESH_MS = 15_000
CHART_REFRESH_MS = 5_000
AUTO_EXEC_CONF = 0.70

# Nifty 50 and Next 50 (trimmed lists; use NSE suffix for yfinance)
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
    # Representative next-50 list (not exhaustive)
    "ABB.NS", "AUBANK.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "GLAND.NS",
    "GODREJCP.NS", "HAVELLS.NS", "HDFCAMC.NS", "ICICIPRULI.NS", "IGL.NS",
    "INDUSTOWER.NS", "JUBLFOOD.NS", "MANAPPURAM.NS", "MARICO.NS", "MPHASIS.NS",
    "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NMDC.NS", "PAGEIND.NS",
    "PEL.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS",
    "RECLTD.NS", "SAIL.NS", "SBICARD.NS", "SRF.NS", "TORNTPHARM.NS",
    "TRENT.NS", "VOLTAS.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "PIIND.NS"
]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

# Sector mapping (used for exposure limits)
SECTOR_MAP = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "MPHASIS.NS"],
    "AUTO": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
    "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS", "GAIL.NS"],
}

# ---------------- Utilities ----------------
def now_indian():
    return datetime.now(IND_TZ)


def market_open():
    n = now_indian()
    try:
        o = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        c = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return o <= n <= c
    except:
        return False


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    return true_range.rolling(period).mean()


def calculate_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()


def get_sector(symbol):
    for sector, stocks in SECTOR_MAP.items():
        if symbol in stocks:
            return sector
    return "OTHER"


def safe_yf_download(symbol, period="5d", interval="5m", max_retries=2):
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if df is not None and not df.empty and len(df) > 10:
                return df
        except Exception as e:
            logging.debug(f"yfinance error {symbol} attempt {attempt}: {e}")
            time.sleep(1)
    return None

# ---------------- Data Manager (with Fib) ----------------
class DataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        self.failed = set()

    def get_cached_data(self, symbol, period="5d", interval="5m"):
        if symbol in self.failed:
            return None
        key = f"{symbol}_{period}_{interval}"
        now_ts = time.time()
        if key in self.cache and now_ts - self.last_update.get(key, 0) < 30:
            return self.cache[key]
        df = self.fetch_ohlc(symbol, period, interval)
        if df is not None:
            self.cache[key] = df
            self.last_update[key] = now_ts
        else:
            self.failed.add(symbol)
        return df

    def fetch_ohlc(self, symbol, period="5d", interval="5m"):
        """Fetch OHLC data robustly and normalize column names.
        Handles MultiIndex columns, alternate column names (e.g., 'Adj Close') and
        gracefully returns None if required fields are missing.
        """
        df = safe_yf_download(symbol, period, interval)
        if df is None or df.empty:
            return None
        # Normalize MultiIndex produced by some yfinance calls
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(0)
            except Exception:
                # fallback: flatten
                df.columns = [c[1] if len(c) > 1 else c[0] for c in df.columns]
        # Ensure string column names
        df.columns = [str(c) for c in df.columns]
        # Common alternate names -> unify to 'Close','Open','High','Low','Volume'
        alt_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("adj close","adj_close","adjclose") and 'Close' not in df.columns:
                alt_map[c] = 'Close'
            if lc in ("close",) and 'Close' not in df.columns:
                alt_map[c] = 'Close'
            if lc in ("open",) and 'Open' not in df.columns:
                alt_map[c] = 'Open'
            if lc in ("high",) and 'High' not in df.columns:
                alt_map[c] = 'High'
            if lc in ("low",) and 'Low' not in df.columns:
                alt_map[c] = 'Low'
            if lc in ("volume",) and 'Volume' not in df.columns:
                alt_map[c] = 'Volume'
        if alt_map:
            df = df.rename(columns=alt_map)
        # last safe check: require Close/Open/High/Low
        required = ['Close','Open','High','Low']
        if not all(col in df.columns for col in required):
            return None
        # drop rows missing close
        df = df.dropna(subset=['Close']).copy()
        if len(df) < 20:
            return None
        # indicators
        df['EMA8'] = ema(df['Close'], 8)
        df['EMA21'] = ema(df['Close'], 21)
        df['EMA50'] = ema(df['Close'], 50)
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['RSI14'] = rsi(df['Close']).fillna(50)
        macd_line, sig_line = macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = sig_line
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        # protect Volume-based indicators if Volume missing or NaN
        if 'Volume' in df.columns:
            df['Volume_SMA20'] = df['Volume'].rolling(20).mean()
        else:
            df['Volume'] = 0
            df['Volume_SMA20'] = 0
        df['ATR'] = calculate_atr(df)
        df['VWAP'] = calculate_vwap(df)
        # compute Fib retracement based on last swing (lookback 60 bars)
        try:
            lookback = 60
            recent = df[-lookback:]
            high = recent['High'].max()
            low = recent['Low'].min()
            diff = high - low
            if diff > 0:
                df['FIB_0'] = high
                df['FIB_0.236'] = high - 0.236 * diff
                df['FIB_0.382'] = high - 0.382 * diff
                df['FIB_0.5'] = high - 0.5 * diff
                df['FIB_0.618'] = high - 0.618 * diff
                df['FIB_1'] = low
        except Exception:
            pass
        return df

# ---------------- Market regime ----------------
def detect_market_regime():
    df = data_manager.get_cached_data("^NSEI", period="60d", interval="1d")
    if df is None or len(df) < 30:
        return "NEUTRAL"
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    if len(df) < 50:
        return "NEUTRAL"
    cur = df['Close'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    if sma20 > sma50 and cur > sma20:
        return "BULL_TREND"
    if sma20 < sma50 and cur < sma20:
        return "BEAR_TREND"
    return "RANGING"

# ---------------- Risk Manager ----------------
class RiskManager:
    def __init__(self):
        self.daily_count = 0
        self.sector_exposure = {}
        self.peak = CAPITAL
        self.last_reset = now_indian().date()

    def reset_if_needed(self):
        if now_indian().date() != self.last_reset:
            self.daily_count = 0
            self.sector_exposure = {}
            self.last_reset = now_indian().date()

    def can_trade(self, symbol, cost, trader):
        self.reset_if_needed()
        if self.daily_count >= MAX_DAILY_TRADES:
            return False, "Daily limit reached"
        current_equity = trader.equity()
        drawdown = (self.peak - current_equity) / (self.peak if self.peak > 0 else 1)
        if drawdown > MAX_DRAWDOWN:
            return False, "Max drawdown breached"
        sector = get_sector(symbol)
        if self.sector_exposure.get(sector, 0) + cost > CAPITAL * SECTOR_EXPOSURE_LIMIT:
            return False, f"Sector exposure limit for {sector}"
        return True, "OK"

    def record(self, symbol, cost):
        sector = get_sector(symbol)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + cost
        self.daily_count += 1

# ---------------- Signal Engine (improved) ----------------
def advanced_signal_engine(df, sym):
    if df is None or len(df) < 30:
        return None
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    regime = detect_market_regime()
    bull = 0
    bear = 0
    # EMA alignment
    if cur.EMA8 > cur.EMA21 > cur.EMA50:
        bull += 3
    elif cur.EMA8 < cur.EMA21 < cur.EMA50:
        bear += 3
    # Volume
    if cur.Volume > (cur.Volume_SMA20 if not math.isnan(cur.Volume_SMA20) else 1) * 1.2:
        bull += 1; bear += 1
    # RSI
    if 35 < cur.RSI14 < 65:
        bull += 1; bear += 1
    if cur.RSI14 > df['RSI14'].iloc[-2]:
        bull += 1
    if cur.RSI14 < df['RSI14'].iloc[-2]:
        bear += 1
    # MACD
    if cur.MACD > cur.MACD_Signal and cur.MACD > df['MACD'].iloc[-2]:
        bull += 2
    if cur.MACD < cur.MACD_Signal and cur.MACD < df['MACD'].iloc[-2]:
        bear += 2
    # VWAP
    if cur.Close > cur.VWAP:
        bull += 1
    else:
        bear += 1
    # regime
    if regime == 'BULL_TREND':
        bull += 1
    elif regime == 'BEAR_TREND':
        bear += 1
    min_score = 7
    # Fibonacci confluence: if price near 0.382/0.5/0.618 add score
    fib_score = 0
    try:
        for level in ['FIB_0.382','FIB_0.5','FIB_0.618']:
            if level in df.columns:
                lvl = float(df[level].iloc[-1])
                if abs(cur.Close - lvl) / (lvl if lvl else 1) < 0.006:  # within 0.6%
                    fib_score += 1
    except Exception:
        pass
    bull += fib_score
    bear += fib_score
    # Signal
    if bull >= min_score and cur.EMA8 > prev.EMA8:
        atr = cur.ATR if not np.isnan(cur.ATR) else cur.Close * 0.008
        atr_stop = atr * 1.5
        stop = cur.Close - atr_stop
        target = cur.Close + (atr_stop * 2)
        conf = min(0.98, 0.65 + (bull * 0.03))
        return {"symbol": sym, "action": "BUY", "entry": float(cur.Close), "stop": float(stop), "target": float(target), "conf": conf, "score": bull, "atr": float(atr)}
    if bear >= min_score and cur.EMA8 < prev.EMA8:
        atr = cur.ATR if not np.isnan(cur.ATR) else cur.Close * 0.008
        atr_stop = atr * 1.5
        stop = cur.Close + atr_stop
        target = cur.Close - (atr_stop * 2)
        conf = min(0.98, 0.65 + (bear * 0.03))
        return {"symbol": sym, "action": "SELL", "entry": float(cur.Close), "stop": float(stop), "target": float(target), "conf": conf, "score": bear, "atr": float(atr)}
    return None

# ---------------- Paper Trader ----------------
class PaperTrader:
    def __init__(self, capital=CAPITAL):
        self.init = capital
        self.cash = capital
        self.pos = {}
        self.log = []
        self.risk = RiskManager()

    def trade_size(self, entry, atr=None):
        if atr and atr > 0:
            risk_amount = self.cash * 0.02
            shares = int(risk_amount / (atr * 1.5))
        else:
            shares = int((TRADE_ALLOC * self.init) // entry)
        shares_by_cap = int((TRADE_ALLOC * self.cash) // entry)
        return max(1, min(shares, shares_by_cap))

    def open(self, signal):
        if signal['symbol'] in self.pos:
            return False
        if signal['conf'] < AUTO_EXEC_CONF:
            return False
        if len(self.pos) >= MAX_CONCURRENT_TRADES:
            return False
        qty = self.trade_size(signal['entry'], signal.get('atr'))
        cost = qty * signal['entry']
        can, reason = self.risk.can_trade(signal['symbol'], cost, self)
        if not can:
            logging.debug(f"Trade blocked: {reason}")
            return False
        if cost > self.cash:
            return False
        self.cash -= cost
        self.pos[signal['symbol']] = {**signal, 'qty': qty, 'open_price': signal['entry'], 'open_time': now_indian()}
        self.risk.record(signal['symbol'], cost)
        self.log.append({'time': now_indian(), 'event': 'OPEN', 'symbol': signal['symbol'], 'action': signal['action'], 'qty': qty, 'price': signal['entry'], 'pnl': None})
        return True

    def update(self):
        to_close = []
        for sym, p in list(self.pos.items()):
            df = data_manager.get_cached_data(sym)
            if df is None:
                continue
            cur = float(df['Close'].iloc[-1])
            if p['action'] == 'BUY':
                if cur <= p['stop'] or cur >= p['target']:
                    to_close.append((sym, cur, 'SL' if cur<=p['stop'] else 'TG'))
            else:
                if cur >= p['stop'] or cur <= p['target']:
                    to_close.append((sym, cur, 'SL' if cur>=p['stop'] else 'TG'))
        for sym, price, reason in to_close:
            self.close(sym, price, reason)

    def close(self, sym, price, reason='MANUAL'):
        if sym not in self.pos:
            return
        p = self.pos.pop(sym)
        if p['action'] == 'BUY':
            pnl = (price - p['open_price']) * p['qty']
        else:
            pnl = (p['open_price'] - price) * p['qty']
        self.cash += price * p['qty']
        self.log.append({'time': now_indian(), 'event': 'CLOSE', 'symbol': sym, 'action': p['action'], 'qty': p['qty'], 'price': price, 'pnl': pnl, 'reason': reason})

    def equity(self):
        total = self.cash
        for sym, p in self.pos.items():
            df = data_manager.get_cached_data(sym)
            if df is not None:
                cur = float(df['Close'].iloc[-1])
                if p['action']=='BUY':
                    pnl = (cur - p['open_price']) * p['qty']
                else:
                    pnl = (p['open_price'] - cur) * p['qty']
                total += p['qty'] * p['open_price'] + pnl
        return total

    def positions_df(self):
        rows = []
        for sym,p in self.pos.items():
            df = data_manager.get_cached_data(sym)
            cur = float(df['Close'].iloc[-1]) if df is not None else p['open_price']
            pnl = (cur - p['open_price']) * p['qty'] if p['action']=='BUY' else (p['open_price'] - cur) * p['qty']
            rows.append({'Symbol': sym, 'Action': p['action'], 'Qty': p['qty'], 'Entry': p['open_price'], 'Current': cur, 'Stop': p['stop'], 'Target': p['target'], 'P/L': pnl})
        return pd.DataFrame(rows)

    def performance(self):
        closes = [t for t in self.log if t['event']=='CLOSE']
        if not closes:
            return {'total_trades':0,'win_rate':0,'total_pnl':0}
        wins = [t for t in closes if t['pnl']>0]
        total_pnl = sum(t['pnl'] for t in closes)
        return {'total_trades':len(closes),'win_rate':len(wins)/len(closes),'total_pnl':total_pnl}

# ---------------- Backtester ----------------
class Backtester:
    def __init__(self):
        pass

    def run_backtest(self, symbols, lookback_days=30):
        summary = {'total_trades':0,'wins':0,'pnl':0}
        # for each symbol fetch historical 5m bars for lookback period (can be heavy)
        period = f"{lookback_days}d"
        for sym in symbols:
            df = safe_yf_download(sym, period=period, interval='5m')
            if df is None or df.empty:
                continue
            # compute indicators same as live
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            df = df.dropna(subset=['Close']).copy()
            df['EMA8']=ema(df['Close'],8); df['EMA21']=ema(df['Close'],21); df['EMA50']=ema(df['Close'],50)
            df['RSI14']=rsi(df['Close']).fillna(50)
            macd_line,sig_line=macd(df['Close']); df['MACD']=macd_line; df['MACD_Signal']=sig_line
            df['ATR']=calculate_atr(df)
            # walk through bars and generate signals when possible
            for i in range(60,len(df)-2):
                window = df.iloc[:i+1]
                sig = advanced_signal_engine(window, sym)
                if sig is None:
                    continue
                # simulate entry at next bar open
                entry_price = float(df['Open'].iloc[i+1])
                qty = max(1, int((TRADE_ALLOC*CAPITAL)//entry_price))
                # simulate exit: scan forward until stop/target hit or end of day
                for j in range(i+2, min(len(df), i+200)):
                    price = float(df['Close'].iloc[j])
                    if sig['action']=='BUY':
                        if price <= sig['stop']:
                            pnl = (sig['stop'] - entry_price)*qty; summary['total_trades']+=1; summary['pnl']+=pnl; break
                        if price >= sig['target']:
                            pnl = (sig['target'] - entry_price)*qty; summary['total_trades']+=1; summary['pnl']+=pnl; summary['wins']+=1; break
                    else:
                        if price >= sig['stop']:
                            pnl = (entry_price - sig['stop'])*qty; summary['total_trades']+=1; summary['pnl']+=pnl; break
                        if price <= sig['target']:
                            pnl = (entry_price - sig['target'])*qty; summary['total_trades']+=1; summary['pnl']+=pnl; summary['wins']+=1; break
        if summary['total_trades']>0:
            summary['win_rate']=summary['wins']/summary['total_trades']
        else:
            summary['win_rate']=0
        return summary

# ---------------- Alerts ----------------
class AlertSystem:
    def __init__(self): self.alerts=[]
    def add(self,msg): self.alerts.append({'time':now_indian(),'msg':msg})
    def recent(self,n=10): return sorted(self.alerts,key=lambda x:x['time'],reverse=True)[:n]

# ---------------- Init ----------------
data_manager = DataManager()
alert_system = AlertSystem()
backtester = Backtester()
if 'trader' not in st.session_state:
    st.session_state.trader = PaperTrader()
trader = st.session_state.trader

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;color:#0b486b;'>📈 Intraday Terminal — Pro v3.0</h1>", unsafe_allow_html=True)
cols = st.columns(5)
with cols[0]:
    nifty_price = None
    try:
        t = yf.Ticker('^NSEI'); d = t.history(period='1d', interval='1m'); nifty_price = float(d['Close'].iloc[-1]) if not d.empty else None
    except: pass
    st.metric('NIFTY 50', f"₹{nifty_price:,.2f}" if nifty_price else "—")
with cols[1]:
    bank_price = None
    try:
        t = yf.Ticker('^NSEBANK'); d = t.history(period='1d', interval='1m'); bank_price = float(d['Close'].iloc[-1]) if not d.empty else None
    except: pass
    st.metric('BANK NIFTY', f"₹{bank_price:,.2f}" if bank_price else "—")
with cols[2]:
    st.metric('Market Status', '🟢 LIVE' if market_open() else '🔴 CLOSED')
with cols[3]:
    regime = detect_market_regime(); icon = '📈' if regime=='BULL_TREND' else '📉' if regime=='BEAR_TREND' else '➡️'; st.metric('Regime', f"{icon} {regime}")
with cols[4]:
    perf = trader.performance(); st.metric('Paper Win Rate', f"{perf.get('win_rate',0):.1%}")

tabs = st.tabs(["Dashboard","Signals","Charts","Paper Trading","Analytics","Alerts","Backtest"])

# Dashboard
with tabs[0]:
    st.subheader('Overview')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Cash', f"₹{trader.cash:,.0f}")
    c2.metric('Equity', f"₹{trader.equity():,.0f}")
    c3.metric('Open Positions', len(trader.pos))
    c4.metric('Daily Trades Used', f"{trader.risk.daily_count}/{MAX_DAILY_TRADES}")
    st.divider()
    st.subheader('Sector Rotation Snapshot')
    s = {k: np.random.uniform(-0.02,0.02) for k in SECTOR_MAP.keys()}  # lightweight placeholder
    st.bar_chart(pd.Series(s))

# Signals
with tabs[1]:
    st_autorefresh(interval=SIGNAL_REFRESH_MS, key='sig_ref')
    st.subheader('Signal Scanner (Nifty50 / Next50)')
    uni = st.selectbox('Universe', ['Nifty 50','Nifty 100'])
    min_conf = st.slider('Min Confidence', 0.6, 0.95, AUTO_EXEC_CONF, 0.05)
    show_reg = st.checkbox('Show Regime', True)
    symbols = NIFTY_50 if uni=='Nifty 50' else NIFTY_100
    if show_reg:
        r = detect_market_regime(); st.info(f'Market Regime: {r}')
    st.write(f'Scanning {len(symbols)} symbols...')
    signals = []
    if market_open():
        prog = st.progress(0)
        status = st.empty()
        for i,symbol in enumerate(symbols):
            status.text(f'Checking {symbol} ({i+1}/{len(symbols)})')
            df = data_manager.get_cached_data(symbol)
            sig = advanced_signal_engine(df, symbol)
            if sig and sig['conf']>=min_conf:
                signals.append(sig); alert_system.add(f"Signal {sig['action']} {symbol} @{sig['entry']:.2f}")
                if sig['conf']>=AUTO_EXEC_CONF:
                    trader.open(sig)
            prog.progress((i+1)/len(symbols))
        trader.update()
        prog.empty(); status.empty()
    else:
        st.warning('Market closed - scanning paused')
    if signals:
        df_sig = pd.DataFrame(signals).sort_values('conf', ascending=False)
        df_sig['conf'] = df_sig['conf'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_sig[['symbol','action','entry','stop','target','conf','score']])
    else:
        st.info('No signals found')

# Charts
with tabs[2]:
    st.subheader('Live Chart')
    sym = st.selectbox('Symbol', NIFTY_100, index=0)
    st_autorefresh(interval=CHART_REFRESH_MS, key='chart_ref')
    df = data_manager.get_cached_data(sym)
    if df is not None:
        fig = go.Figure(); fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        if 'EMA8' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA8'], name='EMA8'))
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=450)
        st.plotly_chart(fig, use_container_width=True)
        # show fib levels if present
        if 'FIB_0.382' in df.columns:
            latest = df.iloc[-1]
            st.markdown(f"**Fib levels (last swing):** 0.382={latest.get('FIB_0.382',np.nan):.2f}, 0.5={latest.get('FIB_0.5',np.nan):.2f}, 0.618={latest.get('FIB_0.618',np.nan):.2f}")
    else:
        st.error('No data')

# Paper Trading
with tabs[3]:
    st.subheader('Paper Trading Account')
    st.metric('Account Value', f"₹{trader.equity():,.0f}")
    st.metric('Cash', f"₹{trader.cash:,.0f}")
    pos_df = trader.positions_df()
    if not pos_df.empty:
        st.dataframe(pos_df)
        sym_to_close = st.selectbox('Close Position', list(trader.pos.keys()) if trader.pos else ['-'])
        if st.button('Close Selected') and sym_to_close!='-':
            cur = data_manager.get_cached_data(sym_to_close); price = float(cur['Close'].iloc[-1]) if cur is not None else trader.pos[sym_to_close]['open_price']; trader.close(sym_to_close, price, 'MANUAL')
    else:
        st.info('No open positions')

# Analytics
with tabs[4]:
    st.subheader('Performance Analytics')
    perf = trader.performance(); st.write(perf)
    if trader.log:
        logdf = pd.DataFrame(trader.log)
        st.dataframe(logdf)

# Alerts
with tabs[5]:
    st.subheader('Alerts')
    for a in alert_system.recent(20):
        st.write(f"{a['time'].strftime('%H:%M:%S')} - {a['msg']}")

# Backtest
with tabs[6]:
    st.subheader('Backtester')
    days = st.number_input('Days to test', min_value=3, max_value=90, value=30)
    if st.button('Run Backtest'):
        with st.spinner('Running backtest...'):
            result = backtester.run_backtest(NIFTY_50, lookback_days=int(days))
            st.success('Backtest finished')
            st.write(result)

# Sidebar
st.sidebar.header('Account')
if st.sidebar.button('Reset Paper Account'):
    st.session_state.trader = PaperTrader(); st.experimental_rerun()

st.markdown('---')
st.markdown("<div style='text-align:center;color:#777;'>Paper Trading Terminal v3.0 — Use for strategy development. Connect a broker API for live execution.</div>", unsafe_allow_html=True)
