# intraday_live_fixed.py
"""
Intraday Live Trading Dashboard (Streamlit)
- Auto-scan every 10s (Live Mode)
- Chart auto-refresh 5s
- Paper trading capital ₹500,000
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, timedelta, time as dt_time
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
st.set_page_config(page_title="Intraday Live", layout="wide", page_icon="📈")
INDIAN_TZ = pytz.timezone("Asia/Kolkata")
PAPER_INITIAL_CAPITAL = 500_000
SIGNAL_REFRESH_MS = 10_000
CHART_REFRESH_MS = 5_000

# --- Utility helpers ---
def now_indian():
    return datetime.now(INDIAN_TZ)

def market_is_open(now=None):
    if now is None: now = now_indian()
    o = INDIAN_TZ.localize(datetime.combine(now.date(), dt_time(9,15)))
    c = INDIAN_TZ.localize(datetime.combine(now.date(), dt_time(15,30)))
    return o <= now <= c

# --- Indicators ---
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    rs = g / l
    return 100 - (100/(1+rs))

# --- Robust data fetch ---
@st.cache_data(ttl=15)
def fetch_enhanced_ohlc(symbol, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty: return None
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        # Ensure required columns
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns:
                return None
        df = df.dropna(subset=["Close"]).copy()
        if len(df)<10: return None

        # Indicators
        df["SMA20"]=df["Close"].rolling(20,min_periods=1).mean()
        df["SMA50"]=df["Close"].rolling(50,min_periods=1).mean()
        df["EMA8"]=ema(df["Close"],8)
        df["EMA21"]=ema(df["Close"],21)
        df["RSI14"]=rsi(df["Close"])
        return df
    except Exception as e:
        st.session_state.setdefault("_fetch_errors",{})
        st.session_state["_fetch_errors"][symbol]=str(e)
        return None

# --- Signal generator ---
def gen_signal(df,sym):
    if df is None or len(df)<3: return None
    l,p=df.iloc[-1],df.iloc[-2]
    s20,s50=l.SMA20,l.SMA50
    s20p,s50p=p.SMA20,p.SMA50
    e8,e21=l.EMA8,l.EMA21
    e8p,e21p=p.EMA8,p.EMA21
    r=l.RSI14
    c=float(l.Close)

    # SMA cross
    if s20>s50 and s20p<=s50p and r<80:
        return dict(symbol=sym,action="BUY",entry=c,stop_loss=c*0.995,target1=c*1.01,
                    confidence=0.6,reason="SMA20>50 crossover")
    if s20<s50 and s20p>=s50p and r>20:
        return dict(symbol=sym,action="SELL",entry=c,stop_loss=c*1.005,target1=c*0.99,
                    confidence=0.6,reason="SMA20<50 crossover")

    # EMA momentum
    if e8>e21 and e8p<=e21p and r<85:
        return dict(symbol=sym,action="BUY",entry=c,stop_loss=c*0.995,target1=c*1.015,
                    confidence=0.55,reason="EMA8>21 crossover")
    if e8<e21 and e8p>=e21p and r>15:
        return dict(symbol=sym,action="SELL",entry=c,stop_loss=c*1.005,target1=c*0.985,
                    confidence=0.55,reason="EMA8<21 crossover")
    return None

# --- Paper trading ---
class Paper:
    def __init__(self,cap): self.cap0=self.avail=cap; self.pos={}; self.hist=[]
    def buy(self,sig):
        sym=sig["symbol"]
        if sym in self.pos: return False
        risk=self.cap0*0.01
        q=max(1,int(risk/abs(sig["entry"]-sig["stop_loss"])))
        cost=q*sig["entry"]
        if cost>self.avail: return False
        self.avail-=cost
        self.pos[sym]={**sig,"qty":q,"time":now_indian()}
        return True
    def close(self,sym,px):
        if sym not in self.pos: return None
        p=self.pos.pop(sym)
        pnl=(px-p["entry"])*p["qty"] if p["action"]=="BUY" else (p["entry"]-px)*p["qty"]
        self.avail+=px*p["qty"]; p["exit"]=px; p["pnl"]=pnl; p["close"]=now_indian()
        self.hist.append(p); return p

paper=Paper(PAPER_INITIAL_CAPITAL)

# --- UI ---
def main():
    st.title("📈 Intraday Live Dashboard")
    st.caption("Signals refresh every 10 s • Chart every 5 s • Paper capital ₹500 000")
    is_open=market_is_open()
    st.write("🕒 Market:", "🟢 OPEN" if is_open else "🔴 CLOSED")

    syms=["RELIANCE.NS","TCS.NS","ICICIBANK.NS","INFY.NS","ASIANPAINT.NS","HDFCBANK.NS"]
    min_conf=st.sidebar.slider("Min confidence",0.3,0.9,0.5,0.05)

    st_autorefresh(interval=SIGNAL_REFRESH_MS,key="scan")
    sigs=[]
    if is_open:
        for s in syms:
            df=fetch_enhanced_ohlc(s)
            sig=gen_signal(df,s)
            if sig and sig["confidence"]>=min_conf: sigs.append(sig)
    if sigs:
        st.success(f"{len(sigs)} new signal(s)")
        st.dataframe(pd.DataFrame(sigs)[["symbol","action","entry","stop_loss","target1","reason"]])
    else:
        st.info("No active signals (check during live market hours).")

    # Chart
    st.markdown("---")
    st.subheader("Live Chart (5 s auto-refresh)")
    sel=st.selectbox("Symbol",syms)
    st_autorefresh(interval=CHART_REFRESH_MS,key="chart_"+sel)
    df=fetch_enhanced_ohlc(sel)
    if df is not None:
        f=go.Figure(data=[go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close)])
        f.add_trace(go.Scatter(x=df.index,y=df.SMA20,name="SMA20"))
        f.add_trace(go.Scatter(x=df.index,y=df.SMA50,name="SMA50"))
        f.update_layout(height=450,xaxis_rangeslider_visible=False,title=sel)
        st.plotly_chart(f,use_container_width=True)
    else:
        st.warning("No data for selected symbol (market closed or fetch issue).")

if __name__=="__main__":
    main()
