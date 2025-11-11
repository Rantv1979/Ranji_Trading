# intraday_live_final_auto.py
"""
Intraday Live Trading Dashboard — Final Version
- Live Mode with auto paper trading
- 10s signal refresh / 5s chart refresh
- Nifty 50 / Next 50 / Nifty 100 / Nifty 500 universes
- ₹5,00,000 paper account
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

# ------------------ Configuration ------------------
st.set_page_config(page_title="Intraday Live Dashboard", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")
CAPITAL = 500_000
SIGNAL_REFRESH_MS = 10_000
CHART_REFRESH_MS = 5_000

# ------------------ Index Universes ------------------
NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","ICICIBANK.NS","INFY.NS","HDFCBANK.NS","ITC.NS","LT.NS","KOTAKBANK.NS",
    "HINDUNILVR.NS","AXISBANK.NS","BAJFINANCE.NS","BHARTIARTL.NS","SBIN.NS","ASIANPAINT.NS",
    "SUNPHARMA.NS","MARUTI.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","POWERGRID.NS"
]
NIFTY_NEXT_50 = [
    "ABB.NS","ADANIENT.NS","BANKBARODA.NS","BEL.NS","CANBK.NS","CHOLAFIN.NS","DABUR.NS",
    "GAIL.NS","HAL.NS","INDHOTEL.NS","IOC.NS","JINDALSTEL.NS","PIDILITIND.NS","PNB.NS","SHREECEM.NS"
]
NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))
NIFTY_500 = NIFTY_100 + [
    "TATAMOTORS.NS","ONGC.NS","COALINDIA.NS","ADANIPORTS.NS","HAVELLS.NS","BAJAJ-AUTO.NS","TATASTEEL.NS"
]

# ------------------ Helpers ------------------
def now_indian(): return datetime.now(IND_TZ)
def market_open():
    now = now_indian()
    o = IND_TZ.localize(datetime.combine(now.date(), dt_time(9,15)))
    c = IND_TZ.localize(datetime.combine(now.date(), dt_time(15,30)))
    return o <= now <= c

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    rs = g/l
    return 100 - (100/(1+rs))

# ------------------ Data Fetch ------------------
@st.cache_data(ttl=15)
def fetch_ohlc(symbol, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in df.columns: return None
        df = df.dropna(subset=["Close"])
        df["SMA20"]=df["Close"].rolling(20,min_periods=1).mean()
        df["SMA50"]=df["Close"].rolling(50,min_periods=1).mean()
        df["EMA8"]=ema(df["Close"],8)
        df["EMA21"]=ema(df["Close"],21)
        df["RSI14"]=rsi(df["Close"])
        return df
    except: return None

# ------------------ Strategy ------------------
def signal(df,sym):
    if df is None or len(df)<3: return None
    l,p=df.iloc[-1],df.iloc[-2]
    s20,s50=l.SMA20,l.SMA50; s20p,s50p=p.SMA20,p.SMA50
    e8,e21=l.EMA8,l.EMA21; e8p,e21p=p.EMA8,p.EMA21
    r=l.RSI14; c=l.Close
    if s20>s50 and s20p<=s50p: return {"symbol":sym,"action":"BUY","entry":c,"sl":c*0.995,"tgt":c*1.01,"reason":"SMA20>SMA50","conf":0.6}
    if s20<s50 and s20p>=s50p: return {"symbol":sym,"action":"SELL","entry":c,"sl":c*1.005,"tgt":c*0.99,"reason":"SMA20<SMA50","conf":0.6}
    if e8>e21 and e8p<=e21p: return {"symbol":sym,"action":"BUY","entry":c,"sl":c*0.995,"tgt":c*1.015,"reason":"EMA8>EMA21","conf":0.55}
    if e8<e21 and e8p>=e21p: return {"symbol":sym,"action":"SELL","entry":c,"sl":c*1.005,"tgt":c*0.985,"reason":"EMA8<EMA21","conf":0.55}
    return None

# ------------------ Paper Trading ------------------
class Paper:
    def __init__(self,cap): self.cap0=cap; self.cash=cap; self.pos={}; self.hist=[]
    def trade(self,s):
        sym=s["symbol"]
        if sym in self.pos: return False
        q=int((self.cap0*0.01)/abs(s["entry"]-s["sl"]))
        q=max(1,min(q,int(self.cash//s["entry"])))
        if q<=0: return False
        cost=q*s["entry"]; self.cash-=cost
        self.pos[sym]={**s,"qty":q,"open":now_indian()}
        return True
    def close(self,sym,px):
        if sym not in self.pos: return None
        p=self.pos.pop(sym)
        pnl=(px-p["entry"])*p["qty"] if p["action"]=="BUY" else (p["entry"]-px)*p["qty"]
        self.cash+=px*p["qty"]; p["exit"]=px; p["pnl"]=pnl; p["close"]=now_indian(); self.hist.append(p); return p

if "paper" not in st.session_state:
    st.session_state.paper = Paper(CAPITAL)
paper = st.session_state.paper

# ------------------ UI ------------------
def main():
    st.title("📊 Intraday Trading Terminal — Live Mode (Final)")
    tabs = st.tabs(["📈 Dashboard","🎯 Live Signals","📉 Live Charts","💰 Paper Trading","🧮 Backtest"])

    # Dashboard
    with tabs[0]:
        st.metric("Market Status","🟢 OPEN" if market_open() else "🔴 CLOSED")
        st.metric("Server Time (IST)", now_indian().strftime("%H:%M:%S"))
        st.metric("Available Capital", f"₹{paper.cash:,.0f}")
        st.caption("Auto-refresh: Signals 10 s • Charts 5 s • Paper ₹5 L")

    # Live Signals
    with tabs[1]:
        st_autorefresh(interval=SIGNAL_REFRESH_MS,key="sig_refresh")
        idx_choice=st.selectbox("Select Universe",["Nifty 50","Nifty Next 50","Nifty 100","Nifty 500"],index=0)
        if idx_choice=="Nifty 50": syms=NIFTY_50
        elif idx_choice=="Nifty Next 50": syms=NIFTY_NEXT_50
        elif idx_choice=="Nifty 100": syms=NIFTY_100
        else: syms=NIFTY_500

        conf_min=st.slider("Min Confidence",0.3,0.9,0.5,0.05)
        st.info(f"Scanning {len(syms)} stocks… Auto-refresh every 10 s")

        sigs=[]
        executed=[]
        if market_open():
            for s in syms:
                df=fetch_ohlc(s)
                sg=signal(df,s)
                if sg and sg["conf"]>=conf_min:
                    sigs.append(sg)
                    # Auto execute for paper trading
                    if paper.trade(sg): executed.append(sg)
        else:
            st.warning("Market closed — live signals limited.")

        if sigs:
            df=pd.DataFrame(sigs)
            df["entry"]=df["entry"].map(lambda x:f"₹{x:.2f}")
            df["sl"]=df["sl"].map(lambda x:f"₹{x:.2f}")
            df["tgt"]=df["tgt"].map(lambda x:f"₹{x:.2f}")
            st.success(f"{len(sigs)} Signal(s) Found ({len(executed)} Auto-Traded)")
            st.dataframe(df[["symbol","action","entry","sl","tgt","conf","reason"]],use_container_width=True)
        else:
            st.info("No new signals yet.")

    # Charts
    with tabs[2]:
        st_autorefresh(interval=CHART_REFRESH_MS,key="chart_refresh")
        idx_choice_chart=st.selectbox("Chart Universe",["Nifty 50","Nifty Next 50","Nifty 100","Nifty 500"],index=0)
        if idx_choice_chart=="Nifty 50": symbols=NIFTY_50
        elif idx_choice_chart=="Nifty Next 50": symbols=NIFTY_NEXT_50
        elif idx_choice_chart=="Nifty 100": symbols=NIFTY_100
        else: symbols=NIFTY_500

        sym=st.selectbox("Select Symbol for Chart",symbols)
        df=fetch_ohlc(sym)
        if df is None: st.warning("No data (market closed)"); return
        f=go.Figure(data=[go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Price")])
        f.add_trace(go.Scatter(x=df.index,y=df.SMA20,name="SMA20"))
        f.add_trace(go.Scatter(x=df.index,y=df.SMA50,name="SMA50"))
        f.update_layout(height=500,title=f"{sym} — 5m",xaxis_rangeslider_visible=False)
        st.plotly_chart(f,use_container_width=True)
        st.line_chart(df["RSI14"],height=150)

    # Paper Trading
    with tabs[3]:
        st.metric("Cash Balance",f"₹{paper.cash:,.0f}")
        st.metric("Open Positions",len(paper.pos))
        if paper.pos:
            for s,p in paper.pos.items():
                st.write(f"**{s}** {p['action']} {p['qty']} @₹{p['entry']:.2f}")
                if st.button(f"Close {s}",key=s):
                    df=fetch_ohlc(s); px=df["Close"].iloc[-1] if df is not None else p["entry"]
                    r=paper.close(s,px)
                    if r: st.success(f"Closed {s}, PnL ₹{r['pnl']:.2f}")
        if paper.hist:
            st.subheader("Closed Trades")
            st.dataframe(pd.DataFrame(paper.hist)[["symbol","action","qty","entry","exit","pnl","open","close"]],use_container_width=True)

    # Backtest
    with tabs[4]:
        st.info("Backtest feature placeholder. Currently Live Mode with auto paper trading.")

if __name__=="__main__":
    main()
