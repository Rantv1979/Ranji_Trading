"""
Intraday Live Trading Terminal — Pro Edition
---------------------------------------------
Features:
- ₹10 L capital paper trading
- 10 % capital allocation per trade
- Auto exit on target / stop-loss
- Nifty 50 / 100 / 500 universes
- Professional UI + Trading Log tab
- Auto refresh: Signals 10 s / Chart 5 s
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
st.set_page_config(page_title="Intraday Live Terminal Pro", layout="wide", page_icon="📈")
IND_TZ = pytz.timezone("Asia/Kolkata")
CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.10      # 10 % per trade
SIGNAL_REFRESH_MS = 10_000
CHART_REFRESH_MS = 5_000
AUTO_EXEC_CONF = 0.60

# ---------------- Nifty universes ----------------
NIFTY_50 = [s+".NS" for s in [
    "RELIANCE","TCS","HDFCBANK","INFY","HINDUNILVR","ICICIBANK","KOTAKBANK","BHARTIARTL","ITC","LT",
    "SBIN","ASIANPAINT","HCLTECH","AXISBANK","MARUTI","SUNPHARMA","TITAN","ULTRACEMCO","WIPRO",
    "NTPC","NESTLEIND","POWERGRID","M&M","BAJFINANCE","ONGC","TATAMOTORS","TATASTEEL","JSWSTEEL",
    "ADANIPORTS","COALINDIA","HDFCLIFE","DRREDDY","HINDALCO","CIPLA","SBILIFE","GRASIM","TECHM",
    "BAJAJFINSV","BRITANNIA","EICHERMOT","DIVISLAB","SHREECEM","APOLLOHOSP","UPL","BAJAJ-AUTO",
    "HEROMOTOCO","INDUSINDBK","ADANIENT","HDFC"
]]
NIFTY_NEXT_50 = [s+".NS" for s in [
    "ABB","ADANIGREEN","BANKBARODA","BEL","CANBK","CHOLAFIN","DABUR","GAIL","HAL","IOC",
    "JINDALSTEL","PIDILITIND","PNB","TORNTPOWER","VOLTAS","ICICIPRULI","MUTHOOTFIN","COLPAL","DMART"
]]
NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))
NIFTY_500 = NIFTY_100  # extend if needed

# ---------------- Helpers ----------------
def now_indian(): return datetime.now(IND_TZ)
def market_open():
    n = now_indian(); o = IND_TZ.localize(datetime.combine(n.date(), dt_time(9,15)))
    c = IND_TZ.localize(datetime.combine(n.date(), dt_time(15,30)))
    return o <= n <= c

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(close, n=14):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n).mean()
    loss = -diff.clip(upper=0).rolling(n).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

@st.cache_data(ttl=15)
def fetch_ohlc(sym, period="1d", interval="5m"):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna(subset=["Close"])
        df["EMA8"], df["EMA21"] = ema(df["Close"],8), ema(df["Close"],21)
        df["SMA20"], df["SMA50"] = df["Close"].rolling(20).mean(), df["Close"].rolling(50).mean()
        df["RSI14"] = rsi(df["Close"]).fillna(50)
        return df
    except: return None

def signal(df,sym):
    if df is None or len(df)<3: return None
    l,p=df.iloc[-1],df.iloc[-2]
    s20,s50=l.SMA20,l.SMA50; s20p,s50p=p.SMA20,p.SMA50
    e8,e21=l.EMA8,l.EMA21; e8p,e21p=p.EMA8,p.EMA21
    close=float(l.Close)
    if s20>s50 and s20p<=s50p: return {"symbol":sym,"action":"BUY","entry":close,"stop":close*0.995,"target":close*1.01,"conf":0.65}
    if s20<s50 and s20p>=s50p: return {"symbol":sym,"action":"SELL","entry":close,"stop":close*1.005,"target":close*0.99,"conf":0.65}
    if e8>e21 and e8p<=e21p: return {"symbol":sym,"action":"BUY","entry":close,"stop":close*0.996,"target":close*1.012,"conf":0.6}
    if e8<e21 and e8p>=e21p: return {"symbol":sym,"action":"SELL","entry":close,"stop":close*1.004,"target":close*0.988,"conf":0.6}
    return None

# ---------------- Paper Trader ----------------
class PaperTrader:
    def __init__(self,capital=CAPITAL,alloc=TRADE_ALLOC):
        self.init=capital; self.cash=capital; self.alloc=alloc
        self.pos={}; self.log=[]
    def trade_size(self,entry): return max(1,int((self.alloc*self.init)//entry))
    def open(self,s):
        if s["symbol"] in self.pos: return
        if s["conf"]<AUTO_EXEC_CONF: return
        q=self.trade_size(s["entry"]); cost=q*s["entry"]
        if cost>self.cash: return
        self.cash-=cost
        self.pos[s["symbol"]]={**s,"qty":q,"open":now_indian(),"status":"OPEN"}
        self.log.append({"time":now_indian(),"event":"OPEN","symbol":s["symbol"],"action":s["action"],"qty":q,"price":s["entry"]})
    def update(self):
        """auto exit on SL/Target"""
        to_close=[]
        for sym,p in list(self.pos.items()):
            df=fetch_ohlc(sym)
            if df is None: continue
            cur=float(df.Close.iloc[-1])
            if p["action"]=="BUY":
                if cur<=p["stop"] or cur>=p["target"]:
                    self.close(sym,cur)
            else:
                if cur>=p["stop"] or cur<=p["target"]:
                    self.close(sym,cur)
        return to_close
    def close(self,sym,price):
        if sym not in self.pos: return
        p=self.pos.pop(sym)
        pnl=((price-p["entry"]) if p["action"]=="BUY" else (p["entry"]-price))*p["qty"]
        self.cash+=price*p["qty"]
        self.log.append({"time":now_indian(),"event":"CLOSE","symbol":sym,"action":p["action"],
                         "qty":p["qty"],"price":price,"pnl":pnl})
    def positions_df(self):
        rows=[]
        for sym,p in self.pos.items():
            df=fetch_ohlc(sym)
            cur=float(df.Close.iloc[-1]) if df is not None else p["entry"]
            pnl=((cur-p["entry"]) if p["action"]=="BUY" else (p["entry"]-cur))*p["qty"]
            tgt_hit = (cur>=p["target"] if p["action"]=="BUY" else cur<=p["target"])
            rows.append({
                "Symbol":sym,"Action":p["action"],"Qty":p["qty"],
                "Entry":p["entry"],"Current":cur,"Stop":p["stop"],"Target":p["target"],
                "Unreal P/L":pnl,"Target Hit":"YES" if tgt_hit else "NO"
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Symbol","Action"])
    def equity(self):
        eq=self.cash
        for sym,p in self.pos.items():
            df=fetch_ohlc(sym)
            if df is None: continue
            cur=float(df.Close.iloc[-1])
            pnl=((cur-p["entry"]) if p["action"]=="BUY" else (p["entry"]-cur))*p["qty"]
            eq+=p["qty"]*p["entry"]+pnl
        return eq

if "trader" not in st.session_state:
    st.session_state.trader=PaperTrader()
trader=st.session_state.trader

# ---------------- UI ----------------
st.markdown("<h2 style='color:#0077cc;'>📊 Intraday Live Trading Dashboard — Pro Edition</h2>",unsafe_allow_html=True)
tabs=st.tabs(["Dashboard","Signals","Charts","Paper Trading","Trading Log"])

# --- Dashboard ---
with tabs[0]:
    col1,col2,col3=st.columns(3)
    nifty=yf.download("^NSEI",period="1d",interval="1d")
    bn=yf.download("^NSEBANK",period="1d",interval="1d")
    n50=nifty["Close"].iloc[-1] if not nifty.empty else None
    bnk=bn["Close"].iloc[-1] if not bn.empty else None
    col1.metric("NIFTY 50",f"{n50:.2f}" if n50 else "n/a")
    col2.metric("BANK NIFTY",f"{bnk:.2f}" if bnk else "n/a")
    col3.metric("Market", "🟢 OPEN" if market_open() else "🔴 CLOSED")
    st.write("")
    st.metric("Cash Balance",f"₹{trader.cash:,.0f}")
    st.metric("Total Equity",f"₹{trader.equity():,.0f}")
    st.caption(f"Capital ₹{CAPITAL:,.0f} | 10 % allocation per trade | Auto exit active")

# --- Signals ---
with tabs[1]:
    st_autorefresh(interval=SIGNAL_REFRESH_MS,key="sigref")
    uni=st.selectbox("Select Universe",["Nifty 50","Nifty 100","Nifty 500"],index=0)
    syms=NIFTY_50 if "50" in uni else NIFTY_100 if "100" in uni else NIFTY_500
    st.write(f"Scanning {len(syms)} stocks ...")
    sigs=[]
    if market_open():
        prog=st.progress(0)
        for i,s in enumerate(syms):
            df=fetch_ohlc(s)
            sg=signal(df,s)
            if sg and sg["conf"]>=AUTO_EXEC_CONF:
                sigs.append(sg)
                trader.open(sg)
            prog.progress((i+1)/len(syms))
        trader.update()  # auto-exit check
        prog.empty()
    else:
        st.info("Market closed. Signals inactive.")
    if sigs:
        df=pd.DataFrame(sigs)
        st.dataframe(df[["symbol","action","entry","stop","target","conf"]],use_container_width=True)
    else:
        st.info("No new signals in this cycle.")

# --- Charts ---
with tabs[2]:
    st_autorefresh(interval=CHART_REFRESH_MS,key="chartref")
    sym=st.selectbox("Symbol",NIFTY_50)
    df=fetch_ohlc(sym)
    if df is not None:
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Price"))
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA20,name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index,y=df.SMA50,name="SMA50"))
        fig.update_layout(title=f"{sym} – 5 min",xaxis_rangeslider_visible=False,height=480)
        st.plotly_chart(fig,use_container_width=True)
        st.line_chart(df["RSI14"],height=120)
    else:
        st.warning("Data unavailable.")

# --- Paper Trading ---
with tabs[3]:
    st.write("### Active Positions")
    dfp=trader.positions_df()
    if not dfp.empty:
        st.dataframe(dfp,use_container_width=True)
    else:
        st.info("No open positions.")
    st.write("")
    st.metric("Cash Balance",f"₹{trader.cash:,.0f}")
    st.metric("Equity",f"₹{trader.equity():,.0f}")

# --- Trading Log ---
with tabs[4]:
    st.write("### Trade Log (Chronological)")
    if trader.log:
        df=pd.DataFrame(trader.log)
        df["time"]=df["time"].dt.strftime("%H:%M:%S")
        st.dataframe(df,use_container_width=True)
    else:
        st.info("No trades logged yet.")
