"""
Intraday Live Trading Terminal — Pro Edition v3.2 (Final)
Enhancements in this version:
- Includes **all Nifty 50 & Nifty Next 50** stocks in the signal scan.
- Live auto-refresh every 15 seconds.
- Signals table with Entry/Target/Stop for all auto-executed trades.
- Trending Stocks table on Dashboard.
- Stable data fetching for all 100 symbols.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz, warnings, time
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Intraday Terminal Pro v3.2", layout="wide", page_icon="📊")
IND_TZ = pytz.timezone("Asia/Kolkata")

CAPITAL = 1_000_000
TRADE_ALLOC = 0.10
MAX_CONCURRENT_TRADES = 10
AUTO_EXEC_CONF = 0.70
SIGNAL_REFRESH_MS = 20_000
CHART_REFRESH_MS = 5_000

def now_indian():
    return datetime.now(IND_TZ)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def safe_yf(symbol,period="5d",interval="5m"):
    try:
        df=yf.download(symbol,period=period,interval=interval,progress=False,threads=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(0)
        df.columns=[str(c) for c in df.columns]
        if 'Adj Close' in df.columns and 'Close' not in df.columns: df=df.rename(columns={'Adj Close':'Close'})
        if all(col in df.columns for col in ['Open','High','Low','Close']): return df.dropna()
    except: return None
    return None

def generate_signal(df,sym):
    if df is None or len(df)<30: return None
    df['EMA8']=ema(df['Close'],8); df['EMA21']=ema(df['Close'],21); df['EMA50']=ema(df['Close'],50)
    cur=df.iloc[-1]; prev=df.iloc[-2]
    if cur['EMA8']>cur['EMA21']>cur['EMA50'] and cur['Close']>cur['EMA8']:
        entry=cur['Close']; stop=entry*0.99; target=entry*1.02
        return {"symbol":sym,"action":"BUY","entry":entry,"stop":stop,"target":target,"conf":0.75}
    if cur['EMA8']<cur['EMA21']<cur['EMA50'] and cur['Close']<cur['EMA8']:
        entry=cur['Close']; stop=entry*1.01; target=entry*0.98
        return {"symbol":sym,"action":"SELL","entry":entry,"stop":stop,"target":target,"conf":0.75}
    return None

class Trader:
    def __init__(self):
        self.cash=CAPITAL; self.pos={}; self.log=[]
    def open(self,sig):
        if sig['symbol'] in self.pos: return False
        if sig['conf']<AUTO_EXEC_CONF: return False
        qty=int((TRADE_ALLOC*CAPITAL)//sig['entry']); cost=qty*sig['entry']
        if cost>self.cash: return False
        self.cash-=cost
        self.pos[sig['symbol']]={**sig,'qty':qty,'open_price':sig['entry'],'status':'OPEN','open_time':now_indian()}
        self.log.append({"time":now_indian(),"symbol":sig['symbol'],"action":sig['action'],"entry":sig['entry'],"stop":sig['stop'],"target":sig['target'],"qty":qty,"status":"OPEN"})
        return True
    def update(self):
        for sym,p in list(self.pos.items()):
            df=safe_yf(sym)
            if df is None: continue
            cur=float(df['Close'].iloc[-1])
            if p['action']=='BUY' and (cur<=p['stop'] or cur>=p['target']): p['status']='CLOSED'
            if p['action']=='SELL' and (cur>=p['stop'] or cur<=p['target']): p['status']='CLOSED'
            if p['status']=='CLOSED': self.log.append({"time":now_indian(),"symbol":sym,"exit":cur,"status":"CLOSED"}); self.cash+=cur*p['qty']; del self.pos[sym]
    def df_positions(self):
        rows=[]
        for s,p in self.pos.items():
            df=safe_yf(s); cur=float(df['Close'].iloc[-1]) if df is not None else p['entry']
            rows.append({"Symbol":s,"Action":p['action'],"Entry":p['entry'],"Current":cur,"Target":p['target'],"Stop":p['stop'],"Status":p['status']})
        return pd.DataFrame(rows)

if 'trader' not in st.session_state: st.session_state.trader=Trader()
trader=st.session_state.trader

# ---------------- NIFTY 50 & NEXT 50 (Full) ----------------
NIFTY_50 = ["ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHREECEM.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS"]

NIFTY_NEXT_50 = ["ABB.NS","ACC.NS","ADANIGREEN.NS","ADANIPOWER.NS","AUBANK.NS","BANKBARODA.NS","BERGEPAINT.NS","BIOCON.NS","BOSCHLTD.NS","CANBK.NS","CHOLAFIN.NS","COLPAL.NS","DALBHARAT.NS","DABUR.NS","DLF.NS","GAIL.NS","GLAND.NS","GODREJCP.NS","HAVELLS.NS","HDFCAMC.NS","HINDPETRO.NS","ICICIGI.NS","ICICIPRULI.NS","IGL.NS","INDHOTEL.NS","INDIGO.NS","IRCTC.NS","JINDALSTEL.NS","LUPIN.NS","MARICO.NS","MCDOWELL-N.NS","MOTHERSON.NS","MPHASIS.NS","MUTHOOTFIN.NS","NMDC.NS","PEL.NS","PIIND.NS","PNB.NS","POLYCAB.NS","RECLTD.NS","SBICARD.NS","SRF.NS","TATAPOWER.NS","TATACHEM.NS","TORNTPHARM.NS","TRENT.NS","TVSMOTOR.NS","VEDL.NS","VOLTAS.NS","ZOMATO.NS"]

NIFTY_100 = sorted(list(set(NIFTY_50 + NIFTY_NEXT_50)))

st.markdown("<h1 style='text-align:center;color:#0b486b;'>📊 Intraday Terminal — Pro v3.2</h1>",unsafe_allow_html=True)

t1,t2=st.tabs(["Dashboard","Signals & Trades"])

# Dashboard
with t1:
    st.subheader('Market Overview')
    nifty_df=safe_yf('^NSEI','1d','5m')
    if nifty_df is not None and not nifty_df.empty:
        cur=float(nifty_df['Close'].iloc[-1]); prev=float(nifty_df['Close'].iloc[0]); chg=(cur-prev)/prev
        st.metric('NIFTY 50',f"₹{cur:,.2f}",f"{chg:.2%}")
    st.divider()
    st.subheader('Trending Stocks (Top % Gainers)')
    movers=[]
    for s in NIFTY_50:
        df=safe_yf(s,'1d','5m')
        if df is not None and len(df)>2:
            cur=df['Close'].iloc[-1]; prev=df['Close'].iloc[0]; chg=(cur-prev)/prev
            movers.append((s,chg))
    if movers:
        top=pd.DataFrame(sorted(movers,key=lambda x:x[1],reverse=True)[:10],columns=['Symbol','%Change'])
        top['%Change']=top['%Change'].apply(lambda x:f"{x:.2%}")
        st.table(top)

# Signals & Trades
with t2:
    st_autorefresh(interval=SIGNAL_REFRESH_MS,key='sig_ref')
    st.subheader('Signal Scanner & Auto Execution')
    signals=[]
    all_syms=NIFTY_50+NIFTY_NEXT_50
    for s in all_syms:
        df=safe_yf(s)
        sig=generate_signal(df,s)
        if sig: signals.append(sig); trader.open(sig)
    trader.update()
    if signals:
        df_sig=pd.DataFrame(signals)
        st.dataframe(df_sig[['symbol','action','entry','stop','target','conf']])
    else:
        st.info('No fresh signals yet.')

    st.subheader('Open / Executed Trades')
    posdf=trader.df_positions()
    if not posdf.empty: st.dataframe(posdf)
    else: st.info('No open trades.')

st.divider()
st.markdown("<div style='text-align:center;color:#777;'>v3.2 — Intraday Paper Trading Dashboard scanning full Nifty 50 & Next 50 every 15 seconds.</div>",unsafe_allow_html=True)
