import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(page_title="My Stock Screener", page_icon="📊", layout="wide")

IST = ZoneInfo("Asia/Kolkata")

# Nifty 100 symbols (Yahoo/NSE symbols, without .NS)
NIFTY_100 = [
    "ABB","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER",
    "AMBUJACEM","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL","BPCL","BRITANNIA","CIPLA",
    "COALINDIA","COLPAL","DABUR","DIVISLAB","DLF","DRREDDY","EICHERMOT",
    "ETERNAL","GAIL","GODREJCP","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","ICICIBANK","ICICIGI",
    "ICICIPRULI","INDHOTEL","INDUSINDBK","INDUSTOWER","INFY","IOC","IRCTC",
    "ITC","JINDALSTEL","JSWENERGY","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARICO","MAXHEALTH","MOTHERSON","MPHASIS","MUTHOOTFIN","NAUKRI",
    "NESTLEIND","NHPC","NTPC","ONGC","PAGEIND","PIDILITIND","PFC","POWERGRID",
    "RECLTD","RELIANCE","SAIL","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN",
    "SIEMENS","SRF","SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TCS",
    "TECHM","TITAN","TORNTPHARM","TRENT","TVSMOTOR","ULTRACEMCO","UPL",
    "VEDL","WIPRO","ZYDUSLIFE","INDIGO","HINDZINC","ADANIENSOL","CANBK"
]

# Broad F&O universe fallback. This is deliberately separate from Nifty 100.
FNO_STOCKS = sorted(set(NIFTY_100 + [
    "AARTIIND","ABCAPITAL","ABFRL","ACC","ADANIPOWER","ALKEM","ANGELONE",
    "APLAPOLLO","ASHOKLEY","ASTRAL","ATGL","AUROPHARMA","BALKRISIND",
    "BANDHANBNK","BANKBARODA","BATAINDIA","BHEL","BIOCON","BOSCHLTD","CANBK",
    "CDSL","CGPOWER","CHAMBLFERT","CHOLAFIN","COFORGE","CONCOR","CROMPTON",
    "CUMMINSIND","DALBHARAT","DEEPAKNTR","DELHIVERY","DIXON","DMART",
    "EXIDEIND","FEDERALBNK","FORTIS","GLENMARK","GODREJPROP","HAL","HAVELLS",
    "HFCL","HINDCOPPER","HUDCO","IDFCFIRSTB","IEX","INDIANB","INDIAMART",
    "IREDA","IRFC","JUBLFOOD","KALYANKJIL","KEI","LAURUSLABS","LICHSGFIN",
    "LICI","LUPIN","MANAPPURAM","MARUTI","MCX","MGL","M&MFIN","MANKIND",
    "METROPOLIS","MOTILALOFS","MRF","NATIONALUM","NBCC","NCC","NMDC",
    "OBEROIRLTY","OFSS","OIL","PAYTM","PERSISTENT","PETRONET","PHOENIXLTD",
    "PIIND","PNB","POLYCAB","POONAWALLA","PRESTIGE","RBLBANK","SBICARD",
    "SOLARINDS","SONACOMS","STARHEALTH","SUPREMEIND","SYNGENE","TATACHEM",
    "TATACOMM","TATAPOWER","TATAELXSI","TATATECH","TIINDIA","UNOMINDA",
    "VOLTAS","YESBANK","ZEEL"
]))

def now_ist():
    return datetime.now(IST)

def market_open(dt):
    return dt.weekday() < 5 and time(9, 15) <= dt.time() <= time(15, 30)

@st.cache_data(ttl=900, show_spinner=False)
def download_daily(symbols):
    return yf.download(
        [s + ".NS" for s in symbols],
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        prepost=False,
    )

@st.cache_data(ttl=30, show_spinner=False)
def download_intraday(symbols):
    return yf.download(
        [s + ".NS" for s in symbols],
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        prepost=False,
    )

def ticker_frame(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            a = data.columns.get_level_values(0)
            b = data.columns.get_level_values(1)
            if ticker in a:
                df = data[ticker].copy()
            elif ticker in b:
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = data.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[-1] for x in df.columns]

        if "Close" not in df.columns:
            return pd.DataFrame()

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def local_dates(index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    return pd.Series(idx.date, index=index)

def calculate(symbol, daily_all, intraday_all, dt):
    daily = ticker_frame(daily_all, symbol + ".NS")
    if daily.empty or "Close" not in daily.columns:
        return None

    daily = daily.dropna(subset=["Close"])
    if len(daily) < 200:
        return None

    today = dt.date()
    dates = local_dates(daily.index)
    hist = daily.loc[dates < today].copy()
    if len(hist) < 200:
        hist = daily.copy()

    hist = hist.dropna(subset=["Close"])
    if len(hist) < 200:
        return None

    previous_close = float(hist["Close"].iloc[-1])

    intraday = ticker_frame(intraday_all, symbol + ".NS")
    today_intraday = pd.DataFrame()

    if not intraday.empty and "Close" in intraday.columns:
        idates = local_dates(intraday.index)
        today_intraday = intraday.loc[idates == today].dropna(subset=["Close"])

    latest_intraday_date = None

    if market_open(dt):
        # During market hours use today's latest available 5-minute price.
        if today_intraday.empty:
            return None
        price = float(today_intraday["Close"].iloc[-1])
    else:
        # After market close the daily feed may still lag one session.
        # Use the newest intraday session close when available.
        latest_intraday = pd.DataFrame()

        if not intraday.empty and "Close" in intraday.columns:
            idates = local_dates(intraday.index)
            valid_idates = idates.dropna()
            if not valid_idates.empty:
                latest_intraday_date = valid_idates.max()
                latest_intraday = intraday.loc[
                    idates == latest_intraday_date
                ].dropna(subset=["Close"])

        hist_date = local_dates(hist.index).iloc[-1]
        if (
            not latest_intraday.empty
            and latest_intraday_date is not None
            and latest_intraday_date >= hist_date
        ):
            price = float(latest_intraday["Close"].iloc[-1])
        else:
            price = previous_close

    if not np.isfinite(price) or price <= 0:
        return None

    # 1D return uses live price vs previous close during market hours.
    # After market, if price is already the latest daily session, compare
    # against the prior completed trading-session close.
    if market_open(dt):
        day_base = previous_close
    else:
        hist_last_date = local_dates(hist.index).iloc[-1]
        selected_date = latest_intraday_date or hist_last_date
        if selected_date == hist_last_date and len(hist) >= 2:
            day_base = float(hist["Close"].iloc[-2])
        else:
            day_base = previous_close

    one_day = (
        (price / day_base - 1) * 100
        if np.isfinite(day_base) and day_base > 0 else np.nan
    )

    week_base = float(hist["Close"].iloc[-5])
    one_week = (price / week_base - 1) * 100 if week_base > 0 else np.nan

    month_base = float(hist["Close"].iloc[-21])
    one_month = (price / month_base - 1) * 100 if month_base > 0 else np.nan

    # Add current price as the latest observation so EMAs move with price.
    ema_data = hist["Close"].copy()
    ema_data.loc[pd.Timestamp(dt)] = price

    ema21 = float(ema_data.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(ema_data.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(ema_data.ewm(span=200, adjust=False).mean().iloc[-1])

    ema_diff = (price / ema21 - 1) * 100 if ema21 > 0 else np.nan

    if price > ema21 and ema21 > ema50 and ema50 > ema200:
        trend = "Bullish"
    elif price < ema21 and ema21 < ema50 and ema50 < ema200:
        trend = "Bearish"
    else:
        trend = "Neutral"

    return {
        "Stock": symbol,
        "Live Price": price,
        "1D Return %": one_day,
        "1W Return %": one_week,
        "1M Return %": one_month,
        "21 EMA": ema21,
        "50 EMA": ema50,
        "200 EMA": ema200,
        "21 EMA vs Price %": ema_diff,
        "Trend": trend,
    }

def scan_universe(symbols):
    dt = now_ist()
    daily = download_daily(symbols)
    intraday = download_intraday(symbols)

    rows = []
    for symbol in symbols:
        row = calculate(symbol, daily, intraday, dt)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows), dt

def show_table(df):
    if df.empty:
        return

    display = df.copy()

    for c in ["Live Price", "21 EMA", "50 EMA", "200 EMA"]:
        display[c] = display[c].map(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
        )

    for c in ["1D Return %", "1W Return %", "1M Return %", "21 EMA vs Price %"]:
        display[c] = display[c].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )

    def colour_trend(value):
        if value == "Bullish":
            return "background-color: #198754; color: white; font-weight: bold;"
        if value == "Neutral":
            return "background-color: #F5B642; color: black; font-weight: bold;"
        if value == "Bearish":
            return "background-color: #DC3545; color: white; font-weight: bold;"
        return ""

    styled = display.style.map(colour_trend, subset=["Trend"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=650,
    )

st.title("📊 My Stock Screener")
st.subheader("Indian Stock Market Technical Screener")

st.caption(
    "Live Price, 1D/1W/1M returns, 21/50/200 EMA, "
    "21 EMA vs Price and Trend. Auto-refreshes every 5 minutes."
)

universe = st.selectbox(
    "Universe",
    ["Nifty 100", "F&O Stocks"],
)

symbols = NIFTY_100 if universe == "Nifty 100" else FNO_STOCKS

if "selected_universe" not in st.session_state:
    st.session_state.selected_universe = universe

if st.session_state.selected_universe != universe:
    st.session_state.selected_universe = universe
    st.session_state.results = pd.DataFrame()

c1, c2 = st.columns([1, 1])

with c1:
    scan = st.button("🔍 Scan", type="primary", use_container_width=True)

with c2:
    refresh = st.button("🔄 Refresh Now", use_container_width=True)

if scan or refresh or "results" not in st.session_state:
    if refresh:
        download_intraday.clear()
        download_daily.clear()

    with st.spinner("Loading market data..."):
        result, updated = scan_universe(symbols)

    st.session_state.results = result
    st.session_state.updated = updated

result = st.session_state.get("results", pd.DataFrame())
updated = st.session_state.get("updated")

if not result.empty:
    st.caption(
        f"{universe} • {len(result)} stocks • "
        f"Updated {updated.strftime('%d-%b-%Y %I:%M:%S %p')} IST"
    )
    show_table(result)
else:
    if market_open(now_ist()):
        st.warning(
            "Current-day intraday data is not available right now. "
            "The app will not display a previous-session close as a live price."
        )
    else:
        st.info("Click Scan or Refresh Now to load the latest completed data.")

# Automatic five-minute refresh.
@st.fragment(run_every="5m")
def auto_refresh():
    with st.spinner("Updating prices..."):
        result, updated = scan_universe(symbols)

    st.session_state.results = result
    st.session_state.updated = updated

    if not result.empty:
        st.caption(
            f"Auto-updated {updated.strftime('%d-%b-%Y %I:%M:%S %p')} IST"
        )
        show_table(result)

auto_refresh()
