import io
import gzip
import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Nifty 100 & F&O Stock Screener",
    page_icon="📊",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-IN,en;q=0.9,en-US;q=0.8",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

NIFTY100_CSV = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"

# NSE publishes the F&O contract master every trading day. The app tries
# today's file first and then recent dates so a holiday/weekend still works.
FNO_CONTRACT_URLS = [
    "https://nsearchives.nseindia.com/content/fo/NSE_FO_contract_{date}.csv.gz",
    "https://nsearchives.nseindia.com/content/fo/NSE_FO_contract_{date}.CSV.GZ",
    "https://nsearchives.nseindia.com/content/fo/NSE_FO_contract_{date}.csv",
]

def ist_now():
    return datetime.now(IST)

def nse_session_open(dt):
    return dt.weekday() < 5 and time(9, 15) <= dt.time() <= time(15, 30)

def nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    # Warm the NSE session first; this materially improves the chance that
    # nsearchives/NSE endpoints accept the following request.
    try:
        s.get("https://www.nseindia.com/", timeout=12)
    except Exception:
        pass
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty100_universe():
    s = nse_session()
    r = s.get(NIFTY100_CSV, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = next(
        (c for c in df.columns if c.upper() in {"SYMBOL", "SYMBOLS"}),
        None,
    )
    if symbol_col is None:
        raise ValueError(f"NSE Nifty 100 CSV has no SYMBOL column: {list(df.columns)}")

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan})
        .dropna()
        .tolist()
    )
    symbols = list(dict.fromkeys(symbols))

    if len(symbols) != 100:
        raise ValueError(
            f"NSE Nifty 100 source returned {len(symbols)} symbols, not 100. "
            "The app will not substitute a hard-coded list."
        )

    return symbols

def recent_trading_dates(days=10):
    d = ist_now().date()
    dates = []
    for i in range(days):
        dates.append((d - pd.Timedelta(days=i)).strftime("%d%m%Y"))
    return dates

def parse_contract_master(raw):
    # NSE contract master is normally gzip-compressed CSV.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize column lookup.
    norm = {
        re.sub(r"[^A-Z0-9]", "", c.upper()): c
        for c in df.columns
    }

    instrument_col = next(
        (norm[k] for k in ["INSTRUMENTTYPE", "INSTRUMENT"] if k in norm),
        None,
    )
    symbol_col = next(
        (
            norm[k]
            for k in [
                "SYMBOL",
                "UNDERLYINGSYMBOL",
                "UNDERLYING",
                "SYMBOLNAME",
            ]
            if k in norm
        ),
        None,
    )

    if symbol_col is None:
        # Some contract masters identify the underlying inside IDENTIFIER.
        ident_col = next(
            (norm[k] for k in ["IDENTIFIER"] if k in norm),
            None,
        )
        if ident_col is None:
            raise ValueError(
                f"Could not find an underlying symbol column. Columns: {list(df.columns)}"
            )
        # This is a last-resort parser for identifiers such as FUTSTKABC...
        text = df[ident_col].astype(str)
        stocks = []
        for value in text:
            m = re.search(r"(?:FUTSTK|OPTSTK)([A-Z0-9&.-]+)", value.upper())
            if m:
                stocks.append(m.group(1))
        return sorted(set(stocks))

    if instrument_col is not None:
        inst = df[instrument_col].astype(str).str.upper().str.strip()
        df = df[inst.isin(["FUTSTK", "OPTSTK"])].copy()

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan})
        .dropna()
    )

    symbols = [
        s for s in dict.fromkeys(symbols.tolist())
        if s and s not in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
    ]

    if not symbols:
        raise ValueError("NSE F&O contract master contained no stock underlyings.")

    return sorted(symbols)

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_universe():
    s = nse_session()
    errors = []

    for date_text in recent_trading_dates(10):
        for template in FNO_CONTRACT_URLS:
            url = template.format(date=date_text)
            try:
                r = s.get(url, timeout=20)
                if r.status_code != 200 or not r.content:
                    continue

                symbols = parse_contract_master(r.content)

                if len(symbols) < 100:
                    continue

                return symbols
            except Exception as exc:
                errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download the current NSE F&O contract master. "
        "The app deliberately does NOT use a hard-coded F&O list, because "
        "that could miss stocks after NSE changes the universe."
    )

@st.cache_data(ttl=45, show_spinner=False)
def download_daily(symbols):
    return yf.download(
        [s + ".NS" for s in symbols],
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
        prepost=False,
    )

@st.cache_data(ttl=20, show_spinner=False)
def download_intraday(symbols):
    return yf.download(
        [s + ".NS" for s in symbols],
        period="1d",
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
            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)

            if ticker in l0:
                df = data[ticker].copy()
            elif ticker in l1:
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = data.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                x[-1] if isinstance(x, tuple) else x
                for x in df.columns
            ]

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

def latest_today_price(intraday, today):
    if intraday.empty or "Close" not in intraday.columns:
        return np.nan

    dates = local_dates(intraday.index)
    today_df = intraday.loc[dates == today].dropna(subset=["Close"])

    if today_df.empty:
        return np.nan

    return float(today_df["Close"].iloc[-1])

def calculate(symbol, daily_all, intraday_all, dt):
    ticker = symbol + ".NS"

    daily = ticker_frame(daily_all, ticker)

    # A stock remains in the universe even when its price data is unavailable.
    if daily.empty or "Close" not in daily.columns:
        return {
            "Stock": symbol,
            "Live Price": np.nan,
            "1D Return %": np.nan,
            "1W Return %": np.nan,
            "1M Return %": np.nan,
            "21 EMA": np.nan,
            "50 EMA": np.nan,
            "200 EMA": np.nan,
            "21 EMA vs Price %": np.nan,
            "Trend": "Data unavailable",
        }

    daily = daily.dropna(subset=["Close"])
    if len(daily) < 2:
        return {
            "Stock": symbol,
            "Live Price": np.nan,
            "1D Return %": np.nan,
            "1W Return %": np.nan,
            "1M Return %": np.nan,
            "21 EMA": np.nan,
            "50 EMA": np.nan,
            "200 EMA": np.nan,
            "21 EMA vs Price %": np.nan,
            "Trend": "Data unavailable",
        }

    today = dt.date()
    dates = local_dates(daily.index)
    hist = daily.loc[dates < today].copy().dropna(subset=["Close"])

    if hist.empty:
        hist = daily.copy()

    previous_close = float(hist["Close"].iloc[-1])

    intraday = ticker_frame(intraday_all, ticker)
    today_price = latest_today_price(intraday, today)

    if nse_session_open(dt):
        price = today_price
        selected_date = today
        day_base = previous_close
    else:
        # After the market, use the last intraday bar of today's session when
        # available; this prevents Yahoo's daily candle from lagging by one day.
        if np.isfinite(today_price):
            price = today_price
            selected_date = today
            day_base = previous_close
        else:
            price = previous_close
            selected_date = local_dates(hist.index).iloc[-1]
            day_base = (
                float(hist["Close"].iloc[-2])
                if len(hist) >= 2
                else np.nan
            )

    if not np.isfinite(price) or price <= 0:
        return {
            "Stock": symbol,
            "Live Price": np.nan,
            "1D Return %": np.nan,
            "1W Return %": np.nan,
            "1M Return %": np.nan,
            "21 EMA": np.nan,
            "50 EMA": np.nan,
            "200 EMA": np.nan,
            "21 EMA vs Price %": np.nan,
            "Trend": "Data unavailable",
        }

    one_day = (
        (price / day_base - 1) * 100
        if np.isfinite(day_base) and day_base > 0
        else np.nan
    )

    week_base = float(hist["Close"].iloc[-5]) if len(hist) >= 5 else np.nan
    one_week = (
        (price / week_base - 1) * 100
        if np.isfinite(week_base) and week_base > 0
        else np.nan
    )

    month_base = float(hist["Close"].iloc[-21]) if len(hist) >= 21 else np.nan
    one_month = (
        (price / month_base - 1) * 100
        if np.isfinite(month_base) and month_base > 0
        else np.nan
    )

    # Current price is added as the latest observation. This makes all three
    # EMAs and the EMA-distance respond to the live/current session price.
    ema_series = hist["Close"].copy()
    ema_series.loc[pd.Timestamp(dt)] = price

    ema21 = float(ema_series.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(ema_series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(ema_series.ewm(span=200, adjust=False).mean().iloc[-1])

    ema_diff = (
        (price / ema21 - 1) * 100
        if ema21 > 0
        else np.nan
    )

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

def scan(symbols):
    dt = ist_now()
    daily = download_daily(symbols)
    intraday = download_intraday(symbols)

    rows = [calculate(s, daily, intraday, dt) for s in symbols]
    result = pd.DataFrame(rows)

    # Never lose a universe member because data for that stock failed.
    result = result.set_index("Stock").reindex(symbols).reset_index()

    return result, dt

def style_table(df):
    display = df.copy()

    for c in ["Live Price", "21 EMA", "50 EMA", "200 EMA"]:
        display[c] = display[c].map(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
        )

    for c in ["1D Return %", "1W Return %", "1M Return %", "21 EMA vs Price %"]:
        display[c] = display[c].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )

    def trend_text(x):
        if x == "Bullish":
            return "🟢 Bullish"
        if x == "Bearish":
            return "🔴 Bearish"
        if x == "Neutral":
            return "🟡 Neutral"
        return "⚪ Data unavailable"

    display["Trend"] = display["Trend"].map(trend_text)

    def trend_style(value):
        value = str(value)
        if "Bullish" in value:
            return "background-color:#198754;color:white;font-weight:700"
        if "Bearish" in value:
            return "background-color:#dc3545;color:white;font-weight:700"
        if "Neutral" in value:
            return "background-color:#f5b642;color:black;font-weight:700"
        return "background-color:#555;color:white"

    return display.style.map(trend_style, subset=["Trend"])

# -------------------- UI --------------------

st.title("📊 My Stock Screener")
st.subheader("NSE Universe Technical Screener")

st.caption(
    "Universe lists are obtained from NSE. No hard-coded Nifty 100 or F&O "
    "stock list is used. Price data is from Yahoo Finance."
)

universe = st.selectbox(
    "Universe",
    ["Nifty 100", "F&O Stocks"],
)

try:
    if universe == "Nifty 100":
        symbols = get_nifty100_universe()
        source_text = "NSE Nifty 100"
    else:
        symbols = get_fno_universe()
        source_text = "NSE current stock F&O contract master"
except Exception as exc:
    st.error(f"Could not load the current NSE universe: {exc}")
    st.stop()

st.caption(f"{source_text} • {len(symbols)} stocks")

c1, c2 = st.columns(2)
with c1:
    scan_button = st.button("🔍 Scan", type="primary", use_container_width=True)
with c2:
    refresh_button = st.button("🔄 Refresh Now", use_container_width=True)

if scan_button or refresh_button or "result" not in st.session_state:
    if refresh_button:
        download_intraday.clear()
        download_daily.clear()

    with st.spinner(f"Loading {len(symbols)} stocks..."):
        result, updated = scan(symbols)

    st.session_state.result = result
    st.session_state.updated = updated
    st.session_state.universe = universe

result = st.session_state.get("result", pd.DataFrame())
updated = st.session_state.get("updated")

if not result.empty:
    unavailable = int(result["Live Price"].isna().sum())

    if updated:
        st.caption(
            f"Updated {updated.strftime('%d-%b-%Y %I:%M:%S %p')} IST"
        )

    if unavailable:
        st.info(
            f"{len(result)} stocks in the NSE universe are displayed. "
            f"{unavailable} currently have unavailable price data; "
            "they have NOT been removed from the universe."
        )

    st.dataframe(
        style_table(result),
        use_container_width=True,
        hide_index=True,
        height=680,
    )

@st.fragment(run_every="5m")
def automatic_refresh():
    # Refresh only price data every 5 minutes. Universe membership is refreshed
    # hourly, which is enough to catch an NSE constituent change without
    # repeatedly hammering NSE.
    download_intraday.clear()

    try:
        if universe == "Nifty 100":
            current_symbols = get_nifty100_universe()
        else:
            current_symbols = get_fno_universe()

        current_result, current_updated = scan(current_symbols)

        st.session_state.result = current_result
        st.session_state.updated = current_updated

        st.dataframe(
            style_table(current_result),
            use_container_width=True,
            hide_index=True,
            height=680,
        )
    except Exception as exc:
        st.warning(f"Automatic refresh could not complete: {exc}")

automatic_refresh()
