import io
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

NSE_URL = "https://www.nseindia.com"
NIFTY100_CSV = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
)
FNO_UNDERLYINGS_PAGE = (
    "https://www.nseindia.com/products-services/"
    "equity-derivatives-list-underlyings-information"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": NSE_URL + "/",
}


def ist_now():
    return datetime.now(IST)


def market_open(dt):
    return dt.weekday() < 5 and time(9, 15) <= dt.time() <= time(15, 30)


def nse_session():
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    session.get(NSE_URL + "/", timeout=15)
    return session


def clean_symbol(value):
    value = str(value).strip().upper()
    value = re.sub(r"\s+", " ", value)
    return value


# ------------------------------------------------------------
# DYNAMIC NSE UNIVERSES
# ------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty100():
    """Read the current Nifty 100 constituent file published by NSE."""
    session = nse_session()
    response = session.get(NIFTY100_CSV, timeout=20)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = next(
        (c for c in df.columns if c.upper() == "SYMBOL"),
        None,
    )
    if symbol_col is None:
        raise RuntimeError(
            f"NSE Nifty 100 CSV changed format. Columns: {list(df.columns)}"
        )

    symbols = [
        clean_symbol(x)
        for x in df[symbol_col].dropna().tolist()
        if clean_symbol(x)
    ]
    symbols = list(dict.fromkeys(symbols))

    # This prevents a broken/partial NSE response from silently becoming
    # a smaller universe.
    if len(symbols) != 100:
        raise RuntimeError(
            f"NSE returned {len(symbols)} Nifty 100 constituents. "
            "The app stopped rather than using an incomplete list."
        )

    return symbols


@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stocks():
    """
    Get the current individual-stock F&O universe from NSE's JSON API.

    NSE's current website itself uses /api/underlying-information. The API
    response contains:
        data -> UnderlyingList
        data -> IndexList

    We use UnderlyingList only, so indices are never mixed into the F&O
    stock universe.
    """
    api_url = "https://www.nseindia.com/api/underlying-information"

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    })

    # NSE expects a browser-like session before accepting the JSON API.
    bootstrap_urls = [
        "https://www.nseindia.com/option-chain",
        "https://www.nseindia.com/",
    ]

    last_error = None

    for bootstrap in bootstrap_urls:
        try:
            session.get(bootstrap, timeout=15)
            response = session.get(api_url, timeout=20)

            if response.status_code == 200:
                payload = response.json()
                data = payload.get("data", {})

                underlying_list = data.get("UnderlyingList", [])

                symbols = []
                for item in underlying_list:
                    if not isinstance(item, dict):
                        continue

                    symbol = clean_symbol(
                        item.get("symbol", "")
                    )

                    if symbol:
                        symbols.append(symbol)

                symbols = list(dict.fromkeys(symbols))

                if len(symbols) < 100:
                    raise RuntimeError(
                        f"NSE API returned only {len(symbols)} "
                        "individual-stock F&O symbols."
                    )

                return symbols

            last_error = (
                f"NSE API HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(
        "NSE F&O universe API could not be read. "
        "The app will not use a hard-coded fallback list. "
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------
# MARKET DATA
# ------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_daily_data(symbols):
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
def get_intraday_data(symbols):
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
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:
                frame = data[ticker].copy()
            elif ticker in level1:
                frame = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            frame = data.copy()

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                c[-1] if isinstance(c, tuple) else c
                for c in frame.columns
            ]

        if "Close" not in frame.columns:
            return pd.DataFrame()

        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return frame.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def local_dates(index):
    idx = pd.DatetimeIndex(index)

    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)

    return pd.Series(idx.date, index=index)


def unavailable_row(symbol):
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


def calculate_stock(symbol, daily_all, intraday_all, dt):
    daily = ticker_frame(daily_all, symbol + ".NS")

    if daily.empty or "Close" not in daily.columns:
        return unavailable_row(symbol)

    daily = daily.dropna(subset=["Close"]).copy()

    if len(daily) < 2:
        return unavailable_row(symbol)

    today = dt.date()
    dates = local_dates(daily.index)

    completed = daily.loc[dates < today].copy()
    if completed.empty:
        completed = daily.copy()

    completed = completed.dropna(subset=["Close"])

    if completed.empty:
        return unavailable_row(symbol)

    yesterday_close = float(completed["Close"].iloc[-1])

    intraday = ticker_frame(intraday_all, symbol + ".NS")
    today_price = np.nan

    if not intraday.empty and "Close" in intraday.columns:
        intra_dates = local_dates(intraday.index)
        today_rows = intraday.loc[
            intra_dates == today
        ].dropna(subset=["Close"])

        if not today_rows.empty:
            today_price = float(today_rows["Close"].iloc[-1])

    if np.isfinite(today_price) and today_price > 0:
        price = today_price
        day_base = yesterday_close
    elif not market_open(dt):
        # Outside market hours, use today's last intraday bar when available.
        # If the provider has no intraday data, use the latest daily close.
        price = yesterday_close
        day_base = (
            float(completed["Close"].iloc[-2])
            if len(completed) >= 2
            else np.nan
        )
    else:
        # Never call yesterday's close "Live Price" during market hours.
        return unavailable_row(symbol)

    if not np.isfinite(price) or price <= 0:
        return unavailable_row(symbol)

    one_day = (
        (price / day_base - 1) * 100
        if np.isfinite(day_base) and day_base > 0
        else np.nan
    )

    week_base = (
        float(completed["Close"].iloc[-5])
        if len(completed) >= 5
        else np.nan
    )
    one_week = (
        (price / week_base - 1) * 100
        if np.isfinite(week_base) and week_base > 0
        else np.nan
    )

    month_base = (
        float(completed["Close"].iloc[-21])
        if len(completed) >= 21
        else np.nan
    )
    one_month = (
        (price / month_base - 1) * 100
        if np.isfinite(month_base) and month_base > 0
        else np.nan
    )

    # Append current price as today's observation. Thus all EMAs and the
    # EMA-distance respond to the current/live price.
    ema_series = completed["Close"].copy()
    ema_series.loc[pd.Timestamp(dt)] = price

    ema21 = float(
        ema_series.ewm(span=21, adjust=False).mean().iloc[-1]
    )
    ema50 = float(
        ema_series.ewm(span=50, adjust=False).mean().iloc[-1]
    )
    ema200 = float(
        ema_series.ewm(span=200, adjust=False).mean().iloc[-1]
    )

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
    daily = get_daily_data(symbols)
    intraday = get_intraday_data(symbols)

    rows = [
        calculate_stock(symbol, daily, intraday, dt)
        for symbol in symbols
    ]

    result = pd.DataFrame(rows)

    # This is critical: every NSE constituent remains in the output even if
    # its Yahoo price data is temporarily unavailable.
    result = (
        result.set_index("Stock")
        .reindex(symbols)
        .reset_index()
    )

    return result, dt


def format_table(df):
    display = df.copy()

    for col in ["Live Price", "21 EMA", "50 EMA", "200 EMA"]:
        display[col] = display[col].map(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "—"
        )

    for col in [
        "1D Return %",
        "1W Return %",
        "1M Return %",
        "21 EMA vs Price %",
    ]:
        display[col] = display[col].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )

    # Trend colouring ONLY:
    # light green = Bullish, light yellow = Neutral,
    # light red = Bearish, light grey = no data.
    display["Trend"] = display["Trend"].replace({
        "Bullish": "↑ Bullish",
        "Neutral": "→ Neutral",
        "Bearish": "↓ Bearish",
        "Data unavailable": "— N/A",
    })

    def trend_style(value):
        value = str(value)

        if "Bullish" in value:
            return (
                "background-color:#D9F2DD;"
                "color:#16823A;"
                "font-weight:700;"
                "text-align:center;"
            )

        if "Neutral" in value:
            return (
                "background-color:#FFF2CC;"
                "color:#C47A00;"
                "font-weight:700;"
                "text-align:center;"
            )

        if "Bearish" in value:
            return (
                "background-color:#FADBDD;"
                "color:#C62828;"
                "font-weight:700;"
                "text-align:center;"
            )

        return (
            "background-color:#E8E8E8;"
            "color:#666666;"
            "font-weight:700;"
            "text-align:center;"
        )

    return (
        display.style
        .map(trend_style, subset=["Trend"])
        .set_properties(
            subset=["Trend"],
            **{
                "text-align": "center",
                "font-weight": "700",
            },
        )
    )


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

st.title("📊 My Stock Screener")
st.subheader("NSE Dynamic Universe Technical Screener")

st.caption(
    "Nifty 100 and F&O universes are read directly from NSE. "
    "No hard-coded stock list is used."
)

universe = st.selectbox(
    "Universe",
    ["Nifty 100", "F&O Stocks"],
)

try:
    if universe == "Nifty 100":
        symbols = get_nifty100()
        source = "NSE Nifty 100"
    else:
        symbols = get_fno_stocks()
        source = "NSE Individual Securities F&O"
except Exception as exc:
    st.error(
        "NSE universe could not be loaded. "
        "The app has intentionally stopped instead of showing an incomplete list."
    )
    st.code(str(exc))
    st.stop()

st.caption(f"{source} • {len(symbols)} stocks")

col1, col2 = st.columns(2)

with col1:
    scan_button = st.button(
        "🔍 Scan",
        type="primary",
        use_container_width=True,
    )

with col2:
    refresh_button = st.button(
        "🔄 Refresh Now",
        use_container_width=True,
    )

if (
    scan_button
    or refresh_button
    or "result" not in st.session_state
    or st.session_state.get("universe") != universe
):
    if refresh_button:
        get_intraday_data.clear()
        get_daily_data.clear()

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
            f"All {len(result)} NSE universe stocks are shown. "
            f"{unavailable} currently have unavailable price data; "
            "they were not removed."
        )

    st.dataframe(
        format_table(result),
        use_container_width=True,
        hide_index=True,
        height=680,
    )


@st.fragment(run_every="5m")
def auto_refresh():
    # Refresh price data every 5 minutes.
    # Universe lists are cached for one hour and therefore automatically
    # pick up an NSE constituent change within the cache window.
    get_intraday_data.clear()

    try:
        if universe == "Nifty 100":
            current_symbols = get_nifty100()
        else:
            current_symbols = get_fno_stocks()

        current_result, current_updated = scan(current_symbols)

        st.session_state.result = current_result
        st.session_state.updated = current_updated
        st.session_state.universe = universe

        st.dataframe(
            format_table(current_result),
            use_container_width=True,
            hide_index=True,
            height=680,
        )
    except Exception as exc:
        st.warning(f"Automatic refresh failed: {exc}")


auto_refresh()
