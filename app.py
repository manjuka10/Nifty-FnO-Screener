import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="My Stock Screener",
    page_icon="📊",
    layout="wide"
)

st.title("📊 My Stock Screener")
st.subheader("Nifty 100 Technical Screener")
st.caption(
    "Latest available intraday price is used for calculations. "
    "Yahoo Finance data may be delayed during market hours."
)

NIFTY100_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


@st.cache_data(ttl=86400)
def get_nifty100_list():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/"
    }

    response = requests.get(
        NIFTY100_URL,
        headers=headers,
        timeout=20
    )
    response.raise_for_status()

    text = response.content.decode(
        "utf-8-sig",
        errors="replace"
    )
    rows = list(csv.reader(io.StringIO(text)))

    if len(rows) < 2:
        raise ValueError("Nifty 100 CSV returned no data.")

    header = [str(x).strip() for x in rows[0]]
    symbol_index = None

    for i, col in enumerate(header):
        if col.lower() == "symbol":
            symbol_index = i
            break

    if symbol_index is None:
        raise ValueError("Symbol column not found in Nifty 100 CSV.")

    symbols = []
    for row in rows[1:]:
        if len(row) > symbol_index:
            symbol = row[symbol_index].strip()
            if symbol:
                symbols.append(symbol)

    symbols = list(dict.fromkeys(symbols))

    if len(symbols) < 80:
        raise ValueError(
            f"Only {len(symbols)} Nifty 100 stocks were found."
        )

    return symbols


@st.cache_data(ttl=300)
def download_stock_data(symbols):
    tickers = [symbol + ".NS" for symbol in symbols]

    daily_data = yf.download(
        tickers=tickers,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    intraday_data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    return daily_data, intraday_data


def get_ticker_close(data, ticker):
    if data is None or data.empty:
        return pd.Series(dtype="float64")

    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if ticker in level0:
                temp = data[ticker].copy()
                if "Close" in temp.columns:
                    return pd.to_numeric(
                        temp["Close"], errors="coerce"
                    ).dropna()

            if ticker in level1:
                temp = data.xs(
                    ticker, axis=1, level=1
                )
                if "Close" in temp.columns:
                    return pd.to_numeric(
                        temp["Close"], errors="coerce"
                    ).dropna()

        elif "Close" in data.columns:
            return pd.to_numeric(
                data["Close"], errors="coerce"
            ).dropna()

    except Exception:
        pass

    return pd.Series(dtype="float64")


def get_index_date(index_value):
    try:
        timestamp = pd.Timestamp(index_value)

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Kolkata")

        return timestamp.date()
    except Exception:
        return None


def calculate_stock_data(symbol, daily_data, intraday_data):
    ticker = symbol + ".NS"

    try:
        daily_close = get_ticker_close(
            daily_data, ticker
        )
        intraday_close = get_ticker_close(
            intraday_data, ticker
        )

        if len(daily_close) < 220:
            return None, "insufficient historical data"

        if len(intraday_close) == 0:
            return None, "intraday data unavailable"

        live_price = float(intraday_close.iloc[-1])

        if not np.isfinite(live_price) or live_price <= 0:
            return None, "invalid intraday price"

        ist = ZoneInfo("Asia/Kolkata")
        today = datetime.now(ist).date()

        historical_close = daily_close.copy()
        last_daily_date = get_index_date(
            historical_close.index[-1]
        )

        # Remove today's incomplete daily candle.
        if last_daily_date == today:
            historical_close = historical_close.iloc[:-1]

        if len(historical_close) < 220:
            return None, "not enough completed daily history"

        # Previous completed trading day's close.
        previous_close = float(historical_close.iloc[-1])

        if previous_close <= 0:
            return None, "invalid previous closing price"

        # Current-day return.
        one_day_return = (
            live_price / previous_close - 1
        ) * 100

        # Current price versus close from 5 completed sessions earlier.
        one_week_base = float(historical_close.iloc[-6])
        one_week_return = (
            live_price / one_week_base - 1
        ) * 100

        # Current price versus close from about 21 sessions earlier.
        one_month_base = float(historical_close.iloc[-22])
        one_month_return = (
            live_price / one_month_base - 1
        ) * 100

        # Add latest price so EMA calculations incorporate it.
        calc_close = pd.concat([
            historical_close,
            pd.Series(
                [live_price],
                index=[pd.Timestamp.now()]
            )
        ])

        ema21 = float(
            calc_close.ewm(
                span=21,
                adjust=False
            ).mean().iloc[-1]
        )

        ema50 = float(
            calc_close.ewm(
                span=50,
                adjust=False
            ).mean().iloc[-1]
        )

        ema200 = float(
            calc_close.ewm(
                span=200,
                adjust=False
            ).mean().iloc[-1]
        )

        last_252 = calc_close.tail(252)
        week52_high = float(last_252.max())
        week52_low = float(last_252.min())

        from_52w_high = (
            live_price / week52_high - 1
        ) * 100

        from_52w_low = (
            live_price / week52_low - 1
        ) * 100

        from_21_ema = (
            live_price / ema21 - 1
        ) * 100

        if (
            live_price > ema21
            and ema21 > ema50
            and ema50 > ema200
        ):
            trend = "Bullish"
        elif (
            live_price < ema21
            and ema21 < ema50
            and ema50 < ema200
        ):
            trend = "Bearish"
        else:
            trend = "Neutral"

        result = {
            "Stock": symbol,
            "Price": live_price,
            "1D Return %": one_day_return,
            "1W Return %": one_week_return,
            "1M Return %": one_month_return,
            "21 EMA": ema21,
            "50 EMA": ema50,
            "200 EMA": ema200,
            "52W High": week52_high,
            "52W Low": week52_low,
            "From 52W High %": from_52w_high,
            "From 52W Low %": from_52w_low,
            "From 21 EMA %": from_21_ema,
            "Trend": trend
        }

        return result, None

    except Exception as e:
        return None, str(e)


def colour_trend(value):
    if value == "Bullish":
        return (
            "background-color: #198754;"
            "color: white;"
            "font-weight: bold;"
        )
    if value == "Neutral":
        return (
            "background-color: #F5B642;"
            "color: black;"
            "font-weight: bold;"
        )
    if value == "Bearish":
        return (
            "background-color: #DC3545;"
            "color: white;"
            "font-weight: bold;"
        )
    return ""


if st.button("🔍 Scan Nifty 100"):
    try:
        symbols = get_nifty100_list()
        st.info(
            f"Current Nifty 100 list: {len(symbols)} stocks"
        )
    except Exception as e:
        st.error("Unable to get the current Nifty 100 list.")
        st.error(str(e))
        st.stop()

    with st.spinner(
        "Downloading daily and latest intraday data..."
    ):
        try:
            daily_data, intraday_data = download_stock_data(
                symbols
            )
        except Exception as e:
            st.error("Unable to download stock data.")
            st.error(str(e))
            st.stop()

    results = []
    unavailable = []
    progress = st.progress(0)
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        result, reason = calculate_stock_data(
            symbol,
            daily_data,
            intraday_data
        )

        if result is not None:
            results.append(result)
        else:
            unavailable.append(
                f"{symbol} ({reason})"
            )

        progress.progress(
            int(((i + 1) / total) * 100)
        )

    progress.empty()

    if not results:
        st.error("No stock data could be calculated.")
        st.stop()

    columns = [
        "Stock",
        "Price",
        "1D Return %",
        "1W Return %",
        "1M Return %",
        "21 EMA",
        "50 EMA",
        "200 EMA",
        "52W High",
        "52W Low",
        "From 52W High %",
        "From 52W Low %",
        "From 21 EMA %",
        "Trend"
    ]

    df = pd.DataFrame(results)
    df = df[columns]

    df = df.sort_values(
        by="From 21 EMA %",
        ascending=False
    ).reset_index(drop=True)

    ist = ZoneInfo("Asia/Kolkata")
    updated_time = datetime.now(ist).strftime(
        "%d-%m-%Y %I:%M:%S %p IST"
    )

    st.success(
        f"🕐 Last updated: {updated_time}"
    )

    st.info(
        f"Nifty 100: {len(symbols)} stocks | "
        f"Calculated: {len(df)} stocks"
    )

    if unavailable:
        st.warning(
            "Data unavailable for: "
            + ", ".join(unavailable)
        )

    st.subheader(
        f"📋 Results — {len(df)} stocks"
    )

    display_df = df.copy()

    number_columns = [
        "Price",
        "1D Return %",
        "1W Return %",
        "1M Return %",
        "21 EMA",
        "50 EMA",
        "200 EMA",
        "52W High",
        "52W Low",
        "From 52W High %",
        "From 52W Low %",
        "From 21 EMA %"
    ]

    for col in number_columns:
        display_df[col] = pd.to_numeric(
            display_df[col],
            errors="coerce"
        ).round(2)

    styled_df = (
        display_df.style
        .map(
            colour_trend,
            subset=["Trend"]
        )
        .format({
            "Price": "{:.2f}",
            "1D Return %": "{:.2f}",
            "1W Return %": "{:.2f}",
            "1M Return %": "{:.2f}",
            "21 EMA": "{:.2f}",
            "50 EMA": "{:.2f}",
            "200 EMA": "{:.2f}",
            "52W High": "{:.2f}",
            "52W Low": "{:.2f}",
            "From 52W High %": "{:.2f}",
            "From 52W Low %": "{:.2f}",
            "From 21 EMA %": "{:.2f}"
        })
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=650,
        hide_index=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="nifty100_screener.csv",
        mime="text/csv"
    )
