import streamlit as st
import pandas as pd
import requests
import feedparser
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html
import re

st.set_page_config(
    page_title="Indian Stock Market News",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Indian Stock Market News")
st.caption("Multi-source market news dashboard • Automatic refresh every 5 minutes")

IST = ZoneInfo("Asia/Kolkata")

# ------------------------------------------------------------
# RSS / NEWS FEEDS
# ------------------------------------------------------------
RSS_SOURCES = {
    "Indian Stock Market":
        "https://news.google.com/rss/search?q=Indian+stock+market+NSE+BSE+Nifty+Sensex&hl=en-IN&gl=IN&ceid=IN:en",
    "Nifty 100 Stocks":
        "https://news.google.com/rss/search?q=Nifty+100+stocks+India&hl=en-IN&gl=IN&ceid=IN:en",
    "Corporate News":
        "https://news.google.com/rss/search?q=India+company+order+results+acquisition+dividend+buyback&hl=en-IN&gl=IN&ceid=IN:en",
    "SEBI RBI":
        "https://news.google.com/rss/search?q=SEBI+RBI+India+stock+market&hl=en-IN&gl=IN&ceid=IN:en",
    "FII DII / Economy":
        "https://news.google.com/rss/search?q=India+FII+DII+rupee+inflation+economy+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    "Global Markets":
        "https://news.google.com/rss/search?q=India+stocks+Fed+China+oil+global+markets&hl=en-IN&gl=IN&ceid=IN:en",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    )
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def entry_time(entry):
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(
                    parsed.tm_year,
                    parsed.tm_mon,
                    parsed.tm_mday,
                    parsed.tm_hour,
                    parsed.tm_min,
                    parsed.tm_sec,
                    tzinfo=timezone.utc,
                ).astimezone(IST)
            except Exception:
                pass
    return None


def classify_sentiment(text):
    t = text.lower()

    positive = [
        "profit", "order win", "order received", "contract", "approval",
        "dividend", "buyback", "acquisition", "acquires", "growth",
        "upgrade", "beats", "strong", "record", "deal", "expansion",
        "positive", "raises guidance", "stake purchase"
    ]

    negative = [
        "loss", "fraud", "penalty", "fine", "downgrade", "probe",
        "investigation", "default", "resign", "resignation", "decline",
        "weak", "misses", "warning", "lawsuit", "regulatory action",
        "debt", "cut guidance", "slump"
    ]

    p = sum(x in t for x in positive)
    n = sum(x in t for x in negative)

    if p > n:
        return "Positive"
    if n > p:
        return "Negative"
    return "Neutral"


def classify_impact(text):
    t = text.lower()

    very_high = [
        "fraud", "bankruptcy", "default", "merger", "takeover",
        "major acquisition", "sebi action", "ed raid", "income tax raid",
        "arrest", "major order", "record order", "accounting issue"
    ]

    high = [
        "order", "contract", "results", "earnings", "profit", "loss",
        "dividend", "buyback", "stake sale", "promoter", "downgrade",
        "upgrade", "approval", "regulatory", "resign", "resignation"
    ]

    medium = [
        "board meeting", "conference", "outlook", "expansion",
        "capacity", "investment", "partnership", "fund raising"
    ]

    if any(x in t for x in very_high):
        return "Very High"
    if any(x in t for x in high):
        return "High"
    if any(x in t for x in medium):
        return "Medium"
    return "Low"


def source_name(entry, fallback):
    try:
        src = entry.get("source")
        if src:
            title = src.get("title")
            if title:
                return clean_text(title)
    except Exception:
        pass
    return fallback


def extract_topic(title):
    # Simple extraction of a likely stock/company/topic.
    # The full headline remains available, so this is only a label.
    parts = re.split(r"\s[-|:]\s", title, maxsplit=1)
    if parts:
        candidate = parts[0].strip()
        if 1 <= len(candidate) <= 45:
            return candidate
    return "Market"


# ------------------------------------------------------------
# FETCH NEWS
# ------------------------------------------------------------
@st.cache_data(ttl=240, show_spinner=False)
def fetch_news():
    rows = []

    for feed_name, url in RSS_SOURCES.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries[:60]:
                title = clean_text(entry.get("title", ""))
                link = entry.get("link", "")
                summary = clean_text(
                    entry.get("summary", entry.get("description", ""))
                )

                if not title or not link:
                    continue

                published = entry_time(entry)
                if published is None:
                    continue

                combined = f"{title} {summary}"

                rows.append({
                    "Published": published,
                    "Topic": extract_topic(title),
                    "Headline": title,
                    "Summary": summary,
                    "Source": source_name(entry, feed_name),
                    "Impact": classify_impact(combined),
                    "Sentiment": classify_sentiment(combined),
                    "Link": link,
                })
        except Exception:
            continue

    columns = [
        "Published", "Topic", "Headline", "Summary",
        "Source", "Impact", "Sentiment", "Link"
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    # Remove duplicate headlines while retaining the newest copy.
    df["dedupe"] = (
        df["Headline"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = (
        df.sort_values("Published", ascending=False)
        .drop_duplicates("dedupe", keep="first")
        .drop(columns=["dedupe"])
        .reset_index(drop=True)
    )

    return df


# ------------------------------------------------------------
# AUTO REFRESH - 5 MINUTES
# ------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(
        interval=5 * 60 * 1000,
        key="news_auto_refresh"
    )
    refresh_ok = True
except Exception:
    refresh_ok = False

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
with st.spinner("Collecting latest news..."):
    news = fetch_news()

now = datetime.now(IST)

a, b, c = st.columns(3)
a.metric("News items", len(news))
b.metric("Feeds", len(RSS_SOURCES))
c.metric("Auto refresh", "5 min")

st.caption(
    f"Last updated: {now.strftime('%d-%m-%Y %I:%M:%S %p')} IST"
)

if not refresh_ok:
    st.warning(
        "Auto-refresh is unavailable because streamlit-autorefresh "
        "is not installed. Make sure requirements.txt contains it."
    )

if news.empty:
    st.error("No news could be retrieved right now.")
    st.stop()

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------
st.subheader("🔎 Filters")

f1, f2, f3 = st.columns(3)

with f1:
    impact_filter = st.selectbox(
        "Impact",
        ["All", "Very High", "High", "Medium", "Low"]
    )

with f2:
    sentiment_filter = st.selectbox(
        "Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

with f3:
    search = st.text_input(
        "Search company / stock / headline"
    )

filtered = news.copy()

if impact_filter != "All":
    filtered = filtered[
        filtered["Impact"] == impact_filter
    ]

if sentiment_filter != "All":
    filtered = filtered[
        filtered["Sentiment"] == sentiment_filter
    ]

if search.strip():
    q = search.strip().lower()
    mask = (
        filtered["Headline"].str.lower().str.contains(q, na=False)
        | filtered["Summary"].str.lower().str.contains(q, na=False)
        | filtered["Topic"].str.lower().str.contains(q, na=False)
    )
    filtered = filtered[mask]

# ------------------------------------------------------------
# IMPORTANT NEWS
# ------------------------------------------------------------
st.subheader("🔥 Latest Important News")

important = filtered[
    filtered["Impact"].isin(["Very High", "High"])
].head(20)

if important.empty:
    st.info("No high-impact news found for the current filters.")
else:
    for _, row in important.iterrows():
        d = row["Published"]

        sentiment_icon = {
            "Positive": "🟢",
            "Negative": "🔴",
            "Neutral": "🟡",
        }.get(row["Sentiment"], "⚪")

        impact_icon = {
            "Very High": "🔥",
            "High": "🟠",
        }.get(row["Impact"], "🟡")

        st.markdown(
            f"**{impact_icon} {row['Impact']} | "
            f"{sentiment_icon} {row['Sentiment']}**  \n"
            f"**{d.strftime('%d-%m-%Y')} | "
            f"{d.strftime('%I:%M:%S %p')} IST**  \n"
            f"### {row['Headline']}"
        )

        if row["Summary"]:
            st.write(row["Summary"][:1500])

        st.caption(
            f"Source: {row['Source']} | Topic: {row['Topic']}"
        )

        st.link_button(
            "📖 Read Full News / Original Article",
            row["Link"]
        )

        st.divider()

# ------------------------------------------------------------
# ALL NEWS
# ------------------------------------------------------------
st.subheader("📰 All News")

display = filtered.copy()
display["Date"] = display["Published"].dt.strftime("%d-%m-%Y")
display["Time (IST)"] = display["Published"].dt.strftime("%I:%M:%S %p")

display = display[
    [
        "Date",
        "Time (IST)",
        "Topic",
        "Impact",
        "Sentiment",
        "Source",
        "Headline",
        "Link",
    ]
]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    height=650,
    column_config={
        "Link": st.column_config.LinkColumn(
            "Full News",
            display_text="Open"
        )
    },
)

# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------
download_df = filtered.copy()
download_df["Published"] = download_df["Published"].dt.strftime(
    "%d-%m-%Y %I:%M:%S %p IST"
)

st.download_button(
    "⬇️ Download News CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="indian_stock_market_news.csv",
    mime="text/csv",
)

st.caption(
    "News is aggregated from publicly available feeds. "
    "Use the original-source button to read the full article. "
    "Publisher terms and paywalls may apply."
)
