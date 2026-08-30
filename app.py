import streamlit as st
import pandas as pd
import requests
import feedparser
import yfinance as yf
import html
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Indian Market News", page_icon="📰", layout="wide")
st.title("📰 Indian Stock Market News Intelligence")
st.caption("Major Indian market, stock, sector and regulatory news • Auto-refresh every 5 minutes")

IST = ZoneInfo("Asia/Kolkata")
HEADERS = {"User-Agent": "Mozilla/5.0 Indian Market News Dashboard"}

NEWS_FEEDS = {
    "Indian Market": "https://news.google.com/rss/search?q=India+stock+market+Nifty+Sensex&hl=en-IN&gl=IN&ceid=IN:en",
    "Indian Stocks": "https://news.google.com/rss/search?q=Indian+stocks+shares+companies&hl=en-IN&gl=IN&ceid=IN:en",
    "Corporate": "https://news.google.com/rss/search?q=India+corporate+earnings+orders+acquisition+merger&hl=en-IN&gl=IN&ceid=IN:en",
    "SEBI RBI": "https://news.google.com/rss/search?q=SEBI+RBI+India+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "Government": "https://news.google.com/rss/search?q=India+government+policy+markets+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    "Sectors": "https://news.google.com/rss/search?q=India+banking+IT+pharma+auto+energy+defence+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    "Economy": "https://news.google.com/rss/search?q=India+inflation+GDP+rupee+FII+DII+oil+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "Global": "https://news.google.com/rss/search?q=India+stocks+Fed+oil+China+global+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "NSE Corporate": "https://www.nseindia.com/rss-feed/corporate-announcements",
}

SECTOR_WORDS = {
    "Banking": ["bank", "banking", "nbfc", "lending", "credit", "loan"],
    "IT": ["it services", "software", "technology", "tcs", "infosys", "wipro", "hcltech", "tech mahindra"],
    "Pharma": ["pharma", "drug", "pharmaceutical", "healthcare", "hospital"],
    "Auto": ["auto", "automobile", "vehicle", "cars", "two-wheeler"],
    "Energy": ["oil", "gas", "energy", "refinery", "power", "renewable"],
    "Defence": ["defence", "defense", "military", "missile", "hal", "bel"],
    "Metals": ["steel", "aluminium", "metal", "copper", "mining"],
    "FMCG": ["fmcg", "consumer", "foods", "beverages"],
    "Telecom": ["telecom", "mobile", "5g", "jio", "airtel"],
    "Realty": ["real estate", "realty", "property", "housing"],
    "Infrastructure": ["infra", "infrastructure", "roads", "construction", "cement"],
    "Financial Services": ["insurance", "finance", "financial services", "mutual fund", "amc"],
    "Chemicals": ["chemical", "specialty chemicals"],
}

HIGH_IMPACT_WORDS = [
    "fraud", "default", "bankruptcy", "insolvency", "major order", "large order",
    "order win", "acquisition", "merger", "takeover", "ceo", "md", "resignation",
    "resign", "regulatory action", "regulatory", "sebi", "rbi", "penalty", "fine",
    "investigation", "rating downgrade", "credit downgrade", "profit warning",
    "guidance cut", "fund raising", "fundraise", "buyback", "dividend", "debt",
    "plant shutdown", "shutdown", "approval", "license", "ban", "earnings",
    "results", "stake sale", "promoter stake", "ipo", "delisting",
]

MARKET_WORDS = [
    "nifty", "sensex", "rbi", "sebi", "budget", "inflation", "gdp", "fii", "dii",
    "rupee", "crude", "oil", "interest rate", "repo rate", "tariff", "fed",
    "trade war", "government", "tax", "india market",
]

def clean_text(value):
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def get_published_time(entry):
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
            except Exception:
                pass
    return None

def classify_news(text):
    text_lower = text.lower()
    sectors = []
    for sector, words in SECTOR_WORDS.items():
        if any(word in text_lower for word in words):
            sectors.append(sector)

    if any(word in text_lower for word in HIGH_IMPACT_WORDS + MARKET_WORDS):
        priority = "HIGH"
    elif sectors:
        priority = "IMPORTANT"
    else:
        priority = "ROUTINE"

    return priority, ", ".join(dict.fromkeys(sectors)[:3]) if sectors else "Market"

def read_feed(source_name, url):
    results = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        for entry in feed.entries[:100]:
            headline = clean_text(entry.get("title", ""))
            link = entry.get("link", "")
            published = get_published_time(entry)
            summary = clean_text(entry.get("summary", entry.get("description", "")))

            if not headline or not link or published is None:
                continue

            priority, sector = classify_news(headline + " " + summary)

            source = source_name
            try:
                source_value = entry.get("source")
                if isinstance(source_value, dict):
                    source = clean_text(source_value.get("title", source_name))
            except Exception:
                pass

            results.append({
                "Published": published,
                "Headline": headline,
                "Summary": summary,
                "Source": source,
                "Priority": priority,
                "Sector": sector,
                "Link": link,
            })
    except Exception:
        pass

    return results

@st.cache_data(ttl=240, show_spinner=False)
def load_news():
    rows = []
    for source, url in NEWS_FEEDS.items():
        rows.extend(read_feed(source, url))

    if not rows:
        return pd.DataFrame(columns=["Published", "Headline", "Summary", "Source", "Priority", "Sector", "Link"])

    df = pd.DataFrame(rows)
    df["DuplicateKey"] = (
        df["Headline"].str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return (
        df.sort_values("Published", ascending=False)
        .drop_duplicates("DuplicateKey")
        .drop(columns="DuplicateKey")
        .reset_index(drop=True)
    )

@st.cache_data(ttl=240, show_spinner=False)
def detect_sudden_moves():
    symbols = [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "BHARTIARTL",
        "ITC", "LT", "SBIN", "AXISBANK", "KOTAKBANK", "M&M", "MARUTI",
        "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "NTPC", "POWERGRID", "BEL", "HAL"
    ]

    results = []

    try:
        data = yf.download(
            [symbol + ".NS" for symbol in symbols],
            period="10d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        for symbol in symbols:
            try:
                close = data[symbol + ".NS"]["Close"].dropna()
                if len(close) < 2:
                    continue

                previous = float(close.iloc[-2])
                latest = float(close.iloc[-1])
                move = (latest / previous - 1) * 100

                if abs(move) >= 3:
                    results.append({"Stock": symbol, "Move %": round(move, 2)})
            except Exception:
                continue
    except Exception:
        pass

    if not results:
        return pd.DataFrame(columns=["Stock", "Move %"])

    return (
        pd.DataFrame(results)
        .sort_values("Move %", key=lambda x: x.abs(), ascending=False)
        .reset_index(drop=True)
    )

# 5-minute automatic refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="market_news_refresh")
except Exception:
    st.warning("Auto-refresh package is unavailable. Check requirements.txt.")

news = load_news()
moves = detect_sudden_moves()
now = datetime.now(IST)

c1, c2, c3 = st.columns(3)
c1.metric("News collected", len(news))
important_count = int(news["Priority"].isin(["HIGH", "IMPORTANT"]).sum()) if not news.empty else 0
c2.metric("High / Important", important_count)
c3.metric("Last update", now.strftime("%I:%M:%S %p"))

st.caption(now.strftime("%d-%m-%Y") + " IST • Automatic refresh every 5 minutes")

st.subheader("🚨 Sudden Stock Moves")

if moves.empty:
    st.info("No ≥3% daily move detected in the monitored large-cap stocks.")
else:
    st.dataframe(moves, use_container_width=True, hide_index=True)
    st.caption(
        "A sudden price move is NOT automatically assigned a reason. "
        "A reason is shown only when supporting news is available."
    )

st.subheader("🔎 Major & Important News")

f1, f2, f3 = st.columns(3)

with f1:
    priority_filter = st.selectbox("Priority", ["HIGH", "IMPORTANT", "ALL"])

with f2:
    sectors = ["ALL"]
    if not news.empty:
        sector_set = set()
        for value in news["Sector"].dropna():
            sector_set.update(str(value).split(", "))
        sectors += sorted(x for x in sector_set if x)
    sector_filter = st.selectbox("Sector", sectors)

with f3:
    search = st.text_input("Search stock / company / topic")

filtered = news.copy()

if priority_filter != "ALL":
    filtered = filtered[filtered["Priority"] == priority_filter]

if sector_filter != "ALL":
    filtered = filtered[
        filtered["Sector"].str.contains(sector_filter, case=False, na=False)
    ]

if search.strip():
    q = search.strip().lower()
    filtered = filtered[
        filtered["Headline"].str.lower().str.contains(q, na=False)
        | filtered["Summary"].str.lower().str.contains(q, na=False)
    ]

for _, row in filtered.head(60).iterrows():
    icon = "🔴" if row["Priority"] == "HIGH" else "🟠"

    st.markdown(
        f"**{icon} {row['Priority']} | {row['Sector']}**  \n"
        f"**{row['Published'].strftime('%d-%m-%Y %I:%M:%S %p')} IST**  \n"
        f"### {row['Headline']}"
    )

    if row["Summary"]:
        st.write(row["Summary"][:1500])

    st.caption("Source: " + str(row["Source"]))
    st.link_button("📖 Read original / full news", row["Link"])
    st.divider()

st.subheader("⚠️ Coverage note")
st.info(
    "This app is designed for maximum practical coverage using public feeds and "
    "primary-source announcements. Some publishers require paid APIs, authentication "
    "or restrict automated access, so no public-feed system can guarantee literally "
    "every story. The app does not invent a reason for a stock move when supporting "
    "news cannot be found."
)

st.download_button(
    "⬇️ Download News CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    "indian_market_news.csv",
    "text/csv",
)
