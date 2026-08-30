import streamlit as st
import pandas as pd
import requests
import feedparser
import html
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

st.set_page_config(page_title="Indian & Global Stock Market News", page_icon="📰", layout="wide")

IST = ZoneInfo("Asia/Kolkata")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IndianMarketNews/1.0)"}

st.title("📰 Indian & Global Stock Market News")
st.caption("Major & important market-moving news • Indian + global markets • Automatic refresh every 5 minutes")

# Broad coverage. No user-facing filters are used.
SEARCHES = [
    ("Indian Market", "India stock market Nifty Sensex shares"),
    ("Indian Companies", "Indian listed companies stocks earnings results order acquisition"),
    ("Corporate Actions", "India listed company dividend buyback merger stake sale fundraising"),
    ("SEBI RBI", "SEBI RBI India stock market regulation circular"),
    ("India Economy", "India inflation GDP rupee FII DII interest rates economy"),
    ("Indian Sectors", "India banking IT pharma auto energy defence metals stocks"),
    ("US Markets", "US stocks S&P 500 Nasdaq Dow Jones market"),
    ("Global Markets", "global stock markets Europe Asia China Japan"),
    ("Central Banks", "Federal Reserve ECB BOJ interest rates markets"),
    ("Commodities", "crude oil gold commodities markets"),
    ("Currencies Bonds", "dollar rupee treasury yields bonds markets"),
    ("Global Macro", "global economy inflation recession tariffs trade markets"),
    ("Geopolitics", "geopolitics sanctions war tariffs markets stocks"),
]

HIGH_TERMS = [
    "fraud","default","bankruptcy","insolvency","major order","large order",
    "order win","acquisition","merger","takeover","ceo","resignation","regulatory",
    "sebi","rbi","penalty","fine","investigation","downgrade","profit warning",
    "guidance cut","fund raising","fundraise","buyback","dividend","debt",
    "shutdown","approval","license","ban","earnings","results","stake sale",
    "promoter stake","ipo","delisting","record profit","profit falls","profit rises",
    "revenue falls","revenue rises","target price","pledge","lawsuit"
]
MARKET_TERMS = [
    "nifty","sensex","s&p 500","nasdaq","dow jones","federal reserve","fed","ecb","boj",
    "repo rate","interest rate","rate cut","rate hike","inflation","gdp","fii","dii",
    "rupee","crude","oil","gold","treasury yield","bond yield","tariff","trade war",
    "recession","economic growth","geopolitical","war","sanctions","china","japan","europe",
    "global markets","us stocks"
]
IMPORTANT_TERMS = [
    "order","contract","earnings","results","profit","revenue","sales","investment",
    "expansion","launch","approval","stake","partnership","forecast","outlook","guidance",
    "brokerage","rating","capacity"
]

SECTOR_WORDS = {
    "Banking": ["bank","banking","nbfc","loan","credit"],
    "IT": ["tcs","infosys","wipro","hcltech","software","technology"],
    "Pharma": ["pharma","drug","pharmaceutical","healthcare","hospital"],
    "Auto": ["auto","automobile","vehicle","cars","two-wheeler"],
    "Energy": ["oil","gas","energy","refinery","power","renewable"],
    "Defence": ["defence","defense","military","missile","hal","bel"],
    "Metals": ["steel","aluminium","metal","copper","mining"],
    "FMCG": ["fmcg","consumer","foods","beverages"],
    "Telecom": ["telecom","mobile","5g","airtel","jio"],
    "Realty": ["real estate","realty","property","housing"],
    "Infrastructure": ["infra","infrastructure","roads","construction","cement"],
    "Financial Services": ["insurance","finance","mutual fund","amc"],
    "Chemicals": ["chemical","specialty chemicals"],
}


def clean_text(value):
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def published_time(entry):
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
            except Exception:
                pass
    return None


def priority(text):
    t = text.lower()
    if any(x in t for x in HIGH_TERMS) or any(x in t for x in MARKET_TERMS):
        return "HIGH"
    if any(x in t for x in IMPORTANT_TERMS):
        return "IMPORTANT"
    return "ROUTINE"


def sector(text):
    t = text.lower()
    found = [name for name, words in SECTOR_WORDS.items() if any(w in t for w in words)]
    return ", ".join(found[:3]) if found else "Market"


def google_rss(query):
    return "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-IN&gl=IN&ceid=IN:en"


def read_feed(label, url):
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        for e in feed.entries[:100]:
            title = clean_text(e.get("title", ""))
            link = e.get("link", "")
            summary = clean_text(e.get("summary", e.get("description", "")))
            published = published_time(e)
            if not title or not link or published is None:
                continue
            text = title + " " + summary
            rows.append({
                "Published": published,
                "Headline": title,
                "Summary": summary,
                "Source": label,
                "Priority": priority(text),
                "Sector": sector(text),
                "Link": link,
            })
    except Exception:
        pass
    return rows


def dedupe(df):
    if df.empty:
        return df
    df = df.copy()
    df["Key"] = (df["Headline"].str.lower()
                 .str.replace(r"[^a-z0-9 ]", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip())
    return (df.sort_values("Published", ascending=False)
              .drop_duplicates("Key")
              .drop(columns="Key")
              .reset_index(drop=True))


@st.cache_data(ttl=240, show_spinner=False)
def load_news():
    rows = []
    for label, query in SEARCHES:
        rows.extend(read_feed(label, google_rss(query)))

    # NSE's public RSS page documents corporate information feeds and real-time-style
    # publication of announcements. Keep the official NSE announcements page in the
    # source set as a direct public source as well.
    rows.extend(read_feed(
        "NSE Corporate Announcements",
        "https://www.nseindia.com/companies-listing/corporate-filings-application"
    ))

    if not rows:
        return pd.DataFrame()

    df = dedupe(pd.DataFrame(rows))
    return df[df["Priority"].isin(["HIGH", "IMPORTANT"])].reset_index(drop=True)


@st.cache_data(ttl=240, show_spinner=False)
def search_news(term):
    rows = []
    queries = [
        f'"{term}" India stock company',
        f'"{term}" NSE BSE earnings results order',
        f'"{term}" latest market news',
        f'"{term}" acquisition merger investment stake',
    ]
    for q in queries:
        rows.extend(read_feed("Stock Search", google_rss(q)))
    if not rows:
        return pd.DataFrame()
    df = dedupe(pd.DataFrame(rows))
    return df[df["Priority"].isin(["HIGH", "IMPORTANT"])].reset_index(drop=True)


# 5-minute automatic refresh.
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="market_news_refresh")
except Exception:
    st.warning("Auto-refresh is unavailable. Check requirements.txt.")

st.caption("Last updated: " + datetime.now(IST).strftime("%d-%m-%Y %I:%M:%S %p IST") + " • Auto-refresh every 5 minutes")

# ONLY user-facing control: search.
st.subheader("🔎 Search Stock / Company / Sector")
search = st.text_input(
    "Search",
    placeholder="Example: Reliance, Infosys, HDFC Bank, Tata Motors, pharma, defence",
    label_visibility="collapsed",
)

if search.strip():
    news = search_news(search.strip())
    st.subheader(f"📰 News related to {search.strip()}")
else:
    news = load_news()
    st.subheader("📰 Major & Important News")

if news.empty:
    st.info("No major or important news found right now.")
else:
    for _, row in news.head(100).iterrows():
        icon = "🔴" if row["Priority"] == "HIGH" else "🟠"
        st.markdown(f"**{icon} {row['Priority']} • {row['Sector']}**")
        st.markdown("**" + row["Published"].strftime("%d-%m-%Y %I:%M:%S %p IST") + "**")
        st.markdown(f"#### {row['Headline']}")
        if row["Summary"]:
            st.write(row["Summary"][:1800])
        st.caption("Source/feed: " + str(row["Source"]))
        st.link_button("📖 Read original / full article", row["Link"])
        st.divider()

st.download_button(
    "⬇️ Download Current News",
    news.to_csv(index=False).encode("utf-8"),
    "market_news.csv",
    "text/csv",
)

st.info(
    "Coverage is designed to surface major and important developments across Indian stocks, "
    "sectors, SEBI/RBI, corporate actions, the Indian economy and major global markets. "
    "No public-feed app can guarantee every article because some publishers and regulatory "
    "systems require direct or paid APIs. A headline is not treated as the reason for a "
    "stock move unless the source itself provides that explanation."
)
