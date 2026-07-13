import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import re
import math
from datetime import datetime
from dotenv import load_dotenv

# Load modular components
from data_sources.yfinance_client import fetch_single_ticker_fundamentals, fetch_historical_prices, fetch_yahoo_news, fetch_historical_data
from data_sources.alphavantage_client import fetch_news_sentiment
from data_sources.finnhub_client import fetch_finnhub_news
from data_sources.polygon_client import fetch_polygon_news
from llm.llm_client import generate_llm_reasoning, analyze_sentiment_with_llm

# Load environment variables from .env
load_dotenv()

# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look and Feel
st.markdown("""
<style>
    /* Hide Streamlit default top and bottom elements */
    #MainMenu {visibility: hidden; display: none;}
    header {visibility: hidden; display: none;}
    footer {visibility: hidden; display: none;}
    [data-testid="stToolbar"] {visibility: hidden; display: none;}
    [data-testid="stDecoration"] {visibility: hidden; display: none;}
    
    /* Hide Streamlit Cloud "Manage App" Badge */
    [class^="viewerBadge_"] {
        display: none !important;
    }
    #st-bottom {
        display: none !important;
    }

    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Font styles */
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, .stSelectbox, .stSlider {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 198, 255, 0.4);
    }
    
    /* AI Card Styles */
    .ai-analysis-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(0, 198, 255, 0.25);
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
        color: #F8FAFC;
    }
    
    /* News Card Styles */
    .news-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .news-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.12);
        transform: scale(1.005);
    }
    
    /* Sentiment Labels */
    .sentiment-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .sentiment-bullish {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .sentiment-bearish {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .sentiment-neutral {
        background-color: rgba(107, 114, 128, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(107, 114, 128, 0.25);
    }
    
    /* Custom divider line */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 198, 255, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Sidebar Navigation Custom Radio Styling */
    div[role="radiogroup"] {
        flex-direction: column !important;
        gap: 8px !important;
        background-color: transparent !important;
    }
    div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        color: #A0AEC0 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(0, 198, 255, 0.3) !important;
        color: #FFFFFF !important;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 198, 255, 0.12), rgba(0, 114, 255, 0.12)) !important;
        border-color: rgba(0, 198, 255, 0.5) !important;
        color: #00C6FF !important;
        font-weight: 600 !important;
    }
    /* Hide the radio dot circle and helper styling */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] label > div:nth-child(2) {
        margin-left: 0px !important;
        padding-left: 0px !important;
    }
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        margin-left: 0px !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONFIGURATION & STATIC PARAMETERS
# ==============================================================================

# Authentication setup
allowed_users_str = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = [email.strip().lower() for email in allowed_users_str.split(",") if email.strip()]

# Replace single BENCHMARK_TICKERS list with a 220+ ticker map grouped by sector
BENCHMARK_SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "INTU", "IBM", "AMAT", "NOW", "TXN", "INTC", "PANW", "MU", "ADI", "LRCX", "KLAC"],
    "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "ISRG", "PFE", "SYK", "VRTX", "BMY", "BSX", "MDT", "ELV", "GILD", "ZTS", "CVS", "CI"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "AXP", "GS", "SPGI", "C", "SCHW", "BLK", "MMC", "CB", "PGR", "CME", "ICE", "AIG", "TRV"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "MAR", "CMG", "ORLY", "HLT", "AZO", "LVS", "ROST", "YUM", "DHI", "TSCO", "F"],
    "Communication Services": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "ATVI", "EA", "WBD", "FOXA", "NWS", "PARA", "LYV", "OMC", "IPG", "MTCH"],
    "Industrials": ["CAT", "GE", "HON", "UNP", "BA", "UPS", "RTX", "LMT", "DE", "ETN", "ADP", "CSX", "EMR", "NSC", "GD", "PCAR", "CMI", "ROP", "PH", "TDG"],
    "Consumer Staples": ["PG", "COST", "WMT", "PEP", "KO", "PM", "MDLZ", "MO", "TGT", "KHC", "HSY", "MNST", "STZ", "ADM", "GIS", "K", "SYY", "CL", "KMB", "EL"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "WMB", "HES", "KMI", "BKR", "HAL", "DVN", "TRGP", "CTRA", "FANG", "MRO", "OKE"],
    "Utilities": ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "PCG", "ED", "XEL", "PEG", "WEC", "AWK", "ETR", "ES", "FE", "CMS", "AEE", "LNT", "NI"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "VTR", "AVB", "EQR", "ARE", "WY", "IRM", "EXR", "MAA", "ESS", "CPT", "UDR"],
    "Materials": ["LIN", "SHW", "FCX", "ECL", "NEM", "APD", "CTVA", "NUE", "DOW", "VMC", "MLM", "DD", "PPG", "ALB", "CE", "CF", "FMC", "EMN", "MOS", "IFF"]
}

# Flatten for API calls where needed
BENCHMARK_TICKERS = []
for sector, tickers in BENCHMARK_SECTORS.items():
    BENCHMARK_TICKERS.extend(tickers)

# Helper dict for fast sector lookup
SECTOR_MAP = {}
for sector, tickers in BENCHMARK_SECTORS.items():
    for ticker in tickers:
        SECTOR_MAP[ticker] = sector

# Authentication Check
if "authenticated" not in st.session_state:
    # Try to load from query params to persist login across reloads
    cached_user = st.query_params.get("user")
    if cached_user and cached_user in ALLOWED_USERS:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = cached_user
    else:
        st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="background: linear-gradient(135deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Stock Analyst Login</h1>
        <p style="color: #A0AEC0;">Please enter your authorized email to access the screener.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email_input = st.text_input("Email Address")
        if st.button("Login", use_container_width=True):
            if email_input.strip().lower() in ALLOWED_USERS:
                st.query_params["user"] = email_input.strip().lower()
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email_input.strip().lower()
                st.rerun()
            else:
                st.error("Unauthorized email address.")
    st.stop()

# Show user email in top right corner
st.markdown(
    f"""
    <div style="position: fixed; top: 15px; right: 20px; background: rgba(255,255,255,0.05); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #A0AEC0; z-index: 999999; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);">
        👤 {st.session_state['user_email']}
    </div>
    """, 
    unsafe_allow_html=True
)

# Load positive/negative sentiment keywords from environmental variables
pos_kw_str = os.getenv("POSITIVE_KEYWORDS", "good,positive,healthy,growth,upward,stellar,strong,bullish,profit,beats,earnings,success,upgrade,outperforms,gain,rise,expand")
neg_kw_str = os.getenv("NEGATIVE_KEYWORDS", "bad,negative,unhealthy,decline,downward,weak,bearish,loss,misses,deficit,failure,downgrade,underperforms,drop,fall,shrink")

POSITIVE_KEYWORDS = [w.strip().lower() for w in pos_kw_str.split(",") if w.strip()]
NEGATIVE_KEYWORDS = [w.strip().lower() for w in neg_kw_str.split(",") if w.strip()]

# Load credentials strictly from .env
av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
finnhub_key = os.getenv("FINNHUB_API_KEY", "")
polygon_key = os.getenv("POLYGON_API_KEY", "")
llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
llm_key = os.getenv("LLM_API_KEY", "")
llm_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
llm_endpoint = os.getenv("LLM_ENDPOINT", "")

# Initialize Connector Statuses in Session State
if "status_yahoo" not in st.session_state:
    st.session_state["status_yahoo"] = ("🟢 Yahoo Finance: Connected", True, "Successfully synced historical pricing and fundamentals.")

if "status_av" not in st.session_state:
    if not av_key or av_key.strip() == "":
        st.session_state["status_av"] = ("🔴 Vintage: Key Missing", False, "Alpha Vantage API Key is missing from the .env file.")
    else:
        st.session_state["status_av"] = ("🟢 Vintage: Connected", True, "Alpha Vantage key configured successfully.")

if "status_finnhub" not in st.session_state:
    if not finnhub_key or finnhub_key.strip() == "":
        st.session_state["status_finnhub"] = ("🔴 Finnhub: Key Missing", False, "Finnhub API Key is missing from the .env file.")
    else:
        st.session_state["status_finnhub"] = ("🟢 Finnhub: Connected", True, "Finnhub key configured successfully.")

if "status_polygon" not in st.session_state:
    if not polygon_key or polygon_key.strip() == "":
        st.session_state["status_polygon"] = ("🔴 Polygon: Key Missing", False, "Polygon.io API Key is missing from the .env file.")
    else:
        st.session_state["status_polygon"] = ("🟢 Polygon: Connected", True, "Polygon key configured successfully.")

if "status_llm" not in st.session_state:
    if not llm_key or llm_key.strip() == "":
        st.session_state["status_llm"] = ("🔴 LLM: Key Missing", False, "LLM API Key is missing from the .env file.")
    else:
        st.session_state["status_llm"] = (f"🟢 LLM: Configured ({llm_provider.upper()})", True, f"LLM client configured for provider '{llm_provider}' using model '{llm_model}'.")

# Initialize Session State cache for On-Demand LLM Scores
if "llm_scores" not in st.session_state:
    st.session_state["llm_scores"] = {}
if "llm_reasoning" not in st.session_state:
    st.session_state["llm_reasoning"] = {}

# ==============================================================================
# SIDEBAR HEADER & NAVIGATION
# ==============================================================================

# Sidebar Header (Title & Subtitle)
st.sidebar.markdown("""
<div style="margin-bottom:15px; padding-bottom:5px;">
    <h2 style="font-weight:700; margin:0; background: linear-gradient(135deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:1.65rem; line-height:1.2;">STOCK ANALYST</h2>
    <p style="color:#A0AEC0; font-size:0.85rem; margin:5px 0 0 0; line-height:1.3;">Top Moving Ticker Powered by Live Yahoo Finance, Alpha Vantage, Finnhub & Polygon.io</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation choice (Screener Views)
page_choice = st.sidebar.radio(
    "Select Dashboard View",
    options=[
        "🏠 Home",
        "📰 Top Moving Ticker",
        "📈 Ticker Trend & Chart",
        "💬 Mentioned Articles",
        "🔌 Data Source Status"
    ],
    index=0,
    label_visibility="collapsed"
)

def clear_cache_for(source):
    st.cache_data.clear()
    st.session_state["last_fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if source == "all":
        keys_to_delete = ["status_yahoo", "status_av", "status_finnhub", "status_polygon", "status_llm"]
    else:
        keys_to_delete = [f"status_{source}"]
    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# Screener parameters are removed as they are no longer needed. The scanner dynamically
# discovers positive stocks from the general market news feed and Yahoo Finance.

# ==============================================================================
# MAIN PANEL DASHBOARD INTERFACE
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Load News Feed & Process Positive Sentiment Tickers
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# STEP 1: Load News Feed & Process Positive Sentiment Tickers
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def build_screener_data(av_key, finnhub_key, polygon_key, pos_keywords, neg_keywords):
    sentiment_map = {}
    statuses = {}
    
    if not av_key or av_key.strip() == "":
        err_msg = "Alpha Vantage API Key is missing from the .env file. Please configure ALPHA_VANTAGE_API_KEY to run Alpha Vantage news scanner."
        statuses["av"] = ("🔴 Vintage: Key Missing", False, err_msg)
        news_feed_ok = False
    else:
        try:
            news_feed_data = fetch_news_sentiment(av_key, ticker=None)
            news_feed_ok = True
            statuses["av"] = ("🟢 Vintage: Connected", True, "Alpha Vantage key configured successfully.")
        except Exception as e:
            err_msg = str(e)
            statuses["av"] = ("🔴 Vintage: Fetch Failed", False, err_msg)
            news_feed_ok = False

    if news_feed_ok:
        articles = news_feed_data.get("feed", [])
        for art in articles:
            title = art.get("title", "")
            summary = art.get("summary", "")
            text_to_search = (title + " " + summary).lower()
            
            av_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_to_search)) for word in pos_keywords)
            av_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_to_search)) for word in neg_keywords)
            av_net = av_pos_hits - av_neg_hits
            
            matched_pos_set = {word for word in pos_keywords if re.search(r'\b' + re.escape(word) + r'\b', text_to_search)}
            matched_neg_set = {word for word in neg_keywords if re.search(r'\b' + re.escape(word) + r'\b', text_to_search)}
            
            ticker_sentiment_list = art.get("ticker_sentiment", [])
            for tick_item in ticker_sentiment_list:
                ticker_symbol = tick_item.get("ticker", "").upper()
                if ":" in ticker_symbol:
                    ticker_symbol = ticker_symbol.split(":")[-1]
                    
                if re.match(r"^[A-Z\-]{1,6}$", ticker_symbol) and (av_pos_hits > 0 or av_neg_hits > 0):
                    if ticker_symbol not in sentiment_map:
                        sentiment_map[ticker_symbol] = {
                            "mention_count": 0, "keyword_score": 0, "articles": [], "pos_words": set(), "neg_words": set()
                        }
                    sentiment_map[ticker_symbol]["mention_count"] += 1
                    sentiment_map[ticker_symbol]["keyword_score"] += av_net
                    sentiment_map[ticker_symbol]["pos_words"].update(matched_pos_set)
                    sentiment_map[ticker_symbol]["neg_words"].update(matched_neg_set)
                    sentiment_map[ticker_symbol]["articles"].append({
                        "title": title, "summary": summary, "source": art.get("source", "Alpha Vantage"),
                        "time_published": art.get("time_published", ""), "url": art.get("url", "#")
                    })

    # Only fetch additional news for active tickers found in the AV live feed
    active_tickers = list(sentiment_map.keys())
    
    if active_tickers:
        try:
            for ticker_symbol in active_tickers:
                # Yahoo
                yahoo_articles = fetch_yahoo_news(ticker_symbol)
                for y_art in yahoo_articles:
                    if not isinstance(y_art, dict): continue
                    content_dict = y_art.get("content", {})
                    if not content_dict: continue
                    y_title = content_dict.get("title") or ""
                    provider_dict = content_dict.get("provider") or {}
                    y_publisher = provider_dict.get("displayName") or "Yahoo Finance"
                    click_through = content_dict.get("clickThroughUrl") or {}
                    y_link = click_through.get("url") or "#"
                    if y_link == "#":
                        canonical = content_dict.get("canonicalUrl") or {}
                        y_link = canonical.get("url") or "#"
                    y_time_raw = content_dict.get("pubDate") or ""
                    y_summary = content_dict.get("summary") or "Full story available at source link."
                    y_time = ""
                    if y_time_raw:
                        try:
                            dt = datetime.strptime(y_time_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                            y_time = dt.strftime("%Y%m%dT%H%M%S")
                        except Exception:
                            y_time = y_time_raw
                    
                    y_text = (y_title + " " + y_summary).lower()
                    y_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', y_text)) for word in pos_keywords)
                    y_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', y_text)) for word in neg_keywords)
                    y_net = y_pos_hits - y_neg_hits
                    y_matched_pos = {word for word in pos_keywords if re.search(r'\b' + re.escape(word) + r'\b', y_text)}
                    y_matched_neg = {word for word in neg_keywords if re.search(r'\b' + re.escape(word) + r'\b', y_text)}
                    
                    if y_pos_hits > 0 or y_neg_hits > 0:
                        if ticker_symbol not in sentiment_map:
                            sentiment_map[ticker_symbol] = {
                                "mention_count": 0, "keyword_score": 0, "articles": [], "pos_words": set(), "neg_words": set()
                            }
                        sentiment_map[ticker_symbol]["mention_count"] += 1
                        sentiment_map[ticker_symbol]["keyword_score"] += y_net
                        sentiment_map[ticker_symbol]["pos_words"].update(y_matched_pos)
                        sentiment_map[ticker_symbol]["neg_words"].update(y_matched_neg)
                        sentiment_map[ticker_symbol]["articles"].append({
                            "title": y_title, "summary": y_summary, "source": y_publisher,
                            "time_published": y_time, "url": y_link
                        })
                            
                # Finnhub
                if finnhub_key and finnhub_key.strip() != "":
                    fh_articles = fetch_finnhub_news(ticker_symbol, finnhub_key)
                    for fh_art in fh_articles:
                        if not isinstance(fh_art, dict): continue
                        fh_title = fh_art.get("title") or ""
                        fh_summary = fh_art.get("summary") or ""
                        fh_text = (fh_title + " " + fh_summary).lower()
                        fh_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', fh_text)) for word in pos_keywords)
                        fh_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', fh_text)) for word in neg_keywords)
                        fh_net = fh_pos_hits - fh_neg_hits
                        fh_matched_pos = {word for word in pos_keywords if re.search(r'\b' + re.escape(word) + r'\b', fh_text)}
                        fh_matched_neg = {word for word in neg_keywords if re.search(r'\b' + re.escape(word) + r'\b', fh_text)}
                        
                        if fh_pos_hits > 0 or fh_neg_hits > 0:
                            if ticker_symbol not in sentiment_map:
                                sentiment_map[ticker_symbol] = {
                                    "mention_count": 0, "keyword_score": 0, "articles": [], "pos_words": set(), "neg_words": set()
                                }
                            sentiment_map[ticker_symbol]["mention_count"] += 1
                            sentiment_map[ticker_symbol]["keyword_score"] += fh_net
                            sentiment_map[ticker_symbol]["pos_words"].update(fh_matched_pos)
                            sentiment_map[ticker_symbol]["neg_words"].update(fh_matched_neg)
                            sentiment_map[ticker_symbol]["articles"].append(fh_art)
                            
                # Polygon
                if polygon_key and polygon_key.strip() != "":
                    pol_articles = fetch_polygon_news(ticker_symbol, polygon_key)
                    for pol_art in pol_articles:
                        if not isinstance(pol_art, dict): continue
                        pol_title = pol_art.get("title") or ""
                        pol_summary = pol_art.get("summary") or ""
                        pol_text = (pol_title + " " + pol_summary).lower()
                        pol_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', pol_text)) for word in pos_keywords)
                        pol_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', pol_text)) for word in neg_keywords)
                        pol_net = pol_pos_hits - pol_neg_hits
                        pol_matched_pos = {word for word in pos_keywords if re.search(r'\b' + re.escape(word) + r'\b', pol_text)}
                        pol_matched_neg = {word for word in neg_keywords if re.search(r'\b' + re.escape(word) + r'\b', pol_text)}
                        
                        if pol_pos_hits > 0 or pol_neg_hits > 0:
                            if ticker_symbol not in sentiment_map:
                                sentiment_map[ticker_symbol] = {
                                    "mention_count": 0, "keyword_score": 0, "articles": [], "pos_words": set(), "neg_words": set()
                                }
                            sentiment_map[ticker_symbol]["mention_count"] += 1
                            sentiment_map[ticker_symbol]["keyword_score"] += pol_net
                            sentiment_map[ticker_symbol]["pos_words"].update(pol_matched_pos)
                            sentiment_map[ticker_symbol]["neg_words"].update(pol_matched_neg)
                            sentiment_map[ticker_symbol]["articles"].append(pol_art)
                            
            statuses["yahoo"] = ("🟢 Yahoo Finance: Connected", True, "Successfully synced Yahoo Finance news and pricing.")
            if finnhub_key and finnhub_key.strip() != "":
                statuses["finnhub"] = ("🟢 Finnhub: Connected", True, "Finnhub key configured successfully.")
            else:
                statuses["finnhub"] = ("🔴 Finnhub: Key Missing", False, "Finnhub API Key is missing from the .env file.")
            if polygon_key and polygon_key.strip() != "":
                statuses["polygon"] = ("🟢 Polygon: Connected", True, "Polygon key configured successfully.")
            else:
                statuses["polygon"] = ("🔴 Polygon: Key Missing", False, "Polygon API Key is missing from the .env file.")
        except Exception as e:
            statuses["yahoo"] = ("🔴 Yahoo Finance: Failed", False, f"Failed to load Yahoo news: {str(e)}")

    return sentiment_map, statuses

with st.spinner("⚡ Scanning live market news and extracting active tickers..."):
    sentiment_map, statuses = build_screener_data(
        av_key, finnhub_key, polygon_key, tuple(POSITIVE_KEYWORDS), tuple(NEGATIVE_KEYWORDS)
    )

# Apply statuses to session_state
for k, v in statuses.items():
    st.session_state[f"status_{k}"] = v

if not sentiment_map:
    st.warning("⚠️ No active tickers found. Waiting for new market news...")

# Build a Screener Dataframe from sentiment map
screener_list = []
for t_symbol, t_data in sentiment_map.items():
    # Fetch name cached
    fund = fetch_single_ticker_fundamentals(t_symbol)
    
    # Format LLM score output if analyzed
    llm_score = st.session_state.llm_scores.get(t_symbol, "Pending")
    llm_score_str = f"{llm_score:+.2f}" if isinstance(llm_score, float) else str(llm_score)
    
    # Format Matched Keywords display
    pos_w = t_data["pos_words"]
    neg_w = t_data["neg_words"]
    
    pos_part = f"🟢 {', '.join(sorted(pos_w))}" if pos_w else ""
    neg_part = f"🔴 {', '.join(sorted(neg_w))}" if neg_w else ""
    
    if pos_part and neg_part:
        kw_display = f"{pos_part} | {neg_part}"
    elif pos_part:
        kw_display = pos_part
    elif neg_part:
        kw_display = neg_part
    else:
        kw_display = "None"
        
    sec = SECTOR_MAP.get(t_symbol, fund.get("sector", "Other"))
    if not sec or sec == "N/A":
        sec = "Other"
    screener_list.append({
        "Ticker": t_symbol,
        "Company Name": fund["name"],
        "Sector": sec,
        "Mentions": t_data["mention_count"],
        "Keyword Score": t_data["keyword_score"],
        "Price": fund.get("price", 0.0),
        "LLM Score": llm_score_str,
        "raw_llm_score": llm_score,
        "Matched Keywords": kw_display
    })

df_screener = pd.DataFrame(screener_list)
if not df_screener.empty:
    # Handle missing prices for sorting, set to infinity so they drop to bottom if ascending, but here we want lowest to highest
    df_screener["SortPrice"] = df_screener["Price"].apply(lambda x: float(x) if isinstance(x, (int, float)) and float(x) > 0 else float('inf'))
    # Sort by Price Lowest to Highest
    df_screener = df_screener.sort_values(by="SortPrice", ascending=True).reset_index(drop=True)
    df_screener = df_screener.drop(columns=["SortPrice"])
    df_screener["Rank"] = df_screener.index + 1
else:
    df_screener = pd.DataFrame(columns=["Rank", "Ticker", "Company Name", "Mentions", "Keyword Score", "Price", "LLM Score", "Matched Keywords"])

if page_choice == "🏠 Home":
    st.markdown("""
    # 📈 Stock Analyst
    **Stock Analyst** is a real-time, AI-powered financial dashboard designed to help day traders, retail investors, and financial analysts discover trending stocks *before* they break out. 
    
    Instead of relying on lagging price indicators, this tool aggregates live market news across **Alpha Vantage**, **Yahoo Finance**, **Finnhub**, and **Polygon.io** to instantly identify which companies are capturing media attention. It runs on-the-fly sentiment analysis to score articles using custom positive/negative keyword targeting, and leverages Large Language Models to provide on-demand, deep-dive sentiment grading.
    """)
    
    with st.expander("🎯 Who is this for?", expanded=True):
        st.markdown("""
        - **Day Traders & Swing Traders:** Looking for high-momentum plays driven by breaking news.
        - **Retail Investors:** Wanting a consolidated view of market sentiment without checking multiple news platforms.
        - **Financial Analysts:** Needing an automated tool to scrape, summarize, and score thousands of articles daily to find hidden gems.
        """)
        


# Helper to highlight matching keywords in text (case-insensitive)
def highlight_keywords(text, pos_kws, neg_kws):
    highlighted = text
    for kw in pos_kws:
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f'<span style="background-color: rgba(0, 198, 255, 0.25); color: #00C6FF; font-weight: 600; padding: 1px 4px; border-radius: 3px;">{m.group(0)}</span>', highlighted)
    for kw in neg_kws:
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f'<span style="background-color: rgba(239, 68, 68, 0.25); color: #EF4444; font-weight: 600; padding: 1px 4px; border-radius: 3px;">{m.group(0)}</span>', highlighted)
    return highlighted

# ==============================================================================
# PAGE VIEW RENDERERS
# ==============================================================================

# ------------------------------------------------------------------------------
# VIEW 1: NEWS SCREENER
# ------------------------------------------------------------------------------
if page_choice == "📰 Top Moving Ticker":
    st.markdown("### 📰 Top Moving Ticker")
    st.caption("**Keyword Sentiment Score** = (Positive Keyword Matches) - (Negative Keyword Matches) across all aggregated articles from Yahoo Finance & Alpha Vantage news feeds.")
    
    if df_screener.empty:
        st.info("💡 No tickers available.")
    else:
        def render_category(df_cat, cat_name):
            if df_cat.empty:
                st.info(f"No tickers found in {cat_name}.")
                return
                
            df_cat = df_cat.reset_index(drop=True)
            df_cat["Rank"] = df_cat.index + 1
            
            ITEMS_PER_PAGE = 50
            total_items = len(df_cat)
            total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))
            
            page_key = f"page_price_{cat_name.replace(' ', '_').replace('<', 'lt').replace('$', 'usd').replace('+', 'plus')}"
            current_page = st.session_state.get(page_key, 1)
            
            start_idx = (current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            df_page = df_cat.iloc[start_idx:end_idx]
            
            for idx, row in df_page.iterrows():
                ticker_symbol = row["Ticker"]
                
                with st.container(border=True):
                    price_str = f"${row['Price']:.2f}" if isinstance(row['Price'], (int, float)) and row['Price'] > 0 else "Price N/A"
                    st.markdown(f"#### #{row['Rank']} [{ticker_symbol}](https://finance.yahoo.com/quote/{ticker_symbol}) - {row['Company Name']} ({price_str})")
                    
                    c_stat1, c_stat2, c_stat3 = st.columns(3)
                    with c_stat1:
                        st.write(f"**Keyword Score:** `{row['Keyword Score']:+d}`")
                    with c_stat2:
                        score_val = st.session_state.llm_scores.get(ticker_symbol, "Pending")
                        if isinstance(score_val, float):
                            color = "#10B981" if score_val > 0.15 else "#EF4444" if score_val < -0.15 else "#9CA3AF"
                            st.markdown(f"**LLM Score:** <span style='color:{color}; font-weight:600;'>{score_val:+.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.write(f"**LLM Score:** _{score_val}_")
                    with c_stat3:
                        st.caption(f"**Keywords:** {row['Matched Keywords']}")
                        
                    if ticker_symbol in st.session_state.llm_scores:
                        score_val = st.session_state.llm_scores[ticker_symbol]
                        reasoning_val = st.session_state.llm_reasoning[ticker_symbol]
                        
                        badge_style = "sentiment-bullish" if score_val > 0.15 else "sentiment-bearish" if score_val < -0.15 else "sentiment-neutral"
                        st.markdown(
                            f"""<div class="ai-analysis-card" style="margin-top: 10px; margin-bottom: 5px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <strong style="font-size:0.9rem; color:#FFFFFF;">🤖 LLM Report:</strong>
                                    <span class="sentiment-badge {badge_style}">Score: {score_val:+.2f}</span>
                                </div>
                                <p style="margin:0; font-size:0.85rem; color:#F8FAFC; line-height:1.3;">
                                    {reasoning_val}
                                </p>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        st.markdown('<hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.04);"/>', unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'/>", unsafe_allow_html=True)
            col_space1, col_page1, col_page2, col_space2 = st.columns([1.5, 1, 1.5, 1])
            with col_page1:
                st.number_input("Page:", min_value=1, max_value=total_pages, step=1, key=page_key)
            with col_page2:
                start_display = min(total_items, 1 + (current_page - 1) * ITEMS_PER_PAGE)
                end_display = min(total_items, current_page * ITEMS_PER_PAGE)
                st.markdown(f"<div style='padding-top: 35px; color: #A0AEC0; font-weight: 500;'>Showing {start_display} - {end_display} of {total_items} tickers</div>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Filter Tickers")
        selected_tickers = st.multiselect(
            "Search and select tickers to bypass price filters...",
            options=df_screener["Ticker"].unique(),
            default=[],
            help="Type a ticker symbol (e.g. AAPL) to view it directly."
        )
        
        if selected_tickers:
            df_filtered = df_screener[df_screener["Ticker"].isin(selected_tickers)].reset_index(drop=True)
            df_filtered["Rank"] = df_filtered.index + 1
            render_category(df_filtered, "Selected Tickers")
        else:
            # Create tabs based on price
            tab_under_10, tab_10_50, tab_50_100, tab_over_100 = st.tabs(["< $10", "$10 - $50", "$50 - $100", "$100+"])
            
            with tab_under_10:
                render_category(df_screener[(df_screener["Price"] > 0) & (df_screener["Price"] < 10)], "< $10")
            with tab_10_50:
                render_category(df_screener[(df_screener["Price"] >= 10) & (df_screener["Price"] < 50)], "$10 - $50")
            with tab_50_100:
                render_category(df_screener[(df_screener["Price"] >= 50) & (df_screener["Price"] < 100)], "$50 - $100")
            with tab_over_100:
                render_category(df_screener[(df_screener["Price"] >= 100) | (df_screener["Price"] == 0)], "$100+")

# ------------------------------------------------------------------------------
# VIEW 2: TICKER TREND & CHART
# ------------------------------------------------------------------------------
elif page_choice == "📈 Ticker Trend & Chart":
    st.markdown("### 📈 Stock Pricing Details & History")
    st.caption("Displays the historical daily chart and core yfinance fundamentals for tickers identified in the news.")
    
    if df_screener.empty:
        st.info("💡 No tickers available. Load positive tickers using the news screener first.")
    else:
        tab_under_10, tab_10_50, tab_50_100, tab_over_100 = st.tabs(["< $10", "$10 - $50", "$50 - $100", "$100+"])
        
        def render_chart_tab(df_cat, tab_key):
            if df_cat.empty:
                st.info("No tickers found in this price range.")
                return
                
            col_chart_sel, col_chart_space = st.columns([2, 3])
            with col_chart_sel:
                selected_chart_ticker = st.selectbox(
                    "Select Ticker for Price Details",
                    options=df_cat["Ticker"].tolist(),
                    index=0,
                    key=f"chart_ticker_selector_{tab_key}"
                )
                
            # yfinance fetch fundamentals
            fund = fetch_single_ticker_fundamentals(selected_chart_ticker)
            
            # Fetch historical pricing
            with st.spinner(f"🔄 Fetching data for {selected_chart_ticker}..."):
                ticker_data_df = fetch_historical_data([selected_chart_ticker])
                
            if ticker_data_df.empty:
                st.error(f"Failed to load pricing series for {selected_chart_ticker}.")
            else:
                if isinstance(ticker_data_df.columns, pd.MultiIndex):
                    prices_series = ticker_data_df["Close"][selected_chart_ticker].dropna()
                    volume_series = ticker_data_df["Volume"][selected_chart_ticker].dropna()
                else:
                    prices_series = ticker_data_df["Close"].dropna()
                    volume_series = ticker_data_df["Volume"].dropna()
                
                # Calculate quick metrics (using configurable 4% default risk-free rate)
                daily_rets = prices_series.pct_change().dropna()
                ann_ret = daily_rets.mean() * 252
                ann_vol = daily_rets.std() * np.sqrt(252)
                sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else 0
                
                latest_volume = volume_series.iloc[-1] if not volume_series.empty else 0
                
                # Visual columns
                col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                with col_c1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <p style="color:#A0AEC0; font-size:0.9rem; margin:0;">Asset</p>
                            <h3 style="margin:5px 0 0 0; font-size:1.3rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{fund['name']}</h3>
                            <p style="color:#00C6FF; font-size:0.8rem; margin:0;">{fund['sector']}</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                with col_c2:
                    pe_val = fund["pe_ratio"]
                    pe_str = f"{pe_val}" if isinstance(pe_val, (int, float)) else "N/A"
                    st.markdown(
                        f"""<div class="metric-card">
                            <p style="color:#A0AEC0; font-size:0.9rem; margin:0;">Market Capitalization</p>
                            <h3 style="margin:5px 0 0 0; font-size:1.3rem; font-weight:600;">${fund['market_cap']/1e9:.2f}B</h3>
                            <p style="color:#00C6FF; font-size:0.8rem; margin:0;">P/E: {pe_str}</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                with col_c3:
                    st.markdown(
                        f"""<div class="metric-card">
                            <p style="color:#A0AEC0; font-size:0.9rem; margin:0;">Volatility Metrics</p>
                            <h3 style="margin:5px 0 0 0; font-size:1.3rem; font-weight:600; color:{'#10B981' if sharpe > 1 else '#EF4444' if sharpe < 0 else '#FFFFFF'};">{sharpe:.3f}</h3>
                            <p style="color:#00C6FF; font-size:0.8rem; margin:0;">Sharpe Ratio (4% RF)</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                with col_c4:
                    st.markdown(
                        f"""<div class="metric-card">
                            <p style="color:#A0AEC0; font-size:0.9rem; margin:0;">Annualized Return / Vol</p>
                            <h3 style="margin:5px 0 0 0; font-size:1.3rem; font-weight:600;">{ann_ret:.1%}</h3>
                            <p style="color:#00C6FF; font-size:0.8rem; margin:0;">Vol: {ann_vol:.1%}</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                with col_c5:
                    st.markdown(
                        f"""<div class="metric-card">
                            <p style="color:#A0AEC0; font-size:0.9rem; margin:0;">Latest Volume</p>
                            <h3 style="margin:5px 0 0 0; font-size:1.3rem; font-weight:600;">{latest_volume:,.0f}</h3>
                            <p style="color:#00C6FF; font-size:0.8rem; margin:0;">As of Last Close</p>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
                st.write("")
                
                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices_series.index,
                    y=prices_series.values,
                    mode='lines',
                    name=selected_chart_ticker,
                    line=dict(color='#00C6FF', width=2.5)
                ))
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showgrid=False, color="#A0AEC0"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#A0AEC0", side="right"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab_under_10:
            render_chart_tab(df_screener[(df_screener["Price"] > 0) & (df_screener["Price"] < 10)], "lt10")
        with tab_10_50:
            render_chart_tab(df_screener[(df_screener["Price"] >= 10) & (df_screener["Price"] < 50)], "10_50")
        with tab_50_100:
            render_chart_tab(df_screener[(df_screener["Price"] >= 50) & (df_screener["Price"] < 100)], "50_100")
        with tab_over_100:
            render_chart_tab(df_screener[(df_screener["Price"] >= 100) | (df_screener["Price"] == 0)], "gt100")

# ------------------------------------------------------------------------------
# VIEW 3: MENTIONED ARTICLES
# ------------------------------------------------------------------------------
elif page_choice == "💬 Mentioned Articles":
    st.markdown("### 💬 Mentioned News Articles")
    st.caption("Displays the full headlines and summaries of articles mentioning the selected stock from both Yahoo Finance and Alpha Vantage. Positive keywords are highlighted.")
    
    if df_screener.empty:
        st.info("💡 No tickers available. Load positive tickers using the news screener first.")
    else:
        tab_under_10, tab_10_50, tab_50_100, tab_over_100 = st.tabs(["< $10", "$10 - $50", "$50 - $100", "$100+"])
        
        def render_articles_tab(df_cat, tab_key):
            if df_cat.empty:
                st.info("No tickers found in this price range.")
                return
                
            col_art_sel, col_art_space = st.columns([2, 3])
            with col_art_sel:
                selected_art_ticker = st.selectbox(
                    "Select Ticker to View News",
                    options=df_cat["Ticker"].tolist(),
                    index=0,
                    key=f"art_ticker_selector_{tab_key}"
                )
                
            ticker_news_list = sentiment_map[selected_art_ticker]["articles"]
            
            # Display Clickable Source Links List
            st.markdown("#### 🔗 Article Source Links")
            for idx, article in enumerate(ticker_news_list):
                title_str = article.get("title", "Headline Unavailable")
                url_str = article.get("url", "#")
                source_str = article.get("source", "Source")
                st.markdown(f"{idx+1}. **[{source_str}]** [{title_str}]({url_str})")
                
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.write(f"Showing {len(ticker_news_list)} detailed news mentions for **{selected_art_ticker}**:")
            
            # Display cards
            for article in ticker_news_list:
                title = article.get("title", "Headline Unavailable")
                summary = article.get("summary", "Summary Unavailable")
                source = article.get("source", "N/A")
                score = float(article.get("overall_sentiment_score", 0.0)) if "overall_sentiment_score" in article else 0.0
                label = article.get("overall_sentiment_label", "Neutral") if "overall_sentiment_label" in article else "N/A"
                url = article.get("url", "#")
                pub_time = article.get("time_published", "")
                
                # Format date and time
                try:
                    dt = datetime.strptime(pub_time, "%Y%m%dT%H%M%S")
                    date_str = dt.strftime("%b %d, %Y - %I:%M %p")
                except Exception:
                    date_str = pub_time
                
                # Highlight positive/negative keywords in title and summary
                highlighted_title = highlight_keywords(title, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS)
                highlighted_summary = highlight_keywords(summary, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS)
                
                # Source-specific badge tags
                source_tag_style = "background-color:rgba(120,53,190,0.12); color:#A78BFA; border:1px solid rgba(120,53,190,0.25);" if "Yahoo" in source else "background-color:rgba(0,198,255,0.12); color:#00C6FF; border:1px solid rgba(0,198,255,0.25);"
                
                sentiment_class = "sentiment-neutral"
                if label.lower() in ["bullish", "somewhat_bullish", "somewhat bullish"]:
                    sentiment_class = "sentiment-bullish"
                elif label.lower() in ["bearish", "somewhat_bearish", "somewhat bearish"]:
                    sentiment_class = "sentiment-bearish"
                    
                st.markdown(
                    f"""<div class="news-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:5px;">
                            <div>
                                <span style="{source_tag_style} padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-right:8px;">🏷️ {source}</span>
                                <span style="color:#A0AEC0; font-size:0.8rem; font-weight:500;">⏱️ {date_str}</span>
                            </div>
                            {f'<span class="sentiment-badge {sentiment_class}">{label} ({score:+.2f})</span>' if label != "N/A" else ''}
                        </div>
                        <h4 style="margin:0 0 8px 0; font-size:1.05rem; font-weight:600; color:#FFFFFF;">
                            <a href="{url}" target="_blank" style="color:#FFFFFF; text-decoration:none; hover:underline;">{highlighted_title}</a>
                        </h4>
                        <p style="margin:0; font-size:0.9rem; color:#D1D5DB; line-height:1.4;">{highlighted_summary}</p>
                    </div>""",
                    unsafe_allow_html=True
                )

        with tab_under_10:
            render_articles_tab(df_screener[(df_screener["Price"] > 0) & (df_screener["Price"] < 10)], "lt10")
        with tab_10_50:
            render_articles_tab(df_screener[(df_screener["Price"] >= 10) & (df_screener["Price"] < 50)], "10_50")
        with tab_50_100:
            render_articles_tab(df_screener[(df_screener["Price"] >= 50) & (df_screener["Price"] < 100)], "50_100")
        with tab_over_100:
            render_articles_tab(df_screener[(df_screener["Price"] >= 100) | (df_screener["Price"] == 0)], "gt100")
st.sidebar.markdown("""
<div style="padding-top:20px; font-size:0.8rem; color:#718096; text-align:center;">
    Dual News Sentiments Active
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# VIEW 4: DATA SOURCE STATUS
# ------------------------------------------------------------------------------
if page_choice == "🔌 Data Source Status":
    st.markdown("### 🔌 Data Source Status & Refresh")
    st.caption("Manage data connections and refresh source caches.")
    
    if st.button("🔄 Refresh All Sources", use_container_width=True, type="primary"):
        clear_cache_for("all")
        
    st.markdown("---")
    
    def render_source_row(name, source_id, status_key):
        c1, c2, c3 = st.columns([2, 2, 1])
        text, ok, msg = st.session_state.get(status_key, (f"🟢 {name}", True, ""))
        
        with c1:
            if ok:
                st.success(text)
            else:
                st.error(f"{text} - {msg}")
                
        with c2:
            last_run = st.session_state.get("last_fetch_time", "Never")
            st.write(f"**Last Run:** `{last_run}`")
            
        with c3:
            if st.button(f"🔄 Refresh", key=f"refresh_{source_id}_pg", use_container_width=True):
                clear_cache_for(source_id)

    render_source_row("Yahoo Finance", "yahoo", "status_yahoo")
    render_source_row("Alpha Vantage", "av", "status_av")
    render_source_row("Finnhub", "finnhub", "status_finnhub")
    render_source_row("Polygon", "polygon", "status_polygon")
    render_source_row("LLM Provider", "llm", "status_llm")
