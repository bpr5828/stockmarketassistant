import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import re
from datetime import datetime
from dotenv import load_dotenv

# Load modular components
from data_sources.yfinance_client import fetch_single_ticker_fundamentals, fetch_historical_prices, fetch_yahoo_news
from data_sources.alphavantage_client import fetch_news_sentiment
from llm.llm_client import generate_llm_reasoning, analyze_sentiment_with_llm

# Load environment variables from .env
load_dotenv()

# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Equity Analytics & Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look and Feel
st.markdown("""
<style>
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
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="background: linear-gradient(135deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Equity Analytics Login</h1>
        <p style="color: #A0AEC0;">Please enter your authorized email to access the screener.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email_input = st.text_input("Email Address")
        if st.button("Login", use_container_width=True):
            if email_input.strip().lower() in ALLOWED_USERS:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Unauthorized email address.")
    st.stop()

# Load positive/negative sentiment keywords from environmental variables
pos_kw_str = os.getenv("POSITIVE_KEYWORDS", "good,positive,healthy,growth,upward,stellar,strong,bullish,profit,beats,earnings,success,upgrade,outperforms,gain,rise,expand")
neg_kw_str = os.getenv("NEGATIVE_KEYWORDS", "bad,negative,unhealthy,decline,downward,weak,bearish,loss,misses,deficit,failure,downgrade,underperforms,drop,fall,shrink")

POSITIVE_KEYWORDS = [w.strip().lower() for w in pos_kw_str.split(",") if w.strip()]
NEGATIVE_KEYWORDS = [w.strip().lower() for w in neg_kw_str.split(",") if w.strip()]

# Load credentials strictly from .env
av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
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
    <h2 style="font-weight:700; margin:0; background: linear-gradient(135deg, #00C6FF, #0072FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:1.65rem; line-height:1.2;">EQUITY ANALYTICS & SCREEENER</h2>
    <p style="color:#A0AEC0; font-size:0.85rem; margin:5px 0 0 0; line-height:1.3;">Modular Production Stock Screener Powered by Live Yahoo Finance & Alpha Vantage API Endpoints</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation choice (Screener Views)
page_choice = st.sidebar.radio(
    "Select Dashboard View",
    options=[
        "📰 Positive News Screener",
        "📈 Ticker Trend & Chart",
        "💬 Mentioned Articles"
    ],
    index=0,
    label_visibility="collapsed"
)

# Screener parameters are removed as they are no longer needed. The scanner dynamically
# discovers positive stocks from the general market news feed and Yahoo Finance.

# ==============================================================================
# MAIN PANEL DASHBOARD INTERFACE
# ==============================================================================

# ------------------------------------------------------------------------------
# TOP REFRESH CONTROLS & LAST SYNC TIMESTAMP
# ------------------------------------------------------------------------------
if "last_fetch_time" not in st.session_state:
    st.session_state["last_fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

col_header_left, col_header_right = st.columns([4, 1])
with col_header_left:
    st.write(f"⏱️ **Last Data Sync:** `{st.session_state['last_fetch_time']}`")
with col_header_right:
    if st.button("🔄 Refresh Latest Data", key="refresh_data_btn", use_container_width=True):
        st.cache_data.clear()
        st.session_state["last_fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Reset statuses and LLM cache
        if "status_yahoo" in st.session_state:
            del st.session_state["status_yahoo"]
        if "status_av" in st.session_state:
            del st.session_state["status_av"]
        if "status_llm" in st.session_state:
            del st.session_state["status_llm"]
        st.session_state["llm_scores"] = {}
        st.session_state["llm_reasoning"] = {}
        st.rerun()

# ------------------------------------------------------------------------------
# CONNECTOR STATUS BAR
# ------------------------------------------------------------------------------
def render_status_panel():
    y_text, y_ok, y_msg = st.session_state.get("status_yahoo", ("🟢 Yahoo Finance: Connected", True, ""))
    av_text, av_ok, av_msg = st.session_state.get("status_av", ("🟢 Vintage: Connected", True, ""))
    llm_text, llm_ok, llm_msg = st.session_state.get("status_llm", ("🟢 LLM: Configured", True, ""))
    
    def get_status_style(ok):
        if ok:
            return "background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); color: #10B981;"
        else:
            return "background-color: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); color: #EF4444;"
            
    y_style = get_status_style(y_ok)
    av_style = get_status_style(av_ok)
    llm_style = get_status_style(llm_ok)
    
    y_tooltip = f" title=\"{y_msg}\"" if y_msg else ""
    av_tooltip = f" title=\"{av_msg}\"" if av_msg else ""
    llm_tooltip = f" title=\"{llm_msg}\"" if llm_msg else ""
    
    html = f"""
    <div style="display:flex; gap:12px; margin: 8px 0 20px 0; flex-wrap:wrap;">
        <div style="{y_style} padding:6px 12px; border-radius:6px; font-size:0.8rem; font-weight:600; cursor:default;"{y_tooltip}>
            {y_text}
        </div>
        <div style="{av_style} padding:6px 12px; border-radius:6px; font-size:0.8rem; font-weight:600; cursor:default;"{av_tooltip}>
            {av_text}
        </div>
        <div style="{llm_style} padding:6px 12px; border-radius:6px; font-size:0.8rem; font-weight:600; cursor:default;"{llm_tooltip}>
            {llm_text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

render_status_panel()

# ------------------------------------------------------------------------------
# STEP 1: Load News Feed & Process Positive Sentiment Tickers
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# STEP 1: Load News Feed & Process Positive Sentiment Tickers
# ------------------------------------------------------------------------------
news_feed_ok = False
news_feed_data = {}

if not av_key or av_key.strip() == "":
    err_msg = "Alpha Vantage API Key is missing from the .env file. Please configure ALPHA_VANTAGE_API_KEY to run Alpha Vantage news scanner."
    st.warning(f"⚠️ Alpha Vantage: {err_msg}")
    st.session_state["status_av"] = ("🔴 Vintage: Key Missing", False, err_msg)
else:
    with st.spinner("⚡ Scanning live Alpha Vantage market news feed..."):
        try:
            news_feed_data = fetch_news_sentiment(av_key, ticker=None)
            news_feed_ok = True
            st.session_state["status_av"] = ("🟢 Vintage: Connected", True, "Alpha Vantage key configured successfully.")
        except Exception as e:
            err_msg = str(e)
            st.error(f"❌ Alpha Vantage news feed load failed: {err_msg}")
            st.session_state["status_av"] = ("🔴 Vintage: Fetch Failed", False, err_msg)

# Initialize sentiment_map
sentiment_map = {}

# Process Alpha Vantage Tickers if news is available
if news_feed_ok:
    articles = news_feed_data.get("feed", [])
    for art in articles:
        title = art.get("title", "")
        summary = art.get("summary", "")
        text_to_search = (title + " " + summary).lower()
        
        # Calculate positive and negative keyword hits in AV Article
        av_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_to_search)) for word in POSITIVE_KEYWORDS)
        av_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_to_search)) for word in NEGATIVE_KEYWORDS)
        av_net = av_pos_hits - av_neg_hits
        
        # Track which words matched
        matched_pos_set = {word for word in POSITIVE_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', text_to_search)}
        matched_neg_set = {word for word in NEGATIVE_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', text_to_search)}
        
        # Scan tickers mentioned in this article
        ticker_sentiment_list = art.get("ticker_sentiment", [])
        for tick_item in ticker_sentiment_list:
            ticker_symbol = tick_item.get("ticker", "").upper()
            
            # Clean up exchange prefixes (e.g. NYSE:AAPL -> AAPL)
            if ":" in ticker_symbol:
                ticker_symbol = ticker_symbol.split(":")[-1]
                
            # Filter standard US stock symbol format
            if re.match(r"^[A-Z\-]{1,6}$", ticker_symbol) and (av_pos_hits > 0 or av_neg_hits > 0):
                if ticker_symbol not in sentiment_map:
                    sentiment_map[ticker_symbol] = {
                        "mention_count": 0,
                        "keyword_score": 0,
                        "articles": [],
                        "pos_words": set(),
                        "neg_words": set()
                    }
                sentiment_map[ticker_symbol]["mention_count"] += 1
                sentiment_map[ticker_symbol]["keyword_score"] += av_net
                sentiment_map[ticker_symbol]["pos_words"].update(matched_pos_set)
                sentiment_map[ticker_symbol]["neg_words"].update(matched_neg_set)
                sentiment_map[ticker_symbol]["articles"].append({
                    "title": title,
                    "summary": summary,
                    "source": art.get("source", "Alpha Vantage"),
                    "time_published": art.get("time_published", ""),
                    "url": art.get("url", "#")
                })

# Determine which tickers to fetch on Yahoo Finance:
# - If AV loaded successfully: fetch Yahoo news only for tickers found in AV news to optimize speed.
# - If AV key is missing or failed: scan Yahoo news for all BENCHMARK_TICKERS.
tickers_to_fetch_yahoo = list(sentiment_map.keys()) if news_feed_ok else BENCHMARK_TICKERS

if tickers_to_fetch_yahoo:
    yahoo_spinner_msg = "Fetching Yahoo news and compiling dual-source metrics..." if news_feed_ok else "Scanning Yahoo Finance news for S&P components..."
    with st.spinner(f"💼 {yahoo_spinner_msg}"):
        try:
            for ticker_symbol in tickers_to_fetch_yahoo:
                yahoo_articles = fetch_yahoo_news(ticker_symbol)
                for y_art in yahoo_articles:
                    content_dict = y_art.get("content", {})
                    if not content_dict:
                        continue
                        
                    y_title = content_dict.get("title", "")
                    y_publisher = content_dict.get("provider", {}).get("displayName", "Yahoo Finance")
                    y_link = content_dict.get("clickThroughUrl", {}).get("url", "#")
                    if y_link == "#":
                        y_link = content_dict.get("canonicalUrl", {}).get("url", "#")
                    y_time_raw = content_dict.get("pubDate", "")
                    y_summary = content_dict.get("summary", "Full story available at source link.")
                    
                    y_time = ""
                    if y_time_raw:
                        try:
                            dt = datetime.strptime(y_time_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                            y_time = dt.strftime("%Y%m%dT%H%M%S")
                        except Exception:
                            y_time = y_time_raw
                            
                    # Calculate Keyword hits in Yahoo news title or summary
                    y_text = (y_title + " " + y_summary).lower()
                    y_pos_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', y_text)) for word in POSITIVE_KEYWORDS)
                    y_neg_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', y_text)) for word in NEGATIVE_KEYWORDS)
                    y_net = y_pos_hits - y_neg_hits
                    
                    y_matched_pos = {word for word in POSITIVE_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', y_text)}
                    y_matched_neg = {word for word in NEGATIVE_KEYWORDS if re.search(r'\b' + re.escape(word) + r'\b', y_text)}
                    
                    if y_pos_hits > 0 or y_neg_hits > 0:
                        if ticker_symbol not in sentiment_map:
                            sentiment_map[ticker_symbol] = {
                                "mention_count": 0,
                                "keyword_score": 0,
                                "articles": [],
                                "pos_words": set(),
                                "neg_words": set()
                            }
                        sentiment_map[ticker_symbol]["mention_count"] += 1
                        sentiment_map[ticker_symbol]["keyword_score"] += y_net
                        sentiment_map[ticker_symbol]["pos_words"].update(y_matched_pos)
                        sentiment_map[ticker_symbol]["neg_words"].update(y_matched_neg)
                        sentiment_map[ticker_symbol]["articles"].append({
                            "title": y_title,
                            "summary": y_summary,
                            "source": y_publisher,
                            "time_published": y_time,
                            "url": y_link
                        })
            st.session_state["status_yahoo"] = ("🟢 Yahoo Finance: Connected", True, "Successfully synced Yahoo Finance news and pricing.")
        except Exception as e:
            st.session_state["status_yahoo"] = ("🔴 Yahoo Finance: Failed", False, f"Failed to load Yahoo news: {str(e)}")

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
        
    # Only include positive net keyword score tickers as per screener requirement
    if t_data["keyword_score"] > 0:
        sec = SECTOR_MAP.get(t_symbol, fund.get("sector", "Other"))
        if not sec or sec == "N/A":
            sec = "Other"
        screener_list.append({
            "Ticker": t_symbol,
            "Company Name": fund["name"],
            "Sector": sec,
            "Mentions": t_data["mention_count"],
            "Keyword Score": t_data["keyword_score"],
            "LLM Score": llm_score_str,
            "raw_llm_score": llm_score,
            "Matched Keywords": kw_display
        })

df_screener = pd.DataFrame(screener_list)
if not df_screener.empty:
    # Sort by Keyword Score descending
    df_screener = df_screener.sort_values(by="Keyword Score", ascending=False).reset_index(drop=True)
    df_screener["Rank"] = df_screener.index + 1
else:
    df_screener = pd.DataFrame(columns=["Rank", "Ticker", "Company Name", "Mentions", "Keyword Score", "LLM Score", "Matched Keywords"])

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
if page_choice == "📰 Positive News Screener":
    st.markdown("### 📰 Dual-Source Sentiment Stock Screener")
    st.caption("**Keyword Sentiment Score** = (Positive Keyword Matches) - (Negative Keyword Matches) across all aggregated articles from Yahoo Finance & Alpha Vantage news feeds.")
    
    if df_screener.empty:
        st.info("💡 No stocks with positive net keyword mentions found in recent news feeds.")
    else:
        # Group positive stocks by Sector
        unique_sectors = sorted(df_screener["Sector"].unique())
        
        if unique_sectors:
            # Dropdown for sector filtering
            filter_options = ["All Sectors"] + unique_sectors
            selected_filter = st.selectbox("Filter by Sector:", filter_options)
            
            # Filter screener dataframe based on selection
            if selected_filter == "All Sectors":
                df_sector = df_screener.copy().reset_index(drop=True)
                st.write("Showing positive sentiment stocks across **All Sectors**:")
            else:
                df_sector = df_screener[df_screener["Sector"] == selected_filter].reset_index(drop=True)
                st.write(f"Showing positive sentiment stocks in the **{selected_filter}** sector:")
                
            df_sector["Rank"] = df_sector.index + 1
            
            # Responsive Card Layout for Screener Rows
            for idx, row in df_sector.iterrows():
                ticker_symbol = row["Ticker"]
                # Create a unique key for the button to avoid duplication errors across filters
                sector_key = row['Sector'].replace(' ', '_')
                
                with st.container(border=True):
                    c_header, c_action = st.columns([3, 1])
                    with c_header:
                        st.markdown(f"#### #{row['Rank']} [{ticker_symbol}](https://finance.yahoo.com/quote/{ticker_symbol}) - {row['Company Name']}")
                    with c_action:
                        if st.button("🪄 Analyze", key=f"btn_llm_{ticker_symbol}_{sector_key}", use_container_width=True):
                            if not llm_key or llm_key.strip() == "":
                                st.error("LLM Key is missing from the .env file.")
                            else:
                                ticker_articles = sentiment_map[ticker_symbol]["articles"]
                                with st.spinner(f"🤖 Analyzing {ticker_symbol}..."):
                                    try:
                                        res = analyze_sentiment_with_llm(
                                            ticker=ticker_symbol,
                                            articles_list=ticker_articles,
                                            provider=llm_provider,
                                            api_key=llm_key,
                                            model=llm_model,
                                            endpoint=llm_endpoint if llm_endpoint.strip() else None
                                        )
                                        st.session_state.llm_scores[ticker_symbol] = res["score"]
                                        st.session_state.llm_reasoning[ticker_symbol] = res["reasoning"]
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Analysis failed: {str(e)}")
                                        
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
                        
                    # Render LLM Reasoning card if analyzed
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

# ------------------------------------------------------------------------------
# VIEW 2: TICKER TREND & CHART
# ------------------------------------------------------------------------------
elif page_choice == "📈 Ticker Trend & Chart":
    st.markdown("### 📈 Stock Pricing Details & History")
    st.caption("Displays the historical daily chart and core yfinance fundamentals for tickers identified in the news.")
    
    if df_screener.empty:
        st.info("💡 No tickers available. Load positive tickers using the news screener first.")
    else:
        col_chart_sel, col_chart_space = st.columns([2, 3])
        with col_chart_sel:
            selected_chart_ticker = st.selectbox(
                "Select Ticker for Price Details",
                options=df_screener["Ticker"].tolist(),
                index=0,
                key="chart_ticker_selector"
            )
            
        # yfinance fetch fundamentals
        fund = fetch_single_ticker_fundamentals(selected_chart_ticker)
        
        # Fetch historical pricing
        with st.spinner(f"🔄 Fetching prices for {selected_chart_ticker}..."):
            ticker_price_df = fetch_historical_prices([selected_chart_ticker])
            
        if ticker_price_df.empty:
            st.error(f"Failed to load pricing series for {selected_chart_ticker}.")
        else:
            prices_series = ticker_price_df[selected_chart_ticker].dropna()
            
            # Calculate quick metrics (using configurable 4% default risk-free rate)
            daily_rets = prices_series.pct_change().dropna()
            ann_ret = daily_rets.mean() * 252
            ann_vol = daily_rets.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else 0
            
            # Visual columns
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
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

# ------------------------------------------------------------------------------
# VIEW 3: MENTIONED ARTICLES
# ------------------------------------------------------------------------------
elif page_choice == "💬 Mentioned Articles":
    st.markdown("### 💬 Mentioned News Articles")
    st.caption("Displays the full headlines and summaries of articles mentioning the selected stock from both Yahoo Finance and Alpha Vantage. Positive keywords are highlighted.")
    
    if df_screener.empty:
        st.info("💡 No tickers available. Load positive tickers using the news screener first.")
    else:
        col_art_sel, col_art_space = st.columns([2, 3])
        with col_art_sel:
            selected_art_ticker = st.selectbox(
                "Select Ticker to View News",
                options=df_screener["Ticker"].tolist(),
                index=0,
                key="art_ticker_selector"
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
st.sidebar.markdown("""
<div style="padding-top:20px; font-size:0.8rem; color:#718096; text-align:center;">
    Dual News Sentiments Active
</div>
""", unsafe_allow_html=True)
