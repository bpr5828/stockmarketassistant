import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_single_ticker_fundamentals(ticker):
    """
    Fetch and cache fundamentals info for a single ticker to optimize speed and
    caching granularity when new custom tickers are added.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extract name, sector, industry, market cap, PE ratio, and price
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or "N/A"
        industry = info.get("industry") or "N/A"
        mcap = info.get("marketCap") or 0
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0.0
        
        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe is not None:
            pe = round(pe, 2)
        else:
            pe = "N/A"
            
        return {
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": mcap,
            "pe_ratio": pe,
            "price": price
        }
    except Exception as e:
        # Propagation of error or empty dict depending on requirements.
        # Here we return a dictionary showing N/A but containing the error so we can notify the user
        # or log it, while avoiding complete application crash if a single ticker info download fails.
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "pe_ratio": "N/A",
            "price": 0.0,
            "error": str(e)
        }

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_historical_prices(tickers):
    """
    Batch download 1y historical close prices for the full ticker list
    using yfinance. Returns a clean pandas DataFrame.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Download
    df = yf.download(tickers, period="1y", progress=False)
    if df.empty:
        return pd.DataFrame()
        
    if "Close" in df.columns:
        close_df = df["Close"]
        # Handle single ticker case returning a Series
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(name=tickers[0])
        return close_df
        
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_yahoo_news(ticker):
    """
    Fetch news articles for a single ticker from Yahoo Finance.
    """
    try:
        t = yf.Ticker(ticker)
        return t.news or []
    except Exception:
        return []
