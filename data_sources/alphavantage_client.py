import streamlit as st
import requests

@st.cache_data(show_spinner=False, ttl=600)
def fetch_news_sentiment(api_key, ticker=None):
    """
    Fetch news articles and sentiment metrics from Alpha Vantage NEWS_SENTIMENT API.
    Raises ValueError if API Key is missing or invalid.
    Raises RuntimeError if Alpha Vantage returns error, note, or information messages (e.g. rate limit).
    """
    if not api_key or api_key.strip() == "":
        raise ValueError(
            "Alpha Vantage API Key is missing or empty. Please configure ALPHA_VANTAGE_API_KEY in your .env file or enter it in the sidebar."
        )
        
    ticker_param = f"&tickers={ticker}" if ticker else ""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT{ticker_param}&apikey={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Alpha Vantage: {str(e)}")
        
    if response.status_code != 200:
        raise RuntimeError(
            f"Alpha Vantage server returned HTTP Status {response.status_code}."
        )
        
    try:
        data = response.json()
    except Exception:
        raise RuntimeError("Failed to parse response from Alpha Vantage as JSON.")
        
    # Alpha Vantage returns "Note" for rate-limiting, "Information" for other updates, and "Error Message" for bad keys
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage API Rate Limit hit: {data['Note']}")
        
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage API Error: {data['Error Message']}")
        
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage Info: {data['Information']}")
        
    if "feed" not in data:
        raise RuntimeError("Invalid Alpha Vantage API response format (missing 'feed' key).")
        
    return data
