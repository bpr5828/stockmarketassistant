import requests
from datetime import datetime, timedelta

def fetch_finnhub_news(ticker: str, api_key: str, days_back: int = 7) -> list:
    """
    Fetch news from Finnhub for a given ticker over the past `days_back` days.
    Returns a list of dictionaries with standard keys:
    [title, summary, source, time_published, url]
    """
    if not api_key:
        return []
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date_str}&to={end_date_str}&token={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data:
            # Finnhub time is UNIX timestamp
            pub_time = ""
            if "datetime" in item:
                dt = datetime.fromtimestamp(item["datetime"])
                pub_time = dt.strftime("%Y%m%dT%H%M%S")
                
            articles.append({
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "Finnhub"),
                "time_published": pub_time,
                "url": item.get("url", "#")
            })
            
        return articles
    except Exception as e:
        print(f"Error fetching Finnhub news for {ticker}: {e}")
        return []
