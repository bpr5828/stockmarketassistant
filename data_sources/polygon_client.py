import requests
from datetime import datetime

def fetch_polygon_news(ticker: str, api_key: str) -> list:
    """
    Fetch news from Polygon.io for a given ticker.
    Returns a list of dictionaries with standard keys:
    [title, summary, source, time_published, url]
    """
    if not api_key:
        return []
        
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=200&apiKey={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        results = data.get("results", [])
        
        for item in results:
            pub_time_raw = item.get("published_utc", "")
            pub_time = ""
            if pub_time_raw:
                try:
                    # e.g., "2023-11-20T17:15:00Z"
                    dt = datetime.strptime(pub_time_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    pub_time = dt.strftime("%Y%m%dT%H%M%S")
                except Exception:
                    pub_time = pub_time_raw
                    
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("description", ""),
                "source": item.get("publisher", {}).get("name", "Polygon"),
                "time_published": pub_time,
                "url": item.get("article_url", "#")
            })
            
        return articles
    except Exception as e:
        print(f"Error fetching Polygon news for {ticker}: {e}")
        return []
