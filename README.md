# 📈 Stock Analyst: Live News Sentiment Screener Dashboard

**Stock Analyst** is a real-time, AI-powered financial dashboard designed to help day traders, retail investors, and financial analysts discover trending stocks *before* they break out. 

Instead of relying on lagging price indicators, this tool aggregates live market news across **Alpha Vantage**, **Yahoo Finance**, **Finnhub**, and **Polygon.io** to instantly identify which companies are capturing media attention. It runs on-the-fly sentiment analysis to score articles using custom positive/negative keyword targeting, and leverages Large Language Models (LLMs like Gemini or OpenAI) to provide on-demand, deep-dive sentiment grading.

### 🎯 Who is this for?
- **Day Traders & Swing Traders:** Looking for high-momentum plays driven by breaking news.
- **Retail Investors:** Wanting a consolidated view of market sentiment without checking multiple news platforms.
- **Financial Analysts:** Needing an automated tool to scrape, summarize, and score thousands of articles daily to find hidden gems.

---

## 🚀 Key Features

*   **📰 Multi-Source News Screener**: Scans general news feeds, pulls mentioned stocks, merges them with Yahoo Finance, Finnhub, and Polygon.io ticker news, compiles Keyword Sentiment Scores, and displays them alongside mention counts.
*   **🧠 On-Demand LLM Sentiment Analyzer**: Select any stock from the positive feed and click a button to run a real-time LLM sentiment analysis (Gemini or OpenAI) on the article titles and summaries. The model assigns a score from -1.0 to +1.0 and generates a 1-sentence reasoning summary.
*   **📈 Ticker Trend & Chart**: Displays the historical 1-year closing price chart via Plotly and key capital structure metrics from **Yahoo Finance** for the selected ticker.
*   **💬 Mentioned Articles with Highlighter**: Lists all news headlines and summaries referencing the stock from Yahoo, Alpha Vantage, Finnhub, and Polygon.io, utilizing an interactive regex highlighting filter to color-code matching positive keywords.
*   **🔄 Cache-Busting Refresh Control**: Clears internal data caches and triggers a reload from the live endpoints, updating a "Last Data Sync" tracker.
*   **🛡️ Connector Status panel**: Monitor connection health for Yahoo Finance, Alpha Vantage, Finnhub, Polygon.io, and LLM API endpoints directly at the top of the dashboard. Shows detailed tooltips for error codes when keys are missing or rate limits are hit.
*   **🧹 Sleek Left Sidebar Layout**: Completely stripped of parameter fluff. Houses only the dashboard title, description, and page navigation menu.

---

## 📁 Component Architecture

```text
StockMarketAssistant/
├── .env                       # Local API keys and model configuration
├── requirements.txt           # Python library dependencies
├── README.md                  # Project setup and overview documentation
├── app.py                     # Streamlit frontend UI and screener coordinator
├── data_sources/              # Data source integration clients
│   ├── __init__.py
│   ├── yfinance_client.py     # Yahoo Finance historical prices, news, and fundamentals
│   ├── alphavantage_client.py # Alpha Vantage News Sentiment API (No mock data)
│   ├── finnhub_client.py      # Finnhub Company News API
│   └── polygon_client.py      # Polygon.io Ticker News API
└── llm/                       # LLM reasoning clients
    ├── __init__.py
    └── llm_client.py          # Gemini & OpenAI real API completions client
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Activate Environment and Install Requirements
```bash
cd /Users/nrobop/Development/StockMarketAssistant
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root directory (or edit the existing one):
```env
# Alpha Vantage API Key (Get a free key at: https://www.alphavantage.co/support/#api-key)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Finnhub API Key (Get a free key at: https://finnhub.io)
FINNHUB_API_KEY=your_finnhub_key_here

# Polygon.io API Key (Get a free key at: https://polygon.io)
POLYGON_API_KEY=your_polygon_key_here

# LLM Configuration (Choose provider: gemini or openai)
LLM_PROVIDER=gemini
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL=gemini-1.5-flash

# Optional: Custom LLM Endpoint URL override
LLM_ENDPOINT=
```

---

## 🖥️ Running the Application

To start the Streamlit server locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.
