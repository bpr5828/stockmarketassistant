Create a complete, single-file production-ready web application using pure Python and Streamlit that functions as an Advanced Equity Analytics & Stock Screener Dashboard. 

The application must integrate with Yahoo Finance (via the yfinance library) as the primary engine for historical prices and fundamentals, and utilize Alpha Vantage for market news sentiment. It must also feature an extensible mock LLM reasoning layer.

### Core Architecture & Workflow:

1. **Data Ingestion & Safety Rails:**
   - Provide a Streamlit text input field where users can paste a comma-separated list of custom stock tickers (e.g., AAPL, MSFT, NVDA).
   - Maintain a hardcoded backend list of ~50 highly liquid benchmark stocks (e.g., major S&P 500 components). If the user runs the "Top 50 Watch & Trade" engine, combine their custom inputs with this baseline universe to evaluate a sufficiently large pool of assets.
   - To stay strictly within Alpha Vantage free tier limits (5 requests/min), route all heavy historical price downloads and fundamental metrics through yfinance. Use Alpha Vantage exclusively for general market news.

2. **Fundamental & Quantitative Processing:**
   - For all tickers in the target pool, fetch Sector, Industry, Market Capitalization, P/E Ratio, and the last 12 months of daily historical closing prices via yfinance.
   - Calculate the annualized Sharpe Ratio for each stock using the daily closing prices, assuming a risk-free rate configurable by a Streamlit sidebar slider (default: 4%).
     Formula: Annualized Sharpe = (Annualized Mean Daily Return - Risk-Free Rate) / Annualized Daily Volatility.

3. **Multi-Factor Ranking Engines:**
   - **Top 10 High-Efficiency Leaderboard:** Display a sortable Streamlit dataframe filtering the top 10 stocks strictly with the highest Sharpe Ratios from the processed pool.
   - **Top 50 Watch & Trade Recommender:** Display a separate dataframe showing the top 50 stocks ranked by a hybrid scoring matrix: 70% weight on Sharpe Ratio and 30% weight on Market Cap size (representing liquidity/stability).

4. **The Mock LLM Reasoning Layer (Future Proofing):**
   - Create a dedicated Python function `generate_llm_reasoning(ticker, metrics_dict)`.
   - For now, this function should simulate an advanced financial LLM analysis. When a user clicks an "Analyze with AI" button next to a stock from the tables, the function should sleep for 1 second (to mock API latency) and return a dynamic, well-structured text generation like:
     "AI ANALYSIS FOR {ticker}: The