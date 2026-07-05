import requests

def generate_llm_reasoning(ticker, metrics_dict, provider, api_key, model, endpoint=None):
    """
    Perform a real LLM call to Google Gemini or OpenAI APIs to get a financial analysis.
    Raises ValueError if API Key is missing or provider is unsupported.
    Raises RuntimeError if the API call fails or returns error codes.
    """
    if not api_key or api_key.strip() == "":
        raise ValueError(
            "LLM API Key is missing or empty. Please set the LLM_API_KEY in your .env file."
        )
        
    provider = provider.strip().lower() if provider else "gemini"
    
    # Standard prompt for both models
    prompt = f"""You are an advanced financial analyst. Conduct a thorough and professional equity research evaluation for the stock ticker {ticker} ({metrics_dict.get('name', ticker)}).
Sector: {metrics_dict.get('sector', 'N/A')}
Industry: {metrics_dict.get('industry', 'N/A')}
Market Cap: ${metrics_dict.get('market_cap', 0)/1e9:.2f}B
P/E Ratio: {metrics_dict.get('pe_ratio', 'N/A')}
Annualized Return (1y): {metrics_dict.get('ann_return', 0.0)*100:.2f}%
Annualized Volatility (1y): {metrics_dict.get('ann_volatility', 0.0)*100:.2f}%
Sharpe Ratio (1y, risk-free rate of {metrics_dict.get('rf_rate', 0.04)*100:.1f}%): {metrics_dict.get('sharpe', 0.0):.3f}

Please provide:
1. A Quantitative & Volatility Assessment based on the Sharpe ratio and returns.
2. A Fundamental & Valuation Analysis based on P/E multiple and market cap.
3. An Actionable Strategic Outlook / Investment Recommendation (STRONG ACCUMULATE, HOLD/NEUTRAL, or UNDERPERFORM/WATCHLIST) with clear rationale and conviction level.

Structure your response clearly using markdown with clean headers.
"""

    if provider == "gemini":
        url = endpoint if endpoint else f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini API: {str(e)}")
            
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API returned status code {response.status_code}: {response.text}"
            )
            
        try:
            data = response.json()
            generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini response: {str(e)}. Response raw: {response.text}")
            
    elif provider == "openai":
        url = endpoint if endpoint else "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {str(e)}")
            
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API returned status code {response.status_code}: {response.text}"
            )
            
        try:
            data = response.json()
            generated_text = data["choices"][0]["message"]["content"]
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Failed to parse OpenAI response: {str(e)}. Response raw: {response.text}")
            
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Supported providers are 'gemini' or 'openai'."
        )

def analyze_sentiment_with_llm(ticker, articles_list, provider, api_key, model, endpoint=None):
    """
    Perform a real-time LLM sentiment analysis on a list of articles for a given ticker.
    Returns a dictionary with 'score' (float between -1.0 and 1.0) and 'reasoning' (1-sentence string).
    Raises ValueError if API Key is missing.
    Raises RuntimeError if call fails.
    """
    if not api_key or api_key.strip() == "":
        raise ValueError("LLM API Key is missing or empty. Please set the LLM_API_KEY in your .env file.")
        
    provider = provider.strip().lower() if provider else "gemini"
    
    # Format articles for prompt
    formatted_articles = ""
    for idx, art in enumerate(articles_list[:5]): # Analyze top 5 articles
        formatted_articles += f"Article {idx+1}:\nTitle: {art.get('title')}\nSummary: {art.get('summary')}\n\n"
        
    prompt = f"""You are a financial news sentiment analyst. Analyze the following news articles mentioning the stock ticker {ticker}:

{formatted_articles}
Evaluate the overall sentiment of this news specifically for the company {ticker}.
You MUST respond strictly in the following format:
SCORE: [a single float between -1.0 and 1.0, where -1.0 is extremely negative, 0.0 is neutral, and 1.0 is extremely positive]
REASONING: [a brief 1-sentence reasoning explaining the score]

Do not return any other text, markdown formatting, or headers. Just return the SCORE and REASONING lines.
"""

    if provider == "gemini":
        url = endpoint if endpoint else f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=20)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini API: {str(e)}")
            
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API returned status code {response.status_code}: {response.text}")
            
        try:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini response: {str(e)}. Response raw: {response.text}")
            
    elif provider == "openai":
        url = endpoint if endpoint else "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=20)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {str(e)}")
            
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API returned status code {response.status_code}: {response.text}")
            
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse OpenAI response: {str(e)}. Response raw: {response.text}")
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'")
        
    # Parse text response to extract SCORE and REASONING
    score = 0.0
    reasoning = "Unable to extract reasoning."
    
    for line in text.split("\n"):
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
            
    return {"score": score, "reasoning": reasoning}
