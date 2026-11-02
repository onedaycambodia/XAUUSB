# XAUUSD Analyzer

Website that pulls gold price data and fresh news, then asks Claude for a BUY / SELL / NEUTRAL view with reasons, risks and key levels.

## Features
- Live gold price, RSI, MACD, EMA20/50, ATR (yfinance)
- News headlines from Google News RSS (gold, Fed, USD, geopolitics)
- AI analysis via Claude API (Khmer or English)
- Price chart, 5-minute cache to save API credits

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```
Open http://localhost:8000

## Deploy (Railway)
1. Push this repo to GitHub
2. Railway → New Project → Deploy from GitHub repo
3. Add variable `ANTHROPIC_API_KEY`
4. Railway uses the `Procfile` automatically

## Disclaimer
For information only, not financial advice. AI cannot predict markets. Use stop-loss and risk management.
