"""
News Fetcher Configuration

Set environment variables before running:
    export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
    export NEWSAPI_KEY="xxxxxxxxxxxxxxxxxxxx"  # optional, for NewsAPI
"""

import os
from datetime import timedelta

# Load .env file if present
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if not os.path.isfile(_ENV_PATH):
    _ENV_PATH = os.path.join(os.getcwd(), ".env")
if os.path.isfile(_ENV_PATH):
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k not in os.environ:
                    os.environ[_k] = _v

# ── DeepSeek API ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT_SEC = 60
DEEPSEEK_MAX_RETRIES = 3
DEEPSEEK_RATE_LIMIT_RPM = 30  # requests per minute

# ── NewsAPI (optional premium source) ─────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE_URL = "https://newsapi.org/v2"
NEWSAPI_TIMEOUT_SEC = 30

# ── Time Window ───────────────────────────────────────────────
LOOKBACK_DAYS = 7
LOOKBACK_DELTA = timedelta(days=LOOKBACK_DAYS)

# ── News Sources (RSS Feeds) ──────────────────────────────────
RSS_FEEDS = [
    {
        "name": "BBC News",
        "url": "https://feeds.bbci.co.uk/news/rss.xml",
        "lean": "C",
    },
    {
        "name": "Reuters",
        "url": "https://news.google.com/rss/search?q=reuters+politics&hl=en-US&gl=US&ceid=US:en",
        "lean": "C",
    },
    {
        "name": "NPR",
        "url": "https://feeds.npr.org/1001/rss.xml",
        "lean": "C",
    },
    {
        "name": "Associated Press",
        "url": "https://news.google.com/rss/search?q=ap+news+politics&hl=en-US&gl=US&ceid=US:en",
        "lean": "C",
    },
    {
        "name": "The Guardian",
        "url": "https://www.theguardian.com/world/rss",
        "lean": "L",
    },
    {
        "name": "The Hill",
        "url": "https://thehill.com/news/feed/",
        "lean": "C",
    },
    {
        "name": "Politico",
        "url": "https://rss.politico.com/politics-news.xml",
        "lean": "C",
    },
    {
        "name": "Fox News",
        "url": "https://moxie.foxnews.com/google-publisher/politics.xml",
        "lean": "R",
    },
]

# ── Maximum items per feed to avoid overwhelming the API ──────
MAX_ITEMS_PER_FEED = 20

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "weekly_news.json")
OUTPUT_FRONTEND_JS = os.path.join(OUTPUT_DIR, "newsdata_generated.js")
