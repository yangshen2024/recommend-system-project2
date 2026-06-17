"""
News Fetcher — Real-time news scraping + DeepSeek AI processing pipeline.

Usage:
    # Set API key
    export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"

    # Optional: NewsAPI premium source
    export NEWSAPI_KEY="xxxxxxxxxxxxxxxxxxxx"

    # Run pipeline
    python news_fetcher/pipeline.py

    # Or run with options
    python news_fetcher/pipeline.py --query "politics" --batch-size 8

    # Generate frontend HTML with live data
    python news_fetcher/pipeline.py --query "politics"
    python -c "
from news_fetcher.pipeline import run_pipeline
from news_fetcher.frontend_formatter import generate_full_html
articles = run_pipeline()
generate_full_html(articles, 'index.html', 'output/index_live.html')
"
"""
