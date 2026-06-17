#!/usr/bin/env python3
"""Wrapper: loads .env, then runs the pipeline."""
import os, sys

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ[k] = v

print(f"✓ Loaded .env: DEEPSEEK_API_KEY={'SET' if os.environ.get('DEEPSEEK_API_KEY') else 'MISSING'}")
print(f"✓ Loaded .env: NEWSAPI_KEY={'SET' if os.environ.get('NEWSAPI_KEY') else 'MISSING'}")

# Now run the pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from news_fetcher.pipeline import run_pipeline

results = run_pipeline(
    newsapi_query="politics",
    batch_size=6,
)
print(f"\n✓ Pipeline complete: {len(results)} credible articles saved.")
