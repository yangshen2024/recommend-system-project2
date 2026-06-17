"""
Processing Pipeline

Orchestrates the full flow:
  1. Fetch raw articles from news sources
  2. Pass batches to DeepSeek for:
     - Content summarization
     - Political-lean classification
     - Factuality / relevance filtering
     - Multi-perspective headline generation
  3. Output cleaned, sorted article list
"""

import json
import logging
import os
import sys
import time
from typing import Any

# Load .env before config imports resolve API keys
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if not os.path.isfile(_ENV_PATH):
    # Fallback: try relative to CWD
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
else:
    import warnings
    warnings.warn(f".env not found at {_ENV_PATH}")

from news_fetcher.config import OUTPUT_DIR
from news_fetcher.deepseek_client import DeepSeekClient
from news_fetcher.news_scraper import RawArticle, fetch_all_news

logger = logging.getLogger("pipeline")


# ── DeepSeek Prompts ──────────────────────────────────────

SYSTEM_ANALYZE = """You are a professional news analyst. Your task is to process raw news articles and produce structured JSON output.

For each article, you MUST return **only valid JSON** — no markdown fences, no extra commentary.

Output this exact JSON structure for each article:
{
  "id": <sequential integer starting from 1>,
  "title": "<cleaned, concise headline>",
  "summary": "<one-paragraph factual summary, 2-3 sentences>",
  "category": "Politics",
  "subtopic": "<one of: Elections, Policy, Diplomacy, National Security, Immigration, Governance>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "is_credible": <true or false>,
  "credibility_reason": "<brief reason if not credible>",
  "political_lean": "<L, C, or R>",
  "lean_confidence": <0.0 to 1.0>,
  "perspectives": {
    "left": {
      "headline": "<how a left/progressive outlet would frame this>",
      "source": "The Progressive Times",
      "summary": "<2 sentences from left perspective>"
    },
    "center": {
      "headline": "<neutral, factual framing>",
      "source": "Associated Press",
      "summary": "<2 sentences from center perspective>"
    },
    "right": {
      "headline": "<how a right/conservative outlet would frame this>",
      "source": "The National Review",
      "summary": "<2 sentences from right perspective>"
    }
  }
}

Rules:
- Only include articles that are REAL NEWS about politics/policy. Exclude opinion pieces, listicles, sponsored content, entertainment gossip, and sports.
- If an article is NOT credible or is clearly fake/misinformation, set is_credible=false and provide a reason.
- For the subtopic, pick the single most applicable category from the list above.
- The perspectives.left, perspectives.center, perspectives.right MUST reflect how outlets with those leanings would cover the SAME event — different framing, same underlying facts.
- All text MUST be in English."""


def build_batch_prompt(articles: list[RawArticle], start_id: int) -> str:
    """Build the user prompt for a batch of articles."""
    lines = [
        f"Analyze the following {len(articles)} news articles. Return a JSON array of article objects.\n",
        "RAW ARTICLES:",
        "---",
    ]
    for i, art in enumerate(articles):
        lines.append(f"[Article {start_id + i}]")
        lines.append(f"Source: {art.source_name} (known lean: {art.source_lean})")
        lines.append(f"Headline: {art.title}")
        lines.append(f"Summary: {art.summary[:300]}")
        pub = art.published_at.strftime("%Y-%m-%d %H:%M UTC") if art.published_at else "unknown"
        lines.append(f"Published: {pub}")
        lines.append(f"URL: {art.url}")
        lines.append("---")
    return "\n".join(lines)


# ── Batch Processor ───────────────────────────────────────
def process_articles(
    articles: list[RawArticle],
    batch_size: int = 8,
    sleep_between_batches: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Send articles to DeepSeek in batches for analysis.

    Returns a flat list of processed article dicts (credible only, sorted by id).
    """
    if not articles:
        logger.warning("No articles to process")
        return []

    client: DeepSeekClient
    try:
        client = DeepSeekClient()
    except RuntimeError as exc:
        logger.error(str(exc))
        raise

    results: list[dict[str, Any]] = []
    total = len(articles)

    for batch_start in range(0, total, batch_size):
        batch = articles[batch_start : batch_start + batch_size]
        batch_id_offset = batch_start
        logger.info(
            "Processing batch %d-%d / %d articles",
            batch_start + 1,
            min(batch_start + batch_size, total),
            total,
        )

        try:
            prompt = build_batch_prompt(batch, batch_id_offset + 1)
            parsed = client.extract_structured(
                system_prompt=SYSTEM_ANALYZE,
                user_prompt=prompt,
                max_tokens=4096,
            )
            batch_results = _normalize_output(parsed)
            credible = [r for r in batch_results if r.get("is_credible", True)]
            filtered = len(batch_results) - len(credible)
            if filtered:
                logger.info(
                    "Batch: %d results, %d filtered out (not credible)",
                    len(batch_results),
                    filtered,
                )
            results.extend(credible)

        except Exception as exc:
            logger.error("Batch processing failed: %s", exc)
            # Fallback: keep raw articles with minimal processing
            for art in batch:
                results.append(_fallback_article(art, batch_id_offset))

        # Rate-limit between batches
        if batch_start + batch_size < total:
            time.sleep(sleep_between_batches)

    client.close()

    # Final sort by id (which reflects original order)
    results.sort(key=lambda r: r.get("id", 0))
    return results


def _normalize_output(parsed: Any) -> list[dict[str, Any]]:
    """Handle both single-object and array responses."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Could be {"articles": [...]} or a single article
        if "articles" in parsed:
            return parsed["articles"]
        # Single article wrapped
        return [parsed]
    logger.warning("Unexpected DeepSeek output type: %s", type(parsed))
    return []


def _fallback_article(art: RawArticle, offset: int) -> dict[str, Any]:
    """Create a minimal article dict when DeepSeek processing fails."""
    return {
        "id": offset + 1,
        "title": art.title,
        "summary": art.summary[:300],
        "category": "Politics",
        "subtopic": "Policy",
        "tags": ["politics"],
        "source": art.source_name,
        "url": art.url,
        "is_credible": True,
        "political_lean": art.source_lean,
        "lean_confidence": 0.5,
        "perspectives": {
            "left": {
                "headline": art.title,
                "source": "The Progressive Times",
                "summary": art.summary[:200],
            },
            "center": {
                "headline": art.title,
                "source": "Associated Press",
                "summary": art.summary[:200],
            },
            "right": {
                "headline": art.title,
                "source": "The National Review",
                "summary": art.summary[:200],
            },
        },
    }


# ── Save Output ───────────────────────────────────────────
def save_results(results: list[dict[str, Any]], output_path: str) -> str:
    """Save processed articles to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d articles to %s", len(results), output_path)
    return output_path


# ── Main Entry Point ──────────────────────────────────────
def run_pipeline(
    newsapi_query: str = "politics",
    output_path: str = "",
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """
    Run the full pipeline: fetch → process → save → return.

    Args:
        newsapi_query: Search query for NewsAPI (if key is set).
        output_path: Where to save JSON output. Defaults to config.OUTPUT_JSON.
        batch_size: Articles per DeepSeek API call.

    Returns:
        List of processed article dicts (credible only, newest first).
    """
    # ── Step 1: Fetch ──
    logger.info("=" * 50)
    logger.info("STEP 1: Fetching news articles from all sources...")
    raw_articles = fetch_all_news(newsapi_query=newsapi_query)
    logger.info("Fetched %d raw articles", len(raw_articles))

    if not raw_articles:
        logger.warning("No articles fetched. Exiting.")
        return []

    # ── Step 2: Process via DeepSeek ──
    logger.info("STEP 2: Processing articles through DeepSeek...")
    clean_articles = process_articles(raw_articles, batch_size=batch_size)

    # ── Step 3: Sort newest first ──
    clean_articles.sort(
        key=lambda a: a.get("id", 0), reverse=False
    )  # id order = original time order

    # ── Step 4: Save ──
    save_path = output_path or os.path.join(OUTPUT_DIR, "weekly_news.json")
    save_results(clean_articles, save_path)

    logger.info("Pipeline complete: %d credible articles", len(clean_articles))
    return clean_articles


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure .env is loaded before imports resolve config
    _ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(_ENV_PATH):
        with open(_ENV_PATH) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                    if _k not in os.environ:
                        os.environ[_k] = _v

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)-10s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Fetch and filter real news via DeepSeek")
    parser.add_argument(
        "--query",
        default="politics",
        help="NewsAPI search query (default: 'politics')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Articles per DeepSeek API call (default: 8)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: output/weekly_news.json)",
    )
    args = parser.parse_args()

    try:
        results = run_pipeline(
            newsapi_query=args.query,
            output_path=args.output,
            batch_size=args.batch_size,
        )
        print(f"\n✓ Pipeline complete: {len(results)} credible articles saved.")
    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
