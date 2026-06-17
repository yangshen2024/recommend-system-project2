"""
News Scraping Module

Fetches recent news articles from multiple RSS feed sources,
with fallback to NewsAPI when an API key is available.
Returns a normalized list of article dicts.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from news_fetcher.config import (
    RSS_FEEDS,
    NEWSAPI_KEY,
    NEWSAPI_BASE_URL,
    NEWSAPI_TIMEOUT_SEC,
    LOOKBACK_DAYS,
    MAX_ITEMS_PER_FEED,
)

logger = logging.getLogger("scraper")


@dataclass
class RawArticle:
    """Normalized article before DeepSeek processing."""

    title: str
    url: str
    source_name: str
    source_lean: str  # L / C / R
    published_at: Optional[datetime] = None
    summary: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                (self.title + self.url).encode("utf-8")
            ).hexdigest()[:16]


# ── RSS Feed Fetcher ───────────────────────────────────────
class RSSNewsFetcher:
    """Fetch and normalize articles from configured RSS feeds."""

    def __init__(self, http_client: Optional[httpx.Client] = None) -> None:
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(NEWSAPI_TIMEOUT_SEC),
            follow_redirects=True,
        )

    def fetch_all(self) -> list[RawArticle]:
        """Fetch from all configured RSS sources and return deduplicated articles."""
        all_articles: list[RawArticle] = []
        seen_hashes: set[str] = set()

        for feed_cfg in RSS_FEEDS:
            try:
                articles = self._fetch_feed(feed_cfg)
                for art in articles:
                    if art.content_hash not in seen_hashes:
                        seen_hashes.add(art.content_hash)
                        all_articles.append(art)
                logger.info(
                    "Fetched %d articles from %s", len(articles), feed_cfg["name"]
                )
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", feed_cfg["name"], exc)

        logger.info("Total unique articles across all feeds: %d", len(all_articles))
        return all_articles

    def _fetch_feed(self, feed_cfg: dict) -> list[RawArticle]:
        """Fetch and parse a single RSS feed, returning normalized articles."""
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        resp = self._client.get(feed_cfg["url"])
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        if feed.bozo and not feed.entries:
            logger.warning(
                "Feed %s may be malformed (bozo=1), but proceeding with available entries",
                feed_cfg["name"],
            )

        articles: list[RawArticle] = []
        for entry in feed.entries[:MAX_ITEMS_PER_FEED]:
            published = self._extract_date(entry)

            # Filter by time window (if we have a date)
            if published and (cutoff - published).days > LOOKBACK_DAYS:
                continue

            summary = self._extract_summary(entry)
            articles.append(
                RawArticle(
                    title=self._clean_text(entry.get("title", "")),
                    url=entry.get("link", ""),
                    source_name=feed_cfg["name"],
                    source_lean=feed_cfg.get("lean", "C"),
                    published_at=published,
                    summary=summary,
                )
            )

        return articles

    @staticmethod
    def _extract_date(entry: dict) -> Optional[datetime]:
        """Parse publication date from RSS entry."""
        for field in ("published_parsed", "updated_parsed"):
            tp = entry.get(field)
            if tp:
                try:
                    return datetime(*tp[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    pass

        # Fallback: parse string fields
        for field in ("published", "updated"):
            raw = entry.get(field, "")
            if raw:
                try:
                    return date_parser.parse(raw).astimezone(timezone.utc)
                except (ValueError, OverflowError):
                    pass

        return None

    @staticmethod
    def _extract_summary(entry: dict) -> str:
        """Extract summary text, stripping HTML tags."""
        raw = entry.get("summary") or entry.get("description") or ""
        if raw:
            soup = BeautifulSoup(raw, "lxml")
            return RSSNewsFetcher._clean_text(soup.get_text(separator=" ", strip=True))
        return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and strip junk characters."""
        return " ".join(text.split())


# ── NewsAPI Fetcher (optional premium source) ──────────────
class NewsAPIFetcher:
    """Fetcher using the NewsAPI.org service (requires API key)."""

    EVERYTHING_URL = f"{NEWSAPI_BASE_URL}/everything"

    def __init__(self, http_client: Optional[httpx.Client] = None) -> None:
        if not NEWSAPI_KEY:
            raise RuntimeError("NEWSAPI_KEY is not set.")
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(NEWSAPI_TIMEOUT_SEC),
        )

    def fetch_recent(
        self, query: str = "politics", page_size: int = 50
    ) -> list[RawArticle]:
        """Fetch articles from the past 7 days via NewsAPI."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
            "language": "en",
            "apiKey": NEWSAPI_KEY,
        }

        resp = self._http.get(self.EVERYTHING_URL, params=params)
        resp.raise_for_status()
        body = resp.json()

        if body.get("status") != "ok":
            logger.error("NewsAPI error: %s", body.get("message", "unknown"))
            return []

        articles: list[RawArticle] = []
        for item in body.get("articles", []):
            pub_date = None
            if item.get("publishedAt"):
                try:
                    pub_date = date_parser.parse(item["publishedAt"]).astimezone(
                        timezone.utc
                    )
                except (ValueError, OverflowError):
                    pass

            articles.append(
                RawArticle(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source_name=item.get("source", {}).get("name", "Unknown"),
                    source_lean="C",  # Unknown lean, DeepSeek will classify
                    published_at=pub_date,
                    summary=(item.get("description") or ""),
                )
            )

        logger.info("NewsAPI returned %d articles for query '%s'", len(articles), query)
        return articles


# ── Unified Entry Point ────────────────────────────────────
def fetch_all_news(newsapi_query: str = "politics") -> list[RawArticle]:
    """
    Fetch news from all available sources.
    Returns deduplicated list sorted by published_at (newest first).
    """

    client = httpx.Client(
        timeout=httpx.Timeout(NEWSAPI_TIMEOUT_SEC),
        follow_redirects=True,
    )

    all_articles: list[RawArticle] = []

    # ── RSS feeds (always available) ────────────────────────
    try:
        rss = RSSNewsFetcher(http_client=client)
        all_articles.extend(rss.fetch_all())
    except Exception as exc:
        logger.error("RSS fetcher failed: %s", exc)

    # ── NewsAPI (optional) ──────────────────────────────────
    if NEWSAPI_KEY:
        try:
            newsapi = NewsAPIFetcher(http_client=client)
            all_articles.extend(newsapi.fetch_recent(query=newsapi_query))
        except Exception as exc:
            logger.error("NewsAPI fetcher failed: %s", exc)

    # ── Deduplicate by content hash ─────────────────────────
    seen: set[str] = set()
    unique: list[RawArticle] = []
    for art in all_articles:
        if art.content_hash not in seen:
            seen.add(art.content_hash)
            unique.append(art)

    # ── Sort newest first ───────────────────────────────────
    unique.sort(
        key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    logger.info("Final deduplicated count: %d articles", len(unique))
    client.close()
    return unique
