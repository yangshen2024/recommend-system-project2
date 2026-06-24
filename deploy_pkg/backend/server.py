"""
MIND News Recommendation Backend — Flask API Server

Start: python backend/server.py
Port: 5001 (behind nginx reverse proxy on port 80)

API Endpoints:
  GET  /api/news              → Full news list (newsData format)
  GET  /api/news/<id>         → Single article detail
  GET  /api/stats             → Data statistics
  GET  /api/categories        → Category list
  GET  /api/user/<user_id>    → User profile
  GET  /api/health            → Health check
  GET  /api/version           → App version
  POST /api/recommend         → Generate baseline recommendations
  POST /api/rerank            → Re-rank recommendation results
  POST /api/feedback          → Record user click/behavior feedback
  GET  /api/users             → List available user IDs and profile summaries
  POST /api/register          → Cold-start user registration
  GET  /api/onboarding        → Get onboarding flow data
  POST /api/onboarding        → Submit onboarding results
  POST /api/recommend/coldstart → Cold-start recommendations
  POST /api/feedback/coldstart  → Cold-start feedback
"""

import json
import sys
import os
import time
import random
from pathlib import Path
from collections import Counter

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)

INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"

# ─── Data Paths ────────────────────────────────────────────────────────
ARTICLES_JSON = Path(__file__).resolve().parent / "mind_articles.json"
STATS_JSON = Path(__file__).resolve().parent / "mind_stats.json"

# ─── Global Data (loaded at startup) ───────────────────────────────────
DATA = None


def _load_from_json():
    """Load articles from pre-generated JSON (preferred method)"""
    if not ARTICLES_JSON.exists():
        return None
    with open(ARTICLES_JSON, encoding="utf-8") as f:
        articles = json.load(f)
    stats = {}
    if STATS_JSON.exists():
        with open(STATS_JSON, encoding="utf-8") as f:
            stats = json.load(f)
    print(f"[Server] Loaded {len(articles)} articles from pre-generated JSON")
    return {
        "articles": articles,
        "stats": stats.get("stats", {}),
        "user_profiles": {},
        "total_raw_news": stats.get("total_raw_news", len(articles)),
        "total_behaviors": stats.get("total_behaviors", 0),
    }


def _load_from_adapter():
    """Load from MIND TSV in real-time (dev environment fallback)"""
    try:
        from backend.mind_adapter import load_all
        data = load_all(max_articles=120, force_reload=True)
        print(f"[Server] Loaded {len(data['articles'])} articles via MIND adapter")
        return data
    except Exception as e:
        print(f"[Server] MIND adapter unavailable: {e}")
        return None


def get_data(force_reload=False):
    global DATA
    if DATA is None or force_reload:
        print("[Server] Loading data...")
        DATA = _load_from_json() or _load_from_adapter()
        if DATA is None:
            print("[Server] FATAL: No data source available")
            DATA = {"articles": [], "stats": {}, "user_profiles": {},
                    "total_raw_news": 0, "total_behaviors": 0}
    return DATA


# ─── API Endpoints ─────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "server": "MIND News Recommendation Backend"})


@app.route("/api/version")
def api_version():
    """Return current app version from VERSION file"""
    version_path = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        ver = version_path.read_text().strip()
    except Exception:
        ver = "unknown"
    return jsonify({"version": ver})


# ─── Frontend Page Serving ─────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(
        os.path.dirname(INDEX_HTML),
        INDEX_HTML.name,
        mimetype="text/html"
    )


@app.route("/<path:filename>")
def serve_static(filename):
    """Generic static file serving (JS, CSS, images, etc.)"""
    static_dir = os.path.dirname(INDEX_HTML)
    filepath = os.path.join(static_dir, filename)
    if os.path.isfile(filepath):
        return send_from_directory(static_dir, filename)
    # Fallback: return index.html (SPA mode)
    return send_from_directory(static_dir, INDEX_HTML.name, mimetype="text/html")


@app.route("/api/news")
def get_news():
    """Return full news article list"""
    data = get_data()
    articles = data["articles"]

    # Optional filter parameters
    category = request.args.get("category")
    subtopic = request.args.get("subtopic")
    lean = request.args.get("lean")
    limit = request.args.get("limit", type=int)

    filtered = articles
    if category:
        filtered = [a for a in filtered if a["category"].lower() == category.lower()]
    if subtopic:
        filtered = [a for a in filtered if a["subtopic"].lower() == subtopic.lower()]
    if lean:
        filtered = [a for a in filtered if a["sourceLean"] == lean.upper()]

    if limit:
        filtered = filtered[:limit]

    # Remove internal _ fields before returning
    result = [{k: v for k, v in a.items() if not k.startswith("_")} for a in filtered]
    return jsonify({"total": len(result), "articles": result})


@app.route("/api/news/<int:article_id>")
def get_article(article_id):
    """Return single article detail"""
    data = get_data()
    for a in data["articles"]:
        if a["id"] == article_id:
            result = {k: v for k, v in a.items() if not k.startswith("_")}
            return jsonify(result)
    return jsonify({"error": "Article not found"}), 404


@app.route("/api/stats")
def get_stats():
    """Return data statistics"""
    data = get_data()
    return jsonify({
        "stats": data["stats"],
        "raw_news_count": data["total_raw_news"],
        "behavior_records": data["total_behaviors"],
    })


@app.route("/api/categories")
def get_categories():
    """Return all categories and subtopics"""
    data = get_data()
    categories = {}
    for a in data["articles"]:
        cat = a["category"]
        sub = a["subtopic"]
        if cat not in categories:
            categories[cat] = set()
        categories[cat].add(sub)
    return jsonify({k: sorted(list(v)) for k, v in categories.items()})


@app.route("/api/user/<user_id>")
def get_user_profile(user_id):
    """Return reading profile for a specific user"""
    data = get_data()
    profile = data["user_profiles"].get(user_id)
    if not profile:
        return jsonify({"error": "User not found"}), 404
    return jsonify(profile)


@app.route("/api/user-profiles")
def list_user_profiles():
    """List all user IDs"""
    data = get_data()
    return jsonify({
        "total": len(data["user_profiles"]),
        "user_ids": list(data["user_profiles"].keys())[:200],
    })


# ─── Re-Ranking Module ─────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from backend.reranker import Reranker
        data = get_data()
        articles = data["articles"]
        _reranker = Reranker(articles, use_embeddings=False)  # Do not force-load sentence-transformers
    return _reranker


# ─── Cold Start Manager ────────────────────────────────────────────────

_coldstart_mgr = None
_registered_users = {}  # {user_id: {static_profile: {...}, feedback_count: int, onboarding_done: bool}}


def _get_coldstart_manager():
    global _coldstart_mgr
    if _coldstart_mgr is None:
        from backend.coldstart import ColdStartManager
        data = get_data()
        _coldstart_mgr = ColdStartManager(data["articles"], reranker=_get_reranker())
    return _coldstart_mgr


# ─── Simulated User Behavior Records ───────────────────────────────────

_user_behaviors = {}  # {user_id: {clicks: [], total_clicks: int, history: [...]}}

# ─── Preset User Definitions ────────────────────────────────────────────
_PRESET_USER_DEFS = {
    "U_TECH": {
        "preferred_leans": ["B"],
        "liked_tags": ["technology", "ai", "science", "tech", "software",
                       "data", "digital", "innovation", "research", "engineering"],
        "categories": ["Technology", "Science"],
        "min_clicks": 20,
    },
    "U_SPORTS": {
        "preferred_leans": ["B"],
        "liked_tags": ["sports", "game", "play", "football", "basketball",
                       "athletic", "team", "league", "championship", "tournament"],
        "categories": ["Sports"],
        "min_clicks": 20,
    },
    "U_GENERAL": {
        "preferred_leans": ["P", "B"],
        "liked_tags": ["news", "politics", "world", "finance", "health",
                       "policy", "economy", "global", "government", "society"],
        "categories": ["News", "World", "Politics", "Finance"],
        "min_clicks": 15,
    },
}


def _init_preset_users():
    """Initialize preset user profiles by selecting matching articles from DATA"""
    data = get_data()
    articles = data["articles"]

    for uid, defn in _PRESET_USER_DEFS.items():
        preferred_cats = defn["categories"]
        liked_tags = defn["liked_tags"]
        preferred_leans = defn["preferred_leans"]
        min_clicks = defn["min_clicks"]

        # Score articles: both category+tag match > category match > tag match
        scored = []
        for a in articles:
            cat_match = a.get("category", "") in preferred_cats
            tag_match = any(t in liked_tags for t in a.get("tags", []))
            if cat_match and tag_match:
                scored.append((a, 3))
            elif cat_match:
                scored.append((a, 2))
            elif tag_match:
                scored.append((a, 1))

        # Sort by score descending, pick top min_clicks with some randomness
        scored.sort(key=lambda x: (-x[1], random.random()))
        clicked_articles = [a for a, _ in scored[:min_clicks]]
        clicked_ids = [a["id"] for a in clicked_articles]

        _user_behaviors[uid] = {
            "user_id": uid,
            "type": "existing",
            "clicks": clicked_ids,
            "total_clicks": len(clicked_ids),
            "history": clicked_ids,
            "preferred_leans": list(preferred_leans),
            "liked_tags": list(liked_tags),
        }
        print(f"[Server] Initialized preset user {uid}: {len(clicked_ids)} clicks")


def _ensure_user(user_id):
    """Ensure user exists (preset user → predefined profile; otherwise simulate)"""
    # Check if already exists (preset or previously initialized)
    if user_id in _user_behaviors:
        return _user_behaviors[user_id]

    if user_id not in _user_behaviors:
        # Randomly decide whether new or existing (75% probability existing)
        is_new = random.random() < 0.25
        data = get_data()
        articles = data["articles"]

        if is_new:
            _user_behaviors[user_id] = {
                "user_id": user_id,
                "type": "new",
                "clicks": [],
                "total_clicks": 0,
                "history": [],
            }
        else:
            # Simulate existing user: randomly generate some historical clicks
            num_clicks = random.randint(5, 30)
            clicked_ids = random.sample(
                [a["id"] for a in articles],
                min(num_clicks, len(articles))
            )
            clicked_articles = [a for a in articles if a["id"] in clicked_ids]
            _user_behaviors[user_id] = {
                "user_id": user_id,
                "type": "existing",
                "clicks": clicked_ids,
                "total_clicks": len(clicked_ids),
                "history": clicked_ids,
                "preferred_leans": _extract_lean_preference(clicked_articles),
                "liked_tags": _extract_liked_tags(clicked_articles),
            }

    return _user_behaviors[user_id]


def _extract_lean_preference(articles):
    """Extract lean distribution from reading history"""
    counts = {"P": 0, "B": 0, "T": 0}
    for a in articles:
        lean = a.get("sourceLean", "B")
        if lean in counts:
            counts[lean] += 1
    total = max(1, sum(counts.values()))
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return [lean for lean, _ in ranked if counts[lean] > total * 0.1]


def _extract_liked_tags(articles):
    """Extract preference tags from reading history"""
    tag_counts = Counter()
    for a in articles:
        for t in a.get("tags", []):
            tag_counts[t] += 1
    return [t for t, _ in tag_counts.most_common(10)]


# ─── Cold Start APIs ───────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register_user():
    """
    User registration — submit static attributes.

    Request JSON:
      {
        "user_id": "U12345",
        "region": "north",          // north/south/east/west/central/overseas
        "device_type": "mobile",    // mobile/pc/tablet
        "age_group": "25-34",       // 18-24/25-34/35-44/45-54/55+
        "gender": "male",           // male/female/other
        "reg_hour": 9               // 0-23
      }

    Response:
      { "user_id", "registered", "static_profile": {...} }
    """
    from backend.coldstart import StaticProfile

    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U" + str(random.randint(10000, 99999)))

    static = StaticProfile(
        region=body.get("region", "east"),
        device_type=body.get("device_type", "mobile"),
        age_group=body.get("age_group", "25-34"),
        gender=body.get("gender", "unknown"),
        reg_hour=int(body.get("reg_hour", 12)),
    )

    _registered_users[user_id] = {
        "static_profile": static,
        "feedback_count": 0,
        "onboarding_done": False,
        "clicked_articles": [],
    }

    # Also initialize behavior tracking
    _user_behaviors[user_id] = {
        "user_id": user_id,
        "type": "new",
        "clicks": [],
        "total_clicks": 0,
        "history": [],
        "static_profile": static.to_dict(),
    }

    return jsonify({
        "user_id": user_id,
        "registered": True,
        "static_profile": static.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/onboarding", methods=["GET"])
def get_onboarding():
    """
    Get new-user onboarding flow data (interest tags + questionnaire).

    Response:
      {
        "category_tags": [...],        // Top-level interest tags
        "subtopic_tags": [...],        // Secondary interest tags
        "questionnaire": [...],        // Guided questionnaire
        "trending_categories": [...],  // Current trending categories
      }
    """
    mgr = _get_coldstart_manager()
    return jsonify(mgr.get_onboarding_data())


@app.route("/api/onboarding", methods=["POST"])
def submit_onboarding():
    """
    Submit onboarding flow results.

    Request JSON:
      {
        "user_id": "U12345",
        "category_tag_ids": ["cat_tech", "cat_finance"],
        "subtopic_tag_ids": ["sub_tech_ai", "sub_economy"],
        "questionnaire_answers": {"q1": "morning", "q2": "deep", "q3": "balanced"}
      }

    Response:
      { "user_id", "profile": {...}, "next_phase": "Hybrid" }
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    mgr = _get_coldstart_manager()

    profile = mgr.submit_onboarding(
        user_id=user_id,
        category_tag_ids=body.get("category_tag_ids"),
        subtopic_tag_ids=body.get("subtopic_tag_ids"),
        questionnaire_answers=body.get("questionnaire_answers"),
    )

    # Update registered user
    if user_id in _registered_users:
        _registered_users[user_id]["onboarding_done"] = True
        _registered_users[user_id]["feedback_count"] = mgr._user_feedback_counts.get(user_id, 4)

    return jsonify({
        "user_id": user_id,
        "profile": {k: v for k, v in profile.items() if k != "user_id"},
        "next_phase": "Hybrid — Skipped pure E&E phase, entering hybrid blending directly",
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/recommend/coldstart", methods=["POST"])
def coldstart_recommend():
    """
    Cold-start recommendation — auto-select strategy based on user phase.

    Request JSON:
      {
        "user_id": "U12345",
        "top_k": 10,
        "use_coldstart": true       // Whether to force cold-start strategy
      }

    Response:
      {
        "user_id", "phase", "phase_label",
        "feedback_count", "transition_progress",
        "recommendations": [...],
        "strategy_breakdown": {...},
        "bandit_stats": {...},
      }
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    top_k = body.get("top_k", 10)
    mgr = _get_coldstart_manager()

    # Get user registration info
    reg = _registered_users.get(user_id, {})
    static_profile = reg.get("static_profile")
    feedback_count = reg.get("feedback_count",
                             len(_user_behaviors.get(user_id, {}).get("clicks", [])))

    result = mgr.recommend(
        user_id=user_id,
        static_profile=static_profile,
        feedback_count=feedback_count,
        top_k=top_k,
    )

    # Clean output fields
    clean = []
    for a in result["recommendations"]:
        item = {k: v for k, v in a.items()
                if not k.startswith("_entities") and not k.startswith("_mind")}
        item["score"] = a.get("_baseline_score", a.get("_popularity_score", 0))
        item["rank"] = a.get("_rank", 0)
        clean.append(item)

    return jsonify({
        "user_id": user_id,
        "phase": result["phase"],
        "phase_label": result["phase_label"],
        "feedback_count": result["feedback_count"],
        "transition_progress": result["transition_progress"],
        "recommendations": clean,
        "strategy_breakdown": result["strategy_breakdown"],
        "bandit_stats": result["bandit_stats"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/feedback/coldstart", methods=["POST"])
def coldstart_feedback():
    """
    Record cold-start user feedback to drive phase progression.

    Request JSON:
      { "user_id": "U12345", "article_id": 5, "action": "click" }
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    article_id = body.get("article_id")
    action = body.get("action", "click")

    mgr = _get_coldstart_manager()

    # Find full article data
    data = get_data()
    id_to_article = {a["id"]: a for a in data["articles"]}
    article = id_to_article.get(article_id)

    # Record to ColdStartManager (update E&E)
    mgr.record_feedback(user_id, article_id, action, article)

    # Update registered user state
    if user_id in _registered_users:
        _registered_users[user_id]["feedback_count"] = \
            mgr._user_feedback_counts.get(user_id, 0)
        if article_id and article_id not in _registered_users[user_id].get("clicked_articles", []):
            _registered_users[user_id].setdefault("clicked_articles", []).append(article_id)

    # Update global behavior records
    user = _ensure_user(user_id)
    if article_id and article_id not in user["clicks"]:
        user["clicks"].append(article_id)
        user["total_clicks"] = len(user["clicks"])
        user["type"] = "existing" if user["total_clicks"] > 0 else "new"

    feedback_count = mgr._user_feedback_counts.get(user_id, 0)
    phase = mgr.determine_phase(feedback_count)

    return jsonify({
        "user_id": user_id,
        "article_id": article_id,
        "action": action,
        "feedback_count": feedback_count,
        "current_phase": phase,
        "phase_label": mgr.PHASE_LABELS[phase],
        "transition_progress": mgr._transition_progress(feedback_count),
        "bandit_explore_rate": mgr.bandit.get_exploration_rate(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


# ─── Recommendation / Re-rank APIs ─────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Generate baseline recommendation results for a user.

    Request JSON:
      { "user_id": "U12345", "top_k": 10, "preset_user_id": "U_TECH" }

    If preset_user_id is provided, use that preset user's profile for recommendations.

    Response:
      { "user_id", "user_type", "baseline": [...], "total_available": int, "using_preset": str|null }
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    top_k = body.get("top_k", 10)
    preset_user_id = body.get("preset_user_id", None)

    # If a preset user is specified, use its behavior profile
    using_preset = None
    if preset_user_id and preset_user_id in _PRESET_USER_DEFS:
        # Ensure preset user is initialized
        if preset_user_id not in _user_behaviors:
            _init_preset_users()
        if preset_user_id in _user_behaviors:
            user = _user_behaviors[preset_user_id]
            using_preset = preset_user_id
        else:
            user = _ensure_user(user_id)
    else:
        user = _ensure_user(user_id)

    reranker = _get_reranker()

    # Build user profile for reranker
    user_profile = {
        "preferred_leans": user.get("preferred_leans", []),
        "liked_tags": user.get("liked_tags", []),
    }
    baseline = reranker.baseline_rank(user_profile=user_profile, top_k=top_k)

    # Remove internal fields
    clean = []
    for a in baseline:
        item = {k: v for k, v in a.items() if not k.startswith("_entities") and not k.startswith("_mind")}
        # Preserve scoring info
        item["score"] = a.get("_baseline_score", 0)
        item["rank"] = a.get("_rank", 0)
        clean.append(item)

    return jsonify({
        "user_id": user_id,
        "user_type": user["type"],
        "total_clicks": user.get("total_clicks", 0),
        "baseline": clean,
        "total_available": len(get_data()["articles"]),
        "using_preset": using_preset,
        "preferred_leans": user.get("preferred_leans", []),
        "liked_tags": user.get("liked_tags", []),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/rerank", methods=["POST"])
def rerank():
    """
    Re-rank baseline recommendation results.

    Request JSON:
      {
        "user_id": "U12345",
        "baseline_ids": [3,17,8,...],    // Article ID list from previous step
        "method": "entity_mmr",           // entity_mmr | embedding_mmr | hybrid_mmr | calibrated
        "lam": 0.6,                       // MMR λ (0~1)
        "alpha": 0.5,                     // Hybrid blending ratio
        "top_k": 10
      }

    Response:
      { "user_id", "method", "lam", "baseline": [...], "reranked": [...], "coverage_stats": {...} }
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    baseline_ids = body.get("baseline_ids", [])
    method = body.get("method", "entity_mmr")
    lam = float(body.get("lam", 0.7))
    alpha = float(body.get("alpha", 0.5))
    top_k = body.get("top_k", 10)
    preset_user_id = body.get("preset_user_id", None)

    # Use preset user profile if specified
    if preset_user_id and preset_user_id in _PRESET_USER_DEFS:
        if preset_user_id not in _user_behaviors:
            _init_preset_users()
        user = _user_behaviors.get(preset_user_id)
    else:
        user = _ensure_user(user_id)

    reranker = _get_reranker()
    all_articles = get_data()["articles"]
    id_to_article = {a["id"]: a for a in all_articles}

    # 1. If baseline_ids provided, rebuild baseline list
    if baseline_ids:
        baseline = []
        for bid in baseline_ids:
            if bid in id_to_article:
                a = dict(id_to_article[bid])
                a["_baseline_score"] = round(random.uniform(0.5, 0.95), 4)
                baseline.append(a)
        if not baseline:
            baseline = reranker.baseline_rank(top_k=top_k)
    else:
        baseline = reranker.baseline_rank(top_k=top_k)

    baseline = baseline[:top_k]

    # 2. Re-rank
    result = reranker.rerank(baseline, method=method, lam=lam, alpha=alpha)

    # 3. Clean output
    def clean_articles(articles):
        cleaned = []
        for a in articles:
            item = {k: v for k, v in a.items()
                    if not k.startswith("_entities") and not k.startswith("_mind")}
            item["score"] = a.get("_reranked_score", a.get("_baseline_score", 0))
            item["rank"] = a.get("_reranked_rank", a.get("_rank", 0))
            # Surface explanation fields
            item["clusterName"] = a.get("_cluster_name", "")
            item["clusterId"] = a.get("_cluster_group_id", -1)
            item["diversityGain"] = a.get("_diversity_gain", 0)
            item["reason"] = a.get("_reason", "")
            cleaned.append(item)
        return cleaned

    return jsonify({
        "user_id": user_id,
        "method": method,
        "lam": lam,
        "alpha": alpha,
        "baseline": clean_articles(baseline),
        "reranked": clean_articles(result["reranked"]),
        "coverage_stats": result["coverage_stats"],
        "using_preset": preset_user_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Record user click/behavior feedback.

    Request JSON:
      { "user_id": "U12345", "article_id": 5, "action": "click" }

    actions: "click" | "like" | "dislike" | "read"
    """
    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id", "U12345")
    article_id = body.get("article_id")
    action = body.get("action", "click")

    user = _ensure_user(user_id)

    if article_id and article_id not in user["clicks"]:
        user["clicks"].append(article_id)
        user["total_clicks"] = len(user["clicks"])
        user["type"] = "existing" if user["total_clicks"] > 0 else "new"

    return jsonify({
        "user_id": user_id,
        "user_type": user["type"],
        "total_clicks": user["total_clicks"],
        "last_action": action,
        "article_id": article_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    })


@app.route("/api/users")
def list_users():
    """List available user IDs and profile summaries"""
    data = get_data()
    profiles = data.get("user_profiles", {})
    user_list = []
    for uid, profile in list(profiles.items())[:20]:
        user_list.append({
            "user_id": uid,
            "history_size": profile.get("history_size", 0),
            "lean_distribution": profile.get("lean_distribution", {}),
        })
    return jsonify({
        "total": len(user_list),
        "users": user_list,
    })


@app.route("/api/users/presets")
def list_preset_users():
    """List preset user profiles with their stats"""
    # Ensure preset users are initialized
    if not _user_behaviors:
        _init_preset_users()

    presets = []
    for uid, defn in _PRESET_USER_DEFS.items():
        user = _user_behaviors.get(uid, {})
        presets.append({
            "user_id": uid,
            "name": uid,
            "label": uid.replace("U_", "").replace("_", " ").title(),
            "categories": defn["categories"],
            "preferred_leans": defn["preferred_leans"],
            "total_clicks": user.get("total_clicks", defn["min_clicks"]),
            "clicked_ids": user.get("clicks", [])[:5],  # sample of clicked articles
            "is_preset": True,
        })

    return jsonify({
        "total": len(presets),
        "presets": presets,
    })


@app.route("/api/articles/clusters")
def get_article_clusters():
    """Return cluster/group info for all articles"""
    reranker = _get_reranker()
    data = get_data()
    articles = data["articles"]

    # Build cluster mapping
    clusters = {}
    for a in articles:
        gid = a.get("_cluster_group_id", -1)
        if gid < 0:
            continue
        if gid not in clusters:
            clusters[gid] = {"cluster_id": gid, "article_ids": [], "articles": []}
        clusters[gid]["article_ids"].append(a["id"])
        clusters[gid]["articles"].append({
            "id": a["id"],
            "title": a.get("title", ""),
            "subtopic": a.get("subtopic", ""),
            "source": a.get("source", ""),
            "sourceLean": a.get("sourceLean", "B"),
            "tags": a.get("tags", []),
        })

    # Only return clusters with 2+ articles
    result = [v for v in clusters.values() if len(v["article_ids"]) >= 2]
    result.sort(key=lambda c: -len(c["article_ids"]))

    return jsonify({
        "total_clusters": len(result),
        "clusters": result,
    })


@app.route("/api/search")
def search_articles():
    """
    Search/filter articles by keyword, category, subtopic, lean.

    Query params:
      q: keyword search (matches title, summary, tags)
      category: filter by category
      subtopic: filter by subtopic
      lean: filter by sourceLean (P/B/T)
      limit: max results (default 20)
    """
    data = get_data()
    articles = data["articles"]

    q = request.args.get("q", "").lower()
    category = request.args.get("category", "").lower()
    subtopic = request.args.get("subtopic", "").lower()
    lean = request.args.get("lean", "").upper()
    limit = request.args.get("limit", 20, type=int)

    results = []
    for a in articles:
        if category and a.get("category", "").lower() != category:
            continue
        if subtopic and a.get("subtopic", "").lower() != subtopic:
            continue
        if lean and a.get("sourceLean", "") != lean:
            continue
        if q:
            title = (a.get("title", "") or "").lower()
            summary = (a.get("summary", "") or "").lower()
            tags = " ".join(t.get("t", t) if isinstance(t, dict) else str(t) for t in a.get("tags", [])).lower()
            if q not in title and q not in summary and q not in tags:
                continue
        results.append({k: v for k, v in a.items() if not k.startswith("_")})

    results = results[:limit]
    return jsonify({
        "query": q,
        "category": category or None,
        "subtopic": subtopic or None,
        "lean": lean or None,
        "total": len(results),
        "articles": results,
    })


# ─── Startup ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MIND News Recommendation Backend")
    print("  Endpoints: /api/news, /api/stats, /api/categories,")
    print("             /api/recommend, /api/rerank, /api/feedback,")
    print("             /api/register, /api/onboarding, coldstart APIs")
    print("=" * 60)
    get_data()  # Preload data
    # Initialize preset users
    _init_preset_users()
    # Pre-initialize reranker
    _get_reranker()
    app.run(host="0.0.0.0", port=5001, debug=False)
