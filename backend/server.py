"""
MIND 新闻推荐后端服务 — Flask API Server

启动: python backend/server.py
端口: 5001

API 端点:
  GET  /api/news              → 全部新闻列表 (newsData 格式)
  GET  /api/news/<id>         → 单篇新闻详情
  GET  /api/stats             → 数据统计
  GET  /api/categories        → 分类列表
  GET  /api/user/<user_id>    → 用户画像
  GET  /api/health            → 健康检查
"""

import json
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)

INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"

# ─── 数据路径 ─────────────────────────────────────────────────────────
ARTICLES_JSON = Path(__file__).resolve().parent / "mind_articles.json"
STATS_JSON = Path(__file__).resolve().parent / "mind_stats.json"

# ─── 全局数据（启动时加载） ─────────────────────────────────────────
DATA = None


def _load_from_json():
    """从预生成的 JSON 加载文章数据（优先方式）"""
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
    """从 MIND TSV 实时加载（开发环境回退）"""
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


# ─── API 端点 ─────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "server": "MIND News Recommendation Backend"})


# ─── 前端页面服务 ─────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(
        os.path.dirname(INDEX_HTML),
        INDEX_HTML.name,
        mimetype="text/html"
    )


@app.route("/<path:filename>")
def serve_static(filename):
    """通用静态文件服务（JS、CSS、图片等）"""
    static_dir = os.path.dirname(INDEX_HTML)
    filepath = os.path.join(static_dir, filename)
    if os.path.isfile(filepath):
        return send_from_directory(static_dir, filename)
    # Fallback: 返回 index.html（SPA 模式）
    return send_from_directory(static_dir, INDEX_HTML.name, mimetype="text/html")


@app.route("/api/news")
def get_news():
    """返回全部新闻文章列表"""
    data = get_data()
    articles = data["articles"]

    # 可选筛选参数
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

    # 移除内部 _ 字段后返回
    result = [{k: v for k, v in a.items() if not k.startswith("_")} for a in filtered]
    return jsonify({"total": len(result), "articles": result})


@app.route("/api/news/<int:article_id>")
def get_article(article_id):
    """返回单篇新闻详情"""
    data = get_data()
    for a in data["articles"]:
        if a["id"] == article_id:
            result = {k: v for k, v in a.items() if not k.startswith("_")}
            return jsonify(result)
    return jsonify({"error": "Article not found"}), 404


@app.route("/api/stats")
def get_stats():
    """返回数据统计"""
    data = get_data()
    return jsonify({
        "stats": data["stats"],
        "raw_news_count": data["total_raw_news"],
        "behavior_records": data["total_behaviors"],
    })


@app.route("/api/categories")
def get_categories():
    """返回所有分类和子主题"""
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
    """返回特定用户的阅读画像"""
    data = get_data()
    profile = data["user_profiles"].get(user_id)
    if not profile:
        return jsonify({"error": "User not found"}), 404
    return jsonify(profile)


@app.route("/api/user-profiles")
def list_user_profiles():
    """列出所有用户 ID"""
    data = get_data()
    return jsonify({
        "total": len(data["user_profiles"]),
        "user_ids": list(data["user_profiles"].keys())[:200],
    })


# ─── 启动 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MIND News Recommendation Backend")
    print("  Endpoints: /api/news, /api/stats, /api/categories, etc.")
    print("=" * 60)
    get_data()  # 预加载数据
    app.run(host="0.0.0.0", port=8096, debug=False)
