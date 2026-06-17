"""
MIND Dataset Adapter — 将 MIND 新闻数据转换为前端所需的 newsData 格式。

MIND news.tsv 字段: news_id, category, subcategory, title, abstract
MIND behaviors.tsv 字段: index, user_id, time, history(空格分隔新闻ID), impressions(新闻ID-标签对)

映射为前端格式:
  id, title, summary, category, subtopic, tags, source, time, thumb, url,
  sourceLean (P/B/T), perspectives (left/center/right)
"""

import re
import json
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────────────
MIND_BASE = Path(__file__).resolve().parent.parent / "Recommender-Project-MIND"
NEWS_PATH = MIND_BASE / "mind_rec" / "MINDsmall_train" / "MINDsmall_train" / "news.tsv"
BEHAVIORS_PATH = MIND_BASE / "mind_rec" / "MINDsmall_train" / "MINDsmall_train" / "behaviors.tsv"
DEV_NEWS_PATH = MIND_BASE / "mind_rec" / "MINDsmall_dev" / "MINDsmall_dev" / "news.tsv"

MAX_ARTICLES = 120            # 最终输出的文章数上限
MIN_ABSTRACT_LEN = 20         # 摘要最短字符（太短的跳过）

# 哪些 MIND category 更适合政治倾向标注
POLITICAL_CATEGORIES = {"news", "finance", "weather"}

# 左右派关键词（英文）
PROGRESSIVE_WORDS = [
    "climate", "equality", "diversity", "inclusive", "renewable", "green",
    "progressive", "reform", "healthcare", "education", "worker", "union",
    "immigrant", "refugee", "lgbtq", "abortion", "feminist", "environment",
    "sustainable", "welfare", "minimum wage", "gun control", "voting rights",
    "racial", "social justice", "medicare", "transgender", "climate change",
    "carbon", "clean energy", "net zero", "human rights", "activist"
]

TRADITIONAL_WORDS = [
    "military", "defense", "border", "traditional", "conservative", "patriot",
    "national security", "veteran", "law enforcement", "police", "second amendment",
    "tax cut", "deregulation", "free market", "religious", "christian", "church",
    "family values", "gun rights", "pro-life", "drill", "oil", "sovereignty",
    "constitutional", "fiscal", "small government", "states rights",
    "homeland", "troop", "navy", "air force", "marine"
]

# 媒体源映射
PROGRESSIVE_SOURCES = [
    "The Progressive Times", "MSNBC", "The Guardian", "Vox", "Mother Jones",
    "The Nation", "Democracy Now", "HuffPost"
]
BALANCED_SOURCES = [
    "Associated Press", "Reuters", "BBC News", "NPR", "PBS NewsHour",
    "ABC News", "CBS News", "USA Today"
]
TRADITIONAL_SOURCES = [
    "The National Review", "Fox News", "The Wall Street Journal", "The Hill",
    "The Washington Times", "Daily Caller", "Breitbart", "Newsmax"
]

# 相对时间列表
RELATIVE_TIMES = [
    "10 minutes ago", "30 minutes ago", "1 hour ago", "2 hours ago",
    "3 hours ago", "4 hours ago", "5 hours ago", "6 hours ago",
    "8 hours ago", "10 hours ago", "12 hours ago", "18 hours ago",
    "1 day ago", "2 days ago", "3 days ago"
]

# 政治子主题
POLITICS_SUBTOPICS = [
    "Elections", "Policy", "Diplomacy", "National Security",
    "Immigration", "Governance", "Economy", "Healthcare",
    "Education", "Environment", "Technology", "Civil Rights"
]


def load_news(news_path=NEWS_PATH):
    """加载 MIND news.tsv，返回 {news_id: {category, subcategory, title, abstract}}"""
    news = {}
    path = Path(news_path)
    if not path.exists():
        print(f"[WARN] news.tsv not found at {path}")
        return news

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            nid, cat, subcat, title, abstract = parts[0], parts[1], parts[2], parts[3], parts[4]
            if len(abstract.strip()) < MIN_ABSTRACT_LEN:
                continue
            news[nid] = {
                "category": cat,
                "subcategory": subcat,
                "title": title.strip(),
                "abstract": abstract.strip(),
            }
    print(f"[MIND] Loaded {len(news)} articles from {path}")
    return news


def load_behaviors(behaviors_path=BEHAVIORS_PATH):
    """加载 MIND behaviors.tsv，返回用户行为列表"""
    rows = []
    path = Path(behaviors_path)
    if not path.exists():
        print(f"[WARN] behaviors.tsv not found at {path}")
        return rows

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            history = parts[3].split() if parts[3].strip() else []
            impressions_raw = parts[4].split()
            pos = [i.split("-")[0] for i in impressions_raw if i.endswith("-1")]
            neg = [i.split("-")[0] for i in impressions_raw if i.endswith("-0")]
            rows.append({
                "user_id": parts[1],
                "time": parts[2],
                "history": history,
                "pos": pos,
                "neg": neg,
            })
    print(f"[MIND] Loaded {len(rows)} behavior records from {path}")
    return rows


def classify_political_lean(title, abstract, category):
    """
    基于内容关键词 + 分类 推断政治倾向。
    返回 'P' (Progressive), 'B' (Balanced), 'T' (Traditional)
    """
    text = f"{title} {abstract}".lower()

    progressive_score = sum(1 for w in PROGRESSIVE_WORDS if w.lower() in text)
    traditional_score = sum(1 for w in TRADITIONAL_WORDS if w.lower() in text)

    if progressive_score > traditional_score + 1:
        return "P"
    elif traditional_score > progressive_score + 1:
        return "T"
    elif category in POLITICAL_CATEGORIES and progressive_score == traditional_score == 0:
        # 政治类但没有关键词 → 随机分布以保持均衡
        return random.choice(["P", "B", "T"])
    else:
        return "B"


def extract_tags(title, abstract):
    """从标题和摘要中提取关键词作为标签"""
    text = f"{title} {abstract}".lower()
    text_clean = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text_clean.split() if len(w) > 3]

    # 停用词
    stopwords = {"this", "that", "with", "from", "they", "have", "been",
                 "were", "about", "what", "when", "which", "their", "more",
                 "some", "would", "could", "other", "after", "over", "into"}
    filtered = [w for w in words if w not in stopwords]

    word_freq = Counter(filtered).most_common(5)
    return [w for w, _ in word_freq]


def map_subtopic(category, subcategory):
    """将 MIND 的 category/subcategory 映射为政治子主题（尽可能合理）"""
    combined = f"{category}_{subcategory}".lower()

    mapping = [
        (r"news.*(politic|government|election|campaign)", "Elections"),
        (r"news.*(world|international|diploma)", "Diplomacy"),
        (r"news.*(crime|law|security|military|defense)", "National Security"),
        (r"news.*(immigration|border)", "Immigration"),
        (r"news.*(econom|budget|tax|fiscal)", "Economy"),
        (r"health", "Healthcare"),
        (r"education", "Education"),
        (r"environment|weather|climate", "Environment"),
        (r"finance|business", "Economy"),
        (r"tech|science", "Technology"),
        (r"sports", "Governance"),
        (r"entertainment|movie|music|tv", "Civil Rights"),
        (r"travel|food|lifestyle|auto", "Policy"),
    ]

    for pattern, subtopic in mapping:
        if re.search(pattern, combined):
            return subtopic

    return "Policy"  # 默认


def parse_perspectives(title, abstract, lean):
    """为新闻生成左/中/右三个视角的报道"""
    text = title
    # 提取前几句作为基础内容
    sentences = re.split(r"[.!?]+", abstract)
    snippet = " ".join(sentences[:2]).strip()[:200] if sentences else abstract[:200]

    # 左派视角
    l_headline = f"Progressive View: {title[:80]}"
    l_summary = f"From a progressive standpoint, {snippet} ..."

    # 中立视角
    c_headline = title[:100]
    c_summary = abstract[:250] if abstract else "No abstract available."

    # 右派视角
    t_headline = f"Traditional Perspective: {title[:80]}"
    t_summary = f"Conservative analysts view this as: {snippet} ..."

    return {
        "left": {
            "headline": l_headline,
            "source": "The Progressive Times",
            "summary": l_summary
        },
        "center": {
            "headline": c_headline,
            "source": "Associated Press",
            "summary": c_summary
        },
        "right": {
            "headline": t_headline,
            "source": "The National Review",
            "summary": t_summary
        }
    }


def compute_article_popularity(nid, behaviors):
    """基于用户行为计算文章热度（被多少用户点击）"""
    click_count = 0
    impression_count = 0
    for b in behaviors:
        if nid in b.get("history", []):
            click_count += 1
        if nid in b.get("pos", []):
            click_count += 1
        all_impressions = b.get("pos", []) + b.get("neg", [])
        if nid in all_impressions:
            impression_count += 1
    return click_count, impression_count


def build_news_data(news_dict, behaviors, max_articles=MAX_ARTICLES):
    """
    主转换函数：MIND 原始数据 → 前端 newsData 格式
    优先选择热门且有用户行为数据的文章
    """
    articles = []
    popularity = {}

    # 计算每篇文章的受欢迎程度
    print("[Adapter] Computing article popularity from behaviors...")
    for nid in news_dict:
        clicks, imps = compute_article_popularity(nid, behaviors)
        popularity[nid] = clicks

    # 按热度 + 多样性选择文章
    sorted_by_pop = sorted(popularity.items(), key=lambda x: -x[1])

    # 倾向计数（保证 P/B/T 均衡）
    lean_counts = {"P": 0, "B": 0, "T": 0}

    # 从热门文章中采样
    selected = set()
    nid_to_lean = {}

    # 第一轮：对所有文章预分配倾向
    for nid, info in news_dict.items():
        lean = classify_political_lean(
            info["title"], info["abstract"], info["category"]
        )
        nid_to_lean[nid] = lean

    # 第二轮：按覆盖率选择（保证每种倾向至少有目标数）
    target_per_lean = max_articles // 3

    # 先按热度排序，选择足够多的文章
    candidate_pool = []
    for nid, pop in sorted_by_pop[:max_articles * 3]:
        if nid in news_dict:
            candidate_pool.append(nid)

    # 按倾向分桶
    lean_buckets = {"P": [], "B": [], "T": []}
    for nid in candidate_pool:
        lean = nid_to_lean.get(nid, "B")
        lean_buckets[lean].append(nid)

    # 交错选择
    bucket_indices = {"P": 0, "B": 0, "T": 0}
    while len(selected) < max_articles:
        added = False
        for lean in ["P", "B", "T"]:
            if len(selected) >= max_articles:
                break
            bucket = lean_buckets[lean]
            idx = bucket_indices[lean]
            if idx < len(bucket):
                selected.add(bucket[idx])
                bucket_indices[lean] = idx + 1
                added = True
        if not added:
            break

    # 生成最终的文章列表
    used_sources = {
        "P": list(PROGRESSIVE_SOURCES),
        "B": list(BALANCED_SOURCES),
        "T": list(TRADITIONAL_SOURCES),
    }

    article_id = 0
    for nid in selected:
        article_id += 1
        info = news_dict[nid]
        lean = nid_to_lean.get(nid, "B")

        source_pool = used_sources[lean]
        if not source_pool:
            source_pool = BALANCED_SOURCES
        source = random.choice(source_pool)

        tags = extract_tags(info["title"], info["abstract"])
        subtopic = map_subtopic(info["category"], info["subcategory"])

        article = {
            "id": article_id,
            "title": info["title"],
            "summary": info["abstract"][:300],
            "category": info["category"].capitalize(),
            "subtopic": subtopic,
            "tags": tags,
            "source": source,
            "time": random.choice(RELATIVE_TIMES),
            "thumb": f"https://picsum.photos/seed/mind{article_id}/360/240",
            "url": f"https://example.com/news/{article_id}",
            "sourceLean": lean,
            "perspectives": parse_perspectives(
                info["title"], info["abstract"], lean
            ),
            # 元数据
            "_mind_id": nid,
            "_popularity": popularity.get(nid, 0),
            "_original_category": info["category"],
            "_original_subcategory": info["subcategory"],
        }

        articles.append(article)
        lean_counts[lean] += 1

    print(f"[Adapter] Generated {len(articles)} articles")
    print(f"  P: {lean_counts['P']}  B: {lean_counts['B']}  T: {lean_counts['T']}")

    return articles


def get_stats(articles):
    """生成数据统计信息"""
    categories = Counter(a["category"] for a in articles)
    subtopics = Counter(a["subtopic"] for a in articles)
    leans = Counter(a["sourceLean"] for a in articles)
    sources = Counter(a["source"] for a in articles)

    return {
        "total": len(articles),
        "categories": dict(categories.most_common()),
        "subtopics": dict(subtopics.most_common()),
        "leans": dict(leans),
        "top_sources": dict(sources.most_common(5)),
    }


def get_user_profiles(behaviors, news_dict, max_users=100):
    """
    从行为数据中提取用户画像。
    返回 {user_id: {history_leans, preferred_categories, ...}}
    """
    profiles = {}
    for b in behaviors[:max_users]:
        uid = b["user_id"]
        history = b.get("history", [])
        leans = []
        categories = []
        for nid in history:
            if nid in news_dict:
                info = news_dict[nid]
                lean = classify_political_lean(info["title"], info["abstract"], info["category"])
                leans.append(lean)
                categories.append(info["category"])

        lean_dist = Counter(leans)
        cat_dist = Counter(categories)

        profiles[uid] = {
            "user_id": uid,
            "history_size": len(history),
            "lean_distribution": {
                "P": lean_dist.get("P", 0) / max(len(leans), 1),
                "B": lean_dist.get("B", 0) / max(len(leans), 1),
                "T": lean_dist.get("T", 0) / max(len(leans), 1),
            },
            "top_categories": dict(cat_dist.most_common(5)),
        }

    return profiles


# ─── 缓存机制 ─────────────────────────────────────────────────────────

_cache = {}


def load_all(max_articles=MAX_ARTICLES, force_reload=False):
    """加载全部数据（带缓存）"""
    cache_key = f"all_{max_articles}"
    if cache_key in _cache and not force_reload:
        return _cache[cache_key]

    news = load_news()
    behaviors = load_behaviors()
    articles = build_news_data(news, behaviors, max_articles)
    stats = get_stats(articles)
    user_profiles = get_user_profiles(behaviors, news)

    result = {
        "articles": articles,
        "stats": stats,
        "user_profiles": user_profiles,
        "total_raw_news": len(news),
        "total_behaviors": len(behaviors),
    }

    _cache[cache_key] = result
    return result


if __name__ == "__main__":
    # 测试：打印几篇样本
    data = load_all(max_articles=30)
    for a in data["articles"][:3]:
        print(json.dumps(a, indent=2, ensure_ascii=False))
    print("\n--- Stats ---")
    print(json.dumps(data["stats"], indent=2, ensure_ascii=False))
