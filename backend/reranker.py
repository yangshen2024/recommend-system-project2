"""
News Reranker — Entity MMR / Embedding MMR / Hybrid / Calibrated Reranking

基于 reranking_diversity_refactored.ipynb 的实验结论：
  - Entity MMR 是提升多视角覆盖度 (coverage-diverse) 最有效的策略
  - Embedding MMR (语义多样性) 不等同于事件覆盖度
  - Hybrid MMR 是折中方案
  - Calibrated Reranker 可激进地推动簇分布均匀化

使用方法:
    from backend.reranker import Reranker
    reranker = Reranker(articles)
    baseline = reranker.baseline_rank(user_profile)
    reranked = reranker.rerank(baseline, method="entity_mmr", lam=0.6)
"""

import math
import re
import random
from collections import Counter

import numpy as np

# ─── 可选依赖 ──────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class Reranker:
    """
    新闻重排序器，支持四种策略:
      - entity_mmr:  实体级 MMR（奖励带来新实体的文章）
      - embedding_mmr: 语义级 MMR（奖励语义距离远的文章）
      - hybrid_mmr:  实体 + 语义混合 MMR
      - calibrated:  簇分布校准（KL 散度最小化）
    """

    def __init__(self, articles, use_embeddings=True):
        """
        Args:
            articles: list of article dicts, each with id, title, summary, tags, subtopic, category, sourceLean
            use_embeddings: 是否加载 sentence-transformers 做语义嵌入
        """
        self.articles = articles
        self.n = len(articles)
        self._id_to_idx = {a["id"]: i for i, a in enumerate(articles)}

        # ── 1. 实体提取与 IDF ──
        self._build_entities()

        # ── 2. 语义嵌入 ──
        self._embeddings = None
        if use_embeddings and _ST_AVAILABLE:
            self._build_embeddings()
        elif use_embeddings and _SKLEARN_AVAILABLE:
            self._build_tfidf()

        # ── 3. 邻域图 (co-event neighborhood) ──
        self._build_neighborhood()

        # ── 4. 热度分数 ──
        self._build_popularity_scores()

    # ════════════════════════════════════════════════════════════════
    #  实体构建
    # ════════════════════════════════════════════════════════════════

    def _build_entities(self):
        """从文章 tags + subtopic + category 提取实体并计算 IDF"""
        entity_docs = []
        for a in self.articles:
            entities = self._extract_entities_from_article(a)
            a["_entities"] = entities
            entity_docs.append(" ".join(entities))

        # 计算 IDF
        self.all_entities = sorted(set(e for d in entity_docs for e in d.split()))
        df = Counter()
        for doc in entity_docs:
            for ent in set(doc.split()):
                df[ent] += 1

        N = max(1, self.n)
        self._entity_idf = {
            e: math.log((N + 1) / (df.get(e, 1) + 1)) + 1
            for e in self.all_entities
        }

        # 每个文章的实体向量 (按 IDF 加权)
        self._entity_vectors = []
        for doc in entity_docs:
            ent_set = set(doc.split())
            vec = np.zeros(len(self.all_entities))
            for e in ent_set:
                vec[self.all_entities.index(e)] = self._entity_idf.get(e, 1.0)
            # L2 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._entity_vectors.append(vec)

    def _extract_entities_from_article(self, a):
        """从文章元数据中提取'实体'（tag + subtopic + category + sourceLean 映射）"""
        entities = []
        # tags
        for t in a.get("tags", []):
            entities.append(f"tag:{t}")
        # subtopic
        sub = a.get("subtopic", "")
        if sub:
            entities.append(f"subtopic:{sub}")
        # category
        cat = a.get("category", "")
        if cat:
            entities.append(f"category:{cat}")
        # sourceLean
        lean = a.get("sourceLean", "")
        if lean:
            entities.append(f"lean:{lean}")
        # 标题关键词
        title = a.get("title", "")
        words = re.findall(r"[a-zA-Z]{4,}", title.lower())
        for w in words[:5]:
            if w not in {"this", "that", "with", "from", "they", "have", "been",
                         "were", "about", "what", "when", "which", "their", "more"}:
                entities.append(f"word:{w}")
        return entities

    # ════════════════════════════════════════════════════════════════
    #  语义嵌入
    # ════════════════════════════════════════════════════════════════

    def _build_embeddings(self):
        """使用 sentence-transformers 构建语义嵌入"""
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [f"{a.get('title','')} {a.get('summary','')[:200]}" for a in self.articles]
            self._embeddings = model.encode(texts, show_progress_bar=False)
            print(f"[Reranker] Built {self._embeddings.shape[1]}d embeddings via MiniLM")
        except Exception as e:
            print(f"[Reranker] MiniLM unavailable: {e}")
            if _SKLEARN_AVAILABLE:
                self._build_tfidf()

    def _build_tfidf(self):
        """使用 TF-IDF 作为语义嵌入的退化方案"""
        texts = [f"{a.get('title','')} {a.get('summary','')}" for a in self.articles]
        vec = TfidfVectorizer(max_features=512, stop_words="english")
        self._embeddings = vec.fit_transform(texts).toarray()
        print(f"[Reranker] Built {self._embeddings.shape[1]}d TF-IDF embeddings")

    # ════════════════════════════════════════════════════════════════
    #  邻域图
    # ════════════════════════════════════════════════════════════════

    def _build_neighborhood(self, tau=0.25):
        """
        构建同事件邻域图: 基于 IDF 加权 Jaccard 相似度
        全局阈值 tau 决定两篇文章是否属于同一事件邻域
        """
        self._neighbors = [[] for _ in range(self.n)]
        self._neighborhood_sizes = [0] * self.n

        for i in range(self.n):
            vi = self._entity_vectors[i]
            for j in range(self.n):
                if i == j:
                    continue
                vj = self._entity_vectors[j]
                # 余弦相似度（因为向量已 L2 归一化，即点积）
                sim = float(np.dot(vi, vj))
                if sim >= tau:
                    self._neighbors[i].append(j)
            self._neighborhood_sizes[i] = len(self._neighbors[i])

        sizes = self._neighborhood_sizes
        print(f"[Reranker] Neighborhood built: median={np.median(sizes):.0f}, "
              f"mean={np.mean(sizes):.1f}, max={max(sizes)}")

        # ── Connected Components (Union-Find) ──
        parent = list(range(self.n))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x, y):
            rx, ry = _find(x), _find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(self.n):
            for j in self._neighbors[i]:
                _union(i, j)

        group_roots = {}
        next_gid = 0
        for i in range(self.n):
            root = _find(i)
            if root not in group_roots:
                group_roots[root] = next_gid
                next_gid += 1
            self.articles[i]["_cluster_group_id"] = group_roots[root]

        self._num_clusters = next_gid
        print(f"[Reranker] Connected components: {next_gid} clusters")

    # ════════════════════════════════════════════════════════════════
    #  热度 / 基准分数
    # ════════════════════════════════════════════════════════════════

    def _build_popularity_scores(self):
        """基于用户行为热度构建基准分数 (0~1)"""
        scores = []
        for a in self.articles:
            pop = a.get("_popularity", 0)
            scores.append(min(1.0, pop / max(1, max(
                self.articles, key=lambda x: x.get("_popularity", 0)
            ).get("_popularity", 1))))
        self._popularity_scores = np.array(scores)

    # ════════════════════════════════════════════════════════════════
    #  Baseline 排序
    # ════════════════════════════════════════════════════════════════

    def baseline_rank(self, user_profile=None, top_k=20):
        """
        基准推荐排序（Layer 3 基础推荐器）
        按: 热度 + 用户偏好匹配 → 排序

        Args:
            user_profile: dict with keys like 'preferred_leans', 'liked_tags', etc.
            top_k: 返回前 K 篇

        Returns:
            list of article dicts with '_baseline_score' and '_rank'
        """
        scores = np.zeros(self.n)

        for i, a in enumerate(self.articles):
            s = self._popularity_scores[i] * 0.4  # 40% 热度

            # 用户偏好匹配 (如果有画像)
            if user_profile:
                s += self._user_article_match(user_profile, a)

            # 新鲜度
            s += (1 - i / max(1, self.n)) * 0.1

            scores[i] = s

        # 按分数降序排列
        ranked_indices = np.argsort(-scores)
        ranked = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            article = dict(self.articles[idx])
            article["_baseline_score"] = round(float(scores[idx]), 4)
            article["_rank"] = rank + 1
            article["_cluster_group_id"] = self.articles[idx].get("_cluster_group_id", -1)
            ranked.append(article)

        return ranked

    def _user_article_match(self, profile, article):
        """用户-文章匹配分数"""
        s = 0
        # 倾向匹配
        preferred = profile.get("preferred_leans", [])
        lean = article.get("sourceLean", "")
        if lean in preferred:
            s += 0.3
        # tag 匹配
        liked_tags = profile.get("liked_tags", [])
        article_tags = article.get("tags", [])
        overlap = len(set(liked_tags) & set(article_tags))
        if overlap > 0:
            s += min(0.3, overlap * 0.1)
        return s

    # ════════════════════════════════════════════════════════════════
    #  MMR 重排序核心
    # ════════════════════════════════════════════════════════════════

    def rerank(self, baseline_results, method="entity_mmr", lam=0.6, alpha=0.5):
        """
        对 baseline 结果进行重排序

        Args:
            baseline_results: baseline_rank() 的输出列表
            method: "entity_mmr" | "embedding_mmr" | "hybrid_mmr" | "calibrated"
            lam: MMR λ 参数 (0=纯相关性, 1=纯多样性)
            alpha: Hybrid MMR 中的混合比例 (0=纯embedding, 1=纯entity)

        Returns:
            dict with:
              - reranked: 重排序后文章列表
              - baseline: 原 basline 列表
              - coverage_stats: 覆盖度统计
              - method, lam, alpha
        """
        n_candidates = len(baseline_results)
        if n_candidates == 0:
            return {"reranked": [], "baseline": [], "coverage_stats": {}, "method": method}

        # 构建索引映射
        candidate_indices = []
        for a in baseline_results:
            idx = self._id_to_idx.get(a["id"])
            if idx is not None:
                candidate_indices.append(idx)

        # 相关性分数
        rel_scores = np.array([
            a.get("_baseline_score", self._popularity_scores[self._id_to_idx.get(a["id"], 0)])
            for a in baseline_results
        ])
        rel_max = max(rel_scores.max(), 1e-8)
        rel_scores = rel_scores / rel_max  # 归一化到 [0,1]

        if method == "entity_mmr":
            reranked_indices = self._entity_mmr(candidate_indices, rel_scores, lam)
        elif method == "embedding_mmr":
            reranked_indices = self._embedding_mmr(candidate_indices, rel_scores, lam)
        elif method == "hybrid_mmr":
            reranked_indices = self._hybrid_mmr(candidate_indices, rel_scores, lam, alpha)
        elif method == "calibrated":
            reranked_indices = self._calibrated(candidate_indices, rel_scores, lam)
        else:
            reranked_indices = list(range(n_candidates))

        # 构建输出
        reranked = []
        for rank, cidx in enumerate(reranked_indices):
            article = dict(baseline_results[cidx])
            article["_reranked_score"] = round(float(rel_scores[cidx]), 4)
            article["_reranked_rank"] = rank + 1
            article["_original_rank"] = cidx + 1

            # ── New: per-article explanation fields ──
            article["_cluster_name"] = self._derive_cluster_name(article)
            article["_diversity_gain"] = self._calc_diversity_gain(
                article, reranked, rank, reranked_indices, baseline_results, method, lam
            )
            article["_reason"] = self._generate_reason(
                article, reranked, rank, reranked_indices, method
            )
            reranked.append(article)

        # 覆盖度统计
        coverage_stats = self._compute_coverage(baseline_results, reranked)

        return {
            "reranked": reranked,
            "baseline": baseline_results,
            "coverage_stats": coverage_stats,
            "method": method,
            "lam": lam,
            "alpha": alpha,
        }

    def _entity_mmr(self, candidates, rel_scores, lam):
        """Entity MMR: 贪心选取覆盖最多新实体的文章"""
        n = len(candidates)
        chosen = []
        remaining = list(range(n))

        # 已覆盖实体的计数
        covered_entities = Counter()
        entity_sets = []
        for i in range(n):
            aidx = candidates[i]
            entity_sets.append(set(self.articles[aidx].get("_entities", [])))

        for _ in range(n):
            best_i = -1
            best_score = -1

            for i in remaining:
                # 相关性部分
                rel_score = rel_scores[i]

                # 新实体增益
                new_entities = entity_sets[i] - set(covered_entities.keys())
                entity_gain = sum(self._entity_idf.get(e, 1.0) for e in new_entities)

                # 同事件邻域加权
                aidx = candidates[i]
                in_neighborhood = 0
                for c in chosen:
                    c_aidx = candidates[c]
                    if c_aidx in self._neighbors[aidx]:
                        in_neighborhood = 1
                        break

                # MMR 分数
                bonus = in_neighborhood * (entity_gain / (1 + entity_gain))
                score = (1 - lam) * rel_score + lam * bonus

                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0:
                chosen.append(best_i)
                remaining.remove(best_i)
                # 更新已覆盖实体
                covered_entities.update(entity_sets[best_i])

        return chosen

    def _embedding_mmr(self, candidates, rel_scores, lam):
        """Embedding MMR: 贪心选取语义多样性最大的文章"""
        n = len(candidates)
        if self._embeddings is None or n <= 1:
            return list(range(n))

        chosen = []
        remaining = list(range(n))

        for _ in range(n):
            best_i = -1
            best_score = -1

            for i in remaining:
                rel_score = rel_scores[i]

                # 与已选文章的最大余弦相似度
                max_sim = 0
                if chosen:
                    vi = self._embeddings[candidates[i]]
                    sims = []
                    for c in chosen:
                        vc = self._embeddings[candidates[c]]
                        dot = float(np.dot(vi, vc))
                        na = np.linalg.norm(vi)
                        nb = np.linalg.norm(vc)
                        sims.append(dot / max(na * nb, 1e-8))
                    max_sim = max(sims)

                score = (1 - lam) * rel_score - lam * max_sim

                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0:
                chosen.append(best_i)
                remaining.remove(best_i)

        return chosen

    def _hybrid_mmr(self, candidates, rel_scores, lam, alpha=0.5):
        """Hybrid MMR: 实体多样性 + 语义多样性的混合"""
        n = len(candidates)
        chosen = []
        remaining = list(range(n))

        covered_entities = Counter()
        entity_sets = [set(self.articles[candidates[i]].get("_entities", [])) for i in range(n)]

        has_embeddings = self._embeddings is not None

        for _ in range(n):
            best_i = -1
            best_score = -1

            for i in remaining:
                rel_score = rel_scores[i]

                # 实体增益
                new_entities = entity_sets[i] - set(covered_entities.keys())
                entity_gain = sum(self._entity_idf.get(e, 1.0) for e in new_entities)
                entity_bonus = entity_gain / (1 + entity_gain)

                # 语义多样性
                embedding_bonus = 0
                if has_embeddings and chosen:
                    vi = self._embeddings[candidates[i]]
                    max_sim = 0
                    for c in chosen:
                        vc = self._embeddings[candidates[c]]
                        dot = float(np.dot(vi, vc))
                        sim = dot / max(np.linalg.norm(vi) * np.linalg.norm(vc), 1e-8)
                        max_sim = max(max_sim, sim)
                    embedding_bonus = 1 - max_sim  # 不相似 = 好
                elif not chosen:
                    embedding_bonus = 1.0

                # 邻域加权
                aidx = candidates[i]
                in_neighborhood = any(
                    candidates[c] in self._neighbors[aidx] for c in chosen
                ) if chosen else 1

                # 混合
                diversity_bonus = alpha * entity_bonus + (1 - alpha) * embedding_bonus
                score = (1 - lam) * rel_score + lam * in_neighborhood * diversity_bonus

                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0:
                chosen.append(best_i)
                remaining.remove(best_i)
                covered_entities.update(entity_sets[best_i])

        return chosen

    def _calibrated(self, candidates, rel_scores, lam):
        """
        Calibrated Reranker: 每步选取最小化 KL 散度的文章
        目标: 各实体簇均匀分布
        """
        n = len(candidates)
        if n == 0:
            return []

        # 收集候选文章的所有实体簇
        all_clusters = set()
        article_clusters = []
        for i in range(n):
            aidx = candidates[i]
            # 用 sourceLean + subtopic 作为簇标签
            lean = self.articles[aidx].get("sourceLean", "B")
            subtopic = self.articles[aidx].get("subtopic", "General")
            cluster = f"{lean}|{subtopic}"
            all_clusters.add(cluster)
            article_clusters.append(cluster)

        clusters = sorted(all_clusters)
        K = max(1, len(clusters))
        cluster_to_idx = {c: i for i, c in enumerate(clusters)}
        target = np.ones(K) / K  # 均匀目标分布

        chosen = []
        remaining = list(range(n))
        current_counts = np.zeros(K)

        for _ in range(n):
            best_i = -1
            best_score = float("inf")  # 最小 KL

            for i in remaining:
                rel_score = rel_scores[i]
                c = article_clusters[i]
                cidx = cluster_to_idx[c]
                temp_counts = current_counts.copy()
                temp_counts[cidx] += 1
                dist = temp_counts / max(1, temp_counts.sum())

                # KL 散度
                kl = 0
                for k in range(K):
                    if dist[k] > 0:
                        kl += dist[k] * math.log(dist[k] / max(target[k], 1e-8))

                # 结合相关性
                score = (1 - lam) * (-rel_score) + lam * kl  # 越小越好

                if score < best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0:
                chosen.append(best_i)
                remaining.remove(best_i)
                current_counts[cluster_to_idx[article_clusters[best_i]]] += 1

        return chosen

    # ════════════════════════════════════════════════════════════════
    #  每篇文章的解释性元数据
    # ════════════════════════════════════════════════════════════════

    def _derive_cluster_name(self, article):
        """Derive a human-readable cluster name from article metadata."""
        sub = article.get("subtopic", "")
        tags = article.get("tags", [])
        if sub:
            # Use first meaningful tag as context, subtopic as base
            meaningful = [t for t in tags if len(t) > 2 and t.lower() not in ("news", "general", "world", "us")]
            if meaningful:
                return f"{meaningful[0].title()} / {sub}"
            return sub
        cat = article.get("category", "")
        return cat or "General"

    def _calc_diversity_gain(self, article, already_selected, rank,
                             reranked_indices, baseline_results, method, lam):
        """Approximate diversity gain this article brings vs already-selected ones."""
        if rank == 0:
            return round(lam * 0.8, 2)  # First pick always high diversity

        prev_articles = already_selected[:rank]
        gain = 0.0

        # New source lean
        prev_leans = {a.get("sourceLean") for a in prev_articles}
        cur_lean = article.get("sourceLean", "B")
        if cur_lean not in prev_leans:
            gain += 0.15

        # New subtopic
        prev_subs = {a.get("subtopic") for a in prev_articles}
        cur_sub = article.get("subtopic", "")
        if cur_sub and cur_sub not in prev_subs:
            gain += 0.15

        # New entities fraction
        cur_ents = set(article.get("_entities", []))
        prev_ents = set()
        for pa in prev_articles:
            prev_ents.update(pa.get("_entities", []))
        new_ent_count = len(cur_ents - prev_ents)
        if len(cur_ents) > 0:
            gain += 0.1 * (new_ent_count / len(cur_ents))

        # Scale by lambda
        gain = gain * lam

        # Add small random jitter for visual difference
        gain += random.uniform(-0.02, 0.02)

        return round(min(max(gain, 0.02), 0.35), 2)

    def _generate_reason(self, article, already_selected, rank, reranked_indices, method):
        """Generate a human-readable reason for including this article."""
        orig_rank = article.get("_original_rank", rank + 1)
        promoted = orig_rank > rank + 1

        lean = article.get("sourceLean", "B")
        lean_label = {"P": "Left", "B": "Center", "T": "Right"}.get(lean, lean)
        cluster = article.get("_cluster_name", "")

        if rank == 0:
            return f"Top relevance match with {lean_label} viewpoint on {cluster}"

        prev_articles = already_selected[:rank]
        prev_leans = {a.get("sourceLean") for a in prev_articles}
        prev_subs = {a.get("subtopic") for a in prev_articles}
        cur_sub = article.get("subtopic", "")

        parts = []

        if promoted and orig_rank - (rank + 1) >= 3:
            parts.append("promoted for diverse perspective")

        if lean not in prev_leans:
            parts.append(f"adds {lean_label} viewpoint")
        elif cur_sub and cur_sub not in prev_subs:
            parts.append(f"broadens coverage to {cur_sub}")
        else:
            parts.append("adds a different angle within the same story")

        if not parts:
            parts.append("keeps high relevance with complementary perspective")

        reason = ", ".join(parts)
        return reason[0].upper() + reason[1:]

    # ════════════════════════════════════════════════════════════════
    #  覆盖度评估
    # ════════════════════════════════════════════════════════════════

    def _compute_coverage(self, baseline, reranked):
        """计算 baseline vs reranked 的多视角覆盖度"""
        def lean_dist(articles):
            dist = {"P": 0, "B": 0, "T": 0, "unknown": 0}
            for a in articles:
                l = a.get("sourceLean", "unknown")
                dist[l] = dist.get(l, 0) + 1
            return dist

        def subtopic_set(articles):
            return set(a.get("subtopic", "") for a in articles)

        def entity_coverage(articles):
            all_ents = set()
            for a in articles:
                all_ents.update(a.get("_entities", []))
            return len(all_ents)

        b_lean = lean_dist(baseline)
        r_lean = lean_dist(reranked)
        b_subs = subtopic_set(baseline)
        r_subs = subtopic_set(reranked)
        b_ent = entity_coverage(baseline)
        r_ent = entity_coverage(reranked)

        # ── Retention: reranked top-k 中保留了多少 baseline top-k 文章 ──
        baseline_ids = [a["id"] for a in baseline]
        reranked_ids = [a["id"] for a in reranked]
        overlap_ids = set(baseline_ids) & set(reranked_ids)
        article_overlap = len(overlap_ids)
        overlap_rate = round(article_overlap / max(1, len(reranked)), 4)
        # 平均排名偏移
        b_rank_map = {a["id"]: i + 1 for i, a in enumerate(baseline)}
        rank_shifts = []
        for i, a in enumerate(reranked):
            if a["id"] in b_rank_map:
                shift = abs(b_rank_map[a["id"]] - (i + 1))
                rank_shifts.append(shift)
        avg_rank_shift = round(sum(rank_shifts) / max(1, len(rank_shifts)), 2)

        # ── Angle Diversity: 成对 Jaccard 距离 ──
        angle_diversity = 0.0
        top_n = reranked[:10] if len(reranked) >= 2 else []
        if len(top_n) >= 2:
            entity_sets_list = [set(a.get("_entities", [])) for a in top_n]
            jaccard_dists = []
            for i in range(len(entity_sets_list)):
                for j in range(i + 1, len(entity_sets_list)):
                    union = entity_sets_list[i] | entity_sets_list[j]
                    inter = entity_sets_list[i] & entity_sets_list[j]
                    if len(union) > 0:
                        jaccard_dists.append(1 - len(inter) / len(union))
                    else:
                        jaccard_dists.append(0.0)
            angle_diversity = round(float(np.mean(jaccard_dists)) if jaccard_dists else 0.0, 4)

        return {
            "baseline": {
                "lean_distribution": b_lean,
                "unique_subtopics": len(b_subs),
                "entity_coverage": b_ent,
            },
            "reranked": {
                "lean_distribution": r_lean,
                "unique_subtopics": len(r_subs),
                "entity_coverage": r_ent,
            },
            "delta": {
                "entity_coverage_gain": r_ent - b_ent,
                "subtopic_gain": len(r_subs) - len(b_subs),
            },
            "retention": {
                "article_overlap": article_overlap,
                "overlap_rate": overlap_rate,
                "avg_rank_shift": avg_rank_shift,
            },
            "angle_diversity": angle_diversity,
        }


# ─── 工厂函数 ─────────────────────────────────────────────────────────

_global_reranker = None


def get_reranker(articles=None, force_reload=False, use_embeddings=True):
    global _global_reranker
    if _global_reranker is None or force_reload:
        if articles is None:
            raise ValueError("articles required for first load")
        _global_reranker = Reranker(articles, use_embeddings=use_embeddings)
    return _global_reranker
