/**
 * reranker.js — News Reranker (pure-JS, zero native deps)
 *
 * Strategies:
 *   entity_mmr    – Entity-level MMR (reward articles that bring new entities)
 *   embedding_mmr – Semantic MMR (reward articles with dissimilar text, via TF-IDF)
 *   hybrid_mmr    – Entity + semantic blended MMR
 *   calibrated    – Cluster-distribution calibration (KL-divergence)
 *
 * Based on reranking_diversity_refactored.ipynb findings:
 *   - Entity MMR is the most effective strategy for coverage diversity
 *   - Embedding diversity ≠ event-coverage diversity
 *   - Hybrid MMR is the compromise
 *   - Calibrated reranker aggressively pushes uniform cluster distribution
 */

const {
  zeros, ones, norm, dot, cosineSimilarity, l2Normalise,
  argsort, sum, mean, median, max, min,
  createRNG, randomChoice, shuffle,
  Counter,
  union, intersection, difference, jaccardDistance,
  clamp, roundTo,
} = require('./utils');

// Political subtopics — used for perspective-diversity tracking in MMR
const POLITICAL_SUBTOPICS = new Set([
  'Policy', 'Elections', 'National Security', 'Diplomacy',
  'Governance', 'Civil Rights',
]);

// Strictly-political categories — content under these categories is
// treated as political for diversity & coverage purposes
const POLITICAL_CATEGORIES = new Set([
  'News', 'World', 'Politics',
]);

// ═══════════════════════════════════════════════════════════════════════
//  Lightweight TF-IDF builder (pure JS, no sklearn dependency)
// ═══════════════════════════════════════════════════════════════════════

class TfidfBuilder {
  /**
   * @param {number} maxFeatures  output dimension
   */
  constructor(maxFeatures = 256) {
    this.maxFeatures = maxFeatures;
    this.vocab = [];          // ordered vocabulary
    this.vocabIndex = new Map();
    this.idf = new Float64Array(0);
  }

  /**
   * Fit the vocabulary from a list of document strings.
   * @param {string[]} docs
   */
  fit(docs) {
    const df = new Map();
    for (const doc of docs) {
      const tokens = this._tokenize(doc);
      const seen = new Set();
      for (const t of tokens) {
        if (!seen.has(t)) { df.set(t, (df.get(t) || 0) + 1); seen.add(t); }
      }
    }

    // Sort by descending DF, keep top maxFeatures
    const sorted = [...df.entries()].sort((a, b) => b[1] - a[1]);
    this.vocab = sorted.slice(0, this.maxFeatures).map(e => e[0]);
    this.vocabIndex.clear();
    this.vocab.forEach((w, i) => this.vocabIndex.set(w, i));

    const N = docs.length;
    this.idf = new Float64Array(this.vocab.length);
    for (let i = 0; i < this.vocab.length; i++) {
      const w = this.vocab[i];
      const d = df.get(w) || 1;
      this.idf[i] = Math.log((N + 1) / (d + 1)) + 1;
    }
  }

  /**
   * Transform a single document into a Float64Array vector.
   * @param {string} doc
   * @returns {Float64Array}
   */
  transform(doc) {
    const vec = zeros(this.vocab.length);
    const tokens = this._tokenize(doc);
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);

    for (const [word, count] of tf) {
      const idx = this.vocabIndex.get(word);
      if (idx != null) vec[idx] = count * this.idf[idx];
    }

    return l2Normalise(vec);
  }

  /**
   * Transform multiple documents.
   * @param {string[]} docs
   * @returns {Float64Array[]}
   */
  transformMany(docs) {
    return docs.map(d => this.transform(d));
  }

  _tokenize(text) {
    // Lowercase, split on non-alphanumeric, filter short tokens
    return text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length >= 2);
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  Reranker class
// ═══════════════════════════════════════════════════════════════════════

class Reranker {
  /**
   * @param {Object[]} articles  array of article dicts (same format as Python)
   * @param {Object} opts
   * @param {boolean} [opts.useEmbeddings=true]  build TF-IDF semantic vectors
   */
  constructor(articles, opts = {}) {
    const { useEmbeddings = true } = opts;
    this.articles = articles;
    this.n = articles.length;
    this._idToIdx = new Map(articles.map((a, i) => [a.id, i]));

    // 1. Entity extraction & IDF
    this._buildEntities();

    // 2. Semantic embeddings (lightweight TF-IDF, no heavy sentence-transformers)
    this._embeddings = null;
    if (useEmbeddings) {
      this._buildEmbeddings();
    }

    // 3. Neighborhood graph (co-event neighborhood)
    this._buildNeighborhood();

    // 4. Popularity scores
    this._buildPopularityScores();
  }

  // ── 1. Entity extraction ───────────────────────────────────────────

  _buildEntities() {
    const entityDocs = [];
    for (const a of this.articles) {
      const entities = this._extractEntitiesFromArticle(a);
      a._entities = entities;
      entityDocs.push(entities.join(' '));
    }

    // Collect all unique entities & compute DF
    const allEnts = new Set();
    for (const doc of entityDocs) {
      for (const e of doc.split(' ')) allEnts.add(e);
    }
    this.allEntities = [...allEnts].sort();

    const df = new Counter();
    for (const doc of entityDocs) {
      const seen = new Set(doc.split(' '));
      for (const e of seen) df.inc(e);
    }

    const N = Math.max(1, this.n);
    this._entityIdf = new Map();
    for (const e of this.allEntities) {
      this._entityIdf.set(e, Math.log((N + 1) / (df.get(e) + 1)) + 1);
    }

    // IDF-weighted entity vectors (L2-normalised)
    this._entityVectors = [];
    for (const doc of entityDocs) {
      const entSet = new Set(doc.split(' '));
      const vec = zeros(this.allEntities.length);
      for (const e of entSet) {
        vec[this.allEntities.indexOf(e)] = this._entityIdf.get(e) || 1;
      }
      l2Normalise(vec);
      this._entityVectors.push(vec);
    }
  }

  _extractEntitiesFromArticle(a) {
    const entities = [];
    // tags
    for (const t of (a.tags || [])) entities.push(`tag:${t}`);
    // subtopic
    if (a.subtopic) {
      entities.push(`subtopic:${a.subtopic}`);
      // Political subtopics get extra entity tags for diversity tracking
      if (POLITICAL_SUBTOPICS.has(a.subtopic)) {
        entities.push('political:true');
      }
    }
    // category
    if (a.category) {
      entities.push(`category:${a.category}`);
      // Political categories get extra entity tags
      if (POLITICAL_CATEGORIES.has(a.category)) {
        entities.push('political:true');
      }
    }
    // sourceLean
    if (a.sourceLean) entities.push(`lean:${a.sourceLean}`);
    // title keywords (4+ letter words)
    const title = (a.title || '').toLowerCase();
    const words = title.match(/[a-zA-Z]{4,}/g) || [];
    const stopWords = new Set([
      'this', 'that', 'with', 'from', 'they', 'have', 'been',
      'were', 'about', 'what', 'when', 'which', 'their', 'more',
    ]);
    let kwCount = 0;
    for (const w of words) {
      if (!stopWords.has(w) && kwCount < 5) {
        entities.push(`word:${w}`);
        kwCount++;
      }
    }
    return entities;
  }

  // ── 2. Semantic embeddings (lightweight TF-IDF) ────────────────────

  _buildEmbeddings() {
    try {
      const texts = this.articles.map(a =>
        `${a.title || ''} ${(a.summary || '').slice(0, 200)}`
      );
      const builder = new TfidfBuilder(256);
      builder.fit(texts);
      this._embeddings = builder.transformMany(texts);
      console.log(`[Reranker] Built ${builder.vocab.length}-dim TF-IDF embeddings`);
    } catch (e) {
      console.warn(`[Reranker] TF-IDF build failed: ${e.message}`);
      this._embeddings = null;
    }
  }

  // ── 3. Neighborhood graph ──────────────────────────────────────────

  _buildNeighborhood(tau = 0.25) {
    this._neighbors = Array.from({ length: this.n }, () => []);
    this._neighborhoodSizes = new Int32Array(this.n);

    for (let i = 0; i < this.n; i++) {
      const vi = this._entityVectors[i];
      for (let j = 0; j < this.n; j++) {
        if (i === j) continue;
        const sim = cosineSimilarity(vi, this._entityVectors[j]);
        if (sim >= tau) this._neighbors[i].push(j);
      }
      this._neighborhoodSizes[i] = this._neighbors[i].length;
    }

    console.log(
      `[Reranker] Neighborhood built: median=${median(this._neighborhoodSizes).toFixed(0)}, ` +
      `mean=${mean(this._neighborhoodSizes).toFixed(1)}, ` +
      `max=${max(this._neighborhoodSizes)}`
    );

    // Union-Find for connected components (cluster IDs)
    const parent = Array.from({ length: this.n }, (_, i) => i);
    const find = (x) => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const union = (x, y) => {
      const rx = find(x), ry = find(y);
      if (rx !== ry) parent[rx] = ry;
    };

    for (let i = 0; i < this.n; i++) {
      for (const j of this._neighbors[i]) union(i, j);
    }

    const groupRoots = new Map();
    let nextGid = 0;
    for (let i = 0; i < this.n; i++) {
      const root = find(i);
      if (!groupRoots.has(root)) {
        groupRoots.set(root, nextGid++);
      }
      this.articles[i]._cluster_group_id = groupRoots.get(root);
    }
    this._numClusters = nextGid;
    console.log(`[Reranker] Connected components: ${nextGid} clusters`);
  }

  // ── 4. Popularity scores ───────────────────────────────────────────

  _buildPopularityScores() {
    this._popularityScores = new Float64Array(this.n);
    const pops = this.articles.map(a => a._popularity || 0);
    const maxPop = Math.max(1, ...pops);
    for (let i = 0; i < this.n; i++) {
      this._popularityScores[i] = Math.min(1, pops[i] / maxPop);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Baseline ranking
  // ═══════════════════════════════════════════════════════════════════

  /**
   * @param {Object} [userProfile]  { preferredLeans, likedTags }
   * @param {number} [topK=20]
   * @returns {Object[]}  ranked articles with _baseline_score and _rank
   */
  baselineRank(userProfile = null, topK = 20) {
    const scores = new Float64Array(this.n);

    for (let i = 0; i < this.n; i++) {
      let s = this._popularityScores[i] * 0.4;   // 40% popularity

      if (userProfile) {
        s += this._userArticleMatch(userProfile, this.articles[i]);
      }

      // freshness bonus
      s += (1 - i / Math.max(1, this.n)) * 0.1;

      scores[i] = s;
    }

    const rankedIdx = argsort(scores, 'desc');
    const ranked = [];
    for (let rank = 0; rank < Math.min(topK, rankedIdx.length); rank++) {
      const idx = rankedIdx[rank];
      const article = { ...this.articles[idx] };
      article._baseline_score = roundTo(scores[idx], 4);
      article._rank = rank + 1;
      article._cluster_group_id = this.articles[idx]._cluster_group_id;
      ranked.push(article);
    }

    return ranked;
  }

  _userArticleMatch(profile, article) {
    let s = 0;
    const preferred = profile.preferredLeans || profile.preferred_leans || [];
    const lean = article.sourceLean || '';
    if (preferred.includes(lean)) s += 0.3;

    const likedTags = profile.likedTags || profile.liked_tags || [];
    const articleTags = article.tags || [];
    const overlap = likedTags.filter(t => articleTags.includes(t)).length;
    if (overlap > 0) s += Math.min(0.3, overlap * 0.1);
    return s;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Main rerank entry point
  // ═══════════════════════════════════════════════════════════════════

  /**
   * @param {Object[]} baselineResults
   * @param {Object} opts
   * @param {string} [opts.method='entity_mmr']  entity_mmr | embedding_mmr | hybrid_mmr | calibrated
   * @param {number} [opts.lam=0.6]   MMR λ ∈ [0,1]
   * @param {number} [opts.alpha=0.5] Hybrid mixing ratio
   * @returns {Object}  { reranked, baseline, coverageStats, method, lam, alpha }
   */
  rerank(baselineResults, opts = {}) {
    const { method = 'entity_mmr', lam = 0.6, alpha = 0.5 } = opts;
    const nCandidates = baselineResults.length;
    if (nCandidates === 0) {
      return { reranked: [], baseline: [], coverageStats: {}, method };
    }

    // Build candidate index mapping
    const candidateIndices = [];
    for (const a of baselineResults) {
      const idx = this._idToIdx.get(a.id);
      candidateIndices.push(idx != null ? idx : -1);
    }

    // Relevance scores (normalised to [0,1])
    const relScores = new Float64Array(nCandidates);
    for (let i = 0; i < nCandidates; i++) {
      const a = baselineResults[i];
      const ci = candidateIndices[i];
      relScores[i] = a._baseline_score || (ci >= 0 ? this._popularityScores[ci] : 0);
    }
    const relMax = Math.max(max(relScores), 1e-8);
    for (let i = 0; i < relScores.length; i++) relScores[i] /= relMax;

    // Dispatch strategy
    let rerankedIndices;
    switch (method) {
      case 'entity_mmr':    rerankedIndices = this._entityMmr(candidateIndices, relScores, lam); break;
      case 'embedding_mmr': rerankedIndices = this._embeddingMmr(candidateIndices, relScores, lam); break;
      case 'hybrid_mmr':    rerankedIndices = this._hybridMmr(candidateIndices, relScores, lam, alpha); break;
      case 'calibrated':    rerankedIndices = this._calibrated(candidateIndices, relScores, lam); break;
      default:              rerankedIndices = Array.from({ length: nCandidates }, (_, i) => i);
    }

    // Build output
    const reranked = [];
    for (let rank = 0; rank < rerankedIndices.length; rank++) {
      const cidx = rerankedIndices[rank];
      const article = { ...baselineResults[cidx] };
      article._reranked_score = roundTo(relScores[cidx], 4);
      article._reranked_rank = rank + 1;
      article._original_rank = cidx + 1;

      // Explanation fields
      article._cluster_name = this._deriveClusterName(article);
      article._diversity_gain = this._calcDiversityGain(article, reranked, rank);
      article._reason = this._generateReason(article, reranked, rank);
      reranked.push(article);
    }

    const coverageStats = this._computeCoverage(baselineResults, reranked);

    return { reranked, baseline: baselineResults, coverageStats, method, lam, alpha };
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Strategy: Entity MMR (cluster-aware perspective diversity)
  // ═══════════════════════════════════════════════════════════════════

  _entityMmr(candidates, relScores, lam) {
    const n = candidates.length;
    const chosen = [];
    const remaining = new Set(Array.from({ length: n }, (_, i) => i));
    const coveredEntities = new Counter();
    // Track lean coverage per cluster for perspective diversity
    const clusterLeans = new Map(); // clusterId -> Set of sourceLean values already covered
    // Track political subtopic coverage for political-diversity bonus
    const coveredPolSubtopics = new Set();

    const entitySets = candidates.map(ci =>
      ci >= 0 ? new Set(this.articles[ci]._entities || []) : new Set()
    );

    while (remaining.size > 0) {
      let bestI = -1, bestScore = -Infinity;

      for (const i of remaining) {
        const relScore = relScores[i];

        // New entity gain
        const newEnts = difference(entitySets[i], new Set(coveredEntities.keys()));
        let entityGain = 0;
        for (const e of newEnts) entityGain += this._entityIdf.get(e) || 1;

        // Cluster-aware perspective diversity bonus
        let perspectiveBonus = 0;
        const ci = candidates[i];
        if (ci >= 0 && chosen.length > 0) {
          const article = this.articles[ci];
          const clusterId = article._cluster_group_id;
          const lean = article.sourceLean || 'B';
          if (clusterId != null) {
            const coveredLeans = clusterLeans.get(clusterId);
            if (coveredLeans && coveredLeans.size > 0 && !coveredLeans.has(lean)) {
              // This article brings a new perspective to an already-seen topic cluster
              perspectiveBonus = 0.3;
            } else if (!coveredLeans || coveredLeans.size === 0) {
              // First article from this cluster — no bonus needed
              perspectiveBonus = 0;
            }
          }
        }

        // Political subtopic diversity bonus: reward articles from political
        // subtopics that haven't been covered yet
        let polDiversityBonus = 0;
        if (ci >= 0) {
          const art = this.articles[ci];
          if (art && art.subtopic && POLITICAL_SUBTOPICS.has(art.subtopic)) {
            if (!coveredPolSubtopics.has(art.subtopic)) {
              // New political subtopic — significant bonus
              polDiversityBonus = 0.25;
            } else if (art.sourceLean !== 'B') {
              // Already-covered subtopic but different lean — smaller bonus
              polDiversityBonus = 0.08;
            }
          }
        }

        // Neighborhood bonus
        let inNeighborhood = 0;
        if (chosen.length > 0) {
          if (ci >= 0) {
            for (const c of chosen) {
              const cc = candidates[c];
              if (cc >= 0 && this._neighbors[ci].includes(cc)) {
                inNeighborhood = 1;
                break;
              }
            }
          }
        }

        const bonus = (inNeighborhood * (entityGain / (1 + entityGain))) + perspectiveBonus + polDiversityBonus;
        const score = (1 - lam) * relScore + lam * bonus;

        if (score > bestScore) { bestScore = score; bestI = i; }
      }

      if (bestI >= 0) {
        chosen.push(bestI);
        remaining.delete(bestI);
        coveredEntities.incKeys(entitySets[bestI]);
        // Track lean coverage for the selected article's cluster
        const bestCi = candidates[bestI];
        if (bestCi >= 0) {
          const selected = this.articles[bestCi];
          const cid = selected._cluster_group_id;
          const lean = selected.sourceLean || 'B';
          if (cid != null) {
            if (!clusterLeans.has(cid)) clusterLeans.set(cid, new Set());
            clusterLeans.get(cid).add(lean);
          }
          // Track political subtopic coverage
          if (selected.subtopic && POLITICAL_SUBTOPICS.has(selected.subtopic)) {
            coveredPolSubtopics.add(selected.subtopic);
          }
        }
      }
    }

    return chosen;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Strategy: Embedding MMR
  // ═══════════════════════════════════════════════════════════════════

  _embeddingMmr(candidates, relScores, lam) {
    const n = candidates.length;
    if (!this._embeddings || n <= 1) return Array.from({ length: n }, (_, i) => i);

    const chosen = [];
    const remaining = new Set(Array.from({ length: n }, (_, i) => i));

    while (remaining.size > 0) {
      let bestI = -1, bestScore = -Infinity;

      for (const i of remaining) {
        const relScore = relScores[i];

        // Max cosine similarity to any already-chosen article
        let maxSim = 0;
        if (chosen.length > 0) {
          const vi = this._embeddings[candidates[i]];
          if (vi) {
            for (const c of chosen) {
              const vc = this._embeddings[candidates[c]];
              if (vc) maxSim = Math.max(maxSim, cosineSimilarity(vi, vc));
            }
          }
        }

        const score = (1 - lam) * relScore - lam * maxSim;
        if (score > bestScore) { bestScore = score; bestI = i; }
      }

      if (bestI >= 0) { chosen.push(bestI); remaining.delete(bestI); }
    }

    return chosen;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Strategy: Hybrid MMR
  // ═══════════════════════════════════════════════════════════════════

  _hybridMmr(candidates, relScores, lam, alpha = 0.5) {
    const n = candidates.length;
    const chosen = [];
    const remaining = new Set(Array.from({ length: n }, (_, i) => i));
    const coveredEntities = new Counter();
    // Track lean coverage per cluster for perspective diversity
    const clusterLeans = new Map();
    // Track political subtopic coverage
    const coveredPolSubtopics = new Set();

    const entitySets = candidates.map(ci =>
      ci >= 0 ? new Set(this.articles[ci]._entities || []) : new Set()
    );
    const hasEmbed = this._embeddings != null;

    while (remaining.size > 0) {
      let bestI = -1, bestScore = -Infinity;

      for (const i of remaining) {
        const relScore = relScores[i];

        // Entity bonus
        const newEnts = difference(entitySets[i], new Set(coveredEntities.keys()));
        let entityGain = 0;
        for (const e of newEnts) entityGain += this._entityIdf.get(e) || 1;
        const entityBonus = entityGain / (1 + entityGain);

        // Cluster-aware perspective diversity bonus
        let perspectiveBonus = 0;
        const ci = candidates[i];
        if (ci >= 0 && chosen.length > 0) {
          const article = this.articles[ci];
          const clusterId = article._cluster_group_id;
          const lean = article.sourceLean || 'B';
          if (clusterId != null) {
            const coveredLeans = clusterLeans.get(clusterId);
            if (coveredLeans && coveredLeans.size > 0 && !coveredLeans.has(lean)) {
              perspectiveBonus = 0.3;
            }
          }
        }

        // Political subtopic diversity bonus
        let polDiversityBonus = 0;
        if (ci >= 0) {
          const art = this.articles[ci];
          if (art && art.subtopic && POLITICAL_SUBTOPICS.has(art.subtopic)) {
            if (!coveredPolSubtopics.has(art.subtopic)) {
              polDiversityBonus = 0.25;
            } else if (art.sourceLean !== 'B') {
              polDiversityBonus = 0.08;
            }
          }
        }

        // Embedding bonus
        let embedBonus = 0;
        if (hasEmbed && chosen.length > 0) {
          const vi = this._embeddings[ci];
          let maxSim = 0;
          for (const c of chosen) {
            const vc = this._embeddings[candidates[c]];
            maxSim = Math.max(maxSim, cosineSimilarity(vi, vc));
          }
          embedBonus = 1 - maxSim;
        } else if (!chosen.length) {
          embedBonus = 1;
        }

        // Neighborhood weighting
        let inNeighborhood = chosen.length === 0 ? 1 : 0;
        if (chosen.length > 0) {
          if (ci >= 0) {
            for (const c of chosen) {
              const cc = candidates[c];
              if (cc >= 0 && this._neighbors[ci].includes(cc)) {
                inNeighborhood = 1;
                break;
              }
            }
          }
        }

        const diversityBonus = alpha * entityBonus + (1 - alpha) * embedBonus + perspectiveBonus + polDiversityBonus;
        const score = (1 - lam) * relScore + lam * inNeighborhood * diversityBonus;

        if (score > bestScore) { bestScore = score; bestI = i; }
      }

      if (bestI >= 0) {
        chosen.push(bestI);
        remaining.delete(bestI);
        coveredEntities.incKeys(entitySets[bestI]);
        // Track lean coverage for selected article's cluster
        const bestCi = candidates[bestI];
        if (bestCi >= 0) {
          const selected = this.articles[bestCi];
          const cid = selected._cluster_group_id;
          const lean = selected.sourceLean || 'B';
          if (cid != null) {
            if (!clusterLeans.has(cid)) clusterLeans.set(cid, new Set());
            clusterLeans.get(cid).add(lean);
          }
          // Track political subtopic coverage
          if (selected.subtopic && POLITICAL_SUBTOPICS.has(selected.subtopic)) {
            coveredPolSubtopics.add(selected.subtopic);
          }
        }
      }
    }

    return chosen;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Strategy: Calibrated Reranker (KL-divergence minimisation)
  // ═══════════════════════════════════════════════════════════════════

  _calibrated(candidates, relScores, lam) {
    const n = candidates.length;
    if (n === 0) return [];

    // Use sourceLean + subtopic as cluster labels
    const clustersArr = candidates.map(ci => {
      if (ci < 0) return 'unknown|unknown';
      const lean = this.articles[ci].sourceLean || 'B';
      const sub = this.articles[ci].subtopic || 'General';
      return `${lean}|${sub}`;
    });

    const allClusters = [...new Set(clustersArr)].sort();
    const K = Math.max(1, allClusters.length);
    const clusterToIdx = new Map(allClusters.map((c, i) => [c, i]));
    const target = ones(K).map(v => v / K);  // uniform target distribution

    const chosen = [];
    const remaining = new Set(Array.from({ length: n }, (_, i) => i));
    const currentCounts = zeros(K);

    while (remaining.size > 0) {
      let bestI = -1, bestScore = Infinity;

      for (const i of remaining) {
        const relScore = relScores[i];
        const c = clustersArr[i];
        const cidx = clusterToIdx.get(c);
        if (cidx == null) continue;

        const tempCounts = currentCounts.slice();
        tempCounts[cidx] += 1;
        const dist = tempCounts.map(v => v / Math.max(1, sum(tempCounts)));

        // KL divergence
        let kl = 0;
        for (let k = 0; k < K; k++) {
          if (dist[k] > 0 && target[k] > 0) {
            kl += dist[k] * Math.log(dist[k] / target[k]);
          }
        }

        const score = (1 - lam) * (-relScore) + lam * kl;  // smaller is better
        if (score < bestScore) { bestScore = score; bestI = i; }
      }

      if (bestI >= 0) {
        chosen.push(bestI);
        remaining.delete(bestI);
        const cidx = clusterToIdx.get(clustersArr[bestI]);
        if (cidx != null) currentCounts[cidx] += 1;
      }
    }

    return chosen;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Explanation metadata
  // ═══════════════════════════════════════════════════════════════════

  _deriveClusterName(article) {
    const sub = article.subtopic || '';
    const tags = article.tags || [];
    if (sub) {
      const meaningful = tags.filter(t =>
        t.length > 2 && !['news', 'general', 'world', 'us'].includes(t.toLowerCase())
      );
      if (meaningful.length > 0) {
        const label = meaningful[0].charAt(0).toUpperCase() + meaningful[0].slice(1);
        return `${label} / ${sub}`;
      }
      return sub;
    }
    return article.category || 'General';
  }

  _calcDiversityGain(article, alreadySelected, rank) {
    if (rank === 0) return roundTo(0.6 * 0.8, 2);

    const prevArticles = alreadySelected.slice(0, rank);
    let gain = 0;

    const prevLeans = new Set(prevArticles.map(a => a.sourceLean));
    if (!prevLeans.has(article.sourceLean || 'B')) gain += 0.15;

    const prevSubs = new Set(prevArticles.map(a => a.subtopic));
    if (article.subtopic && !prevSubs.has(article.subtopic)) gain += 0.15;

    const curEnts = new Set(article._entities || []);
    const prevEnts = new Set();
    for (const pa of prevArticles) {
      for (const e of (pa._entities || [])) prevEnts.add(e);
    }
    const newEntCount = difference(curEnts, prevEnts).size;
    if (curEnts.size > 0) gain += 0.1 * (newEntCount / curEnts.size);

    gain += (Math.random() - 0.5) * 0.04;  // small jitter
    return roundTo(clamp(gain, 0.02, 0.35), 2);
  }

  _generateReason(article, alreadySelected, rank) {
    const origRank = article._original_rank || (rank + 1);
    const promoted = origRank > rank + 1;

    const lean = article.sourceLean || 'B';
    const leanLabel = { P: 'Left', B: 'Center', T: 'Right' }[lean] || lean;
    const cluster = article._cluster_name || '';

    if (rank === 0) {
      return `Top relevance match with ${leanLabel} viewpoint on ${cluster}`;
    }

    const prevArticles = alreadySelected.slice(0, rank);
    const prevLeans = new Set(prevArticles.map(a => a.sourceLean));
    const prevSubs = new Set(prevArticles.map(a => a.subtopic));
    const curSub = article.subtopic || '';

    const parts = [];
    if (promoted && origRank - (rank + 1) >= 3) parts.push('promoted for diverse perspective');
    if (!prevLeans.has(lean)) parts.push(`adds ${leanLabel} viewpoint`);
    else if (curSub && !prevSubs.has(curSub)) parts.push(`broadens coverage to ${curSub}`);
    else parts.push('adds a different angle within the same story');

    const reason = parts.join(', ');
    return reason.charAt(0).toUpperCase() + reason.slice(1);
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Coverage evaluation
  // ═══════════════════════════════════════════════════════════════════

  _computeCoverage(baseline, reranked) {
    const leanDist = (articles) => {
      const d = { P: 0, B: 0, T: 0, unknown: 0 };
      for (const a of articles) { const l = a.sourceLean || 'unknown'; d[l] = (d[l] || 0) + 1; }
      return d;
    };
    const subtopicSet = (articles) => new Set(articles.map(a => a.subtopic || ''));
    const entityCoverage = (articles) => {
      const s = new Set();
      for (const a of articles) for (const e of (a._entities || [])) s.add(e);
      return s.size;
    };

    const bLean = leanDist(baseline);
    const rLean = leanDist(reranked);
    const bSubs = subtopicSet(baseline);
    const rSubs = subtopicSet(reranked);
    const bEnt = entityCoverage(baseline);
    const rEnt = entityCoverage(reranked);

    // Political subtopic coverage
    const polSubtopics = (articles) => {
      const s = new Set();
      for (const a of articles) {
        if (a.subtopic && POLITICAL_SUBTOPICS.has(a.subtopic)) s.add(a.subtopic);
      }
      return s;
    };
    const bPolSubs = polSubtopics(baseline);
    const rPolSubs = polSubtopics(reranked);

    // Retention
    const baselineIds = new Set(baseline.map(a => a.id));
    const rerankedIds = reranked.map(a => a.id);
    const overlapCount = rerankedIds.filter(id => baselineIds.has(id)).length;
    const overlapRate = roundTo(overlapCount / Math.max(1, reranked.length), 4);

    // Average rank shift
    const bRankMap = new Map(baseline.map((a, i) => [a.id, i + 1]));
    const rankShifts = [];
    reranked.forEach((a, i) => {
      if (bRankMap.has(a.id)) rankShifts.push(Math.abs(bRankMap.get(a.id) - (i + 1)));
    });
    const avgRankShift = roundTo(rankShifts.length > 0 ? sum(rankShifts) / rankShifts.length : 0, 2);

    // Angle diversity (pairwise Jaccard distance)
    let angleDiversity = 0;
    const topN = reranked.slice(0, Math.min(10, reranked.length));
    if (topN.length >= 2) {
      const entSets = topN.map(a => new Set(a._entities || []));
      const dists = [];
      for (let i = 0; i < entSets.length; i++) {
        for (let j = i + 1; j < entSets.length; j++) {
          dists.push(jaccardDistance(entSets[i], entSets[j]));
        }
      }
      angleDiversity = roundTo(dists.length > 0 ? sum(dists) / dists.length : 0, 4);
    }

    return {
      baseline: {
        leanDistribution: bLean,
        uniqueSubtopics: bSubs.size,
        entityCoverage: bEnt,
        politicalSubtopics: bPolSubs.size,
      },
      reranked: {
        leanDistribution: rLean,
        uniqueSubtopics: rSubs.size,
        entityCoverage: rEnt,
        politicalSubtopics: rPolSubs.size,
      },
      delta: {
        entityCoverageGain: rEnt - bEnt,
        subtopicGain: rSubs.size - bSubs.size,
        politicalSubtopicGain: rPolSubs.size - bPolSubs.size,
      },
      retention: {
        articleOverlap: overlapCount,
        overlapRate,
        avgRankShift,
      },
      angleDiversity,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  Counter helper: inc multiple keys at once
// ═══════════════════════════════════════════════════════════════════════

// Extend Counter prototype
const { Counter: _Counter } = require('./utils');
_Counter.prototype.incKeys = function (keys) {
  for (const k of keys) this.inc(k);
  return this;
};

module.exports = { Reranker, TfidfBuilder };
