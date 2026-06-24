/**
 * server.js — MIND News Recommendation Backend (Express)
 *
 * Drop-in replacement for backend/server.py + dev_proxy.py.
 *
 * Start:  npm start        (node server.js)
 *         npm run dev      (node --watch server.js)
 * Port:   8096  (configurable via PORT env var)
 *
 * Endpoints (18) — same shape as Python backend.
 */

const path = require('path');
const fs = require('fs/promises');

const express = require('express');
const cors = require('cors');

const { loadData, getData, getVersion } = require('./data');
const { Reranker } = require('./reranker');
const { StaticProfile, ColdStartManager } = require('./coldstart');

// ─── Config ───────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || '8096', 10);
const ROOT = path.resolve(__dirname, '..');
const INDEX_HTML = path.join(ROOT, 'index.html');

// ─── Express setup ────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json());

// ── JSON parse error handler (must come BEFORE api routes) ──────────
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    // Return 400 instead of letting it propagate as uncaught exception
    return res.status(400).json({ error: 'Malformed JSON in request body' });
  }
  next(err);
});

// ─── Global state (lazy-initialised, similar to Python singletons) ────
let _dataPromise = null;       // resolved once
let _reranker = null;          // Reranker instance
let _coldstartMgr = null;      // ColdStartManager instance

// Registered users: { userId: { staticProfile, feedbackCount, onboardingDone, clickedArticles } }
const registeredUsers = new Map();
// User behaviours: { userId: { user_id, type, clicks, total_clicks, history, preferred_leans, liked_tags } }
const userBehaviors = new Map();

// ─── Preset user definitions ──────────────────────────────────────────
const PRESET_USER_DEFS = {
  U_TECH: {
    preferredLeans: ['B'],
    likedTags: ['technology', 'ai', 'science', 'tech', 'software',
                'data', 'digital', 'innovation', 'research', 'engineering'],
    categories: ['Technology', 'Science'],
    minClicks: 20,
  },
  U_SPORTS: {
    preferredLeans: ['B'],
    likedTags: ['sports', 'game', 'play', 'football', 'basketball',
                'athletic', 'team', 'league', 'championship', 'tournament'],
    categories: ['Sports'],
    minClicks: 20,
  },
  U_GENERAL: {
    preferredLeans: ['P', 'B'],
    likedTags: ['news', 'politics', 'world', 'finance', 'health',
                'policy', 'economy', 'global', 'government', 'society'],
    categories: ['News', 'World', 'Politics', 'Finance'],
    minClicks: 15,
  },
};

// ═══════════════════════════════════════════════════════════════════════
//  Lazy initialisers
// ═══════════════════════════════════════════════════════════════════════

async function ensureData() {
  if (_dataPromise) return _dataPromise;
  _dataPromise = loadData();
  return _dataPromise;
}

async function getReranker() {
  if (_reranker) return _reranker;
  await ensureData();
  const data = getData();
  _reranker = new Reranker(data.articles, { useEmbeddings: false }); // lightweight, no sentence-transformers
  return _reranker;
}

async function getColdstartManager() {
  if (_coldstartMgr) return _coldstartMgr;
  await ensureData();
  const data = getData();
  _coldstartMgr = new ColdStartManager(data.articles, await getReranker());
  return _coldstartMgr;
}

function initPresetUsers() {
  if (userBehaviors.size >= Object.keys(PRESET_USER_DEFS).length) return;

  const data = getData();
  const articles = data ? data.articles : [];

  for (const [uid, defn] of Object.entries(PRESET_USER_DEFS)) {
    if (userBehaviors.has(uid)) continue;

    const preferredCats = defn.categories;
    const likedTags = new Set(defn.likedTags);
    const preferredLeans = defn.preferredLeans;
    const minClicks = defn.minClicks;

    const scored = articles.map(a => {
      let s = 0;
      const catMatch = preferredCats.includes(a.category || '');
      const tagMatch = (a.tags || []).some(t => likedTags.has(t));
      if (catMatch && tagMatch) s = 3;
      else if (catMatch) s = 2;
      else if (tagMatch) s = 1;
      return { article: a, score: s };
    });

    scored.sort((a, b) => b.score - a.score || Math.random() - 0.5);
    const clickedArticles = scored.slice(0, minClicks).map(x => x.article);
    const clickedIds = clickedArticles.map(a => a.id);

    userBehaviors.set(uid, {
      user_id: uid,
      type: 'existing',
      clicks: clickedIds,
      total_clicks: clickedIds.length,
      history: clickedIds,
      preferred_leans: [...preferredLeans],
      liked_tags: [...likedTags],
    });

    console.log(`[Server] Initialized preset user ${uid}: ${clickedIds.length} clicks`);
  }
}

function ensureUser(userId) {
  if (userBehaviors.has(userId)) return userBehaviors.get(userId);

  const data = getData();
  const articles = data ? data.articles : [];
  const isNew = Math.random() < 0.25;

  let user;
  if (isNew) {
    user = {
      user_id: userId, type: 'new', clicks: [], total_clicks: 0, history: [],
    };
  } else {
    const numClicks = Math.floor(Math.random() * 26) + 5; // 5..30
    const allIds = articles.map(a => a.id);
    // Shuffle and pick
    const shuffled = [...allIds].sort(() => Math.random() - 0.5);
    const clickedIds = shuffled.slice(0, Math.min(numClicks, allIds.length));
    const clickedArticles = articles.filter(a => clickedIds.includes(a.id));

    user = {
      user_id: userId, type: 'existing',
      clicks: clickedIds, total_clicks: clickedIds.length, history: clickedIds,
      preferred_leans: extractLeanPreference(clickedArticles),
      liked_tags: extractLikedTags(clickedArticles),
    };
  }

  userBehaviors.set(userId, user);
  return user;
}

function extractLeanPreference(articles) {
  const counts = new Map();
  for (const a of articles) {
    const lean = a.sourceLean || 'B';
    counts.set(lean, (counts.get(lean) || 0) + 1);
  }
  const total = Math.max(1, [...counts.values()].reduce((s, v) => s + v, 0));
  return [...counts.entries()]
    .filter(([, v]) => v > total * 0.1)
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => k);
}

function extractLikedTags(articles) {
  const tagCounts = new Map();
  for (const a of articles) {
    for (const t of (a.tags || [])) {
      tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
    }
  }
  return [...tagCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([k]) => k);
}

// ─── Utility: strip internal fields ───────────────────────────────────
function cleanArticle(a) {
  const cleaned = {};
  for (const [k, v] of Object.entries(a)) {
    if (!k.startsWith('_mind') && !k.startsWith('_entities')) {
      cleaned[k] = v;
    }
  }
  return cleaned;
}

function cleanArticles(arr) {
  return arr.map(a => cleanArticle(a));
}

// ═══════════════════════════════════════════════════════════════════════
//  Static file serving (replaces dev_proxy.py)
// ═══════════════════════════════════════════════════════════════════════

app.get('/', async (_req, res) => {
  try {
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    const html = await fs.readFile(INDEX_HTML, 'utf-8');
    res.send(html);
  } catch {
    res.status(404).json({ error: 'index.html not found' });
  }
});

// Catch-all static file serving (JS, CSS, images, etc.)
app.get('/:filename(*)', async (req, res, next) => {
  const { filename } = req.params;
  // Skip API routes
  if (filename.startsWith('api/')) return next();

  if (filename === '' || filename === 'index.html') return next();

  const filePath = path.join(ROOT, filename);
  try {
    await fs.access(filePath);
    const ext = path.extname(filename).toLowerCase();
    const mimeTypes = {
      '.html': 'text/html; charset=utf-8',
      '.htm': 'text/html; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.js': 'application/javascript; charset=utf-8',
      '.json': 'application/json; charset=utf-8',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml; charset=utf-8',
      '.ico': 'image/x-icon',
      '.txt': 'text/plain; charset=utf-8',
    };
    res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
    const content = await fs.readFile(filePath);
    res.send(content);
  } catch {
    // Fallback: serve index.html (SPA mode)
    try {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      const html = await fs.readFile(INDEX_HTML, 'utf-8');
      res.send(html);
    } catch {
      res.status(404).json({ error: 'Not found' });
    }
  }
});

// ═══════════════════════════════════════════════════════════════════════
//  API Routes
// ═══════════════════════════════════════════════════════════════════════

const api = express.Router();

// ── Health ────────────────────────────────────────────────────────────
api.get('/health', (_req, res) => {
  res.json({ status: 'ok', server: 'MIND News Recommendation Backend (Node.js)' });
});

// ── Version ───────────────────────────────────────────────────────────
api.get('/version', async (_req, res) => {
  const version = await getVersion();
  res.json({ version });
});

// ── News ──────────────────────────────────────────────────────────────
api.get('/news', async (_req, res) => {
  try {
    await ensureData();
    const data = getData();
    let articles = data.articles;
    const { category, subtopic, lean, limit } = _req.query;

    let filtered = articles;
    if (category) filtered = filtered.filter(a => (a.category || '').toLowerCase() === category.toLowerCase());
    if (subtopic) filtered = filtered.filter(a => (a.subtopic || '').toLowerCase() === subtopic.toLowerCase());
    if (lean)    filtered = filtered.filter(a => a.sourceLean === lean.toUpperCase());
    if (limit)   filtered = filtered.slice(0, parseInt(limit, 10) || 20);

    res.json({ total: filtered.length, articles: cleanArticles(filtered) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

api.get('/news/:id(\\d+)', async (req, res) => {
  try {
    await ensureData();
    const id = parseInt(req.params.id, 10);
    const data = getData();
    const article = data.articles.find(a => a.id === id);
    if (!article) return res.status(404).json({ error: 'Article not found' });
    res.json(cleanArticle(article));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Stats ─────────────────────────────────────────────────────────────
api.get('/stats', async (_req, res) => {
  await ensureData();
  const data = getData();
  res.json({
    stats: data.stats,
    rawNewsCount: data.totalRawNews,
    behaviorRecords: data.totalBehaviors,
  });
});

// ── Categories ────────────────────────────────────────────────────────
api.get('/categories', async (_req, res) => {
  await ensureData();
  const data = getData();
  const cats = {};
  for (const a of data.articles) {
    const cat = a.category || 'Unknown';
    const sub = a.subtopic || 'General';
    if (!cats[cat]) cats[cat] = new Set();
    cats[cat].add(sub);
  }
  const result = {};
  for (const [k, v] of Object.entries(cats)) result[k] = [...v].sort();
  res.json(result);
});

// ── User profiles ─────────────────────────────────────────────────────
api.get('/user/:userId', async (req, res) => {
  await ensureData();
  const data = getData();
  const profile = data.userProfiles[req.params.userId];
  if (!profile) return res.status(404).json({ error: 'User not found' });
  res.json(profile);
});

api.get('/user-profiles', async (_req, res) => {
  await ensureData();
  const data = getData();
  const ids = Object.keys(data.userProfiles).slice(0, 200);
  res.json({ total: ids.length, userIds: ids });
});

// ── Recommend ─────────────────────────────────────────────────────────
api.post('/recommend', async (req, res) => {
  try {
    const { user_id, top_k = 10, preset_user_id } = req.body;
    const userId = user_id || 'U12345';

    await ensureData();
    const reranker = await getReranker();

    let usingPreset = null;
    let user;
    if (preset_user_id && PRESET_USER_DEFS[preset_user_id]) {
      initPresetUsers();
      user = userBehaviors.get(preset_user_id);
      usingPreset = preset_user_id;
    }
    if (!user) user = ensureUser(userId);

    const userProfile = {
      preferredLeans: user.preferred_leans || [],
      likedTags: user.liked_tags || [],
    };

    const baseline = reranker.baselineRank(userProfile, top_k);

    const clean = baseline.map(a => {
      const item = cleanArticle(a);
      item.score = a._baseline_score || 0;
      item.rank = a._rank || 0;
      return item;
    });

    res.json({
      user_id: userId,
      user_type: user.type,
      total_clicks: user.total_clicks,
      baseline: clean,
      total_available: (getData()?.articles || []).length,
      using_preset: usingPreset,
      preferred_leans: user.preferred_leans || [],
      liked_tags: user.liked_tags || [],
      timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19),
    });
  } catch (e) {
    console.error('[recommend] Error:', e);
    res.status(500).json({ error: e.message });
  }
});

// ── Rerank ────────────────────────────────────────────────────────────
api.post('/rerank', async (req, res) => {
  try {
    const {
      user_id, baseline_ids, method = 'entity_mmr',
      lam = 0.6, alpha = 0.5, top_k = 10, preset_user_id,
    } = req.body;
    const userId = user_id || 'U12345';

    await ensureData();
    const reranker = await getReranker();
    const data = getData();
    const idToArticle = new Map(data.articles.map(a => [a.id, a]));

    // Get user
    let user;
    if (preset_user_id && PRESET_USER_DEFS[preset_user_id]) {
      initPresetUsers();
      user = userBehaviors.get(preset_user_id);
    }
    if (!user) user = ensureUser(userId);

    // Build baseline
    let baseline;
    if (baseline_ids && baseline_ids.length > 0) {
      baseline = baseline_ids
        .filter(id => idToArticle.has(id))
        .map(id => {
          const a = { ...idToArticle.get(id) };
          a._baseline_score = Math.round((0.5 + Math.random() * 0.45) * 10000) / 10000;
          return a;
        });
      if (baseline.length === 0) baseline = reranker.baselineRank({}, top_k);
    } else {
      baseline = reranker.baselineRank({}, top_k);
    }
    baseline = baseline.slice(0, top_k);

    // Rerank
    const result = reranker.rerank(baseline, { method, lam, alpha });

    const clean = (articles) => articles.map(a => {
      const item = cleanArticle(a);
      item.score = a._reranked_score || a._baseline_score || 0;
      item.rank = a._reranked_rank || a._rank || 0;
      item.clusterName = a._cluster_name || '';
      item.clusterId = a._cluster_group_id != null ? a._cluster_group_id : -1;
      item.diversityGain = a._diversity_gain || 0;
      item.reason = a._reason || '';
      return item;
    });

    res.json({
      user_id: userId,
      method, lam, alpha,
      baseline: clean(baseline),
      reranked: clean(result.reranked),
      coverage_stats: result.coverageStats,
      using_preset: preset_user_id || null,
      timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19),
    });
  } catch (e) {
    console.error('[rerank] Error:', e);
    res.status(500).json({ error: e.message });
  }
});

// ── Feedback ──────────────────────────────────────────────────────────
api.post('/feedback', async (req, res) => {
  try {
    const { user_id, article_id, action = 'click' } = req.body;
    const userId = user_id || 'U12345';

    await ensureData();
    const data = getData();
    const user = ensureUser(userId);

    if (article_id != null && !user.clicks.includes(article_id)) {
      user.clicks.push(article_id);
      user.total_clicks = user.clicks.length;
      user.type = user.total_clicks > 0 ? 'existing' : 'new';

      // Rebuild user profile from all clicked articles
      const clickedArticles = data.articles.filter(a => user.clicks.includes(a.id));
      if (clickedArticles.length > 0) {
        user.preferred_leans = extractLeanPreference(clickedArticles);
        user.liked_tags = extractLikedTags(clickedArticles);
      }
    }

    // Find the last clicked article for client-side boosting
    let lastClicked = null;
    if (article_id != null) {
      lastClicked = data.articles.find(a => a.id === article_id) || null;
    }

    res.json({
      user_id: userId,
      user_type: user.type,
      total_clicks: user.total_clicks,
      last_action: action,
      article_id,
      preferred_leans: user.preferred_leans || [],
      liked_tags: user.liked_tags || [],
      last_clicked: lastClicked ? {
        id: lastClicked.id,
        subtopic: lastClicked.subtopic || null,
        category: lastClicked.category || null,
        sourceLean: lastClicked.sourceLean || null,
        clusterId: lastClicked._cluster_group_id != null ? lastClicked._cluster_group_id : null,
        tags: (lastClicked.tags || []).slice(0, 10),
      } : null,
      timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Users ─────────────────────────────────────────────────────────────
api.get('/users', async (_req, res) => {
  await ensureData();
  const data = getData();
  const profiles = data.userProfiles || {};
  const userList = Object.entries(profiles).slice(0, 20).map(([uid, p]) => ({
    userId: uid,
    historySize: p.historySize || 0,
    leanDistribution: p.leanDistribution || {},
  }));
  res.json({ total: userList.length, users: userList });
});

// ── Preset users ──────────────────────────────────────────────────────
api.get('/users/presets', async (_req, res) => {
  await ensureData();
  initPresetUsers();

  const presets = Object.entries(PRESET_USER_DEFS).map(([uid, defn]) => {
    const user = userBehaviors.get(uid) || {};
    return {
      userId: uid,
      name: uid,
      label: uid.replace('U_', '').replace('_', ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase()),
      categories: defn.categories,
      preferredLeans: defn.preferredLeans,
      totalClicks: user.total_clicks || defn.minClicks,
      clickedIds: (user.clicks || []).slice(0, 5),
      isPreset: true,
    };
  });

  res.json({ total: presets.length, presets });
});

// ── Article clusters ──────────────────────────────────────────────────
api.get('/articles/clusters', async (_req, res) => {
  await ensureData();
  const reranker = await getReranker();
  const data = getData();
  const clusters = new Map();

  for (const a of data.articles) {
    const gid = a._cluster_group_id;
    if (gid == null || gid < 0) continue;
    if (!clusters.has(gid)) clusters.set(gid, { clusterId: gid, articleIds: [], articles: [] });
    const c = clusters.get(gid);
    c.articleIds.push(a.id);
    c.articles.push({
      id: a.id, title: a.title || '', subtopic: a.subtopic || '',
      source: a.source || '', sourceLean: a.sourceLean || 'B', tags: a.tags || [],
    });
  }

  const result = [...clusters.values()].filter(c => c.articleIds.length >= 2);
  result.sort((a, b) => b.articleIds.length - a.articleIds.length);

  res.json({ totalClusters: result.length, clusters: result });
});

// ── Search ────────────────────────────────────────────────────────────
api.get('/search', async (req, res) => {
  await ensureData();
  const data = getData();
  let articles = data.articles;
  const { q, category, subtopic, lean, limit = 20 } = req.query;

  let results = articles;
  if (category) results = results.filter(a => (a.category || '').toLowerCase() === category.toLowerCase());
  if (subtopic) results = results.filter(a => (a.subtopic || '').toLowerCase() === subtopic.toLowerCase());
  if (lean)    results = results.filter(a => a.sourceLean === lean.toUpperCase());
  if (q) {
    const ql = q.toLowerCase();
    results = results.filter(a => {
      const title = (a.title || '').toLowerCase();
      const summary = (a.summary || '').toLowerCase();
      const tags = (a.tags || []).map(t => (typeof t === 'object' ? t.t || t : t).toLowerCase()).join(' ');
      return title.includes(ql) || summary.includes(ql) || tags.includes(ql);
    });
  }

  results = results.slice(0, parseInt(limit, 10) || 20);
  res.json({
    query: q, category: category || null, subtopic: subtopic || null, lean: lean || null,
    total: results.length, articles: cleanArticles(results),
  });
});

// ── Register (cold start) ─────────────────────────────────────────────
api.post('/register', async (req, res) => {
  try {
    const body = req.body || {};
    const userId = body.user_id || `U${Math.floor(10000 + Math.random() * 90000)}`;

    const profile = new StaticProfile({
      region: body.region || 'east',
      device_type: body.device_type || 'mobile',
      age_group: body.age_group || '25-34',
      gender: body.gender || 'unknown',
      reg_hour: body.reg_hour != null ? parseInt(body.reg_hour, 10) : 12,
    });

    registeredUsers.set(userId, {
      staticProfile: profile,
      feedbackCount: 0,
      onboardingDone: false,
      clickedArticles: [],
    });

    userBehaviors.set(userId, {
      user_id: userId, type: 'new', clicks: [], total_clicks: 0, history: [],
      static_profile: profile.toDict(),
    });

    const now = new Date();
    res.json({
      user_id: userId, registered: true,
      static_profile: profile.toDict(),
      timestamp: now.toISOString().replace('T', ' ').slice(0, 16),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Onboarding GET ────────────────────────────────────────────────────
api.get('/onboarding', async (_req, res) => {
  try {
    const mgr = await getColdstartManager();
    res.json(mgr.getOnboardingData());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Onboarding POST ───────────────────────────────────────────────────
api.post('/onboarding', async (req, res) => {
  try {
    const body = req.body || {};
    const userId = body.user_id || 'U12345';
    const mgr = await getColdstartManager();

    const profile = mgr.submitOnboarding(userId, {
      categoryTagIds: body.category_tag_ids,
      subtopicTagIds: body.subtopic_tag_ids,
    });

    // Update registered user state
    if (registeredUsers.has(userId)) {
      const reg = registeredUsers.get(userId);
      reg.onboardingDone = true;
      reg.feedbackCount = mgr._userFeedbackCounts.get(userId) || 4;
    }

    const result = {};
    for (const [k, v] of Object.entries(profile)) {
      if (k !== 'userId') result[k] = v;
    }

    const now = new Date();
    res.json({
      user_id: userId,
      profile: result,
      next_phase: 'Hybrid — Skipped pure E&E phase, entering hybrid blending directly',
      timestamp: now.toISOString().replace('T', ' ').slice(0, 16),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Cold-start recommend ──────────────────────────────────────────────
api.post('/recommend/coldstart', async (req, res) => {
  try {
    const body = req.body || {};
    const userId = body.user_id || 'U12345';
    const topK = body.top_k || 10;

    const mgr = await getColdstartManager();
    const reg = registeredUsers.get(userId) || {};
    const staticProfile = reg.staticProfile;
    const feedbackCount = reg.feedbackCount || (userBehaviors.get(userId)?.total_clicks || 0);

    const result = mgr.recommend(userId, {
      staticProfile, feedbackCount, topK,
    });

    const clean = result.recommendations.map(a => {
      const item = { ...a };
      delete item._entities;
      // Also delete internal _mind* fields
      for (const k of Object.keys(item)) {
        if (k.startsWith('_mind')) delete item[k];
      }
      item.score = a._baseline_score || a._popularity_score || 0;
      item.rank = a._rank || 0;
      return item;
    });

    const now = new Date();
    res.json({
      user_id: userId,
      phase: result.phase,
      phase_label: result.phaseLabel,
      feedback_count: result.feedbackCount,
      transition_progress: result.transitionProgress,
      recommendations: clean,
      strategy_breakdown: result.strategyBreakdown,
      bandit_stats: result.banditStats,
      timestamp: now.toISOString().replace('T', ' ').slice(0, 16),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Cold-start feedback ───────────────────────────────────────────────
api.post('/feedback/coldstart', async (req, res) => {
  try {
    const body = req.body || {};
    const userId = body.user_id || 'U12345';
    const articleId = body.article_id;
    const action = body.action || 'click';

    const mgr = await getColdstartManager();
    await ensureData();
    const data = getData();
    const idToArticle = new Map(data.articles.map(a => [a.id, a]));
    const article = idToArticle.get(articleId);

    mgr.recordFeedback(userId, articleId, action, article || null);

    // Update registered user state
    if (registeredUsers.has(userId)) {
      const reg = registeredUsers.get(userId);
      reg.feedbackCount = mgr._userFeedbackCounts.get(userId) || 0;
      if (articleId && !reg.clickedArticles.includes(articleId)) {
        reg.clickedArticles.push(articleId);
      }
    }

    // Update global behaviours
    const user = ensureUser(userId);
    if (articleId != null && !user.clicks.includes(articleId)) {
      user.clicks.push(articleId);
      user.total_clicks = user.clicks.length;
      user.type = user.total_clicks > 0 ? 'existing' : 'new';
    }

    const feedbackCount = mgr._userFeedbackCounts.get(userId) || 0;
    const phase = mgr.determinePhase(feedbackCount);

    const now = new Date();
    res.json({
      user_id: userId, article_id: articleId, action,
      feedback_count: feedbackCount,
      current_phase: phase,
      phase_label: ColdStartManager.PHASE_LABELS[phase],
      transition_progress: mgr._transitionProgress(feedbackCount),
      bandit_explore_rate: mgr.bandit.getExplorationRate(),
      timestamp: now.toISOString().replace('T', ' ').slice(0, 16),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Mount API routes ──────────────────────────────────────────────────
app.use('/api', api);

// ═══════════════════════════════════════════════════════════════════════
//  Global error handler
// ═══════════════════════════════════════════════════════════════════════

app.use((err, _req, res, _next) => {
  console.error('[Server] Uncaught error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// ═══════════════════════════════════════════════════════════════════════
//  Startup
// ═══════════════════════════════════════════════════════════════════════

let _server = null; // keep reference for graceful shutdown

async function startup() {
  console.log('='.repeat(60));
  console.log('  MIND News Recommendation Backend (Node.js)');
  console.log('  Port:', PORT);
  console.log('='.repeat(60));

  // Preload data & initialise singletons
  await ensureData();
  initPresetUsers();
  await getReranker();

  _server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] Listening on http://0.0.0.0:${PORT}`);
    console.log(`[Server] Endpoints: /api/news, /api/stats, /api/recommend, /api/rerank, /api/feedback`);
  });
}

startup().catch(err => {
  console.error('[Server] Fatal startup error:', err);
  process.exit(1);
});

// ═══════════════════════════════════════════════════════════════════════
//  Process-level safety nets (prevent silent crashes from stale healtcheck)
// ═══════════════════════════════════════════════════════════════════════

process.on('unhandledRejection', (reason, promise) => {
  console.error('[Server] Unhandled Rejection — Promise:', promise, '— Reason:', reason);
  // Do NOT exit — keep serving.  The rejection is logged; the healtcheck
  // will remain reachable unless the event-loop itself is blocked.
});

process.on('uncaughtException', (err) => {
  console.error('[Server] Uncaught Exception:', err);
  // Log & keep running.  Express's built-in error handler will serve 5xx
  // for subsequent requests on the affected route.
});

// ── Graceful shutdown (SIGTERM / SIGINT) ───────────────────────────
function shutdown(signal) {
  console.log(`[Server] Received ${signal}, shutting down gracefully...`);
  if (_server) {
    _server.close(() => {
      console.log('[Server] HTTP server closed.');
      process.exit(0);
    });
    // Force exit after 5 s if connections are still open
    setTimeout(() => {
      console.error('[Server] Forced shutdown after timeout.');
      process.exit(1);
    }, 5000);
  } else {
    process.exit(0);
  }
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
