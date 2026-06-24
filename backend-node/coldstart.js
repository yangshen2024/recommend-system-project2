/**
 * coldstart.js — Cold Start Recommendation Engine (pure JS)
 *
 * Four-phase progressive personalisation:
 *   Phase 0 — BOOTSTRAP    (0 interactions):  100% popular + static profile
 *   Phase 1 — EXPLORE      (1–3 interactions): 60% popular + 40% E&E
 *   Phase 2 — HYBRID       (4–7 interactions): 30% popular + 40% E&E + 30% personal
 *   Phase 3 — PERSONALIZED (8+ interactions):  100% personalised
 *
 * Replaces coldstart.py in full.
 */

const { zeros, dot, norm, randomChoice, shuffle, sum, max, roundTo, clamp } = require('./utils');

// ═══════════════════════════════════════════════════════════════════════
//  StaticProfile
// ═══════════════════════════════════════════════════════════════════════

class StaticProfile {
  constructor(opts = {}) {
    this.region = opts.region || 'east';
    this.deviceType = opts.device_type || opts.deviceType || 'mobile';
    this.ageGroup = opts.age_group || opts.ageGroup || '25-34';
    this.gender = opts.gender || 'unknown';
    this.regHour = opts.reg_hour != null ? opts.reg_hour : (opts.regHour || 12);
  }

  toDict() {
    return {
      region: this.region,
      deviceType: this.deviceType,
      ageGroup: this.ageGroup,
      gender: this.gender,
      regHour: this.regHour,
    };
  }

  toFeatureVector() {
    const vec = zeros(12);

    // Region → lean mapping
    const regionMap = {
      north:  [0.7, 0.5, 0.0],
      south:  [0.5, 0.6, 0.2],
      east:   [0.6, 0.5, 0.1],
      west:   [0.3, 0.7, 0.5],
      central:[0.4, 0.6, 0.3],
    };
    const r = regionMap[this.region] || [0.5, 0.5, 0.2];
    [vec[0], vec[1], vec[2]] = r;

    // Device
    const deviceMap = {
      mobile: [0.0, 0.8, -0.3],
      pc:     [0.6, 0.5, 0.4],
      tablet: [0.3, 0.8, 0.2],
    };
    const d = deviceMap[this.deviceType] || [0.3, 0.6, 0.1];
    [vec[3], vec[4], vec[5]] = d;

    // Age group
    const ageMap = {
      '18-24': [1.0, 0.8, 0.3],
      '25-34': [0.8, 0.7, 0.6],
      '35-44': [0.5, 0.6, 0.8],
      '45-54': [0.3, 0.5, 0.7],
      '55+':   [0.2, 0.4, 0.6],
    };
    const a = ageMap[this.ageGroup] || [0.5, 0.6, 0.5];
    [vec[6], vec[7], vec[8]] = a;

    // Registration hour
    vec[9] = this.regHour < 12 ? 1.0 : 0.3;
    vec[10] = this.regHour >= 18 ? 1.0 : 0.3;

    // Gender
    const genderMap = { male: 0.5, female: -0.5, other: 0.0 };
    vec[11] = genderMap[this.gender] || 0;

    return vec;
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  PopularRecommender
// ═══════════════════════════════════════════════════════════════════════

const CATEGORY_POPULARITY = {
  News: 0.88, Finance: 0.72, Technology: 0.78, Sports: 0.80,
  Entertainment: 0.90, Lifestyle: 0.65, Health: 0.60, Education: 0.55,
  Movies: 0.75, Tv: 0.82, Music: 0.70, Travel: 0.58,
  Food: 0.68, Science: 0.62, Politics: 0.73, World: 0.71,
  Foodanddrink: 0.65, Video: 0.70, Weather: 0.62, Autos: 0.55,
};

const SUBTOPIC_WEIGHTS = {
  // Political / governance subtopics — ensure high coverage for diversity
  Policy: 0.78,
  Elections: 0.80,
  'National Security': 0.75,
  Diplomacy: 0.73,
  Governance: 0.74,
  // Other subtopics
  Economy: 0.75,
  'Civil Rights': 0.72,
  Technology: 0.85,
  Education: 0.58,
  Entertainment: 0.85,
  Environment: 0.58,
};

// Subtopics that are explicitly political — used for perspective-diversity tracking
const POLITICAL_SUBTOPICS = new Set([
  'Policy', 'Elections', 'National Security', 'Diplomacy',
  'Governance', 'Civil Rights',
]);

function selectedByCat(selectedArr, cat) {
  let count = 0;
  for (const a of selectedArr) {
    if ((a.category || '') === cat) count++;
  }
  return count;
}

class PopularRecommender {
  constructor(articles) {
    this.articles = articles;
    this._computeScores();
  }

  _computeScores() {
    this.scores = new Float64Array(this.articles.length);

    for (let i = 0; i < this.articles.length; i++) {
      const a = this.articles[i];
      let s = 0;

      // Category popularity 30%
      s += (CATEGORY_POPULARITY[a.category] || 0.5) * 0.3;

      // Subtopic 20%
      s += (SUBTOPIC_WEIGHTS[a.subtopic] || 0.5) * 0.2;

      // Title length 10%
      s += Math.min((a.title || '').length / 120, 1) * 0.1;

      // Summary length 15%
      s += Math.min((a.summary || '').length / 500, 1) * 0.15;

      // Tag richness 10%
      s += Math.min((a.tags || []).length / 10, 1) * 0.1;

      // Lean bonus 5%
      s += (a.sourceLean === 'B' ? 0.8 : 0.5) * 0.05;

      // Random jitter 10%
      s += Math.random() * 0.1;

      this.scores[i] = Math.min(s, 1);
    }
  }

  recommend(topK = 10, categoryBalanced = true) {
    if (!categoryBalanced) {
      const ranked = this.articles
        .map((a, i) => ({ ...a, _popularity_score: roundTo(this.scores[i], 4) }))
        .sort((a, b) => b._popularity_score - a._popularity_score)
        .slice(0, topK)
        .map((a, i) => ({ ...a, _rank: i + 1 }));
      return ranked;
    }

    // Category-balanced round-robin with proportional allocation
    // Larger categories (e.g. News with 71 articles) get proportionally
    // more slots than tiny categories (e.g. Autos with 1 article).
    const byCat = new Map();
    for (let i = 0; i < this.articles.length; i++) {
      const cat = this.articles[i].category || 'News';
      if (!byCat.has(cat)) byCat.set(cat, []);
      byCat.get(cat).push({ idx: i, score: this.scores[i] });
    }
    for (const [, items] of byCat) {
      items.sort((a, b) => b.score - a.score);
    }

    // Compute slot allocation proportional to sqrt(catSize) to prevent
    // dominance while giving larger categories a fairer share.
    const catKeys = [...byCat.keys()].sort();
    const catSizes = catKeys.map(k => byCat.get(k).length);
    const sqrtSizes = catSizes.map(s => Math.sqrt(Math.max(1, s)));
    const totalSqrt = sqrtSizes.reduce((s, v) => s + v, 0);
    const catTargets = sqrtSizes.map(ss => Math.max(1, Math.round(ss / totalSqrt * topK)));

    const selected = [];
    const seen = new Set();
    const catPtr = new Array(catKeys.length).fill(0);
    const maxRounds = Math.max(...catSizes);
    let round = 0;

    while (selected.length < topK && round < maxRounds) {
      for (let ci = 0; ci < catKeys.length; ci++) {
        if (selected.length >= topK) break;
        const pool = byCat.get(catKeys[ci]);
        // Skip categories that already met their target
        if (selectedByCat(selected, catKeys[ci]) >= catTargets[ci]) continue;
        if (catPtr[ci] < pool.length) {
          const { idx, score } = pool[catPtr[ci]];
          if (!seen.has(idx)) {
            seen.add(idx);
            selected.push({
              ...this.articles[idx],
              _popularity_score: roundTo(score, 4),
              _rank: selected.length + 1,
            });
          }
        }
        catPtr[ci]++;
      }
      round++;
    }

    // Fill remaining
    if (selected.length < topK) {
      const remaining = this.articles
        .map((_, i) => ({ idx: i, score: this.scores[i] }))
        .filter(({ idx }) => !seen.has(idx))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK - selected.length);
      for (const { idx, score } of remaining) {
        selected.push({
          ...this.articles[idx],
          _popularity_score: roundTo(score, 4),
          _rank: selected.length + 1,
        });
      }
    }

    return selected.slice(0, topK);
  }

  getTrendingCategories() {
    const catScores = new Map();
    for (let i = 0; i < this.articles.length; i++) {
      const cat = this.articles[i].category || 'News';
      catScores.set(cat, (catScores.get(cat) || 0) + this.scores[i]);
    }
    return [...catScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(e => e[0]);
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  InterestCapture
// ═══════════════════════════════════════════════════════════════════════

class InterestCapture {
  static CATEGORY_TAGS = [
    { id: 'cat_news',         label: 'News & Politics',   categories: ['News'] },
    { id: 'cat_tech',         label: 'Technology',         categories: ['Technology', 'Science'] },
    { id: 'cat_finance',      label: 'Finance',            categories: ['Finance'] },
    { id: 'cat_entertainment',label: 'Entertainment',      categories: ['Entertainment', 'Movies', 'Tv', 'Music', 'Video'] },
    { id: 'cat_sports',       label: 'Sports',             categories: ['Sports'] },
    { id: 'cat_lifestyle',    label: 'Lifestyle',          categories: ['Lifestyle', 'Foodanddrink'] },
    { id: 'cat_weather',      label: 'Weather & Science',  categories: ['Weather'] },
    { id: 'cat_autos',        label: 'Autos',              categories: ['Autos'] },
  ];

  static SUBTOPIC_TAGS = [
    { id: 'sub_policy',         label: 'Policy & Governance',  subtopics: ['Policy', 'Governance'] },
    { id: 'sub_elections',      label: 'Elections',             subtopics: ['Elections'] },
    { id: 'sub_national_sec',   label: 'National Security',     subtopics: ['National Security'] },
    { id: 'sub_diplomacy',      label: 'Global Diplomacy',      subtopics: ['Diplomacy'] },
    { id: 'sub_civil_rights',   label: 'Civil & Human Rights',  subtopics: ['Civil Rights', 'Policy'] },
    { id: 'sub_international',  label: 'International',         subtopics: ['Diplomacy'] },
    { id: 'sub_economy',        label: 'Economy & Markets',     subtopics: ['Economy'] },
    { id: 'sub_ai_ml',          label: 'AI & ML',              subtopics: ['Technology', 'Science'] },
    { id: 'sub_cybersecurity',  label: 'Cybersecurity',        subtopics: ['Technology'] },
    { id: 'sub_consumer_tech',  label: 'Consumer Tech',        subtopics: ['Technology'] },
    { id: 'sub_space',          label: 'Space Exploration',    subtopics: ['Science', 'Technology'] },
    { id: 'sub_blockchain',     label: 'Blockchain & Web3',    subtopics: ['Technology', 'Finance'] },
    { id: 'sub_robotics',       label: 'Robotics',             subtopics: ['Technology', 'Science'] },
    { id: 'sub_stocks',         label: 'Stock Markets',        subtopics: ['Finance'] },
    { id: 'sub_personal_finance', label: 'Personal Finance',   subtopics: ['Finance'] },
    { id: 'sub_real_estate',    label: 'Real Estate',          subtopics: ['Finance', 'Economy'] },
    { id: 'sub_crypto',         label: 'Cryptocurrency',       subtopics: ['Finance', 'Technology'] },
    { id: 'sub_banking',        label: 'Banking',              subtopics: ['Finance'] },
    { id: 'sub_trade',          label: 'Global Trade',         subtopics: ['Economy', 'Finance'] },
    { id: 'sub_movies',         label: 'Movies & Cinema',      subtopics: ['Entertainment', 'Movies'] },
    { id: 'sub_television',     label: 'Television',           subtopics: ['Entertainment', 'Tv'] },
    { id: 'sub_music',          label: 'Music Industry',       subtopics: ['Entertainment', 'Music'] },
    { id: 'sub_gaming',         label: 'Gaming & Esports',     subtopics: ['Entertainment', 'Technology'] },
    { id: 'sub_celebrity',      label: 'Celebrity Culture',    subtopics: ['Entertainment'] },
    { id: 'sub_theater',        label: 'Theater & Arts',       subtopics: ['Entertainment', 'Education'] },
    { id: 'sub_football',       label: 'Football & Soccer',    subtopics: ['Sports'] },
    { id: 'sub_basketball',     label: 'Basketball',           subtopics: ['Sports'] },
    { id: 'sub_tennis',         label: 'Racquet Sports',       subtopics: ['Sports'] },
    { id: 'sub_olympics',       label: 'Olympic Sports',       subtopics: ['Sports'] },
    { id: 'sub_motorsports',    label: 'Motorsports',          subtopics: ['Sports'] },
    { id: 'sub_extreme',        label: 'Extreme Sports',       subtopics: ['Sports'] },
    { id: 'sub_wellness',       label: 'Health & Wellness',    subtopics: ['Health'] },
    { id: 'sub_travel',         label: 'Travel & Adventure',   subtopics: ['Travel'] },
    { id: 'sub_food',           label: 'Food & Dining',        subtopics: ['Food', 'Lifestyle'] },
    { id: 'sub_fashion',        label: 'Fashion & Style',      subtopics: ['Lifestyle'] },
    { id: 'sub_fitness',        label: 'Fitness',              subtopics: ['Health', 'Lifestyle'] },
    { id: 'sub_home',           label: 'Home & Living',        subtopics: ['Lifestyle'] },
    { id: 'sub_higher_ed',      label: 'Higher Education',     subtopics: ['Education'] },
    { id: 'sub_online_learning',label: 'Online Learning',      subtopics: ['Education', 'Technology'] },
    { id: 'sub_research',       label: 'Academic Research',    subtopics: ['Education', 'Science'] },
    { id: 'sub_career_dev',     label: 'Career Development',   subtopics: ['Education'] },
    { id: 'sub_books',          label: 'Books & Literature',   subtopics: ['Education'] },
    { id: 'sub_history',        label: 'History & Heritage',   subtopics: ['Education'] },
    { id: 'sub_environment',    label: 'Environment & Climate',subtopics: ['Environment'] },
  ];

  constructor() {
    this._userSelections = new Map();
  }

  getCategoryTags() { return InterestCapture.CATEGORY_TAGS; }
  getSubtopicTags() { return InterestCapture.SUBTOPIC_TAGS; }

  recordSelections(userId, { categoryTagIds, subtopicTagIds } = {}) {
    const preferredCategories = [];
    const preferredSubtopics = [];

    if (categoryTagIds && categoryTagIds.length > 0) {
      const tagMap = new Map(InterestCapture.CATEGORY_TAGS.map(t => [t.id, t]));
      for (const tid of categoryTagIds) {
        const t = tagMap.get(tid);
        if (t) preferredCategories.push(...t.categories);
      }
    }

    if (subtopicTagIds && subtopicTagIds.length > 0) {
      const subMap = new Map(InterestCapture.SUBTOPIC_TAGS.map(t => [t.id, t]));
      for (const sid of subtopicTagIds) {
        const t = subMap.get(sid);
        if (t) preferredSubtopics.push(...t.subtopics);
      }
    }

    const profile = {
      userId,
      preferredCategories: [...new Set(preferredCategories)],
      preferredSubtopics: [...new Set(preferredSubtopics)],
      preferredLeans: ['B'],
      pctScore: (categoryTagIds || []).length * 0.1 + (subtopicTagIds || []).length * 0.15,
    };

    this._userSelections.set(userId, profile);
    return profile;
  }

  getPseudoProfile(userId) {
    return this._userSelections.get(userId) || null;
  }

  convertToUserProfile(userId) {
    const sel = this._userSelections.get(userId) || {};
    return {
      preferredLeans: sel.preferredLeans || ['B'],
      preferredCategories: sel.preferredCategories || [],
      preferredSubtopics: sel.preferredSubtopics || [],
      likedTags: [],
      isColdstart: true,
      pctScore: sel.pctScore || 0,
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  BanditExplorer — epsilon-greedy / UCB1 / Thompson Sampling
// ═══════════════════════════════════════════════════════════════════════

class BanditExplorer {
  /**
   * @param {string[]} arms
   * @param {string} [algorithm='epsilon_greedy']
   * @param {number} [epsilon=0.3]
   * @param {number} [decayRate=0.05]
   */
  constructor(arms, algorithm = 'epsilon_greedy', epsilon = 0.3, decayRate = 0.05) {
    this.arms = arms;
    this.algorithm = algorithm;
    this.epsilon = epsilon;
    this.decayRate = decayRate;

    this.counts = new Map(arms.map(a => [a, 0]));
    this.rewards = new Map(arms.map(a => [a, 0]));
    this.means = new Map(arms.map(a => [a, 0.5]));
    this.alpha = new Map(arms.map(a => [a, 1]));
    this.beta = new Map(arms.map(a => [a, 1]));
    this.totalPulls = 0;
  }

  selectArm() {
    this.totalPulls++;
    switch (this.algorithm) {
      case 'ucb1':     return this._ucb1Select();
      case 'thompson': return this._thompsonSelect();
      default:         return this._epsilonGreedySelect();
    }
  }

  update(arm, reward) {
    if (!this.counts.has(arm)) return;
    this.counts.set(arm, this.counts.get(arm) + 1);
    this.rewards.set(arm, this.rewards.get(arm) + reward);
    this.means.set(arm, this.rewards.get(arm) / Math.max(1, this.counts.get(arm)));

    if (reward > 0.5) this.alpha.set(arm, this.alpha.get(arm) + 1);
    else this.beta.set(arm, this.beta.get(arm) + 1);

    this.epsilon = Math.max(0.02, this.epsilon * (1 - this.decayRate));
  }

  _epsilonGreedySelect() {
    if (Math.random() < this.epsilon) return randomChoice(this.arms);
    return this.arms.reduce((best, arm) => this.means.get(arm) > this.means.get(best) ? arm : best);
  }

  _ucb1Select() {
    let bestArm = this.arms[0], bestUcb = -Infinity;
    for (const arm of this.arms) {
      if (this.counts.get(arm) === 0) return arm;
      const exploit = this.means.get(arm);
      const explore = Math.sqrt(2 * Math.log(this.totalPulls) / this.counts.get(arm));
      const ucb = exploit + explore;
      if (ucb > bestUcb) { bestUcb = ucb; bestArm = arm; }
    }
    return bestArm;
  }

  _thompsonSelect() {
    let bestArm = this.arms[0], bestSample = -Infinity;
    for (const arm of this.arms) {
      // Marsaglia-Tsang gamma approximation for Beta sampling
      const a = this.alpha.get(arm), b = this.beta.get(arm);
      const gammaX = this._gammaSample(a), gammaY = this._gammaSample(b);
      const sample = gammaX / (gammaX + gammaY);
      if (sample > bestSample) { bestSample = sample; bestArm = arm; }
    }
    return bestArm;
  }

  /** Simple Marsaglia-Tsang gamma variate for shape > 0 */
  _gammaSample(shape) {
    if (shape < 1) {
      const u = Math.random();
      return this._gammaSample(shape + 1) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        x = this._normalSample();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** Box-Muller transform for N(0,1) */
  _normalSample() {
    let u1, u2;
    do { u1 = Math.random(); } while (u1 === 0);
    u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  getArmDistribution() {
    const result = {};
    for (const arm of this.arms) {
      result[arm] = {
        count: this.counts.get(arm),
        mean: roundTo(this.means.get(arm), 4),
        exploreProb: this.algorithm === 'epsilon_greedy' ? roundTo(this.epsilon, 4) : 0,
      };
    }
    return result;
  }

  getExplorationRate() {
    if (this.algorithm === 'epsilon_greedy') return this.epsilon;
    if (this.totalPulls === 0) return 1;
    let unplayed = 0;
    for (const arm of this.arms) if (this.counts.get(arm) === 0) unplayed++;
    return unplayed / this.arms.length;
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  ColdStartManager — orchestrator
// ═══════════════════════════════════════════════════════════════════════

class ColdStartManager {
  static PHASE_BOOTSTRAP = 0;
  static PHASE_EXPLORE = 1;
  static PHASE_HYBRID = 2;
  static PHASE_PERSONALIZED = 3;

  static PHASE_LABELS = {
    0: 'Bootstrap — Popular',
    1: 'Explore — Exploration-led',
    2: 'Hybrid — Blending',
    3: 'Personalized — Personalized',
  };

  /**
   * @param {Object[]} articles
   * @param {import('./reranker').Reranker} [reranker]
   */
  constructor(articles, reranker = null) {
    this.articles = articles;
    this.reranker = reranker;
    this.popular = new PopularRecommender(articles);
    this.interestCapture = new InterestCapture();

    const categories = [...new Set(articles.map(a => a.category || 'News'))];
    this.bandit = new BanditExplorer(categories, 'epsilon_greedy', 0.4, 0.08);

    this._userPhase = new Map();
    this._userFeedbackCounts = new Map();
    this._previousRecs = new Map();
  }

  determinePhase(feedbackCount) {
    if (feedbackCount === 0) return ColdStartManager.PHASE_BOOTSTRAP;
    if (feedbackCount <= 3) return ColdStartManager.PHASE_EXPLORE;
    if (feedbackCount <= 7) return ColdStartManager.PHASE_HYBRID;
    return ColdStartManager.PHASE_PERSONALIZED;
  }

  recommend(userId, { staticProfile, feedbackCount = 0, topK = 10, smoothTransition = true } = {}) {
    const phase = this.determinePhase(feedbackCount);
    this._userPhase.set(userId, phase);
    this._userFeedbackCounts.set(userId, feedbackCount);

    const staticVec = staticProfile ? staticProfile.toFeatureVector() : null;

    const result = this._dispatchPhase(userId, phase, staticVec, feedbackCount, topK);

    if (smoothTransition && feedbackCount > 0) {
      result.recommendations = this._smoothResults(userId, result.recommendations, topK);
    }

    this._previousRecs.set(userId, result.recommendations.map(a => a.id));

    result.userId = userId;
    result.phase = phase;
    result.phaseLabel = ColdStartManager.PHASE_LABELS[phase];
    result.feedbackCount = feedbackCount;
    result.transitionProgress = this._transitionProgress(feedbackCount);

    return result;
  }

  _dispatchPhase(userId, phase, staticVec, feedbackCount, topK) {
    switch (phase) {
      case ColdStartManager.PHASE_BOOTSTRAP:    return this._phaseBootstrap(staticVec, topK);
      case ColdStartManager.PHASE_EXPLORE:      return this._phaseExplore(userId, staticVec, topK);
      case ColdStartManager.PHASE_HYBRID:       return this._phaseHybrid(userId, staticVec, topK);
      default:                                  return this._phasePersonalized(userId, topK);
    }
  }

  _phaseBootstrap(staticVec, topK) {
    let hot = this.popular.recommend(Math.ceil(topK * 1.5), true);
    if (staticVec) hot = this._rerankByStaticProfile(hot, staticVec);
    return {
      recommendations: hot.slice(0, topK),
      strategyBreakdown: { popular: 1, staticProfile: staticVec ? 1 : 0, banditExplore: 0, personalized: 0 },
      banditStats: this.bandit.getArmDistribution(),
    };
  }

  _phaseExplore(userId, staticVec, topK) {
    const nHot = Math.max(1, Math.floor(topK * 0.6));
    const nExplore = topK - nHot;
    const hot = this.popular.recommend(nHot, true);
    const exploreArticles = [];

    const exploredCats = new Set();
    for (let k = 0; k < nExplore; k++) {
      let arm = this.bandit.selectArm();
      let attempts = 0;
      while (exploredCats.has(arm) && attempts < 5) { arm = this.bandit.selectArm(); attempts++; }
      exploredCats.add(arm);

      const catArticles = this.articles.filter(a => a.category === arm);
      if (catArticles.length > 0) {
        const best = catArticles.reduce((prev, cur) => {
          const pi = this.articles.indexOf(prev), ci = this.articles.indexOf(cur);
          return (this.popular.scores[pi] || 0) > (this.popular.scores[ci] || 0) ? prev : cur;
        });
        const a = { ...best, _explore_arm: arm, _explore_epsilon: roundTo(this.bandit.epsilon, 4), _baseline_score: roundTo(0.3 + Math.random() * 0.3, 4) };
        exploreArticles.push(a);
      }
    }

    const combined = shuffle([...hot, ...exploreArticles]);
    combined.forEach((a, i) => { a._rank = i + 1; });

    for (const a of exploreArticles) {
      if (a._explore_arm) this.bandit.update(a._explore_arm, 0.1);
    }

    return {
      recommendations: combined.slice(0, topK),
      strategyBreakdown: { popular: 0.6, staticProfile: staticVec ? 0.3 : 0, banditExplore: 0.4, personalized: 0 },
      banditStats: this.bandit.getArmDistribution(),
    };
  }

  _phaseHybrid(userId, staticVec, topK) {
    const nHot = Math.max(1, Math.floor(topK * 0.3));
    const nExplore = Math.max(1, Math.floor(topK * 0.4));
    const nPersonalized = topK - nHot - nExplore;

    const hot = this.popular.recommend(nHot, true);
    const exploreArticles = [];
    const exploredCats = new Set();

    for (let k = 0; k < nExplore; k++) {
      let arm = this.bandit.selectArm();
      let attempts = 0;
      while (exploredCats.has(arm) && attempts < 5) { arm = this.bandit.selectArm(); attempts++; }
      exploredCats.add(arm);
      const catArticles = this.articles.filter(a => a.category === arm);
      if (catArticles.length > 0) {
        const best = randomChoice(catArticles.slice(0, 10));
        const a = { ...best, _explore_arm: arm, _baseline_score: roundTo(0.3 + Math.random() * 0.4, 4) };
        exploreArticles.push(a);
      }
    }

    let personalized = [];
    if (this.reranker) {
      try {
        const profile = this.interestCapture.convertToUserProfile(userId);
        if (!profile.preferredCategories || profile.preferredCategories.length === 0) {
          profile.preferredLeans = ['B'];
        }
        personalized = this.reranker.baselineRank(profile, nPersonalized);
      } catch {
        personalized = this.popular.recommend(nPersonalized, true);
      }
    } else {
      personalized = this.popular.recommend(nPersonalized, true);
    }

    const combined = shuffle([...hot, ...exploreArticles, ...personalized]);
    combined.forEach((a, i) => { a._rank = i + 1; });

    return {
      recommendations: combined.slice(0, topK),
      strategyBreakdown: { popular: 0.3, banditExplore: 0.4, personalized: 0.3 },
      banditStats: this.bandit.getArmDistribution(),
    };
  }

  _phasePersonalized(userId, topK) {
    if (this.reranker) {
      try {
        const profile = this.interestCapture.convertToUserProfile(userId);
        const baseline = this.reranker.baselineRank(profile, topK);
        return {
          recommendations: baseline,
          strategyBreakdown: { popular: 0, banditExplore: 0.05, personalized: 0.95 },
          banditStats: this.bandit.getArmDistribution(),
        };
      } catch { /* fallback */ }
    }
    const hot = this.popular.recommend(topK, true);
    return {
      recommendations: hot,
      strategyBreakdown: { popular: 1, banditExplore: 0, personalized: 0 },
      banditStats: this.bandit.getArmDistribution(),
    };
  }

  _rerankByStaticProfile(articles, staticVec) {
    const scored = articles.map(a => {
      const av = this._articleToStaticVec(a);
      const sim = dot(staticVec, av) / (Math.max(norm(staticVec), 1e-8) * Math.max(norm(av), 1e-8));
      return { ...a, _static_similarity: roundTo(sim, 4) };
    });
    scored.sort((a, b) => b._static_similarity - a._static_similarity);
    scored.forEach((a, i) => { a._rank = i + 1; a._baseline_score = roundTo(0.4 + a._static_similarity * 0.5, 4); });
    return scored;
  }

  _articleToStaticVec(article) {
    const vec = zeros(12);
    const catMap = {
      News: [0.6, 0.5, 0.3], World: [0.7, 0.4, 0.2], Politics: [0.8, 0.3, 0.5],
      Finance: [0.3, 0.8, 0.6], Technology: [0.5, 0.6, 0.4], Science: [0.4, 0.7, 0.3],
      Entertainment: [0.1, 0.4, 0.1], Movies: [0.0, 0.5, 0.2], Tv: [0.0, 0.5, 0.2],
      Music: [0.1, 0.4, 0.3], Sports: [0.5, 0.5, 0.3], Lifestyle: [0.2, 0.5, 0.4],
      Health: [0.3, 0.6, 0.7], Education: [0.5, 0.5, 0.5], Food: [0.2, 0.5, 0.5],
      Travel: [0.3, 0.5, 0.4],
    };
    const cr = catMap[article.category] || [0.5, 0.5, 0.3];
    [vec[0], vec[1], vec[2]] = cr;

    const tl = (article.title || '').length;
    if (tl < 40) [vec[3], vec[4], vec[5]] = [0.0, 0.8, -0.3];
    else if (tl > 80) [vec[3], vec[4], vec[5]] = [0.6, 0.5, 0.4];

    const subMap = {
      Technology: [0.8, 0.7, 0.6], Economy: [0.5, 0.6, 0.8], Health: [0.3, 0.5, 0.7],
      Entertainment: [1.0, 0.8, 0.3], Policy: [0.5, 0.6, 0.5], Education: [0.5, 0.6, 0.8],
      'Civil Rights': [0.6, 0.5, 0.6],
    };
    const sr = subMap[article.subtopic] || [0.5, 0.6, 0.5];
    [vec[6], vec[7], vec[8]] = sr;

    const leanMap = { P: 0.8, B: 0.5, T: 0.2 };
    vec[11] = leanMap[article.sourceLean] || 0.5;

    return vec;
  }

  _smoothResults(userId, current, topK) {
    const prevIds = this._previousRecs.get(userId) || [];
    if (prevIds.length === 0) return current;

    const idMap = new Map(this.articles.map(a => [a.id, a]));
    const keepCount = Math.max(1, Math.floor(topK * 0.2));
    const kept = prevIds.slice(0, keepCount)
      .filter(id => idMap.has(id))
      .map(id => ({
        ...idMap.get(id),
        _from_previous_round: true,
        _baseline_score: roundTo((idMap.get(id)._baseline_score || 0.5) * 0.9, 4),
      }));

    const keptIds = new Set(kept.map(a => a.id));
    const newFiltered = current.filter(a => !keptIds.has(a.id));
    const combined = [...kept, ...newFiltered];
    combined.slice(0, topK).forEach((a, i) => { a._rank = i + 1; });
    return combined.slice(0, topK);
  }

  _transitionProgress(feedbackCount) {
    return Math.min(1, feedbackCount / 8);
  }

  recordFeedback(userId, articleId, action = 'click', article = null) {
    const rewardMap = { click: 1, like: 1.5, dislike: -0.5, skip: -0.1 };
    const reward = rewardMap[action] || 0.5;

    if (article) {
      const cat = article.category || 'News';
      this.bandit.update(cat, Math.max(0, reward));
    }

    this._userFeedbackCounts.set(userId, (this._userFeedbackCounts.get(userId) || 0) + 1);
  }

  getOnboardingData() {
    return {
      category_tags: this.interestCapture.getCategoryTags(),
      subtopic_tags: this.interestCapture.getSubtopicTags(),
      trending_categories: this.popular.getTrendingCategories(),
    };
  }

  submitOnboarding(userId, { categoryTagIds, subtopicTagIds } = {}) {
    const profile = this.interestCapture.recordSelections(userId, { categoryTagIds, subtopicTagIds });
    this._userFeedbackCounts.set(userId, Math.max(this._userFeedbackCounts.get(userId) || 0, 4));
    return profile;
  }
}

module.exports = { StaticProfile, PopularRecommender, InterestCapture, BanditExplorer, ColdStartManager, POLITICAL_SUBTOPICS };
