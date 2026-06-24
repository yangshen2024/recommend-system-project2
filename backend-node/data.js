/**
 * data.js — Data loading layer
 *
 * Replaces mind_adapter.py + server.py data-loading logic.
 * Loads articles from pre-generated JSON files (same format as Python backend).
 *
 * Usage:
 *   const { loadData } = require('./data');
 *   const data = await loadData();
 */

const fs = require('fs/promises');
const path = require('path');
const { createReadStream } = require('fs');

// ─── Paths ────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, '..');
const ARTICLES_PATH = path.join(ROOT, 'backend', 'mind_articles.json');
const STATS_PATH    = path.join(ROOT, 'backend', 'mind_stats.json');
const VERSION_PATH  = path.join(ROOT, 'VERSION');

// ─── Global cache ─────────────────────────────────────────────────────
let _dataCache = null;
let _loadPromise = null;

/**
 * Load all data (articles + stats).
 *
 * Uses a shared promise to avoid concurrent loads (only the first call
 * does actual I/O; subsequent calls await the same promise).
 *
 * @returns {Promise<{articles: Array, stats: Object, totalRawNews: number, totalBehaviors: number}>}
 */
async function loadData() {
  if (_dataCache) return _dataCache;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    const articles = await loadJSON(ARTICLES_PATH);
    let stats = {};
    try {
      stats = await loadJSON(STATS_PATH);
    } catch {
      console.warn('[data] mind_stats.json not found, stats will be empty');
    }

    _dataCache = {
      articles: Array.isArray(articles) ? articles : [],
      stats: stats.stats || {},
      userProfiles: {},
      totalRawNews: stats.total_raw_news || (Array.isArray(articles) ? articles.length : 0),
      totalBehaviors: stats.total_behaviors || 0,
    };

    console.log(`[data] Loaded ${_dataCache.articles.length} articles`);
    return _dataCache;
  })();

  return _loadPromise;
}

/**
 * Force reload (clears cache then calls loadData again).
 */
async function reloadData() {
  _dataCache = null;
  _loadPromise = null;
  return loadData();
}

/**
 * Get current data without triggering I/O. Returns null if not loaded yet.
 */
function getData() {
  return _dataCache;
}

/**
 * Read app version from VERSION file.
 */
async function getVersion() {
  try {
    const ver = await fs.readFile(VERSION_PATH, 'utf-8');
    return ver.trim();
  } catch {
    return 'unknown';
  }
}

// ─── Internal helpers ─────────────────────────────────────────────────

/** Read & parse a JSON file. */
async function loadJSON(filePath) {
  const raw = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(raw);
}

module.exports = { loadData, reloadData, getData, getVersion };
