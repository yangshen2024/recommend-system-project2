/**
 * utils.js — Pure-JS math utilities replacing numpy / scikit-learn operations.
 *
 * All functions are synchronous and zero-dependency.  Vector ops use plain
 * JS arrays and standard Math.*.  Randomness is provided via Math.random()
 * with an optional deterministic seed wrapper for reproducibility.
 */

// ═══════════════════════════════════════════════════════════════════════
//  Vector / Array helpers
// ═══════════════════════════════════════════════════════════════════════

/** Create a Float64Array of `n` zeros. */
function zeros(n) {
  return new Float64Array(n);
}

/** Create a Float64Array of `n` ones. */
function ones(n) {
  return Float64Array.from({ length: n }, () => 1);
}

/** L2 norm (Euclidean) */
function norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** Dot product of two vectors (must be same length). */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Cosine similarity = dot(a,b) / (|a| * |b|), clamped to [-1, 1]. */
function cosineSimilarity(a, b) {
  const na = norm(a);
  const nb = norm(b);
  if (na < 1e-12 || nb < 1e-12) return 0;
  return Math.max(-1, Math.min(1, dot(a, b) / (na * nb)));
}

/** L2-normalise a vector in-place (mutates). Returns the array for chaining. */
function l2Normalise(v) {
  const n = norm(v);
  if (n > 1e-12) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

// ═══════════════════════════════════════════════════════════════════════
//  Array utilities
// ═══════════════════════════════════════════════════════════════════════

/**
 * argsort: return indices that would sort `arr` in-place.
 * @param {number[]|Float64Array} arr
 * @param {'asc'|'desc'} order  default 'asc'
 * @returns {number[]}
 */
function argsort(arr, order = 'asc') {
  const indices = Array.from({ length: arr.length }, (_, i) => i);
  if (order === 'desc') {
    indices.sort((a, b) => arr[b] - arr[a]);
  } else {
    indices.sort((a, b) => arr[a] - arr[b]);
  }
  return indices;
}

/** Array sum */
function sum(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s;
}

/** Array mean */
function mean(arr) {
  if (arr.length === 0) return 0;
  return sum(arr) / arr.length;
}

/** Array median (requires sorting a copy) */
function median(arr) {
  if (arr.length === 0) return 0;
  const sorted = arr.slice().sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

/** Array max */
function max(arr) {
  if (arr.length === 0) return -Infinity;
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

/** Array min */
function min(arr) {
  if (arr.length === 0) return Infinity;
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] < m) m = arr[i];
  return m;
}

// ═══════════════════════════════════════════════════════════════════════
//  Random helpers
// ═══════════════════════════════════════════════════════════════════════

/**
 * Simple deterministic PRNG (Mulberry32) for reproducible randomness.
 * Call `createRNG(seed)` to get a function `() => number in [0,1)`.
 */
function createRNG(seed) {
  let s = seed | 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random integer in [lo, hi] (inclusive). Uses Math.random() unless rng is given. */
function randInt(lo, hi, rng) {
  const rand = rng ? rng() : Math.random();
  return lo + Math.floor(rand * (hi - lo + 1));
}

/** Random element from an array. */
function randomChoice(arr, rng) {
  return arr[randInt(0, arr.length - 1, rng)];
}

/**
 * Fisher-Yates shuffle (in-place). Returns the same array.
 * @param {Array} arr
 * @param {Function} [rng]  Optional PRNG function
 */
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = randInt(0, i, rng);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ═══════════════════════════════════════════════════════════════════════
//  Counter (collections.Counter equivalent)
// ═══════════════════════════════════════════════════════════════════════

class Counter {
  constructor(iterable) {
    this._map = new Map();
    if (iterable) for (const item of iterable) this.inc(item);
  }

  inc(key, delta = 1) {
    this._map.set(key, (this._map.get(key) || 0) + delta);
    return this;
  }

  get(key) { return this._map.get(key) || 0; }
  set(key, val) { this._map.set(key, val); }
  has(key) { return this._map.has(key); }
  keys() { return this._map.keys(); }
  entries() { return this._map.entries(); }

  /** Return [item, count] pairs sorted by count descending. */
  mostCommon(n) {
    const sorted = [...this._map.entries()].sort((a, b) => b[1] - a[1]);
    return n != null ? sorted.slice(0, n) : sorted;
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  Set operations
// ═══════════════════════════════════════════════════════════════════════

function union(a, b) { return new Set([...a, ...b]); }
function intersection(a, b) { return new Set([...a].filter(x => b.has(x))); }
function difference(a, b) { return new Set([...a].filter(x => !b.has(x))); }
function jaccardDistance(a, b) {
  const u = union(a, b);
  const i = intersection(a, b);
  return u.size > 0 ? 1 - i.size / u.size : 0;
}

// ═══════════════════════════════════════════════════════════════════════
//  Numeric helpers
// ═══════════════════════════════════════════════════════════════════════

function clamp(val, lo, hi) {
  return Math.max(lo, Math.min(hi, val));
}

function roundTo(val, decimals = 4) {
  const factor = 10 ** decimals;
  return Math.round(val * factor) / factor;
}

module.exports = {
  zeros, ones,
  norm, dot, cosineSimilarity, l2Normalise,
  argsort, sum, mean, median, max, min,
  createRNG, randInt, randomChoice, shuffle,
  Counter,
  union, intersection, difference, jaccardDistance,
  clamp, roundTo,
};
