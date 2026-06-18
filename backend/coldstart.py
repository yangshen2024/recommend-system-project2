"""
Cold Start Recommendation Engine

Modular strategy that smoothly transitions new users from cold-start to personalized recommendations:

  Phase 0 — Static Profile + Popular Recommendations     (0 interactions)
  Phase 1 — Explore-Exploit Hybrid                       (1~3 interactions)
  Phase 2 — Interest Capture Boost + Personalization Seed (4~7 interactions)
  Phase 3 — Full Personalized Recommendations            (8+ interactions, transition complete)

Strategy Modules:
  - StaticProfile         Build static profile from registration info
  - PopularRecommender    Global trending / high-CTR / quality content recommendations
  - InterestCapture       Initial interest tag selection & guided questionnaire
  - BanditExplorer        Epsilon-Greedy / UCB1 / Thompson Sampling
  - ColdStartManager      Orchestrator: phase management + smooth transition

Usage:
    from backend.coldstart import ColdStartManager
    mgr = ColdStartManager(articles, reranker)
    result = mgr.recommend(user_id="U12345", static_profile={...}, feedback_count=0)
"""

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ==========================================================================
#  Data Classes
# ==========================================================================

@dataclass
class StaticProfile:
    """User static profile (built at registration)"""
    region: str = "unknown"          # Region: north/south/east/west/central/overseas
    device_type: str = "unknown"     # Device: mobile/pc/tablet
    age_group: str = "unknown"       # Age group: 18-24/25-34/35-44/45-54/55+
    gender: str = "unknown"          # Gender: male/female/other
    reg_hour: int = 12               # Registration hour (0-23)
    preferred_language: str = "zh"   # Preferred language

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "device_type": self.device_type,
            "age_group": self.age_group,
            "gender": self.gender,
            "reg_hour": self.reg_hour,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Convert static attributes into a computable preference feature vector"""
        vec = np.zeros(12)

        # Region → lean mapping (index 0-2)
        region_map = {
            "north": [0.7, 0.5, 0.0],   # Leans left, interested in policy
            "south": [0.5, 0.6, 0.2],   # Leans center, interested in economy/tech
            "east":  [0.6, 0.5, 0.1],   # Leans center-left, interested in culture/education
            "west":  [0.3, 0.7, 0.5],   # Leans center-right, interested in livelihood
            "central":[0.4, 0.6, 0.3],  # Leans center, balanced
        }
        r = region_map.get(self.region, [0.5, 0.5, 0.2])
        vec[0], vec[1], vec[2] = r

        # Device type → content preference (index 3-5)
        device_map = {
            "mobile": [0.0, 0.8, -0.3],  # Fast-consumption, short content
            "pc":     [0.6, 0.5, 0.4],  # In-depth content, finance/tech
            "tablet": [0.3, 0.8, 0.2],  # Video/entertainment, lifestyle
        }
        d = device_map.get(self.device_type, [0.3, 0.6, 0.1])
        vec[3], vec[4], vec[5] = d

        # Age group → content preference (index 6-8)
        age_map = {
            "18-24": [1.0, 0.8, 0.3],  # Entertainment, gaming, social
            "25-34": [0.8, 0.7, 0.6],  # Tech, career, finance
            "35-44": [0.5, 0.6, 0.8],  # Real estate, education, health
            "45-54": [0.3, 0.5, 0.7],  # Politics, health, wellness
            "55+":   [0.2, 0.4, 0.6],  # Wellness, history, culture
        }
        a = age_map.get(self.age_group, [0.5, 0.6, 0.5])
        vec[6], vec[7], vec[8] = a

        # Registration hour (index 9-10): morning vs evening preference
        vec[9] = 1.0 if self.reg_hour < 12 else 0.3   # Morning registration → news-oriented
        vec[10] = 1.0 if self.reg_hour >= 18 else 0.3  # Evening registration → entertainment-oriented

        # Gender (index 11)
        gender_map = {"male": 0.5, "female": -0.5, "other": 0.0}
        vec[11] = gender_map.get(self.gender, 0.0)

        return vec


# ==========================================================================
#  Strategy 1: Non-personalized Popular Recommendations
# ==========================================================================

class PopularRecommender:
    """
    Non-personalized recommendations based on global trending,
    high-CTR content, and quality signals.

    Features:
      - Time-decay weighted popularity score
      - Category-balanced sampling to avoid filter bubbles
      - Quality content weighting (long-form / in-depth preferred)
    """

    # Simulated CTR-based category weights (should be aggregated from DB in production)
    CATEGORY_POPULARITY = {
        "News":       0.85,
        "Finance":    0.72,
        "Technology": 0.78,
        "Sports":     0.80,
        "Entertainment": 0.90,
        "Lifestyle":  0.65,
        "Health":     0.60,
        "Education":  0.55,
        "Movies":     0.75,
        "Tv":         0.82,
        "Music":      0.70,
        "Travel":     0.58,
        "Food":       0.68,
        "Science":    0.62,
        "Politics":   0.73,
        "World":      0.71,
    }

    SUBTOPIC_WEIGHTS = {
        "Technology":  0.9,
        "Policy":      0.7,
        "Civil Rights": 0.65,
        "Economy":     0.75,
        "Education":   0.55,
        "Entertainment": 0.85,
    }

    def __init__(self, articles: List[dict]):
        self.articles = articles
        self._compute_scores()

    def _compute_scores(self):
        """Compute popularity score (0~1) for each article"""
        now = time.time()
        self.scores = np.zeros(len(self.articles))

        for i, a in enumerate(self.articles):
            s = 0.0

            # 1. Category popularity (30%)
            cat = a.get("category", "News")
            s += self.CATEGORY_POPULARITY.get(cat, 0.5) * 0.3

            # 2. Subtopic weight (20%)
            sub = a.get("subtopic", "")
            s += self.SUBTOPIC_WEIGHTS.get(sub, 0.5) * 0.2

            # 3. Title length signal — longer titles often indicate higher quality (10%)
            title_len = len(a.get("title", ""))
            s += min(title_len / 120, 1.0) * 0.1

            # 4. Summary length signal — in-depth content (15%)
            summary_len = len(a.get("summary", ""))
            s += min(summary_len / 500, 1.0) * 0.15

            # 5. Tag richness — information density (10%)
            tags = a.get("tags", [])
            s += min(len(tags) / 10, 1.0) * 0.1

            # 6. Lean diversity bonus — slightly prefer neutral content (5%)
            lean = a.get("sourceLean", "B")
            s += (0.8 if lean == "B" else 0.5) * 0.05

            # 7. Random jitter to avoid full determinism (10%)
            np.random.seed(hash(a["id"]) % 2**31)
            s += np.random.random() * 0.1

            self.scores[i] = min(s, 1.0)

    def recommend(self, top_k: int = 10, category_balanced: bool = True) -> List[dict]:
        """
        Generate a list of popular recommendations.

        Args:
            top_k: Number of articles to return
            category_balanced: Whether to apply category-balanced sampling

        Returns:
            List of recommended articles
        """
        if not category_balanced:
            # Pure popularity-sorted
            ranked = np.argsort(-self.scores)
            return [
                {**self.articles[i],
                 "_popularity_score": round(float(self.scores[i]), 4),
                 "_rank": rank + 1}
                for rank, i in enumerate(ranked[:top_k])
            ]

        # Category-balanced sampling: pick top 1-2 from each category for diversity
        by_category = defaultdict(list)
        for i, a in enumerate(self.articles):
            cat = a.get("category", "News")
            by_category[cat].append((i, self.scores[i]))

        # Sort within each category by score
        for cat in by_category:
            by_category[cat].sort(key=lambda x: -x[1])

        # Round-robin sampling
        selected = []
        seen = set()
        rounds = 0
        max_rounds = max(len(v) for v in by_category.values())

        while len(selected) < top_k and rounds < max_rounds:
            for cat in sorted(by_category.keys()):
                if len(selected) >= top_k:
                    break
                pool = by_category[cat]
                if rounds < len(pool):
                    idx, score = pool[rounds]
                    if idx not in seen:
                        seen.add(idx)
                        selected.append({
                            **self.articles[idx],
                            "_popularity_score": round(float(score), 4),
                            "_rank": len(selected) + 1,
                        })
            rounds += 1

        # If still short of top_k, fill with remaining articles
        if len(selected) < top_k:
            remaining = [(i, s) for i, s in enumerate(self.scores) if i not in seen]
            remaining.sort(key=lambda x: -x[1])
            for idx, score in remaining[:top_k - len(selected)]:
                selected.append({
                    **self.articles[idx],
                    "_popularity_score": round(float(score), 4),
                    "_rank": len(selected) + 1,
                })

        return selected[:top_k]

    def get_trending_categories(self) -> List[str]:
        """Get current trending categories (for onboarding tags)"""
        cat_scores = defaultdict(float)
        for i, a in enumerate(self.articles):
            cat = a.get("category", "News")
            cat_scores[cat] += self.scores[i]
        ranked = sorted(cat_scores.items(), key=lambda x: -x[1])
        return [cat for cat, _ in ranked[:8]]


# ==========================================================================
#  Strategy 2: Interest Capture — Guided Questionnaire & Tag Selection
# ==========================================================================

class InterestCapture:
    """
    Quickly capture user preferences through initial interactions.

    Mechanisms:
      - Predefined interest tag taxonomy (categories + subtopics + lean)
      - Progressive guidance: broad categories → fine-grained topics
      - Convert selected tags into pseudo-behavior signals to accelerate personalization
    """

    # Top-level interest tags (emoji-free, pure English labels)
    CATEGORY_TAGS = [
        {"id": "cat_news",        "label": "News",         "categories": ["News", "World", "Politics"]},
        {"id": "cat_tech",        "label": "Technology",    "categories": ["Technology", "Science"]},
        {"id": "cat_finance",     "label": "Finance",       "categories": ["Finance", "Economy"]},
        {"id": "cat_entertainment","label": "Entertainment", "categories": ["Entertainment", "Movies", "Tv", "Music"]},
        {"id": "cat_sports",      "label": "Sports",        "categories": ["Sports"]},
        {"id": "cat_lifestyle",   "label": "Lifestyle",     "categories": ["Lifestyle", "Health", "Food", "Travel"]},
        {"id": "cat_education",   "label": "Education",     "categories": ["Education"]},
        {"id": "cat_society",     "label": "Society",       "categories": ["News", "World"]},
    ]

    # Sub-topic tags — 6 per main category (48 total), emoji-free
    SUBTOPIC_TAGS = [
        # News sub-topics (6)
        {"id": "sub_breaking",      "label": "Breaking News",         "subtopics": ["News"]},
        {"id": "sub_international",  "label": "International",         "subtopics": ["World"]},
        {"id": "sub_politics",       "label": "Politics",              "subtopics": ["Politics"]},
        {"id": "sub_investigative",  "label": "Investigative",         "subtopics": ["News"]},
        {"id": "sub_opinion",        "label": "Opinion",               "subtopics": ["News"]},
        {"id": "sub_diplomacy",      "label": "Global Diplomacy",      "subtopics": ["World", "Politics"]},
        # Technology sub-topics (6)
        {"id": "sub_ai_ml",          "label": "AI & Machine Learning", "subtopics": ["Technology", "Science"]},
        {"id": "sub_cybersecurity",  "label": "Cybersecurity",         "subtopics": ["Technology"]},
        {"id": "sub_consumer_tech",  "label": "Consumer Tech",         "subtopics": ["Technology"]},
        {"id": "sub_space",          "label": "Space Exploration",     "subtopics": ["Science", "Technology"]},
        {"id": "sub_blockchain",     "label": "Blockchain & Web3",     "subtopics": ["Technology", "Finance"]},
        {"id": "sub_robotics",       "label": "Robotics",              "subtopics": ["Technology", "Science"]},
        # Finance sub-topics (6)
        {"id": "sub_stocks",         "label": "Stock Markets",         "subtopics": ["Finance"]},
        {"id": "sub_personal_finance","label": "Personal Finance",     "subtopics": ["Finance"]},
        {"id": "sub_real_estate",    "label": "Real Estate",           "subtopics": ["Finance", "Economy"]},
        {"id": "sub_crypto",         "label": "Cryptocurrency",        "subtopics": ["Finance", "Technology"]},
        {"id": "sub_banking",        "label": "Banking",               "subtopics": ["Finance"]},
        {"id": "sub_trade",          "label": "Global Trade",          "subtopics": ["Economy", "Finance"]},
        # Entertainment sub-topics (6)
        {"id": "sub_movies",         "label": "Movies & Cinema",       "subtopics": ["Entertainment", "Movies"]},
        {"id": "sub_television",     "label": "Television",            "subtopics": ["Entertainment", "Tv"]},
        {"id": "sub_music",          "label": "Music Industry",        "subtopics": ["Entertainment", "Music"]},
        {"id": "sub_gaming",         "label": "Gaming & Esports",      "subtopics": ["Entertainment", "Technology"]},
        {"id": "sub_celebrity",      "label": "Celebrity Culture",     "subtopics": ["Entertainment"]},
        {"id": "sub_theater",        "label": "Theater & Arts",        "subtopics": ["Entertainment", "Education"]},
        # Sports sub-topics (6)
        {"id": "sub_football",       "label": "Football & Soccer",     "subtopics": ["Sports"]},
        {"id": "sub_basketball",     "label": "Basketball",            "subtopics": ["Sports"]},
        {"id": "sub_tennis",         "label": "Racquet Sports",        "subtopics": ["Sports"]},
        {"id": "sub_olympics",       "label": "Olympic Sports",        "subtopics": ["Sports"]},
        {"id": "sub_motorsports",    "label": "Motorsports",           "subtopics": ["Sports"]},
        {"id": "sub_extreme",        "label": "Extreme Sports",        "subtopics": ["Sports"]},
        # Lifestyle sub-topics (6)
        {"id": "sub_wellness",       "label": "Health & Wellness",     "subtopics": ["Health"]},
        {"id": "sub_travel",         "label": "Travel & Adventure",    "subtopics": ["Travel"]},
        {"id": "sub_food",           "label": "Food & Dining",         "subtopics": ["Food", "Lifestyle"]},
        {"id": "sub_fashion",        "label": "Fashion & Style",       "subtopics": ["Lifestyle"]},
        {"id": "sub_fitness",        "label": "Fitness",               "subtopics": ["Health", "Lifestyle"]},
        {"id": "sub_home",           "label": "Home & Living",         "subtopics": ["Lifestyle"]},
        # Education sub-topics (6)
        {"id": "sub_higher_ed",      "label": "Higher Education",      "subtopics": ["Education"]},
        {"id": "sub_online_learning","label": "Online Learning",       "subtopics": ["Education", "Technology"]},
        {"id": "sub_research",       "label": "Academic Research",     "subtopics": ["Education", "Science"]},
        {"id": "sub_career_dev",     "label": "Career Development",    "subtopics": ["Education"]},
        {"id": "sub_books",          "label": "Books & Literature",    "subtopics": ["Education"]},
        {"id": "sub_history",        "label": "History & Heritage",    "subtopics": ["Education"]},
        # Society sub-topics (6)
        {"id": "sub_civil_rights",   "label": "Civil & Human Rights",  "subtopics": ["Civil Rights", "Policy"]},
        {"id": "sub_urban",          "label": "Urban Development",     "subtopics": ["News"]},
        {"id": "sub_environment",    "label": "Environment & Climate", "subtopics": ["Science"]},
        {"id": "sub_social_justice", "label": "Social Justice",        "subtopics": ["Civil Rights"]},
        {"id": "sub_public_safety",  "label": "Public Safety",         "subtopics": ["News"]},
        {"id": "sub_demographics",   "label": "Demographics",          "subtopics": ["News"]},
    ]

    # Onboarding questionnaire
    QUESTIONNAIRE = [
        {
            "id": "q1",
            "question": "When do you usually read the news?",
            "options": [
                {"id": "morning", "label": "Morning commute", "lean": "B"},
                {"id": "noon",    "label": "Lunch break",    "lean": "B"},
                {"id": "evening", "label": "Evening unwind", "lean": "P"},
                {"id": "night",   "label": "Late night",     "lean": "B"},
            ],
        },
        {
            "id": "q2",
            "question": "What content style do you prefer?",
            "options": [
                {"id": "deep",    "label": "In-depth long-form analysis", "lean": "B"},
                {"id": "quick",   "label": "Quick news summaries",        "lean": "B"},
                {"id": "opinion", "label": "Opinion & commentary",        "lean": "P"},
                {"id": "data",    "label": "Data & charts",               "lean": "B"},
            ],
        },
        {
            "id": "q3",
            "question": "What's your leaning on news perspectives?",
            "options": [
                {"id": "progressive", "label": "Progressive / Reform-focused",  "lean": "P"},
                {"id": "balanced",    "label": "Neutral & objective",            "lean": "B"},
                {"id": "traditional", "label": "Traditional / Conservative",     "lean": "T"},
                {"id": "diverse",     "label": "Want to see diverse viewpoints", "lean": "B"},
            ],
        },
    ]

    def __init__(self):
        self._user_selections: Dict[str, dict] = {}

    def get_category_tags(self) -> List[dict]:
        """Return top-level interest tags (for frontend display)"""
        return self.CATEGORY_TAGS

    def get_subtopic_tags(self) -> List[dict]:
        """Return secondary interest tags"""
        return self.SUBTOPIC_TAGS

    def get_questionnaire(self) -> List[dict]:
        """Return the guided questionnaire"""
        return self.QUESTIONNAIRE

    def record_selections(self, user_id: str,
                          category_tag_ids: List[str] = None,
                          subtopic_tag_ids: List[str] = None,
                          questionnaire_answers: Dict[str, str] = None) -> dict:
        """
        Record user selections and generate pseudo-behavior profile.

        Returns:
            dict with preferred_categories, preferred_subtopics, preferred_lean, pseudo_score
        """
        preferred_categories = []
        preferred_subtopics = []
        lean_counts = Counter()

        # Parse top-level tag selections
        if category_tag_ids:
            tag_map = {t["id"]: t for t in self.CATEGORY_TAGS}
            for tid in category_tag_ids:
                if tid in tag_map:
                    preferred_categories.extend(tag_map[tid]["categories"])

        # Parse secondary tag selections
        if subtopic_tag_ids:
            sub_map = {t["id"]: t for t in self.SUBTOPIC_TAGS}
            for sid in subtopic_tag_ids:
                if sid in sub_map:
                    preferred_subtopics.extend(sub_map[sid]["subtopics"])

        # Parse questionnaire → lean inference
        if questionnaire_answers:
            q_map = {q["id"]: q for q in self.QUESTIONNAIRE}
            for qid, answer_id in questionnaire_answers.items():
                if qid in q_map:
                    for opt in q_map[qid]["options"]:
                        if opt["id"] == answer_id:
                            lean_counts[opt["lean"]] += 1

        # Merge & deduplicate
        preferred_categories = list(set(preferred_categories))
        preferred_subtopics = list(set(preferred_subtopics))

        # Infer preferred lean (from tag preferences)
        if not lean_counts:
            # Default to center
            lean_counts["B"] = 3

        total_lean = sum(lean_counts.values())
        lean_dist = {k: round(v / total_lean, 2) for k, v in lean_counts.most_common()}
        preferred_leans = [k for k, _ in lean_counts.most_common(2)]

        profile = {
            "user_id": user_id,
            "preferred_categories": preferred_categories,
            "preferred_subtopics": preferred_subtopics,
            "preferred_leans": preferred_leans,
            "lean_distribution": lean_dist,
            "pseudo_score": len(category_tag_ids or []) * 0.1 + len(subtopic_tag_ids or []) * 0.15,
        }

        self._user_selections[user_id] = profile
        return profile

    def get_pseudo_profile(self, user_id: str) -> Optional[dict]:
        """Get the user's pseudo-behavior profile"""
        return self._user_selections.get(user_id)

    def convert_to_user_profile(self, user_id: str) -> dict:
        """
        Convert interest capture results into a reranker-compatible user_profile format.

        This allows cold-start users to directly feed into baseline_rank() for
        personalized ranking.
        """
        selections = self._user_selections.get(user_id, {})
        return {
            "preferred_leans": selections.get("preferred_leans", ["B"]),
            "preferred_categories": selections.get("preferred_categories", []),
            "preferred_subtopics": selections.get("preferred_subtopics", []),
            "liked_tags": [],  # No click history yet
            "is_coldstart": True,
            "pseudo_score": selections.get("pseudo_score", 0),
        }


# ==========================================================================
#  Strategy 3: Explore & Exploit (E&E) — Bandit Algorithms
# ==========================================================================

class BanditExplorer:
    """
    Multi-Armed Bandit explore-exploit strategy.

    Supports three algorithms:
      - epsilon-greedy: Explore randomly with probability ε, exploit best with 1-ε
      - UCB1: Upper Confidence Bound, balances exploration and exploitation
      - Thompson Sampling: Bayesian method, samples from each arm's distribution

    Arms: Content categories
    Reward: User click = 1, no click = 0
    """

    def __init__(self, arms: List[str], algorithm: str = "epsilon_greedy",
                 epsilon: float = 0.3, decay_rate: float = 0.05):
        """
        Args:
            arms: List of available arms (e.g., article categories)
            algorithm: "epsilon_greedy" | "ucb1" | "thompson"
            epsilon: Initial exploration rate for ε-greedy
            decay_rate: ε decay rate per interaction
        """
        self.arms = arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        # Per-arm statistics
        self.counts = {arm: 0 for arm in arms}     # Selection count
        self.rewards = {arm: 0.0 for arm in arms}   # Cumulative reward
        self.means = {arm: 0.5 for arm in arms}     # Estimated mean (used by Thompson)

        # Thompson Sampling parameters (Beta distribution)
        self.alpha = {arm: 1.0 for arm in arms}
        self.beta = {arm: 1.0 for arm in arms}

        # Global interaction counter
        self.total_pulls = 0

    def select_arm(self) -> str:
        """Select an arm according to the current strategy"""
        self.total_pulls += 1

        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy_select()
        elif self.algorithm == "ucb1":
            return self._ucb1_select()
        elif self.algorithm == "thompson":
            return self._thompson_select()
        else:
            return random.choice(self.arms)

    def update(self, arm: str, reward: float):
        """
        Update the statistics for a given arm.

        Args:
            arm: The selected arm
            reward: The reward received (0 or 1)
        """
        if arm not in self.counts:
            return

        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.means[arm] = self.rewards[arm] / max(1, self.counts[arm])

        # Thompson update
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

        # ε decay
        self.epsilon = max(0.02, self.epsilon * (1 - self.decay_rate))

    def _epsilon_greedy_select(self) -> str:
        """ε-greedy: explore randomly with probability ε"""
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(self.arms, key=lambda a: self.means[a])

    def _ucb1_select(self) -> str:
        """UCB1: select the arm with the highest upper confidence bound"""
        best_arm = self.arms[0]
        best_ucb = -float("inf")

        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm  # Prioritize untried arms

            # Exploitation term + exploration term
            exploitation = self.means[arm]
            exploration = math.sqrt(2 * math.log(self.total_pulls) / self.counts[arm])
            ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm

    def _thompson_select(self) -> str:
        """Thompson Sampling: sample from Beta distribution for each arm"""
        best_arm = self.arms[0]
        best_sample = -float("inf")

        for arm in self.arms:
            sample = np.random.beta(self.alpha[arm], self.beta[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def get_arm_distribution(self) -> Dict[str, dict]:
        """Get statistics for all arms (for frontend visualization)"""
        return {
            arm: {
                "count": self.counts[arm],
                "mean": round(self.means[arm], 4),
                "explore_prob": round(self.epsilon, 4) if self.algorithm == "epsilon_greedy" else 0,
            }
            for arm in self.arms
        }

    def get_exploration_rate(self) -> float:
        """Current exploration rate"""
        if self.algorithm == "epsilon_greedy":
            return self.epsilon
        if self.total_pulls == 0:
            return 1.0
        unplayed = sum(1 for a in self.arms if self.counts[a] == 0)
        return unplayed / len(self.arms)


# ==========================================================================
#  Strategy 4: ColdStartManager — Orchestrator + Smooth Transition
# ==========================================================================

class ColdStartManager:
    """
    Cold-start recommendation orchestrator.

    Manages four phases from cold-start to personalized recommendations:

        Phase 0 — BOOTSTRAP    (0 interactions):  100% popular + static profile ranking
        Phase 1 — EXPLORE      (1~3 interactions): 60% popular + 40% E&E (exploration-led)
        Phase 2 — HYBRID       (4~7 interactions): 30% popular + 40% E&E + 30% personalization seed
        Phase 3 — PERSONALIZED (8+ interactions):  100% personalized (baseline_rank + rerank)

    Smooth transition mechanism:
      - Each phase's results are a weighted blend of the previous phase's results
      - Transition band uses exponential decay weights to avoid abrupt result changes
    """

    PHASE_BOOTSTRAP = 0
    PHASE_EXPLORE = 1
    PHASE_HYBRID = 2
    PHASE_PERSONALIZED = 3

    PHASE_LABELS = {
        0: "Bootstrap — Popular",
        1: "Explore — Exploration-led",
        2: "Hybrid — Blending",
        3: "Personalized — Personalized",
    }

    def __init__(self, articles: List[dict], reranker=None):
        """
        Args:
            articles: All article list
            reranker: Reranker instance (for personalized ranking)
        """
        self.articles = articles
        self.reranker = reranker

        # Sub-strategies
        self.popular = PopularRecommender(articles)
        self.interest_capture = InterestCapture()

        # E&E — use article categories as arms
        categories = list(set(a.get("category", "News") for a in articles))
        self.bandit = BanditExplorer(
            arms=categories,
            algorithm="epsilon_greedy",
            epsilon=0.4,
            decay_rate=0.08,
        )

        # User phase state
        self._user_phase: Dict[str, int] = {}  # {user_id: phase}
        self._user_feedback_counts: Dict[str, int] = defaultdict(int)
        self._previous_recs: Dict[str, List[int]] = {}  # Previous round article IDs for smoothing

    def determine_phase(self, feedback_count: int) -> int:
        """Determine current phase based on interaction count"""
        if feedback_count == 0:
            return self.PHASE_BOOTSTRAP
        elif feedback_count <= 3:
            return self.PHASE_EXPLORE
        elif feedback_count <= 7:
            return self.PHASE_HYBRID
        else:
            return self.PHASE_PERSONALIZED

    def recommend(self,
                  user_id: str,
                  static_profile: Optional[StaticProfile] = None,
                  feedback_count: int = 0,
                  top_k: int = 10,
                  smooth_transition: bool = True) -> dict:
        """
        Core recommendation interface.

        Args:
            user_id: User ID
            static_profile: Static profile (used in Phase 0)
            feedback_count: User's accumulated interaction count
            top_k: Number of articles to return
            smooth_transition: Whether to enable smooth transition

        Returns:
            {
                "user_id": str,
                "phase": int,
                "phase_label": str,
                "feedback_count": int,
                "recommendations": [articles...],
                "strategy_breakdown": {...},     # Contribution ratios of each strategy
                "bandit_stats": {...},           # E&E statistics
                "transition_progress": float,    # Transition progress 0~1
            }
        """
        phase = self.determine_phase(feedback_count)
        self._user_phase[user_id] = phase
        self._user_feedback_counts[user_id] = feedback_count

        static_vec = None
        if static_profile:
            static_vec = static_profile.to_feature_vector()

        results = self._dispatch_phase(
            user_id, phase, static_vec, feedback_count, top_k
        )

        # Smooth transition: blend current results with previous round
        if smooth_transition and feedback_count > 0:
            results["recommendations"] = self._smooth_results(
                user_id, results["recommendations"], top_k
            )

        self._previous_recs[user_id] = [a["id"] for a in results["recommendations"]]

        results["user_id"] = user_id
        results["phase"] = phase
        results["phase_label"] = self.PHASE_LABELS[phase]
        results["feedback_count"] = feedback_count
        results["transition_progress"] = self._transition_progress(feedback_count)

        return results

    def _dispatch_phase(self, user_id: str, phase: int,
                        static_vec: Optional[np.ndarray],
                        feedback_count: int, top_k: int) -> dict:
        """Dispatch recommendation strategy by phase"""
        if phase == self.PHASE_BOOTSTRAP:
            return self._phase_bootstrap(static_vec, top_k)
        elif phase == self.PHASE_EXPLORE:
            return self._phase_explore(user_id, static_vec, top_k)
        elif phase == self.PHASE_HYBRID:
            return self._phase_hybrid(user_id, static_vec, top_k)
        else:
            return self._phase_personalized(user_id, top_k)

    # ── Phase 0: Bootstrap ──────────────────────────────────────────

    def _phase_bootstrap(self, static_vec: Optional[np.ndarray],
                         top_k: int) -> dict:
        """
        Pure popular + static profile ranking.

        Strategy:
          - 100% PopularRecommender (category-balanced)
          - If static profile available, fine-tune order of popular results by profile features
        """
        hot = self.popular.recommend(top_k=int(top_k * 1.5), category_balanced=True)

        if static_vec is not None:
            # Re-rank popular results using static profile features
            hot = self._rerank_by_static_profile(hot, static_vec)

        return {
            "recommendations": hot[:top_k],
            "strategy_breakdown": {
                "popular": 1.0,
                "static_profile": 1.0 if static_vec is not None else 0,
                "bandit_explore": 0,
                "personalized": 0,
            },
            "bandit_stats": self.bandit.get_arm_distribution(),
        }

    def _rerank_by_static_profile(self, articles: List[dict],
                                   static_vec: np.ndarray) -> List[dict]:
        """Weight and re-rank articles using the static profile vector"""
        scored = []
        for a in articles:
            art_vec = self._article_to_static_vec(a)
            # Cosine similarity
            similarity = float(np.dot(static_vec, art_vec) / (
                max(np.linalg.norm(static_vec), 1e-8) *
                max(np.linalg.norm(art_vec), 1e-8)
            ))
            scored.append((a, similarity))

        scored.sort(key=lambda x: -x[1])
        result = []
        for rank, (article, sim) in enumerate(scored):
            art = dict(article)
            art["_static_similarity"] = round(sim, 4)
            art["_rank"] = rank + 1
            art["_baseline_score"] = round(0.4 + sim * 0.5, 4)
            result.append(art)
        return result

    def _article_to_static_vec(self, article: dict) -> np.ndarray:
        """Map an article into the same feature space as the static profile"""
        vec = np.zeros(12)
        cat = article.get("category", "News")
        sub = article.get("subtopic", "")
        lean = article.get("sourceLean", "B")
        title_len = len(article.get("title", ""))

        # Category mapping
        cat_map = {
            "News": [0.6, 0.5, 0.3], "World": [0.7, 0.4, 0.2],
            "Politics": [0.8, 0.3, 0.5], "Finance": [0.3, 0.8, 0.6],
            "Technology": [0.5, 0.6, 0.4], "Science": [0.4, 0.7, 0.3],
            "Entertainment": [0.1, 0.4, 0.1], "Movies": [0.0, 0.5, 0.2],
            "Tv": [0.0, 0.5, 0.2], "Music": [0.1, 0.4, 0.3],
            "Sports": [0.5, 0.5, 0.3], "Lifestyle": [0.2, 0.5, 0.4],
            "Health": [0.3, 0.6, 0.7], "Education": [0.5, 0.5, 0.5],
            "Food": [0.2, 0.5, 0.5], "Travel": [0.3, 0.5, 0.4],
        }
        vec[:3] = cat_map.get(cat, [0.5, 0.5, 0.3])

        # Device-related (mobile/desktop content differentiation)
        if title_len < 40:
            vec[3:6] = [0.0, 0.8, -0.3]  # Short content → mobile-oriented
        elif title_len > 80:
            vec[3:6] = [0.6, 0.5, 0.4]   # Long content → PC-oriented

        # Subtopic + age association
        sub_map = {
            "Technology": [0.8, 0.7, 0.6],
            "Economy": [0.5, 0.6, 0.8],
            "Health": [0.3, 0.5, 0.7],
            "Entertainment": [1.0, 0.8, 0.3],
            "Policy": [0.5, 0.6, 0.5],
            "Education": [0.5, 0.6, 0.8],
            "Civil Rights": [0.6, 0.5, 0.6],
        }
        vec[6:9] = sub_map.get(sub, [0.5, 0.6, 0.5])

        # Lean
        lean_map = {"P": 0.8, "B": 0.5, "T": 0.2}
        vec[11] = lean_map.get(lean, 0.5)

        return vec

    # ── Phase 1: Explore ────────────────────────────────────────────

    def _phase_explore(self, user_id: str,
                       static_vec: Optional[np.ndarray],
                       top_k: int) -> dict:
        """
        E&E led: 60% popular + 40% E&E.

        Uses BanditExplorer to select categories, then picks top-scoring
        articles from each selected category.
        """
        n_hot = max(1, int(top_k * 0.6))
        n_explore = top_k - n_hot

        # Popular portion
        hot = self.popular.recommend(top_k=n_hot, category_balanced=True)

        # E&E exploration portion
        explore_articles = []
        explored_cats = set()
        for _ in range(n_explore):
            arm = self.bandit.select_arm()
            # Try to pick different categories each time
            attempts = 0
            while arm in explored_cats and attempts < 5:
                arm = self.bandit.select_arm()
                attempts += 1
            explored_cats.add(arm)

            # Select highest-popularity article from that category
            cat_articles = [a for a in self.articles if a.get("category") == arm]
            if cat_articles:
                best = max(cat_articles, key=lambda a: self.popular.scores[
                    self.articles.index(a) if a in self.articles else 0
                ] if a in self.articles else 0)
                a = dict(best)
                a["_explore_arm"] = arm
                a["_explore_epsilon"] = round(self.bandit.epsilon, 4)
                a["_baseline_score"] = round(0.3 + random.random() * 0.3, 4)
                explore_articles.append(a)

        # Shuffle to avoid all popular items clustered at the top
        combined = hot + explore_articles
        random.shuffle(combined)

        # Add rank
        for i, a in enumerate(combined):
            a["_rank"] = i + 1

        # Record to pseudo-behavior (so E&E can learn)
        for a in explore_articles:
            arm = a.get("_explore_arm", "")
            if arm:
                # Simulate weak reward — impression = 0.1 reward
                self.bandit.update(arm, 0.1)

        return {
            "recommendations": combined[:top_k],
            "strategy_breakdown": {
                "popular": 0.6,
                "static_profile": 0.3 if static_vec is not None else 0,
                "bandit_explore": 0.4,
                "personalized": 0,
            },
            "bandit_stats": self.bandit.get_arm_distribution(),
        }

    # ── Phase 2: Hybrid ─────────────────────────────────────────────

    def _phase_hybrid(self, user_id: str,
                      static_vec: Optional[np.ndarray],
                      top_k: int) -> dict:
        """
        Hybrid transition: 30% popular + 40% E&E + 30% personalization seed.

        Personalization seed = calls baseline_rank based on interest capture results.
        """
        n_hot = max(1, int(top_k * 0.3))
        n_explore = max(1, int(top_k * 0.4))
        n_personalized = top_k - n_hot - n_explore

        hot = self.popular.recommend(top_k=n_hot, category_balanced=True)

        # E&E
        explore_articles = []
        explored_cats = set()
        for _ in range(n_explore):
            arm = self.bandit.select_arm()
            attempts = 0
            while arm in explored_cats and attempts < 5:
                arm = self.bandit.select_arm()
                attempts += 1
            explored_cats.add(arm)
            cat_articles = [a for a in self.articles if a.get("category") == arm]
            if cat_articles:
                best = random.choice(cat_articles[:10])
                a = dict(best)
                a["_explore_arm"] = arm
                a["_baseline_score"] = round(0.3 + random.random() * 0.4, 4)
                explore_articles.append(a)

        # Personalization seed
        personalized = []
        if self.reranker is not None:
            user_profile = self.interest_capture.convert_to_user_profile(user_id)
            if not user_profile.get("preferred_categories"):
                # No interest capture data, fallback to popular
                user_profile = {"preferred_leans": ["B"]}
            try:
                bl = self.reranker.baseline_rank(
                    user_profile=user_profile, top_k=n_personalized
                )
                personalized = bl
            except Exception:
                personalized = self.popular.recommend(top_k=n_personalized, category_balanced=True)
        else:
            personalized = self.popular.recommend(top_k=n_personalized, category_balanced=True)

        combined = hot + explore_articles + personalized
        random.shuffle(combined)
        for i, a in enumerate(combined):
            a["_rank"] = i + 1

        return {
            "recommendations": combined[:top_k],
            "strategy_breakdown": {
                "popular": 0.3,
                "bandit_explore": 0.4,
                "personalized": 0.3,
            },
            "bandit_stats": self.bandit.get_arm_distribution(),
        }

    # ── Phase 3: Personalized ───────────────────────────────────────

    def _phase_personalized(self, user_id: str, top_k: int) -> dict:
        """
        Full personalized recommendations: using baseline_rank.

        At this point the user already has preference data from the interest
        capture phase and can directly transition to personalized ranking.
        """
        if self.reranker is not None:
            user_profile = self.interest_capture.convert_to_user_profile(user_id)
            try:
                baseline = self.reranker.baseline_rank(
                    user_profile=user_profile, top_k=top_k
                )
                return {
                    "recommendations": baseline,
                    "strategy_breakdown": {
                        "popular": 0,
                        "bandit_explore": 0.05,
                        "personalized": 0.95,
                    },
                    "bandit_stats": self.bandit.get_arm_distribution(),
                }
            except Exception:
                pass

        # Fallback: popular
        hot = self.popular.recommend(top_k=top_k, category_balanced=True)
        return {
            "recommendations": hot,
            "strategy_breakdown": {
                "popular": 1.0,
                "bandit_explore": 0,
                "personalized": 0,
            },
            "bandit_stats": self.bandit.get_arm_distribution(),
        }

    # ── Smooth Transition ───────────────────────────────────────────

    def _smooth_results(self, user_id: str, current: List[dict],
                        top_k: int) -> List[dict]:
        """
        Smooth transition: weighted blend of current results with previous round.

        - With existing feedback → keep 20% old articles + 80% new results
        - Avoids "recommendation list mutation" that confuses users
        """
        prev_ids = self._previous_recs.get(user_id, [])
        if not prev_ids:
            return current

        id_to_article = {a["id"]: a for a in self.articles}

        # Retain a portion of previous articles
        keep_count = max(1, int(top_k * 0.2))
        kept = []
        for pid in prev_ids[:keep_count]:
            if pid in id_to_article:
                a = dict(id_to_article[pid])
                a["_from_previous_round"] = True
                a["_baseline_score"] = a.get("_baseline_score", 0.5) * 0.9
                kept.append(a)

        # Deduplicate and merge
        kept_ids = {a["id"] for a in kept}
        new_filtered = [a for a in current if a["id"] not in kept_ids]
        combined = kept + new_filtered

        for i, a in enumerate(combined[:top_k]):
            a["_rank"] = i + 1

        return combined[:top_k]

    def _transition_progress(self, feedback_count: int) -> float:
        """Compute transition progress (0~1) for frontend display"""
        # 8 interactions to complete transition
        return min(1.0, feedback_count / 8.0)

    # ── User Behavior Updates ───────────────────────────────────────

    def record_feedback(self, user_id: str, article_id: int,
                        action: str = "click", article: dict = None):
        """
        Record user feedback and update the E&E model.

        Args:
            user_id: User ID
            article_id: Article ID
            action: "click" | "like" | "dislike" | "skip"
            article: Full article data (optional)
        """
        reward = {
            "click": 1.0,
            "like": 1.5,
            "dislike": -0.5,
            "skip": -0.1,
        }.get(action, 0.5)

        # Update Bandit: find the article's category
        if article:
            cat = article.get("category", "News")
            self.bandit.update(cat, max(0, reward))

        # Increment interaction count
        self._user_feedback_counts[user_id] += 1

    def get_onboarding_data(self) -> dict:
        """Get all data needed for the new-user onboarding flow"""
        return {
            "category_tags": self.interest_capture.get_category_tags(),
            "subtopic_tags": self.interest_capture.get_subtopic_tags(),
            "questionnaire": self.interest_capture.get_questionnaire(),
            "trending_categories": self.popular.get_trending_categories(),
        }

    def submit_onboarding(self, user_id: str,
                          category_tag_ids: List[str] = None,
                          subtopic_tag_ids: List[str] = None,
                          questionnaire_answers: Dict[str, str] = None) -> dict:
        """Submit onboarding flow results"""
        profile = self.interest_capture.record_selections(
            user_id=user_id,
            category_tag_ids=category_tag_ids,
            subtopic_tag_ids=subtopic_tag_ids,
            questionnaire_answers=questionnaire_answers,
        )
        # Force-promote to Phase 2 (skip pure E&E phase, interest signals available)
        self._user_feedback_counts[user_id] = max(
            self._user_feedback_counts.get(user_id, 0), 4
        )
        return profile
