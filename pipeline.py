"""
Shared training pipeline.

Builds the synthetic dataset, performs the leave-one-out split, trains every
model, and evaluates them. Both the command-line entry point (main.py) and the
Streamlit web app (app.py) call run_pipeline() so they always use the exact same
models and numbers.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SEED, K_VALUES, TOP_K_REPORT, CATEGORIES
from data_generation import generate_dataset, Dataset
from recommender import (
    build_interaction_matrix, CollaborativeFilter, ContentScorer, HybridRanker,
)
from baselines import RandomRecommender, PopularityRecommender
from evaluation import leave_one_out_split, training_popularity, evaluate


@dataclass
class Pipeline:
    ds: Dataset
    train_by_user: dict
    test_target: dict
    train_pop: np.ndarray
    cf: CollaborativeFilter
    content: ContentScorer
    hybrid: HybridRanker
    models: dict             # name -> object exposing .score(user_idx, card_indices)
    results: pd.DataFrame     # metrics, one row per model


class _MatrixAdapter:
    """Wrap a precomputed (n_users, n_cards) score matrix in the common interface."""
    def __init__(self, matrix):
        self.m = matrix

    def score(self, user_idx, card_indices):
        return self.m[user_idx, np.asarray(card_indices)]


class _CFAdapter:
    """Score via the collaborative filter's leave-one-out user representation."""
    def __init__(self, cf, train_by_user):
        self.cf = cf
        self.tbu = train_by_user

    def score(self, user_idx, card_indices):
        return self.cf.score_user(self.tbu[user_idx], card_indices)


def run_pipeline(seed=SEED):
    """Generate data, train all models, evaluate, and return everything bundled."""
    ds = generate_dataset(seed=seed)
    n_users, n_cards = ds.spend_matrix.shape[0], ds.card_reward_matrix.shape[0]

    train_holdings, train_by_user, test_target = leave_one_out_split(
        ds.holdings, n_users, seed=seed)
    train_matrix = build_interaction_matrix(train_holdings, n_users, n_cards)
    train_pop = training_popularity(train_holdings, n_cards)

    cf = CollaborativeFilter().fit(train_matrix)
    content = ContentScorer(ds.card_reward_matrix, ds.annual_fees, ds.signup_bonuses)
    random_rec = RandomRecommender(n_cards, seed=seed)
    pop_rec = PopularityRecommender().fit(train_pop)
    hybrid = HybridRanker(ds, cf, content, train_pop, train_by_user, seed=seed).fit()

    value_all = content.expected_values(ds.spend_matrix)
    models = {
        "Random": random_rec,
        "Popularity": pop_rec,
        "Collaborative Filtering": _CFAdapter(cf, train_by_user),
        "Content (value)": _MatrixAdapter(value_all),
        "Hybrid Ranker": hybrid,
    }

    rows = {name: evaluate(m, ds, train_by_user, test_target, seed=seed)
            for name, m in models.items()}
    results = pd.DataFrame(rows).T
    metric_cols = (
        [f"HitRate@{k}" for k in K_VALUES]
        + [f"NDCG@{k}" for k in K_VALUES]
        + ["MRR", f"MAP@{TOP_K_REPORT}",
           f"AvgValue@{TOP_K_REPORT}", f"Coverage@{TOP_K_REPORT}",
           f"Personalization@{TOP_K_REPORT}"]
    )
    results = results[metric_cols]

    return Pipeline(ds, train_by_user, test_target, train_pop,
                    cf, content, hybrid, models, results)


def recommend(pipeline, user_idx, k=TOP_K_REPORT, model_name="Hybrid Ranker"):
    """Return the top-k recommended card indices for a user from a chosen model."""
    ds = pipeline.ds
    n_cards = ds.card_reward_matrix.shape[0]
    held = pipeline.train_by_user.get(user_idx, set())
    all_cards = np.arange(n_cards)
    candidates = all_cards[~np.isin(all_cards, list(held))]
    scores = np.asarray(pipeline.models[model_name].score(user_idx, candidates), dtype=float)
    ranked = candidates[np.argsort(-scores)]
    return ranked[:k], sorted(int(c) for c in held)


def explain_card(dataset, user_idx, card, top_cats=None):
    """A one-line, plain-English reason a card was recommended to a user."""
    if top_cats is None:
        top_cats = np.argsort(-dataset.spend_matrix[user_idx])[:3]
    rewards = dataset.card_reward_matrix[card]
    hits = [CATEGORIES[c] for c in top_cats if rewards[c] >= 0.03]
    if hits:
        return "boosts your top spending: " + ", ".join(hits)
    if dataset.annual_fees[card] == 0:
        return "no annual fee, solid all-round rewards"
    return "popular among similar users"
