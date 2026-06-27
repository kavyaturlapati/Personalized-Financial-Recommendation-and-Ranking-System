"""
Baseline recommenders. A model is only impressive if it beats these.

  RandomRecommender     - ranks cards in random order. The absolute floor; tells
                          you what each metric looks like with zero signal.

  PopularityRecommender - ranks every user by global card popularity (how many
                          users hold each card in the training set). A deceptively
                          strong baseline and the one real recsys teams must beat,
                          because popularity correlates with quality.
"""
import numpy as np


class RandomRecommender:
    def __init__(self, n_cards, seed=0):
        self.n_cards = n_cards
        self.rng = np.random.default_rng(seed)

    def score(self, user_idx, card_indices):
        return self.rng.random(len(card_indices))


class PopularityRecommender:
    def fit(self, train_popularity_counts):
        # Same score vector for every user -> not personalized at all.
        self.popularity = train_popularity_counts.astype(float)
        return self

    def score(self, user_idx, card_indices):
        return self.popularity[np.asarray(card_indices)]
