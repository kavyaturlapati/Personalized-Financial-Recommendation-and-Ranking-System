"""
Recommendation & ranking models.

Three model families, plus a hybrid that fuses them:

  CollaborativeFilter  - implicit-feedback matrix factorization (TruncatedSVD).
                         Learns latent user & card embeddings from the holdings
                         matrix. Captures "users like you also hold card X".

  ContentScorer        - scores cards by the expected $ value to a user (and a
                         cosine match between the user's spend profile and the
                         card's reward profile). Captures personal value and,
                         crucially, handles cold-start users with no history.

  HybridRanker         - a gradient-boosted learning-to-rank model. For each
                         (user, card) pair it builds features from the CF
                         embedding score, the content signals, and statistical
                         card features, then learns to rank cards a user is
                         likely to want. This is the headline model: it combines
                         semantic embeddings with statistical scoring.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier

from config import N_LATENT_FACTORS, N_NEGATIVES_PER_POS, SEED


def build_interaction_matrix(holdings, n_users, n_cards):
    """Sparse binary user x card holdings matrix from a list of (user, card) pairs."""
    if not holdings:
        return csr_matrix((n_users, n_cards))
    rows, cols = zip(*holdings)
    data = np.ones(len(holdings), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_cards))


class CollaborativeFilter:
    """Matrix factorization on implicit feedback via truncated SVD."""

    def __init__(self, n_factors=N_LATENT_FACTORS, seed=SEED):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=seed)

    def fit(self, interaction_matrix):
        # Item embeddings carry the co-holding structure; we build each user's
        # representation on the fly (below) so scoring can be leave-one-out aware.
        self.svd.fit(interaction_matrix)
        self.item_embeddings = self.svd.components_.T                       # (n_cards, k)
        return self

    def score_user(self, held_cards, target_cards):
        """Leave-one-out-aware affinity score.

        Represents a user by the MEAN embedding of the cards they hold, then dots
        that against each target card's embedding. If a target card is itself in
        the held set (only happens for training positives), it is excluded from
        the user representation. This keeps the signal consistent between training
        positives and held-out test items -- the fix for the leakage that made an
        earlier version of the ranker memorize the holdings matrix.
        """
        held = np.array(sorted(held_cards), dtype=int)
        target_cards = np.asarray(target_cards)
        V = self.item_embeddings
        if len(held) == 0:
            return np.zeros(len(target_cards))
        base_sum = V[held].sum(axis=0)                      # (k,)
        base_count = len(held)
        in_held = np.isin(target_cards, held)
        ctx_sum = base_sum[None, :] - in_held[:, None] * V[target_cards]
        cnt = base_count - in_held.astype(int)
        cnt_safe = np.where(cnt > 0, cnt, 1)
        ctx_mean = ctx_sum / cnt_safe[:, None]
        scores = np.einsum("ij,ij->i", ctx_mean, V[target_cards])
        return np.where(cnt > 0, scores, 0.0)


class ContentScorer:
    """Value- and similarity-based scoring from spend & reward profiles."""

    def __init__(self, card_reward_matrix, annual_fees, signup_bonuses):
        self.reward_matrix = card_reward_matrix
        self.fees = annual_fees
        self.bonuses = signup_bonuses
        # Pre-normalize card reward vectors for cosine similarity.
        norms = np.linalg.norm(card_reward_matrix, axis=1, keepdims=True) + 1e-12
        self._reward_unit = card_reward_matrix / norms

    def expected_values(self, spend_matrix):
        """(n_users, n_cards) expected annual $ value."""
        annual_spend = spend_matrix * 12.0
        reward_back = annual_spend @ self.reward_matrix.T
        return reward_back - self.fees[None, :] + self.bonuses[None, :] / 2.0

    def cosine_match(self, spend_matrix):
        """(n_users, n_cards) cosine similarity between spend and reward profiles."""
        norms = np.linalg.norm(spend_matrix, axis=1, keepdims=True) + 1e-12
        spend_unit = spend_matrix / norms
        return spend_unit @ self._reward_unit.T


class HybridRanker:
    """Gradient-boosted learning-to-rank fusing collaborative, content & statistical signals."""

    FEATURE_NAMES = [
        "cf_score", "content_value", "content_cosine", "annual_fee",
        "signup_bonus", "card_popularity", "card_generosity",
        "user_total_spend", "reward_rate_in_top_category",
    ]

    def __init__(self, dataset, cf, content, train_popularity, train_by_user, seed=SEED):
        self.ds = dataset
        self.cf = cf
        self.train_by_user = train_by_user
        self.seed = seed
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=seed,
        )

        # Pre-compute the content signal matrices once (these do NOT depend on the
        # holdings labels, so they are identical at train and eval -> no leakage).
        # The CF score is computed per-call in _features() so it can be LOO-aware.
        self._value_all = content.expected_values(dataset.spend_matrix)  # (n_users, n_cards)
        self._cosine_all = content.cosine_match(dataset.spend_matrix)    # (n_users, n_cards)

        # Card-level (broadcast) features.
        self._fees = dataset.annual_fees
        self._bonus = dataset.signup_bonuses
        pop = train_popularity.astype(float)
        self._pop_norm = pop / (pop.max() + 1e-12)
        self._generosity = dataset.card_reward_matrix.sum(axis=1)

        # User-level features.
        self._user_total = dataset.spend_matrix.sum(axis=1)
        self._top_cat = dataset.spend_matrix.argmax(axis=1)
        # Reward rate each card offers in a user's single biggest spend category.
        self._reward_in_top = dataset.card_reward_matrix[:, self._top_cat].T  # (n_users, n_cards)

    def _features(self, user_idx, card_indices):
        """Build a feature matrix for one user across a set of candidate cards."""
        card_indices = np.asarray(card_indices)
        n = len(card_indices)
        feats = np.empty((n, len(self.FEATURE_NAMES)), dtype=np.float32)
        feats[:, 0] = self.cf.score_user(self.train_by_user[user_idx], card_indices)
        feats[:, 1] = self._value_all[user_idx, card_indices] / 1000.0
        feats[:, 2] = self._cosine_all[user_idx, card_indices]
        feats[:, 3] = self._fees[card_indices] / 100.0
        feats[:, 4] = self._bonus[card_indices] / 100.0
        feats[:, 5] = self._pop_norm[card_indices]
        feats[:, 6] = self._generosity[card_indices]
        feats[:, 7] = self._user_total[user_idx] / 1000.0
        feats[:, 8] = self._reward_in_top[user_idx, card_indices]
        return feats

    def fit(self):
        """Train the ranker using positives (held cards) + sampled negatives."""
        rng = np.random.default_rng(self.seed)
        all_cards = np.arange(self.ds.card_reward_matrix.shape[0])

        X_parts, y_parts = [], []
        for user, held in self.train_by_user.items():
            held = np.array(sorted(held))
            if len(held) == 0:
                continue
            negative_pool = np.setdiff1d(all_cards, held, assume_unique=False)
            n_neg = min(len(negative_pool), len(held) * N_NEGATIVES_PER_POS)
            negatives = rng.choice(negative_pool, size=n_neg, replace=False)

            X_parts.append(self._features(user, held))
            y_parts.append(np.ones(len(held), dtype=np.int8))
            X_parts.append(self._features(user, negatives))
            y_parts.append(np.zeros(len(negatives), dtype=np.int8))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self.clf.fit(X, y)
        return self

    def score(self, user_idx, card_indices):
        """Ranking score (P(hold)) for one user's candidate cards."""
        feats = self._features(user_idx, card_indices)
        return self.clf.predict_proba(feats)[:, 1]

    def feature_importance(self):
        return dict(sorted(
            zip(self.FEATURE_NAMES, self.clf.feature_importances_),
            key=lambda kv: kv[1], reverse=True,
        ))

