"""
Synthetic data generation for the financial recommender.

Produces three things:
  1. A catalog of credit cards, each with per-category reward rates, an annual
     fee, and a sign-up bonus (procedurally generated, but realistic: more
     generous rewards -> higher fees).
  2. Users, each assigned a spending *archetype* (traveler, foodie, family, ...)
     that shapes how their monthly spend is distributed across categories.
  3. Holdings: which cards each user owns. A user's probability of holding a card
     is a weighted mix of (a) the card's $ value to that user, (b) the card's
     overall popularity, and (c) a segment-level "taste" affinity. This mixture
     is what lets different models capture different parts of the signal.

The generator also returns a full user x card value matrix, used later as
ground truth for business-impact metrics.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    CATEGORIES, CATEGORY_INDEX, N_CATEGORIES, N_USERS, N_CARDS,
    MIN_CARDS_PER_USER, MAX_CARDS_PER_USER, W_VALUE, W_POPULARITY,
    W_AFFINITY, HOLD_TEMPERATURE, HOLD_NOISE, SEED,
)
from utils import zscore, softmax

# Spending archetypes: relative weight per category. Missing categories get a
# small baseline weight so every user still spends a little everywhere.
ARCHETYPES = {
    "traveler":       {"travel": 4.0, "dining": 3.0, "entertainment": 2.0, "online_shopping": 1.5},
    "foodie":         {"dining": 4.0, "groceries": 3.0, "entertainment": 2.5, "drugstores": 1.0},
    "family":         {"groceries": 4.0, "utilities": 3.0, "gas": 2.5, "drugstores": 2.0, "transit": 1.0},
    "commuter":       {"transit": 4.0, "gas": 3.5, "dining": 2.0, "online_shopping": 1.5},
    "online_shopper": {"online_shopping": 4.0, "entertainment": 3.0, "dining": 2.0, "drugstores": 1.5},
    "homebody":       {"groceries": 3.0, "utilities": 3.0, "online_shopping": 2.5, "entertainment": 2.0},
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())


@dataclass
class Dataset:
    cards: pd.DataFrame              # card catalog (metadata)
    card_reward_matrix: np.ndarray   # (n_cards, n_categories) reward rate per category
    annual_fees: np.ndarray          # (n_cards,)
    signup_bonuses: np.ndarray       # (n_cards,)
    card_popularity: np.ndarray      # (n_cards,) latent popularity weight
    users: pd.DataFrame              # user metadata (archetype, total spend)
    spend_matrix: np.ndarray         # (n_users, n_categories) monthly $ per category
    user_archetype_idx: np.ndarray   # (n_users,) archetype index
    value_matrix: np.ndarray         # (n_users, n_cards) expected annual $ value
    holdings: list                   # list of (user_idx, card_idx) tuples


def _archetype_distribution(name, rng):
    """A normalized monthly-spend distribution for one user of a given archetype."""
    base = np.full(N_CATEGORIES, 0.4)
    for cat, w in ARCHETYPES[name].items():
        base[CATEGORY_INDEX[cat]] = w
    base = base * rng.uniform(0.8, 1.2, size=N_CATEGORIES)  # per-user jitter
    return base / base.sum()


def _generate_cards(rng):
    """Procedurally build a diverse catalog of credit cards."""
    rate_choices = np.array([0.02, 0.03, 0.04, 0.05])
    base_choices = np.array([0.010, 0.015, 0.020])
    fee_tiers = np.array([0, 49, 95, 120, 250, 395, 550])

    reward_matrix = np.zeros((N_CARDS, N_CATEGORIES))
    fees, bonuses = np.zeros(N_CARDS), np.zeros(N_CARDS)
    names, kinds = [], []

    for i in range(N_CARDS):
        n_boost = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
        boosted = rng.choice(N_CATEGORIES, size=n_boost, replace=False)
        base_rate = float(rng.choice(base_choices))
        reward_matrix[i, :] = base_rate
        for c in boosted:
            reward_matrix[i, c] = float(rng.choice(rate_choices))

        # Fee scales with total reward generosity, then snaps to a realistic tier.
        generosity = reward_matrix[i, boosted].sum() + base_rate
        raw_fee = max(0.0, generosity * 1800 - 130 + rng.normal(0, 35))
        fees[i] = fee_tiers[np.argmin(np.abs(fee_tiers - raw_fee))]
        # Sign-up bonus loosely tracks the fee (premium cards buy you in).
        bonuses[i] = max(0.0, fees[i] * 2.0 + rng.normal(120, 60)) if fees[i] > 0 else \
            max(0.0, rng.normal(120, 80)) * (rng.random() < 0.5)

        boosted_names = [CATEGORIES[c].replace("_", " ").title() for c in boosted]
        names.append(f"Card {i:02d} \u00b7 " + " / ".join(boosted_names))
        kinds.append("+".join(CATEGORIES[c] for c in boosted))

    # Latent popularity: cheaper cards are more widely held, plus multiplicative noise.
    popularity = np.exp(-fees / 250.0) * rng.lognormal(0.0, 0.5, size=N_CARDS)

    cards = pd.DataFrame({
        "card_id": np.arange(N_CARDS),
        "name": names,
        "boosted_categories": kinds,
        "annual_fee": fees.astype(int),
        "signup_bonus": bonuses.round(0).astype(int),
        "base_rate": reward_matrix.min(axis=1).round(3),
        "max_rate": reward_matrix.max(axis=1).round(3),
        "popularity_weight": popularity.round(3),
    })
    return cards, reward_matrix, fees, bonuses, popularity


def _generate_users(rng):
    """Build users with archetype-driven monthly spend profiles."""
    spend_matrix = np.zeros((N_USERS, N_CATEGORIES))
    arch_idx = np.zeros(N_USERS, dtype=int)
    totals = np.zeros(N_USERS)

    for u in range(N_USERS):
        a = rng.integers(len(ARCHETYPE_NAMES))
        arch_idx[u] = a
        dist = _archetype_distribution(ARCHETYPE_NAMES[a], rng)
        total_monthly = float(rng.lognormal(mean=np.log(2500), sigma=0.4))
        totals[u] = total_monthly
        spend_matrix[u] = dist * total_monthly

    users = pd.DataFrame({
        "user_id": np.arange(N_USERS),
        "archetype": [ARCHETYPE_NAMES[i] for i in arch_idx],
        "monthly_spend": totals.round(2),
    })
    return users, spend_matrix, arch_idx


def _compute_value_matrix(spend_matrix, reward_matrix, fees, bonuses):
    """Expected annual $ value of each card for each user.

    value = (annual spend . reward rates) - annual fee + half the sign-up bonus
    (the bonus is amortized over ~2 years).
    """
    annual_spend = spend_matrix * 12.0                  # (n_users, n_cat)
    reward_back = annual_spend @ reward_matrix.T        # (n_users, n_cards)
    return reward_back - fees[None, :] + bonuses[None, :] / 2.0


def _generate_holdings(rng, value_matrix, popularity, arch_idx):
    """Sample the cards each user holds from a mixture of value/popularity/affinity."""
    # Segment-level "taste": each archetype idiosyncratically prefers some cards.
    # CF can recover this from co-holding patterns; value/popularity cannot.
    affinity = rng.normal(0.0, 1.0, size=(len(ARCHETYPE_NAMES), N_CARDS))

    value_z = zscore(value_matrix, axis=1)              # per-user across cards
    pop_z = zscore(np.log(popularity + 1e-8))           # per-card

    holdings = []
    for u in range(N_USERS):
        aff_z = zscore(affinity[arch_idx[u]])
        logits = (W_VALUE * value_z[u]
                  + W_POPULARITY * pop_z
                  + W_AFFINITY * aff_z
                  + rng.normal(0.0, HOLD_NOISE, size=N_CARDS))
        probs = softmax(logits / HOLD_TEMPERATURE)
        n_hold = int(rng.integers(MIN_CARDS_PER_USER, MAX_CARDS_PER_USER + 1))
        chosen = rng.choice(N_CARDS, size=n_hold, replace=False, p=probs)
        holdings.extend((u, int(c)) for c in chosen)
    return holdings


def generate_dataset(seed=SEED):
    """Generate the full synthetic dataset (deterministic given the seed)."""
    rng = np.random.default_rng(seed)
    cards, reward_matrix, fees, bonuses, popularity = _generate_cards(rng)
    users, spend_matrix, arch_idx = _generate_users(rng)
    value_matrix = _compute_value_matrix(spend_matrix, reward_matrix, fees, bonuses)
    holdings = _generate_holdings(rng, value_matrix, popularity, arch_idx)

    return Dataset(
        cards=cards,
        card_reward_matrix=reward_matrix,
        annual_fees=fees,
        signup_bonuses=bonuses,
        card_popularity=popularity,
        users=users,
        spend_matrix=spend_matrix,
        user_archetype_idx=arch_idx,
        value_matrix=value_matrix,
        holdings=holdings,
    )


if __name__ == "__main__":
    ds = generate_dataset()
    print(f"Users: {len(ds.users):,} | Cards: {len(ds.cards)} | Holdings: {len(ds.holdings):,}")
    print(f"Avg cards/user: {len(ds.holdings) / len(ds.users):.2f}")
    print("\nSample cards:")
    print(ds.cards[["name", "annual_fee", "signup_bonus", "max_rate"]].head(8).to_string(index=False))
    print("\nArchetype counts:")
    print(ds.users["archetype"].value_counts().to_string())
