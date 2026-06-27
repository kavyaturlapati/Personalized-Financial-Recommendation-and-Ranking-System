"""
Evaluation harness.

Protocol: leave-one-out.
  For every user we hold out ONE of their cards as the test target and keep the
  rest as training history. At test time the model ranks all cards the user does
  NOT already hold (the held-out card plus every never-held card). We then check
  where the held-out card lands.

Two metric families:
  Ranking metrics  -> is the held-out card ranked highly? (Hit Rate, NDCG, MRR, MAP)
  Business metrics -> are the recommendations actually good for the business/user?
                      (expected $ value of the top-K, catalog coverage, personalization)

Ties are broken with a tiny seeded jitter so models with many equal scores
(e.g. popularity) are ranked fairly rather than by array order.
"""
import numpy as np

from config import K_VALUES


def leave_one_out_split(holdings, n_users, seed=0):
    """Split holdings into a training set and one held-out card per eligible user."""
    rng = np.random.default_rng(seed)
    by_user = {}
    for u, c in holdings:
        by_user.setdefault(u, []).append(c)

    train_by_user, test_target = {}, {}
    train_holdings = []
    for u, cards in by_user.items():
        cards = list(cards)
        if len(cards) >= 2:                      # need >=1 left for training
            held = int(rng.choice(cards))
            test_target[u] = held
            remaining = [c for c in cards if c != held]
        else:
            remaining = cards
        train_by_user[u] = set(remaining)
        train_holdings.extend((u, c) for c in remaining)
    return train_holdings, train_by_user, test_target


def training_popularity(train_holdings, n_cards):
    """Number of users holding each card in the training set."""
    counts = np.zeros(n_cards)
    for _, c in train_holdings:
        counts[c] += 1
    return counts


def _ranked_candidates(model, user, candidates, jitter_rng):
    """Return candidate cards ordered best-first for a model."""
    scores = np.asarray(model.score(user, candidates), dtype=float)
    scores = scores + jitter_rng.random(len(scores)) * 1e-9   # deterministic tie-break
    order = np.argsort(-scores)
    return candidates[order]


def evaluate(model, dataset, train_by_user, test_target, k_values=K_VALUES, seed=0):
    """Compute ranking + business metrics for a model under leave-one-out."""
    n_cards = dataset.card_reward_matrix.shape[0]
    all_cards = np.arange(n_cards)
    jitter_rng = np.random.default_rng(seed)

    max_k = max(k_values)
    hits = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    ap = {k: [] for k in k_values}
    rr = []
    value_at_k = {k: [] for k in k_values}      # business: $ value of top-K
    topk_sets = {k: [] for k in k_values}        # for coverage / personalization

    for user, target in test_target.items():
        held = train_by_user.get(user, set())
        candidates = all_cards[~np.isin(all_cards, list(held))]   # not already held
        ranked = _ranked_candidates(model, user, candidates, jitter_rng)

        rank = int(np.where(ranked == target)[0][0]) + 1          # 1-based rank of target
        rr.append(1.0 / rank)
        for k in k_values:
            in_topk = rank <= k
            hits[k].append(1.0 if in_topk else 0.0)
            ndcg[k].append(1.0 / np.log2(rank + 1) if in_topk else 0.0)
            ap[k].append(1.0 / rank if in_topk else 0.0)         # single-relevant AP
            topk = ranked[:k]
            value_at_k[k].append(dataset.value_matrix[user, topk].mean())
            topk_sets[k].append(set(int(c) for c in topk))

    results = {"MRR": float(np.mean(rr))}
    for k in k_values:
        results[f"HitRate@{k}"] = float(np.mean(hits[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcg[k]))
        results[f"MAP@{k}"] = float(np.mean(ap[k]))
    # Business metrics reported at the headline K.
    report_k = k_values[1] if len(k_values) > 1 else k_values[0]
    results[f"AvgValue@{report_k}"] = float(np.mean(value_at_k[report_k]))
    results[f"Coverage@{report_k}"] = _coverage(topk_sets[report_k], n_cards)
    results[f"Personalization@{report_k}"] = _personalization(topk_sets[report_k], seed)
    return results


def _coverage(topk_sets, n_cards):
    """Fraction of the catalog that appears in at least one user's top-K."""
    recommended = set().union(*topk_sets) if topk_sets else set()
    return len(recommended) / n_cards


def _personalization(topk_sets, seed, n_pairs=4000):
    """Mean (1 - Jaccard) over random user pairs: how different are recommendation lists?"""
    if len(topk_sets) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = len(topk_sets)
    diffs = []
    for _ in range(n_pairs):
        i, j = rng.integers(n), rng.integers(n)
        if i == j:
            continue
        a, b = topk_sets[i], topk_sets[j]
        union = a | b
        jaccard = len(a & b) / len(union) if union else 0.0
        diffs.append(1.0 - jaccard)
    return float(np.mean(diffs)) if diffs else 0.0
