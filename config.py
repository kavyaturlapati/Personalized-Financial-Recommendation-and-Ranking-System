"""
Central configuration for the personalized financial recommender.

The signal-weight constants (W_VALUE / W_POPULARITY / W_AFFINITY) control *why*
a synthetic user holds a card. Each weight maps to a different model family, so
tuning them changes which model is expected to win:

    W_VALUE       -> reward-value alignment      -> Content recommender
    W_POPULARITY  -> how popular a card is        -> Popularity baseline
    W_AFFINITY    -> shared taste within a segment -> Collaborative filtering

Because holding depends on a *mix* of all three, only the Hybrid ranker (which
combines collaborative + content + statistical signals) can capture all of it.
"""

SEED = 42

# Spending categories. Used for BOTH user spend profiles and card reward rates,
# so the two live in the same vector space (this is what makes content matching work).
CATEGORIES = [
    "groceries", "dining", "travel", "gas", "online_shopping",
    "entertainment", "utilities", "transit", "drugstores", "other",
]
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES)

# Dataset size
N_USERS = 3000
N_CARDS = 40

# How many cards a user holds (drives the implicit-feedback matrix density)
MIN_CARDS_PER_USER = 2
MAX_CARDS_PER_USER = 6

# Relative strength of each holding signal (see module docstring)
W_VALUE = 1.0
W_POPULARITY = 0.6
W_AFFINITY = 0.9
HOLD_TEMPERATURE = 0.85   # lower = users hold value/affinity-aligned cards more deterministically
HOLD_NOISE = 0.5          # std of random noise added to holding logits

# Model
N_LATENT_FACTORS = 16     # CF embedding dimensionality (must be < N_CARDS)
N_NEGATIVES_PER_POS = 6   # negative samples per positive when training the ranker

# Evaluation
K_VALUES = [3, 5, 10]
TOP_K_REPORT = 5
