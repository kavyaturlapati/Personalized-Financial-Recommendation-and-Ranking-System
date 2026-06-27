"""
CardRank - interactive web frontend for the personalized card recommender.

Run:  streamlit run app.py

Reuses the exact models from pipeline.py. Pick a customer, choose how many cards
to recommend, and see the ranked picks with plain-English reasons, plus the model
leaderboard and how the ranker makes decisions.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from config import CATEGORIES, TOP_K_REPORT
from pipeline import run_pipeline, recommend, explain_card

# ---- Palette (credit-card native: deep navy + premium gold) ----------------
INK, BLUE, GOLD, SLATE, MUTE = "#11213A", "#1F5FE0", "#C2912F", "#6B7785", "#C2C9D4"

st.set_page_config(page_title="CardRank", page_icon="\U0001F4B3", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown { font-family:'IBM Plex Sans',sans-serif; }
.app-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.05rem;
             color:#11213A; letter-spacing:-.02em; margin:0; }
.app-sub { color:#6B7785; font-size:1.0rem; margin:.25rem 0 0; }
.app-rule { height:3px; width:64px; background:#C2912F; border-radius:2px; margin:.7rem 0 1rem; }
.user-head { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.2rem;
             color:#11213A; margin:.1rem 0 .15rem; }
.section { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.02rem;
           color:#11213A; margin:.4rem 0 .5rem; }
.muted { color:#6B7785; font-size:.86rem; }
.hold { display:inline-block; background:#EEF1F6; color:#3A475A; border:1px solid #E1E6EE;
        border-radius:999px; padding:2px 10px; font-size:.78rem;
        font-family:'IBM Plex Mono',monospace; margin:0 6px 6px 0; }
.rec { display:flex; gap:14px; align-items:flex-start; background:#FFFFFF; border:1px solid #E6E9EF;
       border-radius:14px; padding:14px 16px; margin-bottom:10px; box-shadow:0 1px 2px rgba(17,33,58,.04); }
.rec-rank { flex:0 0 auto; width:30px; height:30px; border-radius:50%; background:#11213A; color:#fff;
            font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:.95rem;
            display:flex; align-items:center; justify-content:center; margin-top:2px; }
.rec-body { flex:1 1 auto; min-width:0; }
.rec-top { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
.rec-name { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.02rem; color:#11213A; }
.rec-value { font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1.16rem;
             color:#C2912F; white-space:nowrap; }
.rec-value-unit { font-size:.72rem; color:#9aa3b0; margin-left:1px; }
.rec-chips { margin:7px 0 5px; }
.chip { display:inline-block; background:#EAF1FF; color:#1F5FE0; border-radius:6px; padding:2px 8px;
        font-size:.74rem; font-weight:500; margin:0 6px 6px 0; }
.rec-meta { color:#6B7785; font-size:.84rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the recommender (first load only, ~30s)\u2026")
def load_pipeline():
    return run_pipeline()


def _chips(boosted):
    return [s.replace("_", " ").title() for s in boosted.split("+")]


def _rec_card(rank, name, value, fee, chips, reason):
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    fee_txt = "No annual fee" if fee == 0 else f"${fee} annual fee"
    return (
        f'<div class="rec"><div class="rec-rank">{rank}</div><div class="rec-body">'
        f'<div class="rec-top"><span class="rec-name">{name}</span>'
        f'<span class="rec-value">${value:,.0f}<span class="rec-value-unit">/yr</span></span></div>'
        f'<div class="rec-chips">{chip_html}</div>'
        f'<div class="rec-meta">{fee_txt} &middot; {reason}</div></div></div>'
    )


def _bar(df, value_col, label, highlight=None, color=BLUE):
    if highlight:
        enc_color = alt.condition(alt.datum.Model == highlight,
                                  alt.value(color), alt.value(MUTE))
    else:
        enc_color = alt.value(color)
    return (
        alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X(f"{value_col}:Q", title=label),
            y=alt.Y("Model:N", sort=list(df["Model"]), title=None),
            color=enc_color,
            tooltip=["Model", value_col],
        ).properties(height=230)
    )


p = load_pipeline()
ds, results, hybrid = p.ds, p.results, p.hybrid

st.markdown(
    '<div class="app-title">CardRank</div>'
    '<div class="app-sub">Pick a customer and see which credit cards the model ranks '
    'highest \u2014 and why.</div><div class="app-rule"></div>',
    unsafe_allow_html=True,
)

# ---- Sidebar controls ------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section">Controls</div>', unsafe_allow_html=True)
    archetypes = ["All"] + sorted(ds.users["archetype"].unique())
    arch = st.selectbox("Spending profile", archetypes, index=0)
    pool = (list(ds.users.index) if arch == "All"
            else list(ds.users.index[ds.users["archetype"] == arch]))

    if "user_id" not in st.session_state:
        trav = ds.users.index[ds.users["archetype"] == "traveler"]
        st.session_state.user_id = int(trav[0]) if len(trav) else int(ds.users.index[0])

    if st.button("\U0001F3B2  Pick a random customer", width='stretch'):
        st.session_state.user_id = int(np.random.default_rng().choice(pool))

    if st.session_state.user_id not in pool:
        st.session_state.user_id = int(pool[0])

    user_idx = st.selectbox(
        "Customer", pool, index=pool.index(st.session_state.user_id),
        format_func=lambda u: (f"User {u} \u00b7 {ds.users.loc[u, 'archetype']} "
                               f"\u00b7 ${ds.users.loc[u, 'monthly_spend']:,.0f}/mo"),
    )
    st.session_state.user_id = user_idx

    st.divider()
    model_name = st.selectbox("Ranking model", list(p.models.keys()),
                              index=len(p.models) - 1)
    k = st.slider("How many to recommend", 3, 10, TOP_K_REPORT)
    st.divider()
    st.caption("Synthetic data \u00b7 models trained on first load and cached.")

# ---- Per-user computations -------------------------------------------------
u = user_idx
spend = ds.spend_matrix[u]
top_cats = np.argsort(-spend)[:3]
top_cards, held = recommend(p, u, k=k, model_name=model_name)
values = p.content.expected_values(ds.spend_matrix)[u]

tab_rec, tab_perf, tab_how = st.tabs(
    ["Recommendations", "Model performance", "How it works"])

# ===== Recommendations ======================================================
with tab_rec:
    st.markdown(
        f'<div class="user-head">User {u} \u00b7 {ds.users.loc[u, "archetype"]}</div>'
        f'<div class="muted">${ds.users.loc[u, "monthly_spend"]:,.0f} monthly spend '
        f'\u00b7 ranked by <b>{model_name}</b></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="section">Spending fingerprint</div>', unsafe_allow_html=True)
        order = np.argsort(-spend)[:6]
        sp_df = pd.DataFrame({
            "Category": [CATEGORIES[c].replace("_", " ").title() for c in order],
            "Monthly": [float(spend[c]) for c in order],
        })
        chart = alt.Chart(sp_df).mark_bar(color=GOLD, cornerRadiusEnd=4).encode(
            x=alt.X("Monthly:Q", title="Monthly spend ($)", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("Category:N", sort="-x", title=None),
            tooltip=["Category", alt.Tooltip("Monthly:Q", format="$,.0f")],
        ).properties(height=215)
        st.altair_chart(chart, width='stretch')
    with right:
        st.markdown('<div class="section">Currently holds</div>', unsafe_allow_html=True)
        if held:
            st.markdown("".join(f'<span class="hold">Card {c:02d}</span>' for c in held),
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="muted">No cards yet \u2014 a cold-start customer.</span>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:1rem;">'
                f'Top {k} recommended cards</div>', unsafe_allow_html=True)
    html = ""
    for rank, c in enumerate(top_cards, 1):
        row = ds.cards.loc[c]
        html += _rec_card(rank, row["name"], values[c], int(row["annual_fee"]),
                          _chips(row["boosted_categories"]),
                          explain_card(ds, u, c, top_cats))
    st.markdown(html, unsafe_allow_html=True)

# ===== Model performance ====================================================
with tab_perf:
    kk = TOP_K_REPORT
    hyb, pop = results.loc["Hybrid Ranker"], results.loc["Popularity"]
    lift = (hyb[f"NDCG@{kk}"] - pop[f"NDCG@{kk}"]) / pop[f"NDCG@{kk}"] * 100

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Hybrid NDCG@{kk}", f"{hyb[f'NDCG@{kk}']:.3f}", f"+{lift:.0f}% vs popularity")
    m2.metric(f"Hit Rate@{kk}", f"{hyb[f'HitRate@{kk}']:.1%}")
    m3.metric(f"Avg value of top-{kk}", f"${hyb[f'AvgValue@{kk}']:,.0f}/yr")

    st.write("")
    order = ["Random", "Popularity", "Collaborative Filtering",
             "Content (value)", "Hybrid Ranker"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        df = pd.DataFrame({"Model": order,
                           f"NDCG@{kk}": [results.loc[m, f"NDCG@{kk}"] for m in order]})
        st.altair_chart(_bar(df, f"NDCG@{kk}", f"Ranking quality (NDCG@{kk})",
                             highlight="Hybrid Ranker", color=BLUE),
                        width='stretch')
    with c2:
        df = pd.DataFrame({"Model": order,
                           f"AvgValue@{kk}": [results.loc[m, f"AvgValue@{kk}"] for m in order]})
        st.altair_chart(_bar(df, f"AvgValue@{kk}", f"Avg expected value of top-{kk} ($)",
                             highlight="Hybrid Ranker", color=GOLD),
                        width='stretch')

    st.markdown('<div class="section">Full leaderboard</div>', unsafe_allow_html=True)
    fmt = {c: (lambda v: f"${v:,.0f}") if c.startswith("AvgValue")
           else (lambda v: f"{v:.3f}") for c in results.columns}
    st.dataframe(results.style.format(fmt), width='stretch')
    st.caption("Leave-one-out evaluation over all customers. The Hybrid Ranker wins "
               "because it is the only model that combines collaborative, content, "
               "and statistical signals.")

# ===== How it works =========================================================
with tab_how:
    st.markdown('<div class="section">Why a hybrid?</div>', unsafe_allow_html=True)
    st.markdown(
        "A customer holds a card for a **mix** of reasons, and each reason is captured "
        "by a different model:\n\n"
        "- **Reward value** \u2014 the card pays well for how they spend \u2192 *content scorer*\n"
        "- **Popularity** \u2014 lots of people hold it \u2192 *popularity baseline*\n"
        "- **Segment taste** \u2014 people who spend similarly favor it \u2192 *collaborative filtering*\n\n"
        "The hybrid learning-to-rank model fuses all three (plus card-level features), "
        "which is why it generalizes best. The bars below show how much the trained "
        "ranker leans on each signal."
    )
    fi = hybrid.feature_importance()
    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
    chart = alt.Chart(fi_df).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        x=alt.X("Importance:Q"),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
    ).properties(height=300)
    st.altair_chart(chart, width='stretch')
    st.caption("Models are evaluated with a leave-one-out protocol and the collaborative "
               "signal is computed leave-one-out-aware to avoid train/eval leakage.")
