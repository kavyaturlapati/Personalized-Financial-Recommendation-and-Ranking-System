"""
Command-line pipeline for the personalized financial recommender.

Run:  python main.py

Trains all models (via pipeline.py), evaluates them, prints a comparison table
and feature importances, saves a chart and CSVs to ./outputs/, and shows a worked
recommendation demo. For an interactive version, run the web app instead:

    streamlit run app.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SEED, CATEGORIES, TOP_K_REPORT, K_VALUES
from pipeline import run_pipeline, recommend, explain_card

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating data and training models...")
    p = run_pipeline(SEED)
    ds, results, hybrid = p.ds, p.results, p.hybrid
    print(f"  {len(ds.users):,} users | {len(ds.cards)} cards | "
          f"{len(ds.holdings):,} holdings ({len(ds.holdings)/len(ds.users):.1f}/user)\n")

    results.to_csv(os.path.join(OUTPUT_DIR, "results.csv"))
    _print_table(results)
    _print_lift(results)
    _save_charts(results)
    _feature_importance(hybrid)
    _demo(p)

    ds.cards.to_csv(os.path.join(OUTPUT_DIR, "card_catalog.csv"), index=False)
    ds.users.head(200).to_csv(os.path.join(OUTPUT_DIR, "sample_users.csv"), index=False)
    print(f"\nArtifacts written to ./{OUTPUT_DIR}/")


def _print_table(results):
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    show = results.copy()
    for c in show.columns:
        if c.startswith("AvgValue"):
            show[c] = show[c].map(lambda v: f"${v:,.0f}")
        else:
            show[c] = show[c].map(lambda v: f"{v:.3f}")
    print(show.to_string())
    print()


def _print_lift(results):
    k = TOP_K_REPORT
    base, hyb = results.loc["Popularity"], results.loc["Hybrid Ranker"]
    print("LIFT OF HYBRID RANKER OVER POPULARITY BASELINE")
    print("-" * 60)
    for metric in [f"NDCG@{k}", f"HitRate@{k}", "MRR", f"AvgValue@{k}"]:
        b, h = base[metric], hyb[metric]
        lift = (h - b) / b * 100 if b else float("nan")
        unit = "$" if metric.startswith("AvgValue") else ""
        print(f"  {metric:<14} {unit}{b:,.3f} -> {unit}{h:,.3f}   (+{lift:.1f}%)")
    print()


def _save_charts(results):
    k = TOP_K_REPORT
    order = ["Random", "Popularity", "Collaborative Filtering",
             "Content (value)", "Hybrid Ranker"]
    colors = ["#9aa0a6", "#bdc1c6", "#7baaf7", "#669df6", "#1a73e8"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], f"NDCG@{k}", f"Ranking quality (NDCG@{k})"),
        (axes[1], f"AvgValue@{k}", f"Avg expected value of top-{k} ($)"),
    ]:
        vals = [results.loc[m, metric] for m in order]
        bars = ax.bar(range(len(order)), vals, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in order], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            label = f"${v:,.0f}" if metric.startswith("AvgValue") else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Personalized Card Recommender \u2014 Model Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {path}\n")


def _feature_importance(hybrid):
    print("HYBRID RANKER \u2014 FEATURE IMPORTANCE")
    print("-" * 60)
    for feat, imp in hybrid.feature_importance().items():
        bar = "\u2588" * int(round(imp * 40))
        print(f"  {feat:<28} {imp:.3f} {bar}")
    print()


def _demo(p):
    ds = p.ds
    travelers = ds.users.index[ds.users["archetype"] == "traveler"]
    user_idx = int(travelers[0]) if len(travelers) else 0

    print("=" * 100)
    print(f"RECOMMENDATION DEMO \u2014 user #{user_idx} "
          f"({ds.users.loc[user_idx, 'archetype']})")
    print("=" * 100)
    spend = ds.spend_matrix[user_idx]
    top_cats = np.argsort(-spend)[:3]
    print("Top spending categories (monthly):")
    for c in top_cats:
        print(f"  {CATEGORIES[c]:<16} ${spend[c]:,.0f}")

    top_cards, held = recommend(p, user_idx, k=TOP_K_REPORT)
    values = p.content.expected_values(ds.spend_matrix)[user_idx]
    print(f"\nAlready holds: {held}")
    print(f"\nTop {TOP_K_REPORT} recommended cards:")
    for rank, card in enumerate(top_cards, 1):
        row = ds.cards.loc[card]
        reason = explain_card(ds, user_idx, card, top_cats)
        print(f"  {rank}. {row['name']}")
        print(f"     est. value ${values[card]:,.0f}/yr | fee ${row['annual_fee']} | {reason}")
    print()


if __name__ == "__main__":
    main()
