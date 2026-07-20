"""
Multi-agent hallucination-detector trace analysis.

Reads the two raw JSONL trace files, computes flag rates / finding
category counts / wrong-answer overlap with each error type, and
renders a 4-panel matplotlib figure.

Usage:
    python detector_analysis.py

Expects these files in the same directory (or edit PATHS below):
    detector_traces_qa_symmetric_debate.jsonl
    detector_traces_qa_specialized_board.jsonl
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATHS = {
    "symmetric_debate": "detector_traces_qa_symmetric_debate.jsonl",
    "specialized_board": "detector_traces_qa_specialized_board.jsonl",
}


def load_records():
    records = []
    for _, path in PATHS.items():
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                cats = {finding["category"] for finding in d["llm_findings"]}
                records.append({
                    "case_id": d["case_id"],
                    "method": d["method"],
                    "is_wrong": d["is_wrong"],
                    "hallucinated_evidence_rate": d["hallucinated_evidence_rate"],
                    "has_contradiction": len(d["contradiction_flags"]) > 0,
                    "consensus_illusion_flag": d["consensus_illusion_flag"],
                    "has_sycophantic_flip": len(d["sycophantic_flip_agents"]) > 0,
                    "propagation_traced": d["propagation_origin_agent"] is not None,
                    "finding_hallucination": "hallucination" in cats,
                    "finding_contradiction": "contradiction" in cats,
                    "finding_error_propagation": "error_propagation" in cats,
                    "finding_sycophancy": "sycophancy" in cats,
                    "n_findings": len(d["llm_findings"]),
                })
    return pd.DataFrame(records)


def grouped_bar(ax, cat_labels, vals_a, vals_b, ylabel, title, colors, series_labels, pct=True):
    x = np.arange(len(cat_labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, vals_a, w, label=series_labels[0], color=colors[0])
    b2 = ax.bar(x + w / 2, vals_b, w, label=series_labels[1], color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fmt = (lambda v: f"{v:.1f}%") if pct else (lambda v: f"{v:.0f}")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(fmt(h), (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, color="#52514e")
    return b1, b2


def main():
    df = load_records()
    methods = ["specialized_board", "symmetric_debate"]
    colors = {"specialized_board": "#2a78d6", "symmetric_debate": "#eda100"}
    pretty = {"specialized_board": "Specialized board", "symmetric_debate": "Symmetric debate"}

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.titlesize": 11.5,
        "axes.titleweight": "medium",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Multi-agent detector trace analysis: specialized board vs symmetric debate (n=500 each)",
        fontsize=13, fontweight="medium", y=0.995,
    )

    # --- Panel A: detector flag rates by architecture ---
    df["high_halluc_flag"] = df["hallucinated_evidence_rate"] >= 0.15
    flag_cols = ["is_wrong", "high_halluc_flag", "has_contradiction",
                 "consensus_illusion_flag", "has_sycophantic_flip", "propagation_traced"]
    flag_names = ["Wrong", "High\nhallucination", "Contradiction",
                  "Consensus\nillusion", "Sycophantic\nflip", "Propagation\ntraced"]
    vals = {m: [df[df.method == m][c].mean() * 100 for c in flag_cols] for m in methods}
    b1, b2 = grouped_bar(axes[0, 0], flag_names, vals["specialized_board"], vals["symmetric_debate"],
                          "Share of cases (%)", "A. Detector flag rates by architecture",
                          [colors["specialized_board"], colors["symmetric_debate"]],
                          [pretty["specialized_board"], pretty["symmetric_debate"]])
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=9)

    # --- Panel B: llm_findings category counts by architecture ---
    cat_cols = ["finding_hallucination", "finding_contradiction",
                "finding_error_propagation", "finding_sycophancy"]
    cat_names = ["Hallucination", "Contradiction", "Error\npropagation", "Sycophancy"]
    counts = {m: [df[df.method == m][c].sum() for c in cat_cols] for m in methods}
    grouped_bar(axes[0, 1], cat_names, counts["specialized_board"], counts["symmetric_debate"],
                "Number of cases flagged", "B. Cases with an LLM-identified finding, by category",
                [colors["specialized_board"], colors["symmetric_debate"]],
                [pretty["specialized_board"], pretty["symmetric_debate"]], pct=False)
    axes[0, 1].legend(frameon=False, loc="upper right", fontsize=9)

    # --- Panel C: wrong rate by whether any finding present ---
    df["has_any_finding"] = df["n_findings"] > 0
    wr = df.groupby(["method", "has_any_finding"])["is_wrong"].mean().unstack() * 100
    grouped_bar(axes[1, 0], ["No finding", "Has finding"],
                [wr.loc["specialized_board", False], wr.loc["specialized_board", True]],
                [wr.loc["symmetric_debate", False], wr.loc["symmetric_debate", True]],
                "Wrong-answer rate (%)", "C. Wrong-answer rate, with vs without any finding",
                [colors["specialized_board"], colors["symmetric_debate"]],
                [pretty["specialized_board"], pretty["symmetric_debate"]])
    axes[1, 0].legend(frameon=False, loc="upper left", fontsize=9)

    # --- Panel D: overlap of wrong answers with each error type ---
    overlap_cols = {
        "Hallucination\nfinding": "finding_hallucination",
        "Contradiction\nfinding": "finding_contradiction",
        "Error\npropagation": "finding_error_propagation",
        "Sycophancy\nfinding": "finding_sycophancy",
        "High halluc.\nevidence rate": "high_halluc_flag",
        "Consensus\nillusion": "consensus_illusion_flag",
    }
    names4 = list(overlap_cols.keys())
    wrong_df = df[df.is_wrong]
    correct_df = df[~df.is_wrong]
    wrong_pct = [wrong_df[overlap_cols[n]].mean() * 100 for n in names4]
    correct_pct = [correct_df[overlap_cols[n]].mean() * 100 for n in names4]

    axp = axes[1, 1]
    grouped_bar(axp, names4, wrong_pct, correct_pct,
                "Share of cases carrying flag (%)",
                "D. Overlap: error-type prevalence in wrong vs correct answers",
                ["#e34948", "#199e70"],
                [f"Wrong answers (n={len(wrong_df)})", f"Correct answers (n={len(correct_df)})"])
    axp.legend(frameon=False, loc="upper right", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("detector_analysis_matplotlib.png", dpi=180, bbox_inches="tight")
    print("Saved detector_analysis_matplotlib.png")


if __name__ == "__main__":
    main()
