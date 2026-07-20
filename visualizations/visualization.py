"""
Additional detector diagnostics: confusion matrix, wrong rate by finding
category, and severity mix in wrong vs correct answers.

Usage:
    python detector_diagnostics_extra.py

Expects these files in the same directory (or edit PATHS below):
    detector_traces_qa_symmetric_debate.jsonl
    detector_traces_qa_specialized_board.jsonl
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATHS = [
    "detector_traces\qa\detector_traces_qa_symmetric_debate.jsonl",
    "detector_traces\qa\detector_traces_qa_specialized_board.jsonl",
]


def load_records():
    records = []
    for path in PATHS:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                records.append({
                    "method": d["method"],
                    "is_wrong": d["is_wrong"],
                    "n_findings": len(d["llm_findings"]),
                    "cats": [fi["category"] for fi in d["llm_findings"]],
                    "sevs": [fi.get("severity") for fi in d["llm_findings"]],
                })
    df = pd.DataFrame(records)
    df["has_finding"] = df["n_findings"] > 0
    return df


def main():
    df = load_records()

    plt.rcParams.update({
        "font.size": 10.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.titlesize": 11.5, "axes.titleweight": "medium",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: confusion matrix heatmap ---
    ax = axes[0]
    tp = df[(df.is_wrong) & (df.has_finding)].shape[0]
    fn_ = df[(df.is_wrong) & (~df.has_finding)].shape[0]
    fp = df[(~df.is_wrong) & (df.has_finding)].shape[0]
    tn = df[(~df.is_wrong) & (~df.has_finding)].shape[0]
    mat = np.array([[tn, fp], [fn_, tp]])
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No finding", "Has finding"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Correct", "Wrong"])
    ax.set_xlabel("Detector output")
    ax.set_ylabel("Ground truth")
    ax.set_title("A. Detector confusion matrix", loc="left")
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            ax.text(j, i, f"{v}\n({v / mat.sum() * 100:.1f}%)", ha="center", va="center",
                    fontsize=11, color="white" if v > mat.max() * 0.5 else "#0b0b0b")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn_)
    ax.text(0.5, -0.32, f"Precision = {precision:.2f}   Recall = {recall:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Panel B: wrong rate by finding category ---
    ax = axes[1]
    cats_all = ["hallucination", "contradiction", "error_propagation", "sycophancy"]
    cat_names = ["Hallucination", "Contradiction", "Error\npropagation", "Sycophancy"]
    wrong_rates, ns = [], []
    for c in cats_all:
        mask = df["cats"].apply(lambda l: c in l)
        ns.append(mask.sum())
        wrong_rates.append(df[mask]["is_wrong"].mean() * 100 if mask.sum() else 0)
    baseline = df["is_wrong"].mean() * 100
    bars = ax.bar(cat_names, wrong_rates, color="#e34948", width=0.55)
    ax.axhline(baseline, color="#52514e", linestyle="--", linewidth=1.2)
    ax.text(3.45, baseline + 1, f"Overall wrong rate ({baseline:.1f}%)",
            ha="right", fontsize=8.5, color="#52514e")
    for rect, v, n in zip(bars, wrong_rates, ns):
        ax.annotate(f"{v:.0f}%\n(n={n})", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wrong-answer rate given category flagged (%)")
    ax.set_title("B. Wrong rate by finding category", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(wrong_rates) * 1.3)

    # --- Panel C: severity mix in wrong vs correct cases ---
    ax = axes[2]
    rows = []
    for _, r in df.iterrows():
        for s in r["sevs"]:
            rows.append({"is_wrong": r["is_wrong"], "severity": s})
    sdf = pd.DataFrame(rows)
    sev_order = ["high", "medium"]
    correct_counts = [sdf[(~sdf.is_wrong) & (sdf.severity == s)].shape[0] for s in sev_order]
    wrong_counts = [sdf[(sdf.is_wrong) & (sdf.severity == s)].shape[0] for s in sev_order]
    correct_pct = [100 * c / sum(correct_counts) for c in correct_counts]
    wrong_pct = [100 * c / sum(wrong_counts) for c in wrong_counts]
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2, correct_pct, w,
                label=f"Correct answers (n={sum(correct_counts)} findings)", color="#199e70")
    b2 = ax.bar(x + w / 2, wrong_pct, w,
                label=f"Wrong answers (n={sum(wrong_counts)} findings)", color="#e34948")
    ax.set_xticks(x); ax.set_xticklabels(["High severity", "Medium severity"])
    ax.set_ylabel("Share of findings (%)")
    ax.set_title("C. Finding severity mix: correct vs wrong", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("detector_diagnostics_extra.png", dpi=180, bbox_inches="tight")
    print("Saved detector_diagnostics_extra.png")


if __name__ == "__main__":
    main()