"""
Round-over-round agreement and LLM-identified error-source analysis.

Reads the two raw JSONL trace files and renders a 4-panel matplotlib
figure:
  A. Reasoning alignment score, round 1 vs round 2 (boxplot)
  B. Answer agreement score, round 1 vs round 2 (boxplot)
  C. Which agent role is implicated in findings (source of error)
  D. Error category by round (r1 vs r2), source of error over time

Usage:
    python agreement_and_error_source.py

Expects these files in the same directory (or edit PATHS below):
    detector_traces_qa_symmetric_debate.jsonl
    detector_traces_qa_specialized_board.jsonl
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATHS = {
    "symmetric_debate": "detector_traces/qausmle/detector_traces_qa_specialized_board.jsonl",
}

METHODS = ["specialized_board"]
COLORS = {"specialized_board": "#2a78d6", "symmetric_debate": "#eda100"}
PRETTY = {"specialized_board": "Specialized board", "symmetric_debate": "Symmetric debate"}
ROLE_ORDER = {
    "specialized_board": ["Agent_Diagnostician", "Agent_Evidence", "Agent_Treatment"],
    # "symmetric_debate": ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"],
}


def load_records():
    records, findings_rows = [], []
    for _, path in PATHS.items():
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                records.append({
                    "case_id": d["case_id"],
                    "method": d["method"],
                    "is_wrong": d["is_wrong"],
                    "reasoning_alignment_r1": d["reasoning_alignment_r1"],
                    "reasoning_alignment_r2": d["reasoning_alignment_r2"],
                    "answer_agreement_r1": d["answer_agreement_r1"],
                    "answer_agreement_r2": d["answer_agreement_r2"],
                })
                for finding in d["llm_findings"]:
                    findings_rows.append({
                        "method": d["method"],
                        "agent_id": finding["agent_id"],
                        "round": finding["round"],
                        "category": finding["category"],
                        "severity": finding.get("severity"),
                    })
    return pd.DataFrame(records), pd.DataFrame(findings_rows)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def agreement_boxplot(ax, df, col_r1, col_r2, ylabel, title):
    positions, data, box_colors, labels = [], [], [], []
    pos = 0
    for m in METHODS:
        for rnd, col in [("R1", col_r1), ("R2", col_r2)]:
            data.append(df[df.method == m][col].values)
            positions.append(pos)
            box_colors.append(COLORS[m])
            labels.append(rnd)
            pos += 0.8
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color="#0b0b0b", linewidth=1.4))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor(c)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    style_ax(ax)
    ax.text(np.mean(positions[0:2]), -0.14, PRETTY["specialized_board"], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9.5, color=COLORS["specialized_board"])
    ax.text(np.mean(positions[2:4]), -0.14, PRETTY["symmetric_debate"], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=9.5, color=COLORS["symmetric_debate"])


def main():
    df, fdf = load_records()

    plt.rcParams.update({
        "font.size": 10.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.titlesize": 11.5, "axes.titleweight": "medium",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Round-over-round agreement and LLM-identified error sources",
                 fontsize=13, fontweight="medium", y=0.995)

    # A. reasoning alignment r1 vs r2
    agreement_boxplot(axes[0, 0], df, "reasoning_alignment_r1", "reasoning_alignment_r2",
                       "Reasoning alignment score", "A. Reasoning alignment: round 1 vs round 2")

    # B. answer agreement r1 vs r2
    agreement_boxplot(axes[0, 1], df, "answer_agreement_r1", "answer_agreement_r2",
                       "Answer agreement score", "B. Answer agreement: round 1 vs round 2")

    # C. source of error by agent role
    axC = axes[1, 0]
    x = np.arange(3)
    w = 0.36
    sb_counts = [fdf[(fdf.method == "specialized_board") & (fdf.agent_id == a)].shape[0]
                 for a in ROLE_ORDER["specialized_board"]]
    sd_counts = [fdf[(fdf.method == "symmetric_debate") & (fdf.agent_id == a)].shape[0]
                 for a in ROLE_ORDER["symmetric_debate"]]
    b1 = axC.bar(x - w / 2, sb_counts, w, label=PRETTY["specialized_board"], color=COLORS["specialized_board"])
    b2 = axC.bar(x + w / 2, sd_counts, w, label=PRETTY["symmetric_debate"], color=COLORS["symmetric_debate"])
    axC.set_xticks(x)
    axC.set_xticklabels(["Agent 1\n(Diagnostician / Alpha)", "Agent 2\n(Evidence / Beta)",
                          "Agent 3\n(Treatment / Gamma)"], fontsize=9)
    axC.set_ylabel("Number of findings attributed to agent")
    axC.set_title("C. Source of error: which agent is implicated", loc="left")
    style_ax(axC)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            axC.annotate(f"{int(h)}", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=8.5, color="#52514e")
    axC.legend(frameon=False, loc="upper right", fontsize=9)

    # D. error category by round
    axD = axes[1, 1]
    cats = ["hallucination", "contradiction", "error_propagation", "sycophancy"]
    cat_names = ["Hallucination", "Contradiction", "Error\npropagation", "Sycophancy"]
    x = np.arange(len(cats))
    w = 0.2
    combos = [("specialized_board", "r1", "#2a78d6", 0.55), ("specialized_board", "r2", "#2a78d6", 1.0),
              ("symmetric_debate", "r1", "#eda100", 0.55), ("symmetric_debate", "r2", "#eda100", 1.0)]
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for (m, rnd, c, alpha), off in zip(combos, offsets):
        vals = [fdf[(fdf.method == m) & (fdf["round"] == rnd) & (fdf.category == cat)].shape[0]
                for cat in cats]
        axD.bar(x + off, vals, w * 0.95, color=c, alpha=alpha, label=f"{PRETTY[m]} — {rnd.upper()}")
    axD.set_xticks(x)
    axD.set_xticklabels(cat_names)
    axD.set_ylabel("Number of findings")
    axD.set_title("D. Source of error: category by round", loc="left")
    style_ax(axD)
    axD.legend(frameon=False, loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig("agreement_and_error_source.png", dpi=180, bbox_inches="tight")
    print("Saved agreement_and_error_source.png")


if __name__ == "__main__":
    main()
