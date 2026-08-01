"""
detector_trace_analysis.py
===========================
End-to-end analysis + visualization of multi-agent LLM debate/board detector
traces (2 architectures x 3 judge/detector backend models x 500 QA cases).

Usage:
    python3 detector_trace_analysis.py

Reads   : /mnt/user-data/uploads/detector_traces_*.jsonl
Writes  : ./cases_tidy.csv, ./findings_tidy.csv, ./summary_by_model_method.csv,
          ./illusion_vs_error.csv, ./corr_with_error.csv,
          ./category_corr_by_model.csv, ./severity_corr_by_model.csv,
          ./dashboard.png, ./charts/*.png
"""
import json
import re
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where to look for the input .jsonl files. Priority:
#   1. DETECTOR_TRACES_DIR environment variable, if set
#   2. the same folder as this script
#   3. the sandbox path used during development (harmless if absent)
_CANDIDATE_DIRS = [
    os.environ.get("DETECTOR_TRACES_DIR"),
    SCRIPT_DIR,
    "/mnt/user-data/uploads",
]

DATA_DIR = None
FILES = []
for cand in _CANDIDATE_DIRS:
    if not cand:
        continue
    matches = sorted(glob.glob(os.path.join(cand, "detector_traces_*.jsonl")))
    if matches:
        DATA_DIR = cand
        FILES = matches
        break

OUT_DIR = SCRIPT_DIR
CHART_DIR = os.path.join(OUT_DIR, "charts")
FNAME_RE = re.compile(r"detector_traces_(?P<dataset>[a-z]+)_(?P<method>.+)_(?P<model>qwen|openai|deepseek)\.jsonl")

MODEL_LABELS = {"openai": "OpenAI", "qwen": "Qwen", "deepseek": "DeepSeek"}
METHOD_LABELS = {"specialized_board": "Role-Specialist Board", "symmetric_debate": "Symmetric Debate"}

COLOR_METHOD = {"Role-Specialist Board": "#4C72B0", "Symmetric Debate": "#DD8452"}
COLOR_MODEL = {"OpenAI": "#2CA02C", "Qwen": "#9467BD", "DeepSeek": "#D62728"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.7,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def pct(ax, axis="y"):
    fmt = mticker.PercentFormatter(xmax=1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# 1. Load & tidy
# ---------------------------------------------------------------------------

def load_records():
    if not FILES:
        raise FileNotFoundError(
            "No files matching 'detector_traces_*.jsonl' were found.\n"
            f"Looked in: {[c for c in _CANDIDATE_DIRS if c]}\n"
            "Fix: either place the .jsonl files in the same folder as this script, "
            "or set the DETECTOR_TRACES_DIR environment variable to their folder, e.g.\n"
            '  (Windows PowerShell)  $env:DETECTOR_TRACES_DIR = "C:\\path\\to\\jsonl_folder"\n'
            "  (cmd.exe)             set DETECTOR_TRACES_DIR=C:\\path\\to\\jsonl_folder\n"
            "  (bash)                export DETECTOR_TRACES_DIR=/path/to/jsonl_folder"
        )

    rows, findings_rows = [], []
    for fp in FILES:
        base = os.path.basename(fp)
        m = FNAME_RE.match(base)
        if not m:
            print(f"WARNING: filename didn't match pattern, skipping: {base}")
            continue
        model = MODEL_LABELS.get(m.group("model"), m.group("model"))
        method = METHOD_LABELS.get(m.group("method"), m.group("method"))

        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                llm_findings = d.get("llm_findings") or []
                other_findings = d.get("other_findings") or []
                n_sycophantic = len(d.get("sycophantic_flip_agents") or [])

                rows.append({
                    "file": base, "model": model, "method": method,
                    "case_id": d.get("case_id"), "dataset": d.get("dataset"),
                    "gold_label": d.get("gold_label"), "aggregated_answer": d.get("aggregated_answer"),
                    "is_wrong": bool(d.get("is_wrong")),
                    "detection_mode": d.get("detection_mode"),
                    "hallucinated_evidence_rate": d.get("hallucinated_evidence_rate"),
                    "reasoning_alignment_r1": d.get("reasoning_alignment_r1"),
                    "reasoning_alignment_r2": d.get("reasoning_alignment_r2"),
                    "answer_agreement_r1": d.get("answer_agreement_r1"),
                    "answer_agreement_r2": d.get("answer_agreement_r2"),
                    "consensus_illusion_flag": bool(d.get("consensus_illusion_flag")),
                    "n_sycophantic_flip_agents": n_sycophantic,
                    "n_contradiction_flags": len(d.get("contradiction_flags") or {}),
                    "n_llm_findings": len(llm_findings),
                    "n_other_findings": len(other_findings),
                    "n_findings_total": len(llm_findings) + len(other_findings),
                    "n_agents": len(d.get("claims_r1") or {}),
                })

                for fd in llm_findings:
                    if not isinstance(fd, dict):
                        continue
                    findings_rows.append({
                        "file": base, "model": model, "method": method,
                        "case_id": d.get("case_id"), "is_wrong": bool(d.get("is_wrong")),
                        "category": fd.get("category"), "severity": fd.get("severity"),
                        "agent_id": fd.get("agent_id"), "round": fd.get("round"),
                    })

    if not rows:
        raise ValueError(
            f"Found {len(FILES)} file(s) matching the pattern, but parsed 0 case records.\n"
            "Every filename must match: detector_traces_<dataset>_<method>_<qwen|openai|deepseek>.jsonl\n"
            f"Files seen: {[os.path.basename(f) for f in FILES]}"
        )

    df = pd.DataFrame(rows)
    findings_df = pd.DataFrame(findings_rows)
    df["reasoning_alignment_delta"] = df["reasoning_alignment_r2"] - df["reasoning_alignment_r1"]
    df["answer_agreement_delta"] = df["answer_agreement_r2"] - df["answer_agreement_r1"]
    return df, findings_df


# ---------------------------------------------------------------------------
# 2. Summaries
# ---------------------------------------------------------------------------

def build_summaries(df, findings_df):
    summary = df.groupby(["model", "method"]).agg(
        n_cases=("case_id", "count"),
        error_rate=("is_wrong", "mean"),
        mean_hallucination_rate=("hallucinated_evidence_rate", "mean"),
        mean_reasoning_alignment_r1=("reasoning_alignment_r1", "mean"),
        mean_reasoning_alignment_r2=("reasoning_alignment_r2", "mean"),
        mean_answer_agreement_r1=("answer_agreement_r1", "mean"),
        mean_answer_agreement_r2=("answer_agreement_r2", "mean"),
        consensus_illusion_rate=("consensus_illusion_flag", "mean"),
        mean_sycophantic_flips=("n_sycophantic_flip_agents", "mean"),
        mean_findings_per_case=("n_findings_total", "mean"),
    ).reset_index().sort_values(["method", "model"])

    illusion_vs_error = df.groupby(["model", "method", "consensus_illusion_flag"])["is_wrong"].mean().reset_index()

    numeric_cols = [
        "hallucinated_evidence_rate", "reasoning_alignment_r1", "reasoning_alignment_r2",
        "answer_agreement_r1", "answer_agreement_r2", "reasoning_alignment_delta",
        "answer_agreement_delta", "n_sycophantic_flip_agents", "n_findings_total",
    ]
    corr_rows = []
    for method, sub in df.groupby("method"):
        corr = sub[numeric_cols + ["is_wrong"]].corr()["is_wrong"].drop("is_wrong")
        for metric, val in corr.items():
            corr_rows.append({"method": method, "metric": metric, "corr_with_error": val})
    corr_df = pd.DataFrame(corr_rows)

    # --- per-model correlation between finding TYPE (category/severity) counts and is_wrong ---
    # Build one row per case with a count-per-category column, per (model, method), then correlate.
    def per_case_type_counts(type_col):
        counts = (findings_df.groupby(["file", "model", "method", "case_id", type_col])
                  .size().unstack(fill_value=0))
        counts = counts.reset_index()
        merged = counts.merge(
            df[["file", "case_id", "is_wrong"]], on=["file", "case_id"], how="left"
        )
        # cases with zero findings never appear in findings_df -> add them back as all-zero rows
        key_cols = ["file", "model", "method", "case_id"]
        all_cases = df[key_cols + ["is_wrong"]].drop_duplicates()
        merged = all_cases.merge(merged.drop(columns=["is_wrong"]), on=key_cols, how="left")
        type_cols = [c for c in counts.columns if c not in key_cols]
        merged[type_cols] = merged[type_cols].fillna(0)
        return merged, type_cols

    def corr_by_model(type_col):
        merged, type_cols = per_case_type_counts(type_col)
        out = []
        for model, sub in merged.groupby("model"):
            for t in type_cols:
                if sub[t].std() == 0:
                    r = np.nan
                else:
                    r = sub[[t, "is_wrong"]].corr().iloc[0, 1]
                out.append({"model": model, type_col: t, "corr_with_error": r, "n_flagged": int(sub[t].sum())})
        return pd.DataFrame(out)

    category_corr_by_model = corr_by_model("category")
    severity_corr_by_model = corr_by_model("severity")

    return summary, illusion_vs_error, corr_df, category_corr_by_model, severity_corr_by_model


# ---------------------------------------------------------------------------
# 3. Chart panels
# ---------------------------------------------------------------------------

def panel_error_rate(ax, df):
    g = df.groupby("method")["is_wrong"].mean().reindex(COLOR_METHOD.keys())
    bars = ax.bar(g.index, g.values, color=[COLOR_METHOD[m] for m in g.index], width=0.55)
    for b, v in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Final-answer error rate by architecture")
    ax.set_ylabel("Error rate (wrong aggregated answer)")
    pct(ax)
    ax.set_ylim(0, max(g.values) * 1.35)


def panel_hallucination_by_model(ax, df):
    piv = df.pivot_table(index="method", columns="model", values="hallucinated_evidence_rate", aggfunc="mean")
    piv = piv.reindex(index=list(COLOR_METHOD.keys()), columns=list(COLOR_MODEL.keys()))
    x = np.arange(len(piv.index)); w = 0.25
    for i, model in enumerate(piv.columns):
        ax.bar(x + (i - 1) * w, piv[model].values, width=w, label=model, color=COLOR_MODEL[model])
    ax.set_xticks(x); ax.set_xticklabels(piv.index)
    ax.set_title("Detected hallucinated-evidence rate\n(by judge/detector model)")
    ax.set_ylabel("Mean hallucinated-evidence rate")
    pct(ax)
    ax.legend(title="Judge model", fontsize=8, title_fontsize=8)


def panel_findings_per_case(ax, df):
    piv = df.pivot_table(index="method", columns="model", values="n_findings_total", aggfunc="mean")
    piv = piv.reindex(index=list(COLOR_METHOD.keys()), columns=list(COLOR_MODEL.keys()))
    x = np.arange(len(piv.index)); w = 0.25
    for i, model in enumerate(piv.columns):
        ax.bar(x + (i - 1) * w, piv[model].values, width=w, label=model, color=COLOR_MODEL[model])
    ax.set_xticks(x); ax.set_xticklabels(piv.index)
    ax.set_title("Detector verbosity: findings per case\n(by judge/detector model)")
    ax.set_ylabel("Mean # findings flagged")
    ax.legend(title="Judge model", fontsize=8, title_fontsize=8)


def panel_alignment_convergence(ax, df):
    labels = ["Reasoning\nalignment", "Answer\nagreement"]
    for method, color in COLOR_METHOD.items():
        sub = df[df["method"] == method]
        r1 = [sub["reasoning_alignment_r1"].mean(), sub["answer_agreement_r1"].mean()]
        r2 = [sub["reasoning_alignment_r2"].mean(), sub["answer_agreement_r2"].mean()]
        xs = np.arange(len(labels))
        for i, (a, b) in enumerate(zip(r1, r2)):
            ax.plot([xs[i], xs[i] + 0.35], [a, b], marker="o", color=color, label=method if i == 0 else None)
    ax.set_xticks(np.arange(len(labels)) + 0.175); ax.set_xticklabels(labels)
    ax.set_title("Round 1 -> Round 2 convergence\n(dot=R1, line end=R2)")
    ax.set_ylabel("Mean score"); ax.set_ylim(0, 1.05); pct(ax)
    ax.legend(fontsize=8)


def panel_correlation(ax, corr):
    piv = corr.pivot(index="metric", columns="method", values="corr_with_error")
    piv = piv.reindex(columns=list(COLOR_METHOD.keys()))
    order = piv.mean(axis=1).sort_values().index
    piv = piv.reindex(order)
    y = np.arange(len(piv.index)); h = 0.35
    for i, method in enumerate(piv.columns):
        ax.barh(y + (i - 0.5) * h, piv[method].values, height=h, label=method, color=COLOR_METHOD[method])
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Correlation of per-case signals\nwith a wrong final answer")
    ax.set_xlabel("Pearson r with is_wrong")
    ax.legend(fontsize=8)


def panel_illusion_sycophancy(ax, df):
    g = df.groupby("method").agg(
        consensus_illusion_rate=("consensus_illusion_flag", "mean"),
        mean_sycophantic_flips=("n_sycophantic_flip_agents", "mean"),
    ).reindex(list(COLOR_METHOD.keys()))
    x = np.arange(len(g.index)); w = 0.35
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, g["consensus_illusion_rate"], width=w, color="#4C72B0", label="Consensus-illusion rate")
    b2 = ax2.bar(x + w / 2, g["mean_sycophantic_flips"], width=w, color="#C44E52", label="Mean sycophantic flips/case")
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    ax.set_ylabel("Consensus-illusion rate", color="#4C72B0")
    ax2.set_ylabel("Mean sycophantic flips / case", color="#C44E52")
    pct(ax)
    ax.set_title("False-consensus & sycophancy signals")
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")


def panel_finding_categories(ax, findings):
    piv = pd.crosstab(findings["model"], findings["category"], normalize="index")
    piv = piv.reindex(index=list(COLOR_MODEL.keys()))
    cat_order = piv.sum().sort_values(ascending=False).index
    piv = piv[cat_order]
    bottom = np.zeros(len(piv)); cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(piv.columns):
        ax.bar(piv.index, piv[cat], bottom=bottom, label=cat, color=cmap(i))
        bottom += piv[cat].values
    ax.set_title("Flagged-finding category mix\nby judge model")
    ax.set_ylabel("Share of findings"); pct(ax)
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")


def panel_finding_severity(ax, findings):
    order = ["low", "medium", "high"]
    piv = pd.crosstab(findings["model"], findings["severity"], normalize="index")
    piv = piv.reindex(index=list(COLOR_MODEL.keys()), columns=[c for c in order if c in piv.columns])
    colors = {"low": "#91BFDB", "medium": "#FEE08B", "high": "#D73027"}
    bottom = np.zeros(len(piv))
    for cat in piv.columns:
        ax.bar(piv.index, piv[cat], bottom=bottom, label=cat, color=colors.get(cat))
        bottom += piv[cat].values
    ax.set_title("Flagged-finding severity mix\nby judge model")
    ax.set_ylabel("Share of findings"); pct(ax)
    ax.legend(fontsize=8, title="severity", title_fontsize=8)


def _corr_heatmap(ax, piv, title, cbar_label, annot_fmt="{:.2f}"):
    """Generic small correlation heatmap: rows=finding type, cols=model."""
    vmax = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1
    vmax = max(vmax, 0.05)
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, annot_fmt.format(v), ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax * 0.6 else "black")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)


def panel_category_corr_heatmap(ax, category_corr_by_model):
    piv = category_corr_by_model.pivot(index="category", columns="model", values="corr_with_error")
    piv = piv.reindex(columns=[m for m in COLOR_MODEL if m in piv.columns])
    order = piv.mean(axis=1).sort_values(ascending=False).index
    piv = piv.reindex(order)
    _corr_heatmap(ax, piv, "Finding CATEGORY count vs. wrong answer\n(Pearson r, per judge model)", "corr with is_wrong")


def panel_severity_corr_heatmap(ax, severity_corr_by_model):
    order = ["low", "medium", "high"]
    piv = severity_corr_by_model.pivot(index="severity", columns="model", values="corr_with_error")
    piv = piv.reindex(index=[s for s in order if s in piv.index],
                       columns=[m for m in COLOR_MODEL if m in piv.columns])
    _corr_heatmap(ax, piv, "Finding SEVERITY count vs. wrong answer\n(Pearson r, per judge model)", "corr with is_wrong")


def panel_category_corr_bars(ax, category_corr_by_model):
    """Same data as the heatmap but as grouped bars, easier to read exact magnitude/direction."""
    piv = category_corr_by_model.pivot(index="category", columns="model", values="corr_with_error")
    piv = piv.reindex(columns=[m for m in COLOR_MODEL if m in piv.columns])
    order = piv.mean(axis=1).sort_values().index
    piv = piv.reindex(order)
    y = np.arange(len(piv.index)); h = 0.25
    for i, model in enumerate(piv.columns):
        ax.barh(y + (i - 1) * h, piv[model].values, height=h, label=model, color=COLOR_MODEL[model])
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Per-category correlation with error,\nby judge model (detail view)")
    ax.set_xlabel("Pearson r with is_wrong")
    ax.legend(fontsize=8)


def panel_finding_outcome_heatmaps(fig, df):
    """Grid of 2x2 heatmaps (correct/wrong x has-finding/no-finding), one per
    model x architecture combination. Cell values = row-normalized share
    (i.e. within each model/architecture, what fraction of cases fall in
    each of the 4 quadrants) with raw counts annotated underneath."""
    df = df.copy()
    df["has_finding"] = np.where(df["n_findings_total"] > 0, "Has finding", "No finding")
    df["outcome"] = np.where(df["is_wrong"], "Wrong", "Correct")

    models = [m for m in COLOR_MODEL if m in df["model"].unique()]
    methods = [m for m in COLOR_METHOD if m in df["method"].unique()]

    nrows, ncols = len(methods), len(models)
    axes = fig.subplots(nrows, ncols, squeeze=False)

    row_order = ["Correct", "Wrong"]
    col_order = ["No finding", "Has finding"]

    for i, method in enumerate(methods):
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub = df[(df["method"] == method) & (df["model"] == model)]
            counts = pd.crosstab(sub["outcome"], sub["has_finding"]).reindex(index=row_order, columns=col_order, fill_value=0)
            share = counts / counts.values.sum()

            im = ax.imshow(share.values, cmap="YlOrRd", vmin=0, vmax=max(0.55, share.values.max()))
            ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order, fontsize=8)
            ax.set_yticks(range(len(row_order))); ax.set_yticklabels(row_order, fontsize=8)
            ax.grid(False)
            for r in range(2):
                for c in range(2):
                    v = share.values[r, c]
                    n = counts.values[r, c]
                    ax.text(c, r, f"{v:.1%}\n(n={n})", ha="center", va="center", fontsize=8,
                             color="white" if v > share.values.max() * 0.6 else "black")
            ax.set_title(f"{model} | {method}", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(method, fontsize=9)

    fig.suptitle(
        "Answer correctness x detector-finding presence, by judge model & architecture\n"
        "(cell = share of that model/architecture's 500 cases; raw n in parentheses)",
        fontsize=12, fontweight="bold",
    )


def panel_finding_rate_by_outcome_bars(ax, df):
    """Simpler companion view: P(has finding) for correct vs wrong cases,
    grouped by model, faceted by architecture via color hatching per method."""
    df = df.copy()
    df["has_finding"] = df["n_findings_total"] > 0
    df["outcome"] = np.where(df["is_wrong"], "Wrong", "Correct")

    piv = df.groupby(["model", "method", "outcome"])["has_finding"].mean().reset_index()
    models = [m for m in COLOR_MODEL if m in df["model"].unique()]
    methods = [m for m in COLOR_METHOD if m in df["method"].unique()]

    x = np.arange(len(models))
    w = 0.2
    offsets = {("Role-Specialist Board", "Correct"): -1.5, ("Role-Specialist Board", "Wrong"): -0.5,
               ("Symmetric Debate", "Correct"): 0.5, ("Symmetric Debate", "Wrong"): 1.5}
    hatch = {"Correct": "", "Wrong": "//"}
    for method in methods:
        for outcome in ["Correct", "Wrong"]:
            vals = []
            for model in models:
                r = piv[(piv["model"] == model) & (piv["method"] == method) & (piv["outcome"] == outcome)]
                vals.append(r["has_finding"].values[0] if len(r) else 0)
            off = offsets[(method, outcome)]
            ax.bar(x + off * w, vals, width=w, color=COLOR_METHOD[method], hatch=hatch[outcome],
                   edgecolor="white", label=f"{method} - {outcome}")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("P(at least one finding flagged)")
    pct(ax)
    ax.set_title("Finding-flag rate: correct vs. wrong cases\nby model & architecture")
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")


PANELS = [
    ("error_rate_by_architecture", panel_error_rate, ("df",)),
    ("hallucination_rate_by_judge_model", panel_hallucination_by_model, ("df",)),
    ("findings_per_case_by_judge_model", panel_findings_per_case, ("df",)),
    ("alignment_convergence", panel_alignment_convergence, ("df",)),
    ("correlation_with_error", panel_correlation, ("corr",)),
    ("illusion_and_sycophancy", panel_illusion_sycophancy, ("df",)),
    ("finding_category_mix", panel_finding_categories, ("findings",)),
    ("finding_severity_mix", panel_finding_severity, ("findings",)),
    ("category_corr_heatmap", panel_category_corr_heatmap, ("category_corr",)),
    ("severity_corr_heatmap", panel_severity_corr_heatmap, ("severity_corr",)),
    ("category_corr_bars", panel_category_corr_bars, ("category_corr",)),
]


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    df, findings_df = load_records()
    print(f"Loaded {len(df)} case records from {len(FILES)} files")
    print(df["file"].value_counts().to_string())

    summary, illusion_vs_error, corr_df, category_corr_by_model, severity_corr_by_model = build_summaries(df, findings_df)

    df.to_csv(os.path.join(OUT_DIR, "cases_tidy.csv"), index=False)
    findings_df.to_csv(os.path.join(OUT_DIR, "findings_tidy.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary_by_model_method.csv"), index=False)
    illusion_vs_error.to_csv(os.path.join(OUT_DIR, "illusion_vs_error.csv"), index=False)
    corr_df.to_csv(os.path.join(OUT_DIR, "corr_with_error.csv"), index=False)
    category_corr_by_model.to_csv(os.path.join(OUT_DIR, "category_corr_by_model.csv"), index=False)
    severity_corr_by_model.to_csv(os.path.join(OUT_DIR, "severity_corr_by_model.csv"), index=False)

    print("\n=== Summary by model x method ===")
    print(summary.to_string(index=False))
    print("\n=== Finding CATEGORY count correlation with is_wrong, by judge model ===")
    print(category_corr_by_model.sort_values(["model", "corr_with_error"], ascending=[True, False]).to_string(index=False))
    print("\n=== Finding SEVERITY count correlation with is_wrong, by judge model ===")
    print(severity_corr_by_model.sort_values(["model", "corr_with_error"], ascending=[True, False]).to_string(index=False))

    data = {
        "df": df, "findings": findings_df, "summary": summary, "illusion": illusion_vs_error,
        "corr": corr_df, "category_corr": category_corr_by_model, "severity_corr": severity_corr_by_model,
    }

    # combined dashboard (11 panels -> 3x4 grid, last cell blank)
    n = len(PANELS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.2 * nrows))
    fig.suptitle(
        "Multi-Agent Debate Detector Traces — Error, Hallucination & Consensus Analysis\n"
        "(500 QA cases x 2 architectures x 3 judge/detector models = 3,000 case-traces)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes_flat = axes.flat
    for ax, (name, fn, needs) in zip(axes_flat, PANELS):
        fn(ax, *[data[k] for k in needs])
    for ax in list(axes_flat):  # hide any unused trailing axes
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dashboard.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # standalone panels
    for name, fn, needs in PANELS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fn(ax, *[data[k] for k in needs])
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, f"{name}.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    # standalone: correctness x finding-presence heatmap grid (own figure, own size)
    n_models = df["model"].nunique()
    n_methods = df["method"].nunique()
    fig = plt.figure(figsize=(3.6 * n_models + 1, 3.6 * n_methods + 1))
    panel_finding_outcome_heatmaps(fig, df)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(CHART_DIR, "correctness_vs_finding_heatmap.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # standalone: companion bar chart
    fig, ax = plt.subplots(figsize=(8, 5.5))
    panel_finding_rate_by_outcome_bars(ax, df)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "finding_rate_by_outcome_bars.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote dashboard.png and {len(PANELS)} standalone charts to {CHART_DIR}")
    print("Wrote correctness_vs_finding_heatmap.png and finding_rate_by_outcome_bars.png")


if __name__ == "__main__":
    main()