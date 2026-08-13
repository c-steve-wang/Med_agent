"""
full_report.py
===============
Single-file pipeline: loads every detector_traces_qa_*.jsonl /
improved_detector_traces_qa_*.jsonl trace file sitting next to this script
and produces the full set of requested figures.

KEY DESIGN CHOICE (per request): the "improved" detector run for a given
judge model is NOT merged/hatched into the same bar as the original — it is
given its own distinct label and color, e.g. "Improved GPT-5.6 Luna Pro" /
"Improved Qwen 3.6 Flash", so it shows up as a first-class separate series
in every legend and every model axis.

Usage:
    python3 full_report.py
Figures land in ./figures/run_<timestamp>/

Requires: pandas, numpy, matplotlib, scikit-learn
"""
import glob
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

# ===========================================================================
# CONFIG
# ===========================================================================

# Base display name per model code (used for the ORIGINAL detector run).
BASE_MODEL_RENAME = {
    "openai5.6":  "GPT-5.6 Luna Pro",
    "deepseek":   "DeepSeek V4 Flash 0423",
    "qwen4.6":    "Qwen 3.6 Flash",
}

ARCH_SLUG_TO_NAME = {
    "specialized_board": "Role-Specialist Board",
    "symmetric_debate":  "Symmetric Debate",
}
ARCHITECTURES = ["Role-Specialist Board", "Symmetric Debate"]
ARCH_COLORS = {"Role-Specialist Board": "#4c72b0", "Symmetric Debate": "#dd8452"}

FINDING_CATEGORIES = ["other", "hallucination", "error_propagation", "contradiction", "sycophancy"]
CATEGORY_COLORS = {
    "other": "#1f77b4", "hallucination": "#ff7f0e", "error_propagation": "#2ca02c",
    "contradiction": "#d62728", "sycophancy": "#9467bd",
}
SEVERITIES = ["low", "medium", "high"]
SEVERITY_COLORS = {"low": "#9ecae1", "medium": "#fee391", "high": "#e34a33"}

# Colors for base (original-detector) models
BASE_MODEL_COLORS = {
    "Qwen 3.6 Flash":           "#9467bd",
    "DeepSeek V4 Flash 0423":   "#d62728",
    "GPT-5.6 Luna Pro":         "#17becf",
}
# Lighter/distinct tint used for the "Improved <model>" series
IMPROVED_MODEL_COLORS = {
    "Improved Qwen 3.6 Flash":         "#c5b0d5",
    "Improved DeepSeek V4 Flash 0423": "#ff9896",
    "Improved GPT-5.6 Luna Pro":       "#9edae5",
}

MODEL_ORDER = [
    "Qwen 3.6 Flash", "Improved Qwen 3.6 Flash",
    "DeepSeek V4 Flash 0423", "Improved DeepSeek V4 Flash 0423",
    "GPT-5.6 Luna Pro", "Improved GPT-5.6 Luna Pro",
]
MODEL_COLORS = {**BASE_MODEL_COLORS, **IMPROVED_MODEL_COLORS}

GLOB_DIR = os.environ.get("DETECTOR_TRACES_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(GLOB_DIR, "figures")

VERDICT_ORDER = ["PASS", "NEEDS_REVIEW", "LIKELY_WRONG"]
VERDICT_COLORS = {"PASS": "#2ca02c", "NEEDS_REVIEW": "#eda100", "LIKELY_WRONG": "#d62728"}

MANUAL_REVIEW_FILE = "__Manual_Detector-Finding_Review__.txt"


def _ordered(models_present):
    return [m for m in MODEL_ORDER if m in models_present]


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ===========================================================================
# LOAD DATA
# ===========================================================================
_ARCH_ALT = "|".join(sorted((re.escape(k) for k in ARCH_SLUG_TO_NAME), key=len, reverse=True))
_MODEL_ALT = "|".join(sorted((re.escape(k) for k in BASE_MODEL_RENAME), key=len, reverse=True))
FNAME_RE = re.compile(
    rf"^(?P<improved>improved_)?detector_traces_qa_(?P<arch>{_ARCH_ALT})_(?P<model>{_MODEL_ALT})\.jsonl$"
)


def _discover_files(directory):
    found = []
    for path in glob.glob(os.path.join(directory, "*.jsonl")):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        method = "improved" if m.group("improved") else "original"
        found.append((path, ARCH_SLUG_TO_NAME[m.group("arch")], m.group("model"), method))
    return found


def _display_model(model_code, method):
    base = BASE_MODEL_RENAME[model_code]
    return f"Improved {base}" if method == "improved" else base


def _load_one_file(path, architecture, model_code, method):
    model_name = _display_model(model_code, method)
    case_rows, finding_rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            all_findings = list(d.get("llm_findings") or []) + list(d.get("other_findings") or [])
            n_findings = len(all_findings)
            case_rows.append({
                "case_id": d["case_id"], "architecture": architecture,
                "model": model_name, "base_model": BASE_MODEL_RENAME[model_code],
                "method": method,
                "gold_label": d.get("gold_label"), "aggregated_answer": d.get("aggregated_answer"),
                "is_wrong": bool(d.get("is_wrong")),
                "hallucinated_evidence_rate": d.get("hallucinated_evidence_rate") or 0.0,
                "reasoning_alignment_r1": d.get("reasoning_alignment_r1"),
                "reasoning_alignment_r2": d.get("reasoning_alignment_r2"),
                "answer_agreement_r1": d.get("answer_agreement_r1"),
                "answer_agreement_r2": d.get("answer_agreement_r2"),
                "consensus_illusion_flag": bool(d.get("consensus_illusion_flag")),
                "n_sycophantic_flip_agents": len(d.get("sycophantic_flip_agents") or []),
                "n_findings_total": n_findings,
                "has_finding": n_findings > 0,
                "detector_risk_score": d.get("detector_risk_score"),
                "detector_verdict": d.get("detector_verdict"),
                "would_escalate": d.get("would_escalate"),
            })
            for finding in all_findings:
                finding_rows.append({
                    "case_id": d["case_id"], "architecture": architecture,
                    "model": model_name, "base_model": BASE_MODEL_RENAME[model_code],
                    "method": method, "is_wrong": bool(d.get("is_wrong")),
                    "category": finding.get("category", "other"),
                    "severity": finding.get("severity", "medium"),
                    "agent_id": finding.get("agent_id"), "round": finding.get("round"),
                })
    cases_df = pd.DataFrame(case_rows)
    cases_df["reasoning_alignment_delta"] = cases_df["reasoning_alignment_r2"] - cases_df["reasoning_alignment_r1"]
    cases_df["answer_agreement_delta"] = cases_df["answer_agreement_r2"] - cases_df["answer_agreement_r1"]
    findings_df = pd.DataFrame(finding_rows)
    return cases_df, findings_df


def load_all(directory=GLOB_DIR):
    files = _discover_files(directory)
    if not files:
        all_jsonl = glob.glob(os.path.join(directory, "*.jsonl"))
        raise FileNotFoundError(
            f"No matching trace files found in {os.path.abspath(directory)}.\n"
            f".jsonl files present: {[os.path.basename(p) for p in all_jsonl] or 'none'}\n"
            f"Known model codes: {list(BASE_MODEL_RENAME)}\n"
            f"Known architecture slugs: {list(ARCH_SLUG_TO_NAME)}"
        )
    c_parts, f_parts = [], []
    for path, arch, model_code, method in files:
        c, fnd = _load_one_file(path, arch, model_code, method)
        c_parts.append(c)
        f_parts.append(fnd)
    cases_df = pd.concat(c_parts, ignore_index=True)
    findings_df = pd.concat(f_parts, ignore_index=True) if f_parts else pd.DataFrame(
        columns=["case_id", "architecture", "model", "base_model", "method", "is_wrong", "category", "severity"]
    )
    return cases_df, findings_df


# Maps the short model labels used inside the manual-review txt file to our
# BASE_MODEL_RENAME display names.
MANUAL_REVIEW_MODEL_MAP = {
    "OpenAI": "GPT-5o Mini",
    "OpenAI-5.6": "GPT-5.6 Luna Pro",
    "DeepSeek": "DeepSeek V4 Flash 0423",
    "Qwen": "Qwen 3.6 Flash",
}


def load_manual_review(directory=GLOB_DIR):
    """
    Parses __Manual_Detector-Finding_Review__.txt for per-finding:
      - model (judge model that produced the finding)
      - VERIFICATION (mechanical: VERIFIED / UNVERIFIED)
      - RATING (human judgment: ACCURATE / LIKELY ACCURATE / PARTIALLY ACCURATE / INACCURATE)
    Returns a tidy DataFrame, or an empty one (with a printed note) if the file
    isn't present -- every downstream figure that depends on it degrades
    gracefully rather than crashing.
    """
    path = os.path.join(directory, MANUAL_REVIEW_FILE)
    cols = ["model", "bucket", "category", "severity", "verif", "score", "rating"]
    if not os.path.exists(path):
        print(f"[manual review] {MANUAL_REVIEW_FILE} not found in {directory} -- "
              f"manual-review figures will be skipped.")
        return pd.DataFrame(columns=cols)

    text = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    case_re = re.compile(r"### Case \d+ .{1,3} (\S+) / .+? .{1,3} bucket: (\S+)")
    cases = [(m.start(), m.group(1), m.group(2)) for m in case_re.finditer(text)]
    cases.append((len(text), None, None))

    finding_re = re.compile(
        r"\*\*Finding \d+\*\*.*?category=`(?P<cat>[^`]+)`, severity=`(?P<sev>[^`]+)`.*?\n"
        r".*?VERIFICATION \(mechanical\): (?P<verif>VERIFIED|UNVERIFIED)[^\n]*score=(?P<score>[\d.]+)\)\s*\n"
        r"- \[x\] RATING: \*\*(?P<rating>[A-Z ]+)\*\*",
        re.S,
    )

    rows = []
    for i in range(len(cases) - 1):
        start, model, bucket = cases[i]
        end = cases[i + 1][0]
        chunk = text[start:end]
        for fm in finding_re.finditer(chunk):
            d = fm.groupdict()
            rows.append({
                "model": model, "bucket": bucket,
                "category": d["cat"], "severity": d["sev"],
                "verif": d["verif"], "score": float(d["score"]),
                "rating": d["rating"].strip(),
            })
    df = pd.DataFrame(rows, columns=cols)
    df["model"] = df["model"].map(MANUAL_REVIEW_MODEL_MAP).fillna(df["model"])
    print(f"[manual review] parsed {len(df)} rated findings from {MANUAL_REVIEW_FILE}")
    return df


# ===========================================================================
# METRICS
# ===========================================================================
def error_rate_by_architecture(cases_df):
    # original-detector rows only (is_wrong is a property of the underlying
    # transcript/answer, identical for original & improved on the same case,
    # so using method=="original" avoids double counting)
    sub = cases_df[cases_df["method"] == "original"]
    return sub.groupby("architecture")["is_wrong"].mean().reindex(
        [a for a in ARCHITECTURES if a in sub["architecture"].unique()]
    )


def category_correlation_with_error(cases_df, findings_df):
    cat_counts = (
        findings_df.groupby(["model", "case_id", "architecture", "method", "category"])
        .size().unstack(fill_value=0).reindex(columns=FINDING_CATEGORIES, fill_value=0).reset_index()
    )
    merged = cat_counts.merge(
        cases_df[["model", "case_id", "architecture", "method", "is_wrong"]],
        on=["model", "case_id", "architecture", "method"], how="right"
    ).fillna(0)
    out = {}
    for model, sub in merged.groupby("model"):
        row = {}
        for cat in FINDING_CATEGORIES:
            row[cat] = np.corrcoef(sub[cat], sub["is_wrong"])[0, 1] if sub[cat].std() > 0 else np.nan
        out[model] = row
    return pd.DataFrame(out).T[FINDING_CATEGORIES]


def severity_correlation_with_error(cases_df, findings_df):
    sev_counts = (
        findings_df.groupby(["model", "case_id", "architecture", "method", "severity"])
        .size().unstack(fill_value=0).reindex(columns=SEVERITIES, fill_value=0).reset_index()
    )
    merged = sev_counts.merge(
        cases_df[["model", "case_id", "architecture", "method", "is_wrong"]],
        on=["model", "case_id", "architecture", "method"], how="right"
    ).fillna(0)
    out = {}
    for model, sub in merged.groupby("model"):
        row = {}
        for sev in SEVERITIES:
            row[sev] = np.corrcoef(sub[sev], sub["is_wrong"])[0, 1] if sub[sev].std() > 0 else np.nan
        out[model] = row
    return pd.DataFrame(out).T[SEVERITIES]


def category_auroc(cases_df, findings_df):
    """
    AUROC of each category's per-case count predicting is_wrong, per model.
    For rows that carry the improved detector's `detector_verdict`, two extra
    pseudo-categories are appended: whether the verdict was NEEDS_REVIEW or
    worse ("verdict != PASS"), and whether it was LIKELY_WRONG specifically --
    so the category heatmap directly shows how much signal a non-PASS verdict
    itself carries, next to the finding categories.
    """
    cat_counts = (
        findings_df.groupby(["model", "case_id", "architecture", "method", "category"])
        .size().unstack(fill_value=0).reindex(columns=FINDING_CATEGORIES, fill_value=0).reset_index()
    )
    merged = cat_counts.merge(
        cases_df[["model", "case_id", "architecture", "method", "is_wrong", "detector_verdict"]],
        on=["model", "case_id", "architecture", "method"], how="right"
    ).fillna({c: 0 for c in FINDING_CATEGORIES})
    merged["verdict_not_pass"] = (merged["detector_verdict"].notna() & (merged["detector_verdict"] != "PASS")).astype(int)
    merged["verdict_likely_wrong"] = (merged["detector_verdict"] == "LIKELY_WRONG").astype(int)

    all_cols = FINDING_CATEGORIES + ["verdict_not_pass", "verdict_likely_wrong"]
    out = {}
    for model, sub in merged.groupby("model"):
        has_verdict = sub["detector_verdict"].notna().any()
        row = {}
        for cat in FINDING_CATEGORIES:
            row[cat] = roc_auc_score(sub["is_wrong"], sub[cat]) if sub["is_wrong"].nunique() > 1 else np.nan
        for cat in ["verdict_not_pass", "verdict_likely_wrong"]:
            if has_verdict and sub["is_wrong"].nunique() > 1 and sub[cat].nunique() > 1:
                row[cat] = roc_auc_score(sub["is_wrong"], sub[cat])
            else:
                row[cat] = np.nan  # not applicable for original-detector models
        out[model] = row
    return pd.DataFrame(out).T[all_cols]


def correctness_x_finding_presence(cases_df):
    out = {}
    for (model, arch, method), sub in cases_df.groupby(["model", "architecture", "method"]):
        total = len(sub)
        tab = pd.crosstab(sub["is_wrong"], sub["has_finding"]).reindex(
            index=[False, True], columns=[False, True], fill_value=0
        )
        out[(model, arch, method)] = (tab / total, tab)
    return out


def per_case_signal_correlations(cases_df):
    signals = [
        "n_sycophantic_flip_agents", "answer_agreement_delta", "n_findings_total",
        "reasoning_alignment_delta", "hallucinated_evidence_rate",
        "reasoning_alignment_r2", "answer_agreement_r2",
        "reasoning_alignment_r1", "answer_agreement_r1",
    ]
    out = {}
    for arch, sub in cases_df.groupby("architecture"):
        row = {}
        for s in signals:
            row[s] = np.corrcoef(sub[s], sub["is_wrong"])[0, 1] if sub[s].std() > 0 else np.nan
        out[arch] = row
    present = [a for a in ARCHITECTURES if a in out]
    return pd.DataFrame(out)[present].T[signals].T


def pct_wrong_with_finding(cases_df):
    """Of the cases that are actually wrong, what % had >=1 LLM-identified finding?"""
    wrong = cases_df[cases_df["is_wrong"]]
    return wrong.groupby(["model", "architecture", "method"])["has_finding"].mean()


def verdict_vs_correctness(cases_df):
    sub = cases_df[cases_df["detector_verdict"].notna()]
    return sub.groupby(["model", "architecture", "detector_verdict"])["is_wrong"].agg(["mean", "count"]).reset_index()


def judge_agreement_matrix(cases_df):
    """
    Pairwise agreement between judge models' *finding-flag* decisions
    (has_finding) on shared case_ids, restricted to the original detector run
    so every model is evaluated on the same underlying transcripts.
    Returns (raw agreement matrix, Cohen's kappa matrix), both indexed/cols by base_model.
    """
    sub = cases_df[cases_df["method"] == "original"]
    pivot = sub.pivot_table(index=["architecture", "case_id"], columns="base_model", values="has_finding", aggfunc="first")
    models = [m for m in pivot.columns]
    agree = pd.DataFrame(index=models, columns=models, dtype=float)
    kappa = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                agree.loc[m1, m2] = 1.0
                kappa.loc[m1, m2] = 1.0
                continue
            pair = pivot[[m1, m2]].dropna()
            if len(pair) == 0:
                continue
            v1 = pair[m1].astype(float).values
            v2 = pair[m2].astype(float).values
            agree.loc[m1, m2] = float((v1 == v2).mean())
            if len(set(v1.tolist())) > 1 or len(set(v2.tolist())) > 1:
                kappa.loc[m1, m2] = float(cohen_kappa_score(v1, v2))
            else:
                kappa.loc[m1, m2] = 1.0 if (v1 == v2).all() else np.nan
    return agree, kappa


def scoring_method_auroc(cases_df, findings_df):
    """AUROC of raw finding count per model (used by category_auroc-style summary)."""
    rows = []
    for (model, arch, method), sub in cases_df.groupby(["model", "architecture", "method"]):
        if sub["is_wrong"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(sub["is_wrong"], sub["n_findings_total"])
        rows.append({"model": model, "architecture": arch, "method": method, "auroc": auc})
    return pd.DataFrame(rows)


# ===========================================================================
# FIGURES
# ===========================================================================

def fig_agreement_and_error_source(cases_df):
    """
    Composite: answer_agreement_r2 (final-round consensus) vs error rate,
    broken out by whether the case had a detector finding at all -- i.e. is
    disagreement or agreement the bigger source of the final error?
    """
    sub = cases_df[cases_df["method"] == "original"].copy()
    sub["agreement_bin"] = pd.cut(
        sub["answer_agreement_r2"], bins=[0, 0.5, 0.8, 0.95, 1.01],
        labels=["<0.5 (low)", "0.5-0.8", "0.8-0.95", ">0.95 (near-unanimous)"]
    )
    grp = sub.groupby(["agreement_bin", "has_finding"], observed=True)["is_wrong"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = np.arange(len(grp.index))
    width = 0.35
    for i, col in enumerate([False, True]):
        if col not in grp.columns:
            continue
        ax.bar(x + (i - 0.5) * width, grp[col].values, width=width,
               label="No finding" if not col else "Has finding",
               color="#4c72b0" if not col else "#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=15, ha="right")
    ax.set_ylabel("Error rate (share wrong)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Agreement level vs. error source:\ndoes low consensus or a flagged finding drive errors?", fontweight="bold")
    ax.legend()
    _savefig(fig, "agreement_and_error_source")


def _binned_line(cases_df, x_col):
    out = {}
    sub_all = cases_df[cases_df["method"] == "original"]
    for arch, sub in sub_all.groupby("architecture"):
        for round_label, xcol in [("R1", f"{x_col}_r1"), ("R2", f"{x_col}_r2")]:
            d = sub[[xcol, "is_wrong"]].dropna()
            if len(d) < 8:
                continue
            d["bin"] = pd.qcut(d[xcol], q=min(8, d[xcol].nunique()), duplicates="drop")
            g = d.groupby("bin", observed=True).agg(x=(xcol, "mean"), y=("is_wrong", "mean")).sort_values("x")
            out[(arch, round_label)] = g
    return out


def fig_binned(cases_df, x_col, title, xlabel, fname):
    lines = _binned_line(cases_df, x_col)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for arch in ARCHITECTURES:
        for round_label, ls in [("R1", "solid"), ("R2", "dashed")]:
            key = (arch, round_label)
            if key not in lines:
                continue
            g = lines[key]
            ax.plot(g["x"], g["y"], marker="o", linestyle=ls, color=ARCH_COLORS[arch], label=f"{arch} - {round_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error rate (share wrong)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    _savefig(fig, fname)


def fig_box(cases_df, x_col, title, ylabel, fname):
    sub_all = cases_df[cases_df["method"] == "original"]
    fig, ax = plt.subplots(figsize=(10, 7))
    positions, data, labels, colors, hatches = [], [], [], [], []
    pos = 0
    for arch in ARCHITECTURES:
        sub_arch = sub_all[sub_all["architecture"] == arch]
        for round_label, rcol in [("R1", f"{x_col}_r1"), ("R2", f"{x_col}_r2")]:
            for correct_label, wrong_val in [("Correct", False), ("Wrong", True)]:
                vals = sub_arch[sub_arch["is_wrong"] == wrong_val][rcol].dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals); labels.append(f"{round_label}\n{correct_label}")
                positions.append(pos); colors.append(ARCH_COLORS[arch])
                hatches.append("//" if wrong_val else None)
                pos += 1
        pos += 1
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.7)
    for patch, c, h in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(c); patch.set_alpha(0.85)
        if h:
            patch.set_hatch(h)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    _savefig(fig, fname)


def fig_category_auroc_heatmap(cases_df, findings_df):
    auc = category_auroc(cases_df, findings_df)
    row_labels = FINDING_CATEGORIES + ["verdict != PASS\n(improved only)", "verdict = LIKELY_WRONG\n(improved only)"]
    models = _ordered(auc.index)
    auc = auc.loc[models]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(row_labels) + 2))
    im = ax.imshow(auc.T.values, cmap="RdYlGn", vmin=0.4, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j, model in enumerate(models):
            v = auc.iloc[j, i]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC vs is_wrong")
    ax.set_title(
        "Category AUROC heatmap\n(finding categories + non-PASS verdict signals, per model incl. improved)",
        fontweight="bold"
    )
    _savefig(fig, "category_auroc_heatmap")


def fig_category_corr(cases_df, findings_df):
    corr = category_correlation_with_error(cases_df, findings_df)
    models = _ordered(corr.index)
    corr = corr.loc[models]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(models) + 2))
    im = ax.imshow(corr.T.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(FINDING_CATEGORIES))); ax.set_yticklabels(FINDING_CATEGORIES)
    for i, cat in enumerate(FINDING_CATEGORIES):
        for j, model in enumerate(models):
            ax.text(j, i, f"{corr.loc[model, cat]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="corr with is_wrong")
    ax.set_title("Finding CATEGORY count vs. wrong answer\n(Pearson r, per judge model incl. improved)", fontweight="bold")
    _savefig(fig, "category_corr_heatmap")

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(models) + 3))
    y = np.arange(len(FINDING_CATEGORIES))
    height = 0.8 / len(models)
    for i, model in enumerate(models):
        offs = (i - (len(models) - 1) / 2) * height
        ax.barh(y + offs, corr.loc[model, FINDING_CATEGORIES].values, height=height,
                color=MODEL_COLORS[model], label=model)
    ax.set_yticks(y); ax.set_yticklabels(FINDING_CATEGORIES)
    ax.set_xlabel("Pearson r with is_wrong")
    ax.set_title("Per-category correlation with error,\nby judge model (detail view, incl. improved)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, "category_corr_bars")


def fig_severity_corr(cases_df, findings_df):
    corr = severity_correlation_with_error(cases_df, findings_df)
    models = _ordered(corr.index)
    corr = corr.loc[models]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(models) + 2))
    im = ax.imshow(corr.T.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(SEVERITIES))); ax.set_yticklabels(SEVERITIES)
    for i, sev in enumerate(SEVERITIES):
        for j, model in enumerate(models):
            ax.text(j, i, f"{corr.loc[model, sev]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="corr with is_wrong")
    ax.set_title("Finding SEVERITY count vs. wrong answer\n(Pearson r, per judge model incl. improved)", fontweight="bold")
    _savefig(fig, "severity_corr_heatmap")


def fig_correctness_vs_finding(cases_df):
    cells = correctness_x_finding_presence(cases_df)
    models = _ordered(cases_df["model"].unique())
    methods = sorted(cases_df["method"].unique())
    combos = [(m, a, meth) for m in models for a in ARCHITECTURES for meth in methods if (m, a, meth) in cells]
    n = len(combos)
    ncols = min(4, n) if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("Answer correctness x detector-finding presence\n(cell = share of that combo's cases; raw n in parentheses)",
                 fontweight="bold", fontsize=13)
    for ax, (model, arch, method) in zip(axes, combos):
        share, raw = cells[(model, arch, method)]
        grid = share.values
        vmax = grid.max() * 1.3 if grid.max() else 1
        ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{grid[i, j]:.1%}\n(n={raw.values[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > vmax * 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No finding", "Has finding"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Correct", "Wrong"])
        ax.set_title(f"{model}\n{arch}", fontsize=9, fontweight="bold")
    for ax in axes[len(combos):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "correctness_vs_finding_heatmap")


def fig_correlation_with_error(cases_df):
    corr = per_case_signal_correlations(cases_df)
    signals = corr.index.tolist()
    present_archs = [a for a in ARCHITECTURES if a in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(signals))
    height = 0.8 / max(1, len(present_archs))
    for i, arch in enumerate(present_archs):
        offs = (i - (len(present_archs) - 1) / 2) * height
        ax.barh(y + offs, corr[arch].values, height=height, color=ARCH_COLORS[arch], label=arch)
    ax.set_yticks(y); ax.set_yticklabels(signals)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with is_wrong")
    ax.set_title("Correlation of per-case signals\nwith a wrong final answer", fontweight="bold")
    ax.legend(fontsize=9)
    _savefig(fig, "correlation_with_error")


def fig_detector_analysis(cases_df):
    """
    Overview panel: mean detector_risk_score by verdict bucket, and
    would_escalate rate vs actual error rate -- only meaningful for rows
    that have the improved-detector fields.
    """
    sub = cases_df[cases_df["detector_verdict"].notna()].copy()
    if len(sub) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: risk score distribution by verdict
    ax = axes[0]
    data = [sub[sub["detector_verdict"] == v]["detector_risk_score"].dropna().values for v in VERDICT_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, v in zip(bp["boxes"], VERDICT_ORDER):
        patch.set_facecolor(VERDICT_COLORS[v])
    ax.set_xticklabels(VERDICT_ORDER)
    ax.set_ylabel("detector_risk_score")
    ax.set_title("Risk score by verdict", fontweight="bold")

    # Panel 2: would_escalate rate vs actual error rate, per model
    ax = axes[1]
    models = _ordered(sub["model"].unique())
    esc_rate = sub.groupby("model")["would_escalate"].mean().reindex(models)
    err_rate = sub.groupby("model")["is_wrong"].mean().reindex(models)
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, esc_rate.values, width=w, label="would_escalate rate", color="#8172b2")
    ax.bar(x + w / 2, err_rate.values, width=w, label="Actual error rate", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Escalation rate vs. actual error rate\n(improved detector)", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Detector analysis (improved detector fields)", fontweight="bold", fontsize=14, y=1.03)
    _savefig(fig, "detector_analysis")


def fig_detector_diagnostics_extra(cases_df):
    """Extra diagnostics: mean risk score & escalation rate broken out by architecture, per model."""
    sub = cases_df[cases_df["detector_verdict"].notna()].copy()
    if len(sub) == 0:
        return
    models = _ordered(sub["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    piv = sub.groupby(["model", "architecture"])["detector_risk_score"].mean().unstack().reindex(models)
    piv[[a for a in ARCHITECTURES if a in piv.columns]].plot(
        kind="bar", ax=ax, color=[ARCH_COLORS[a] for a in ARCHITECTURES if a in piv.columns], width=0.7
    )
    ax.set_ylabel("Mean detector_risk_score")
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_title("Mean risk score by architecture", fontweight="bold")

    ax = axes[1]
    piv2 = sub.groupby(["model", "architecture"])["would_escalate"].mean().unstack().reindex(models)
    piv2[[a for a in ARCHITECTURES if a in piv2.columns]].plot(
        kind="bar", ax=ax, color=[ARCH_COLORS[a] for a in ARCHITECTURES if a in piv2.columns], width=0.7
    )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_title("Escalation rate by architecture", fontweight="bold")

    fig.suptitle("Detector diagnostics: extra breakdowns", fontweight="bold", fontsize=14, y=1.03)
    _savefig(fig, "detector_diagnostics_extra")


def fig_would_escalate_heatmap(cases_df):
    """
    2x2 heatmap per improved-detector model: is_wrong x would_escalate,
    same style as correctness_vs_finding_heatmap but using the improved
    detector's would_escalate flag instead of raw finding presence.
    """
    sub = cases_df[cases_df["detector_verdict"].notna()].copy()
    if len(sub) == 0:
        return
    sub["would_escalate"] = sub["would_escalate"].astype(bool)
    models = _ordered(sub["model"].unique())
    combos = [(m, a) for m in models for a in ARCHITECTURES if len(sub[(sub["model"] == m) & (sub["architecture"] == a)]) > 0]
    n = len(combos)
    ncols = min(4, n) if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        "Would-escalate heatmap (improved detector)\nAnswer correctness x escalation decision "
        "(cell = share of that model/architecture's cases; raw n in parentheses)",
        fontweight="bold", fontsize=13,
    )
    for ax, (model, arch) in zip(axes, combos):
        s = sub[(sub["model"] == model) & (sub["architecture"] == arch)]
        total = len(s)
        tab = pd.crosstab(s["is_wrong"], s["would_escalate"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
        share = tab / total
        grid = share.values
        vmax = grid.max() * 1.3 if grid.max() else 1
        ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{grid[i, j]:.1%}\n(n={tab.values[i, j]})", ha="center", va="center", fontsize=9,
                        color="white" if grid[i, j] > vmax * 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No escalate", "Escalate"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Correct", "Wrong"])
        ax.set_title(f"{model}\n{arch}", fontsize=9, fontweight="bold")
    for ax in axes[len(combos):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "would_escalate_heatmap")


def fig_judge_verdict_relationship(cases_df):
    """
    Expanded look at how detector_verdict relates to is_wrong and to the
    AUROC of that relationship, for the improved detector:
      panel 1 - verdict distribution split by actual correctness (normalized within is_wrong)
      panel 2 - AUROC of the ordinal verdict score vs is_wrong, per model x architecture
      panel 3 - error rate within each verdict bucket (calibration curve), per model
    """
    sub = cases_df[cases_df["detector_verdict"].notna()].copy()
    if len(sub) == 0:
        return
    order = {"PASS": 0, "NEEDS_REVIEW": 1, "LIKELY_WRONG": 2}
    sub["_verdict_score"] = sub["detector_verdict"].map(order)
    models = _ordered(sub["model"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel 1: verdict share within correct vs wrong cases (pooled across model)
    ax = axes[0]
    dist = sub.groupby(["is_wrong", "detector_verdict"]).size().unstack(fill_value=0)
    dist = dist.reindex(columns=VERDICT_ORDER, fill_value=0)
    dist_share = dist.div(dist.sum(axis=1), axis=0)
    bottom = np.zeros(2)
    xlabels = ["Correct", "Wrong"]
    for v in VERDICT_ORDER:
        vals = dist_share[v].reindex([False, True]).values
        ax.bar(xlabels, vals, bottom=bottom, color=VERDICT_COLORS[v], label=v, width=0.6)
        bottom += vals
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_ylabel("Share of cases")
    ax.set_title("Verdict distribution,\ncorrect vs. wrong cases (pooled)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: AUROC of ordinal verdict, per model x architecture
    ax = axes[1]
    rows = []
    for (model, arch), g in sub.groupby(["model", "architecture"]):
        auc = roc_auc_score(g["is_wrong"], g["_verdict_score"]) if g["is_wrong"].nunique() > 1 else np.nan
        rows.append({"model": model, "architecture": arch, "auroc": auc})
    auc_df = pd.DataFrame(rows)
    width = 0.8 / len(ARCHITECTURES)
    x = np.arange(len(models))
    for i, arch in enumerate(ARCHITECTURES):
        a = auc_df[auc_df["architecture"] == arch].set_index("model").reindex(models)["auroc"]
        offset = (i - (len(ARCHITECTURES) - 1) / 2) * width
        ax.bar(x + offset, a.values, width=width, color=ARCH_COLORS[arch], label=arch)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC (ordinal verdict vs is_wrong)")
    ax.set_title("Verdict AUROC by model\n& architecture", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: calibration -- error rate within each verdict bucket, per model
    ax = axes[2]
    cal = sub.groupby(["model", "detector_verdict"])["is_wrong"].mean().unstack().reindex(columns=VERDICT_ORDER)
    cal = cal.reindex(models)
    xw = np.arange(len(VERDICT_ORDER))
    width2 = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = cal.loc[model].values
        offset = (i - (len(models) - 1) / 2) * width2
        ax.bar(xw + offset, vals, width=width2, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(xw); ax.set_xticklabels(VERDICT_ORDER)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("P(actually wrong | this verdict)")
    ax.set_title("Calibration: error rate\nwithin each verdict bucket", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Improved detector: how detector_verdict relates to is_wrong & AUROC", fontweight="bold", fontsize=15, y=1.05)
    _savefig(fig, "judge_verdict_relationship")


def fig_finding_category_mix(findings_df):
    counts = findings_df.groupby(["model", "category"]).size().unstack(fill_value=0).reindex(columns=FINDING_CATEGORIES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    models = _ordered(shares.index)
    shares = shares.loc[models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 6))
    bottom = np.zeros(len(models))
    for cat in FINDING_CATEGORIES:
        vals = shares[cat].values
        ax.bar(models, vals, bottom=bottom, label=cat, color=CATEGORY_COLORS[cat], width=0.6)
        bottom += vals
    ax.set_ylabel("Share of findings")
    ax.set_title("Flagged-finding category mix\nby judge model (incl. improved)", fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _savefig(fig, "finding_category_mix_by_judge_model")


def fig_finding_severity_mix(findings_df):
    counts = findings_df.groupby(["model", "severity"]).size().unstack(fill_value=0).reindex(columns=SEVERITIES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    models = _ordered(shares.index)
    shares = shares.loc[models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 6))
    bottom = np.zeros(len(models))
    for sev in SEVERITIES:
        vals = shares[sev].values
        ax.bar(models, vals, bottom=bottom, label=sev, color=SEVERITY_COLORS[sev], width=0.6)
        bottom += vals
    ax.set_ylabel("Share of findings")
    ax.set_title("Flagged-finding severity mix\nby judge model (incl. improved)", fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(title="severity", fontsize=9, title_fontsize=9)
    _savefig(fig, "finding_severity_mix")


def fig_findings_per_case(cases_df):
    data = cases_df.groupby(["architecture", "model"])["n_findings_total"].mean().reset_index()
    models = _ordered(data["model"].unique())
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.3), 6.5))
    x_base = np.arange(len(ARCHITECTURES))
    for i, model in enumerate(models):
        sub = data[data["model"] == model]
        vals = [sub[sub["architecture"] == a]["n_findings_total"].values for a in ARCHITECTURES]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x_base + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x_base); ax.set_xticklabels(ARCHITECTURES)
    ax.set_ylabel("Mean # findings flagged")
    ax.set_title("Detector verbosity: findings per case\n(by judge/detector model, incl. improved)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, "findings_per_case_by_judge_model")


def fig_finding_rate_by_outcome(cases_df):
    data = cases_df.groupby(["model", "architecture", "is_wrong"])["has_finding"].mean().reset_index()
    models = _ordered(data["model"].unique())
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 1.4), 6.5))
    width = 0.8 / (len(ARCHITECTURES) * 2)
    x = np.arange(len(models))
    for ai, arch in enumerate(ARCHITECTURES):
        for oi, wrong in enumerate([False, True]):
            sub = data[(data["architecture"] == arch) & (data["is_wrong"] == wrong)]
            vals = [sub[sub["model"] == m]["has_finding"].values for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (ai * 2 + oi - 1.5) * width
            ax.bar(x + offset, vals, width=width, color=ARCH_COLORS[arch],
                   hatch="//" if wrong else None, edgecolor="black", linewidth=0.4,
                   label=f"{arch} - {'Wrong' if wrong else 'Correct'}")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("P(at least one finding flagged)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Finding-flag rate: correct vs. wrong cases\nby model & architecture (incl. improved)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _savefig(fig, "finding_rate_by_outcome_bars")


def fig_illusion_and_sycophancy(cases_df):
    sub = cases_df[cases_df["method"] == "original"]
    illusion = sub.groupby("architecture")["consensus_illusion_flag"].mean()
    flips = sub.groupby("architecture")["n_sycophantic_flip_agents"].mean()
    present = [a for a in ARCHITECTURES if a in illusion.index]
    fig, ax1 = plt.subplots(figsize=(9, 6.5))
    ax2 = ax1.twinx()
    x = np.arange(len(present))
    w = 0.35
    b1 = ax1.bar(x - w / 2, illusion.reindex(present).values, width=w, color="#4c72b0", label="Consensus-illusion rate")
    b2 = ax2.bar(x + w / 2, flips.reindex(present).values, width=w, color="#c44e52", label="Mean sycophantic flips/case")
    ax1.set_xticks(x); ax1.set_xticklabels(present)
    ax1.set_ylabel("Consensus-illusion rate", color="#4c72b0")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax2.set_ylabel("Mean sycophantic flips / case", color="#c44e52")
    ax1.set_title("False-consensus & sycophancy signals", fontweight="bold")
    ax1.legend(handles=[b1, b2], loc="upper right")
    _savefig(fig, "illusion_and_sycophancy")


def fig_judge_agreement_matrix(cases_df):
    agree, kappa = judge_agreement_matrix(cases_df)
    models = agree.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(6 * len(models) / 2 + 4, 0.6 * len(models) + 4))
    for ax, mat, title, cmap in [
        (axes[0], agree, "Raw agreement rate\n(same has_finding decision)", "YlGnBu"),
        (axes[1], kappa, "Cohen's kappa\n(chance-corrected agreement)", "RdYlGn"),
    ]:
        im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=0 if title.startswith("Raw") else -0.2,
                        vmax=1 if title.startswith("Raw") else 0.6)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(models)):
                v = mat.values[i, j]
                if not pd.isna(v):
                    ax.text(j, i, f"{float(v):.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Judge agreement matrix (original detector, same underlying transcripts)", fontweight="bold", y=1.05)
    _savefig(fig, "judge_agreement_matrix")
    return agree, kappa


def fig_manual_review_kappa(manual_df):
    """
    Real manual-review kappa: Cohen's kappa between the mechanical
    VERIFICATION check (VERIFIED/UNVERIFIED -- does the quote exist in the
    transcript) and the human RATING (binarized: ACCURATE/LIKELY ACCURATE vs
    PARTIALLY ACCURATE/INACCURATE -- is the flagged problem actually real),
    per judge model, from the human-annotated sample in
    __Manual_Detector-Finding_Review__.txt.
    """
    if len(manual_df) == 0:
        return
    df = manual_df.copy()
    df["verif_bin"] = (df["verif"] == "VERIFIED").astype(int)
    df["rating_bin"] = df["rating"].isin(["ACCURATE", "LIKELY ACCURATE"]).astype(int)

    rows = []
    for model, sub in df.groupby("model"):
        if sub["verif_bin"].nunique() > 1 or sub["rating_bin"].nunique() > 1:
            k = cohen_kappa_score(sub["verif_bin"], sub["rating_bin"])
        else:
            k = 1.0 if (sub["verif_bin"] == sub["rating_bin"]).all() else np.nan
        rows.append({"model": model, "kappa": k, "n": len(sub)})
    overall_k = cohen_kappa_score(df["verif_bin"], df["rating_bin"]) if df["verif_bin"].nunique() > 1 else np.nan
    rows.append({"model": "Overall (all models)", "kappa": overall_k, "n": len(df)})
    kdf = pd.DataFrame(rows)
    order = [m for m in MODEL_ORDER if m in kdf["model"].values] + ["Overall (all models)"]
    kdf = kdf.set_index("model").reindex(order).dropna(how="all").reset_index()

    fig, ax = plt.subplots(figsize=(max(8, len(kdf) * 1.2), 6))
    colors = [MODEL_COLORS.get(m, "#555555") for m in kdf["model"]]
    bars = ax.bar(kdf["model"], kdf["kappa"], color=colors, width=0.6)
    for b, v, n in zip(bars, kdf["kappa"], kdf["n"]):
        if not pd.isna(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cohen's kappa")
    ax.set_title(
        "Manual review kappa:\nmechanical VERIFICATION vs. human RATING agreement\n"
        "(does the quote existing in the transcript predict a human judging the finding accurate?)",
        fontweight="bold"
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    _savefig(fig, "manual_review_kappa")


def fig_pct_wrong_with_finding(cases_df):
    data = pct_wrong_with_finding(cases_df).reset_index()
    models = _ordered(data["model"].unique())
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.3), 6.5))
    width = 0.8 / len(ARCHITECTURES)
    x = np.arange(len(models))
    for i, arch in enumerate(ARCHITECTURES):
        sub = data[data["architecture"] == arch]
        vals = [sub[sub["model"] == m]["has_finding"].values for m in models]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - (len(ARCHITECTURES) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=ARCH_COLORS[arch], label=arch)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of wrong-answer cases with >=1 LLM finding")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Recall: percent of wrong answers\nthat have an LLM-identified finding", fontweight="bold")
    ax.legend()
    _savefig(fig, "pct_wrong_answers_with_llm_finding")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    global OUT_DIR
    run_dir = os.path.join(OUT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    OUT_DIR = run_dir
    print(f"Writing figures to: {OUT_DIR}")

    cases_df, findings_df = load_all()
    print(f"Loaded {len(cases_df)} cases, {len(findings_df)} findings")
    print("Models present:", sorted(cases_df['model'].unique()))
    manual_df = load_manual_review()

    fig_agreement_and_error_source(cases_df)
    fig_binned(cases_df, "answer_agreement",
               "Answer agreement vs. error rate,\nRound 1 vs. Round 2 (binned, by architecture)",
               "Answer agreement (binned mean)", "answer_agreement_both_rounds_binned")
    fig_box(cases_df, "answer_agreement",
            "Answer agreement distribution: Round 1 vs Round 2,\nby correctness & architecture",
            "Answer agreement", "answer_agreement_both_rounds_box")
    fig_category_auroc_heatmap(cases_df, findings_df)
    fig_category_corr(cases_df, findings_df)
    fig_correctness_vs_finding(cases_df)
    fig_correlation_with_error(cases_df)
    fig_detector_analysis(cases_df)
    fig_detector_diagnostics_extra(cases_df)
    fig_finding_category_mix(findings_df)
    fig_finding_rate_by_outcome(cases_df)
    fig_finding_severity_mix(findings_df)
    fig_findings_per_case(cases_df)
    fig_illusion_and_sycophancy(cases_df)
    agree, kappa = fig_judge_agreement_matrix(cases_df)
    fig_manual_review_kappa(manual_df)
    fig_would_escalate_heatmap(cases_df)
    fig_judge_verdict_relationship(cases_df)
    fig_binned(cases_df, "reasoning_alignment",
               "Reasoning alignment vs. error rate,\nRound 1 vs. Round 2 (binned, by architecture)",
               "Reasoning alignment (binned mean)", "reasoning_alignment_both_rounds_binned")
    fig_box(cases_df, "reasoning_alignment",
            "Reasoning alignment distribution: Round 1 vs Round 2,\nby correctness & architecture",
            "Reasoning alignment", "reasoning_alignment_both_rounds_box")
    fig_severity_corr(cases_df, findings_df)
    fig_pct_wrong_with_finding(cases_df)

    print("\nAll figures written to", OUT_DIR)


if __name__ == "__main__":
    main()