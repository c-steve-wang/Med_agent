"""
manual_inspect.py
==================
A standalone tool for manually auditing detector ("judge") outputs against
the underlying multi-agent transcripts, so a human can sanity-check whether
each judge model's flagged findings are actually accurate -- something the
automated metrics (AUROC, correlation, etc.) cannot verify on their own,
since those metrics only check whether findings correlate with the gold
is_wrong label, not whether the finding's stated reasoning is itself true.

Two modes:

  1. EXPORT mode (default, no interaction needed):
         python3 manual_inspect.py --mode export --n 40
     Draws a stratified random sample of cases across judge models and
     outcome buckets (true positive / false positive / false negative /
     true negative, using "has >=1 finding" as the judge's implicit
     prediction of is_wrong) and writes a human-readable Markdown file
     (manual_review_sample.md) with the case question, gold label, the
     agents' claims, and every finding the judge raised -- plus a blank
     rating line for you to fill in by hand, on paper or in a spreadsheet.

  2. INTERACTIVE mode (run it, answer prompts in your terminal):
         python3 manual_inspect.py --mode interactive --n 40
     Walks through the same stratified sample one case at a time in your
     terminal, shows the same information, and asks you to rate each
     finding as accurate / partially accurate / inaccurate. Your ratings
     are saved incrementally to manual_review_ratings.csv, so you can quit
     (Ctrl+C) and resume later -- already-rated (file, case_id, finding
     index) triples are skipped on the next run.

Both modes read the same .jsonl files as detector_trace_analysis.py and use
the same DETECTOR_TRACES_DIR / script-folder / sandbox-path lookup order.
"""
import json
import os
import glob
import re
import csv
import random
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [os.environ.get("DETECTOR_TRACES_DIR"), SCRIPT_DIR, "/mnt/user-data/uploads"]

FNAME_RE = re.compile(r"detector_traces_(?P<dataset>[a-z]+)_(?P<method>.+)_(?P<model>qwen|openai|deepseek|5_6)\.jsonl")
MODEL_LABELS = {"openai": "OpenAI", "qwen": "Qwen", "deepseek": "DeepSeek", "5_6": "OpenAI-5.6"}
METHOD_LABELS = {"specialized_board": "Role-Specialist Board", "symmetric_debate": "Symmetric Debate"}

RATINGS_CSV = os.path.join(SCRIPT_DIR, "manual_review_ratings.csv")
EXPORT_MD = os.path.join(SCRIPT_DIR, "manual_review_sample.md")


def find_files():
    for cand in _CANDIDATE_DIRS:
        if not cand:
            continue
        matches = sorted(glob.glob(os.path.join(cand, "detector_traces_*.jsonl")))
        if matches:
            return matches
    raise FileNotFoundError(
        "No detector_traces_*.jsonl files found. Set DETECTOR_TRACES_DIR or place "
        "the files next to this script (same fix as detector_trace_analysis.py)."
    )


def load_all_cases():
    """Returns a list of dicts: raw record + model/method/file tags."""
    records = []
    for fp in find_files():
        base = os.path.basename(fp)
        m = FNAME_RE.match(base)
        if not m:
            continue
        model = MODEL_LABELS.get(m.group("model"), m.group("model"))
        method = METHOD_LABELS.get(m.group("method"), m.group("method"))
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_file"] = base
                d["_model"] = model
                d["_method"] = method
                records.append(d)
    return records


def bucket_of(record):
    """Confusion-matrix-style bucket using 'has >=1 finding' as the judge's
    implicit prediction of is_wrong. Lets us sample evenly across the cases
    that matter most for auditing: where the judge disagreed with itself
    (FP/FN) as well as where it agreed (TP/TN)."""
    n_findings = len(record.get("llm_findings") or []) + len(record.get("other_findings") or [])
    predicted_wrong = n_findings > 0
    actual_wrong = bool(record.get("is_wrong"))
    if predicted_wrong and actual_wrong:
        return "TP (flagged & actually wrong)"
    if predicted_wrong and not actual_wrong:
        return "FP (flagged but actually correct)"
    if not predicted_wrong and actual_wrong:
        return "FN (not flagged but actually wrong)"
    return "TN (not flagged & actually correct)"


def stratified_sample(records, n_total, seed=42):
    """Sample roughly evenly across (judge model x confusion bucket), so the
    reviewer sees a balanced mix rather than mostly-TN cases (which dominate
    the raw data). FP and FN cases are the most diagnostically important --
    they're where the judge's behavior most needs human scrutiny -- so they
    are over-weighted slightly if too few exist naturally."""
    rng = random.Random(seed)
    groups = {}
    for r in records:
        key = (r["_model"], bucket_of(r))
        groups.setdefault(key, []).append(r)

    for v in groups.values():
        rng.shuffle(v)

    # priority: FP/FN buckets first (most informative), then TP, then TN
    priority = {"FP (flagged but actually correct)": 0, "FN (not flagged but actually wrong)": 0,
                "TP (flagged & actually wrong)": 1, "TN (not flagged & actually correct)": 2}
    keys_sorted = sorted(groups.keys(), key=lambda k: priority.get(k[1].split(" ")[0] + " " + " ".join(k[1].split(" ")[1:]), 1))
    # simpler: sort by the bucket's priority prefix
    def pri(k):
        for prefix, p in priority.items():
            if k[1] == prefix:
                return p
        return 1
    keys_sorted = sorted(groups.keys(), key=pri)

    sample = []
    per_group_target = max(1, n_total // max(1, len(groups)))
    for key in keys_sorted:
        take = groups[key][:per_group_target]
        sample.extend(take)
    # top up if under target (small groups ran out)
    if len(sample) < n_total:
        remaining = [r for key in keys_sorted for r in groups[key][per_group_target:]]
        rng.shuffle(remaining)
        sample.extend(remaining[: n_total - len(sample)])

    rng.shuffle(sample)
    return sample[:n_total]


def format_case_header(r):
    lines = [
        f"### Case {r.get('case_id')} — {r['_model']} / {r['_method']} — bucket: {bucket_of(r)}",
        "",
        f"- **file**: `{r['_file']}`",
        f"- **gold_label**: {r.get('gold_label')}",
        f"- **aggregated_answer**: {r.get('aggregated_answer')}",
        f"- **is_wrong**: {r.get('is_wrong')}",
        f"- **hallucinated_evidence_rate**: {r.get('hallucinated_evidence_rate')}",
        f"- **reasoning_alignment (r1 -> r2)**: {r.get('reasoning_alignment_r1')} -> {r.get('reasoning_alignment_r2')}",
        f"- **answer_agreement (r1 -> r2)**: {r.get('answer_agreement_r1')} -> {r.get('answer_agreement_r2')}",
    ]
    return "\n".join(lines)


def format_findings(r):
    findings = (r.get("llm_findings") or []) + (r.get("other_findings") or [])
    if not findings:
        out = ["**No findings were flagged by the judge for this case.**"]
        return findings, "\n".join(out)
    out = []
    for i, fdg in enumerate(findings):
        if not isinstance(fdg, dict):
            continue
        out.append(
            f"**Finding {i}** — category=`{fdg.get('category')}`, severity=`{fdg.get('severity')}`, "
            f"agent=`{fdg.get('agent_id')}`, round=`{fdg.get('round')}`\n"
            f"> Quote: \"{fdg.get('quote', '')}\"\n"
            f"> Explanation: {fdg.get('explanation', '')}"
        )
    return findings, "\n\n".join(out)


def export_mode(n):
    records = load_all_cases()
    sample = stratified_sample(records, n)
    with open(EXPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Manual Detector-Finding Review Sample\n\n")
        f.write(
            f"Stratified sample of {len(sample)} cases across judge models and confusion buckets "
            "(TP/FP/FN/TN, using 'has >=1 finding' as the judge's implicit is_wrong prediction).\n\n"
            "For each finding below, judge for yourself whether the quote/explanation is an ACCURATE "
            "description of a real problem in the transcript, or a hallucinated/overstated/irrelevant flag. "
            "Fill in the blank rating line under each finding (accurate / partially accurate / inaccurate + "
            "optional note), then tally your ratings however you like (e.g. paste into a spreadsheet).\n\n"
            "---\n\n"
        )
        for r in sample:
            f.write(format_case_header(r) + "\n\n")
            findings, txt = format_findings(r)
            f.write(txt + "\n\n")
            for i in range(len(findings)):
                f.write(f"- [ ] Finding {i} rating: ______________  notes: ______________________________\n")
            f.write("\n---\n\n")
    print(f"Wrote {len(sample)} cases to {EXPORT_MD}")
    print("Open it in any Markdown viewer / text editor and fill in the rating lines by hand.")


def load_existing_ratings():
    done = set()
    if os.path.exists(RATINGS_CSV):
        with open(RATINGS_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["file"], row["case_id"], row["finding_index"]))
    return done


def interactive_mode(n):
    records = load_all_cases()
    sample = stratified_sample(records, n)
    done = load_existing_ratings()

    is_new = not os.path.exists(RATINGS_CSV)
    csv_file = open(RATINGS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new:
        writer.writerow(["file", "case_id", "model", "method", "bucket", "finding_index",
                          "category", "severity", "quote", "rating", "note"])

    print(f"Loaded {len(sample)} sampled cases. Ratings save incrementally to {RATINGS_CSV}.")
    print("At each finding, enter: [a]ccurate / [p]artially accurate / [i]naccurate / [s]kip case / [q]uit\n")

    try:
        for r in sample:
            findings, _ = format_findings(r)
            if not findings:
                continue  # nothing to rate for TN cases in this pass
            print("\n" + "=" * 100)
            print(format_case_header(r))
            print()
            skip_case = False
            for i, fdg in enumerate(findings):
                if not isinstance(fdg, dict):
                    continue
                key = (r["_file"], str(r.get("case_id")), str(i))
                if key in done:
                    continue
                print(f"\n--- Finding {i} ---")
                print(f"category={fdg.get('category')}  severity={fdg.get('severity')}  "
                      f"agent={fdg.get('agent_id')}  round={fdg.get('round')}")
                print(f"Quote: \"{fdg.get('quote', '')}\"")
                print(f"Explanation: {fdg.get('explanation', '')}")
                ans = input("Rating [a/p/i/s/q]: ").strip().lower()
                if ans == "q":
                    raise KeyboardInterrupt
                if ans == "s":
                    skip_case = True
                    break
                rating = {"a": "accurate", "p": "partially_accurate", "i": "inaccurate"}.get(ans, "unclear")
                note = input("Optional note (enter to skip): ").strip()
                writer.writerow([r["_file"], r.get("case_id"), r["_model"], r["_method"], bucket_of(r),
                                  i, fdg.get("category"), fdg.get("severity"), fdg.get("quote", ""), rating, note])
                csv_file.flush()
            if skip_case:
                continue
    except KeyboardInterrupt:
        print("\n\nStopped early -- your ratings so far are saved. Resume any time by rerunning this script.")
    finally:
        csv_file.close()
    print(f"\nDone. Ratings are in {RATINGS_CSV}")


def main():
    ap = argparse.ArgumentParser(description="Manually audit detector findings against source transcripts.")
    ap.add_argument("--mode", choices=["export", "interactive"], default="export")
    ap.add_argument("--n", type=int, default=40, help="Number of cases to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "export":
        export_mode(args.n)
    else:
        interactive_mode(args.n)


if __name__ == "__main__":
    main()
