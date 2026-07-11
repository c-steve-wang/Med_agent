"""
run_evaluation.py
==================
Runs `MedicalAuditEvaluator` (src/evaluators.py) over every already-generated
`traces_<dataset>_<method>.jsonl` file, without re-invoking the multi-agent
pipelines. Use this whenever you already have execution traces on disk and
just want (or re-want) the accuracy / agreement / evidence-overlap / revision
metrics -- e.g. after regenerating traces, after cleaning them, or to grade
a batch you collected earlier.
"""

import os
import re
import sys
import json
import time
import random
import argparse
from dataclasses import fields as dataclass_fields
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.getcwd())

from src.datastructures import ExecutionTrace
from src.evaluators import (
    load_qa_dataset,
    load_pubmedqa_dataset,
    load_qausmle_dataset,
    MedicalAuditEvaluator,
)

DATASET_LOADERS = {
    "qa": load_qa_dataset,
    "pubmedqa": load_pubmedqa_dataset,
    "qausmle": load_qausmle_dataset,
}

# Longest-first so "qausmle" isn't mistakenly cut short by "qa" during matching
FILENAME_RE = re.compile(r"^traces_(qausmle|pubmedqa|qa)_(.+)\.jsonl$")

_TRACE_FIELDS = {f.name for f in dataclass_fields(ExecutionTrace)}
_TRACE_DEFAULTS = {
    "round_1_outputs": {},
    "round_2_outputs": {},
    "aggregated_answer": "UNKNOWN",
    "total_tokens_used": 0,
    "estimated_cost": 0.0,
}


def discover_trace_files(traces_dir: str):
    """Finds every traces_<dataset>_<method>.jsonl file and returns
    (dataset, method, filepath) tuples, sorted for stable/reproducible runs."""
    found = []
    if not os.path.isdir(traces_dir):
        return found
    for name in sorted(os.listdir(traces_dir)):
        m = FILENAME_RE.match(name)
        if m:
            dataset, method = m.group(1), m.group(2)
            found.append((dataset, method, os.path.join(traces_dir, name)))
    return found


def load_dataset_cases(dataset: str, data_dir: str):
    """Loads a dataset's MedicalCases and returns a case_id -> MedicalCase map."""
    if dataset == "qa":
        cases = DATASET_LOADERS["qa"](path=os.path.join(data_dir, "QA_data.json"))
    elif dataset == "pubmedqa":
        cases = DATASET_LOADERS["pubmedqa"](
            questions_path=os.path.join(data_dir, "ori_pqal.json"),
            ground_truth_path=os.path.join(data_dir, "test_ground_truth.json"),
        )
    elif dataset == "qausmle":
        cases = DATASET_LOADERS["qausmle"](path=os.path.join(data_dir, "test.jsonl"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return {c.case_id: c for c in cases}


def dedupe_by_case_id(raw_lines, source_label: str, skipped_log: list):
    """Keeps only the LAST row per case_id (most recent run wins) and
    records how many duplicate rows were dropped, per file."""
    by_case = {}
    order = []
    dupes = 0
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            trace_dict = json.loads(line)
        except json.JSONDecodeError as e:
            skipped_log.append({"file": source_label, "reason": f"json_parse_error: {e}"})
            continue
        cid = trace_dict.get("case_id")
        if cid in by_case:
            dupes += 1
        else:
            order.append(cid)
        by_case[cid] = trace_dict
    if dupes:
        print(f"    NOTE: {dupes} duplicate case_id row(s) found in {source_label}; keeping the last occurrence of each.")
    return [by_case[cid] for cid in order]


def resilient_call(func, *args, description="network call", max_retry_delay=300, **kwargs):
    """
    Calls func(*args, **kwargs) and retries INDEFINITELY on any exception,
    with exponential backoff (1, 2, 4, 8... seconds, jittered) capped at
    `max_retry_delay`. There is no retry limit and no permanent-failure
    path -- this is meant for long, unattended runs where the only
    acceptable outcomes are "it eventually succeeds" or "you kill the
    process yourself". A dropped wifi connection, a DNS blip, or a
    laptop waking up from sleep with a stale socket all just look like an
    exception here and get retried the same way.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            delay = min(2 ** min(attempt, 9), max_retry_delay) + random.uniform(0, 1)
            print(f"    [{description}] attempt {attempt} failed ({e!r}); "
                  f"retrying in {delay:.0f}s (will keep retrying until it succeeds)...")
            time.sleep(delay)


def _checkpoint_path(output_dir: str, dataset: str, method: str) -> str:
    ckpt_dir = os.path.join(output_dir, "_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{dataset}_{method}.jsonl")


def load_checkpoint(output_dir: str, dataset: str, method: str):
    """Returns (done_case_ids: set, previously_computed_metrics: list) from
    any prior interrupted run of this exact (dataset, method) file."""
    path = _checkpoint_path(output_dir, dataset, method)
    done_ids, rows = set(), []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line from a killed process -- drop it, it'll be redone
                done_ids.add(row.get("case_id"))
                rows.append(row)
    return done_ids, rows


def append_checkpoint(output_dir: str, dataset: str, method: str, metrics: dict):
    """Persists one scored case immediately so progress survives a crash."""
    path = _checkpoint_path(output_dir, dataset, method)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def to_execution_trace(trace_dict: dict) -> ExecutionTrace:
    """Builds an ExecutionTrace from a raw JSONL dict, tolerating missing
    fields (filling safe defaults) or extra/unknown ones (dropped)."""
    clean = {k: v for k, v in trace_dict.items() if k in _TRACE_FIELDS}
    for key, default in _TRACE_DEFAULTS.items():
        clean.setdefault(key, default)
    clean.setdefault("case_id", trace_dict.get("case_id", "UNKNOWN"))
    clean.setdefault("architecture", trace_dict.get("architecture", "UNKNOWN"))
    return ExecutionTrace(**clean)


def run_evaluation(traces_dir: str, data_dir: str, output_dir: str, use_cara: bool, judge_model: str, judge_temperature: float, max_retry_delay: int = 300):
    os.makedirs(output_dir, exist_ok=True)

    trace_files = discover_trace_files(traces_dir)
    if not trace_files:
        print(f"No traces_<dataset>_<method>.jsonl files found under '{traces_dir}'.")
        return

    dataset_cache = {}
    all_summaries = defaultdict(dict)
    skipped_log = []
    all_detailed_rows = []

    for dataset, method, filepath in trace_files:
        print(f"\n=== {dataset} / {method} ({filepath}) ===")

        if dataset not in dataset_cache:
            print(f"  Loading '{dataset}' gold-label dataset...")
            try:
                dataset_cache[dataset] = load_dataset_cases(dataset, data_dir)
            except Exception as e:
                print(f"  ERROR: could not load dataset '{dataset}': {e}. Skipping this file.")
                skipped_log.append({"file": filepath, "reason": f"dataset_load_failed: {e}"})
                continue
        case_map = dataset_cache[dataset]

        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        trace_dicts = dedupe_by_case_id(raw_lines, filepath, skipped_log)

        evaluator = MedicalAuditEvaluator(judge_model=judge_model, temperature=judge_temperature)
        n_ok, n_skipped = 0, 0

        # Resume support only applies to --cara runs: that's the slow,
        # network-bound path worth surviving an interruption for. A plain
        # (local-only, instant) run always recomputes fresh, so a later
        # --cara run never silently gets skipped by stale non-cara
        # checkpoints from an earlier plain run.
        done_ids, checkpointed_rows = set(), []
        if use_cara:
            done_ids, checkpointed_rows = load_checkpoint(output_dir, dataset, method)
            if checkpointed_rows:
                print(f"  Resuming: {len(checkpointed_rows)} case(s) already checkpointed from a previous --cara run, skipping those.")
            for row in checkpointed_rows:
                evaluator.results.append(row)
                all_detailed_rows.append(row)
                n_ok += 1

        for trace_dict in trace_dicts:
            case_id = trace_dict.get("case_id")
            if case_id in done_ids:
                continue
            case = case_map.get(case_id)
            if case is None:
                n_skipped += 1
                skipped_log.append({"file": filepath, "case_id": case_id, "reason": "case_id_not_in_dataset"})
                continue

            try:
                trace = to_execution_trace(trace_dict)
            except Exception as e:
                n_skipped += 1
                skipped_log.append({"file": filepath, "case_id": case_id, "reason": f"trace_build_failed: {e}"})
                continue

            try:
                metrics = evaluator.calculate_metrics(trace, case.gold_label)
            except Exception as e:
                n_skipped += 1
                skipped_log.append({"file": filepath, "case_id": case_id, "reason": f"calculate_metrics_failed: {e}"})
                continue

            if use_cara:
                # Only the CARA path touches the network, so it's the only
                # part that needs indefinite retry -- if wifi drops or the
                # laptop sleeps mid-run, this just keeps trying rather than
                # failing the case or crashing the script.
                print("Running cara for case id:", case_id)
                cara_res = resilient_call(
                    evaluator.run_cara_llm_evaluation, case, trace,
                    description=f"CARA eval case {case_id}",
                    max_retry_delay=max_retry_delay,
                )
                metrics["cara_pairwise_results"] = cara_res["pairwise_results"]
                metrics["cara_llm_cost"] = cara_res["total_cara_cost"]

            metrics["dataset"] = dataset
            metrics["method"] = method
            all_detailed_rows.append(metrics)
            if use_cara:
                append_checkpoint(output_dir, dataset, method, metrics)
            n_ok += 1

        print(f"  Scored {n_ok} case(s), skipped {n_skipped}.")

        if n_ok == 0:
            continue

        df_results = pd.DataFrame([{k: v for k, v in m.items() if k not in ("dataset", "method")} for m in evaluator.results])
        excel_dir = os.path.join(output_dir, dataset)
        os.makedirs(excel_dir, exist_ok=True)
        excel_path = os.path.join(excel_dir, f"eval_result_{method}.xlsx")
        df_results.to_excel(excel_path, index=False)
        print(f"  Wrote detailed results: {excel_path}")

        all_summaries[dataset][method] = evaluator.get_summary()

    # --- Combined master summary, keyed by dataset so same-named methods
    # across different datasets never overwrite each other. ---
    summary_path = os.path.join(output_dir, "eval_summary_all.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nMaster summary written to: {summary_path}")

    if skipped_log:
        skipped_path = os.path.join(output_dir, "skipped_rows.json")
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped_log, f, indent=2)
        print(f"{len(skipped_log)} row(s) skipped -- details in: {skipped_path}")

    # --- Compact stdout comparison table across every dataset/method ---
    if all_detailed_rows:
        combined_df = pd.DataFrame(all_detailed_rows)
        table = combined_df.groupby(["dataset", "method"]).agg(
            n=("case_id", "count"),
            accuracy=("correct", "mean"),
            agreement_r1=("agreement_r1", "mean"),
            agreement_r2=("agreement_r2", "mean"),
            evidence_overlap=("evidence_overlap", "mean"),
            revision_utility=("revision_utility", "sum"),
            avg_confidence=("avg_confidence", "mean"),
            total_cost=("cost", "sum"),
        ).round(4)
        print("\n=== Summary across all datasets/methods ===")
        print(table.to_string())

        combined_path = os.path.join(output_dir, "all_detailed_results.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\nAll per-case rows (every dataset/method combined) written to: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedicalAuditEvaluator over existing execution traces.")
    parser.add_argument("--traces-dir", type=str, default="logs/execution_traces", help="Directory containing traces_<dataset>_<method>.jsonl files")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing QA_data.json / ori_pqal.json / test_ground_truth.json / test.jsonl")
    parser.add_argument("--output-dir", type=str, default="logs/eval_output", help="Where to write Excel/JSON/CSV results")
    parser.add_argument("--cara", action="store_true",default= "True", help="Also run the LLM-as-judge CARA reasoning-alignment evaluation (requires OPENROUTER_API_KEY, slower/costly)")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-4o-mini", help="Model used for --cara judging")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Temperature used for --cara judging")
    parser.add_argument("--max-retry-delay", type=int, default=300, help="Cap (seconds) on exponential backoff between retries of a failed --cara call. Retries never stop, they just stop growing past this delay.")
    args = parser.parse_args()

    traces_dir = args.traces_dir
    if not os.path.isdir(traces_dir):
        # fall back to the current directory, matching the flexible-path
        # convention used elsewhere in this project
        print(f"'{traces_dir}' not found; looking for trace files in the current directory instead.")
        traces_dir = "."

    run_evaluation(
        traces_dir=traces_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_cara=args.cara,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        max_retry_delay=args.max_retry_delay,
    )