
import os
import re
import json
import difflib
from collections import Counter

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_DIR = "data"
OUTPUT_DIR = "logs/cleaned_traces"

DATASETS = ["qa", "qausmle", "pubmedqa"]
ARCHITECTURES = ["cot", "vote", "workflow", "critic", "symmetric_debate", "specialized_board"]

# Architectures whose final round is a straight majority vote across all
# participating agents, so aggregated_answer can be safely *recomputed*
# from the cleaned per-agent answers rather than trusted as-is.
VOTE_STYLE_ARCHITECTURES = {"Independent Vote", "Symmetric Debate", "Role-Specialist Board"}
SINGLE_AGENT_ARCHITECTURES = {"Single-Agent CoT", "Workflow-Orchestrator"}
# "Critic-Reviewer Board" is intentionally excluded: the critic's own
# output (used as a tie-breaker) is never persisted to the trace by
# pipelines.py, so a tie between the two revised solvers cannot be
# recomputed after the fact from the saved data alone. See README notes
# at the bottom of this file for a suggested pipelines.py fix.

FUZZY_MATCH_THRESHOLD = 0.85  # difflib ratio required for a "loose" text match
MIN_SUBSTRING_LEN = 4         # guard against short numeric/token false-positive substring matches


# --------------------------------------------------------------------------
# Step 1: Build a case_id -> {option_letter: option_text} lookup per dataset
# --------------------------------------------------------------------------

def _extract_options_from_question(q_text: str) -> dict:
    """
    Extract {letter: option_text} from a QA_data.json question blob.
    Handles BOTH observed formats in the source data:
        "...stem...\\nA. Option one\\nB. Option two"          (newline-delimited)
        "...stem... Answer Choices: (A) Option one (B) Two"   (inline parenthetical)
    and does not cap the letter range at E -- this dataset goes up to J
    (and beyond, for combination-style questions).
    """
    options = {}

    # Format 1: "A. text" / "A) text", anchored at start-of-line
    for key, val in re.findall(
        r'(?:^|\n)\s*([A-Z])[\.\)]\s*(.*?)(?=\n\s*[A-Z][\.\)]\s|\Z)',
        q_text, re.DOTALL,
    ):
        val = val.strip().strip('"\'').strip()
        if val:
            options[key.upper()] = val

    # Format 2: "(A) text (B) text ..." inline. Only fills gaps left by
    # format 1 so we never overwrite a cleanly-delimited match with a
    # noisier inline one.
    for key, val in re.findall(
        r'\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)|$)',
        q_text, re.DOTALL,
    ):
        val = val.strip().strip('"\'').strip()
        if val:
            options.setdefault(key.upper(), val)

    return options


def build_answer_cache(input_dir: str = INPUT_DIR) -> dict:
    """
    Builds cache["qa"][case_id]      -> {letter: option_text}   (from QA_data.json)
          cache["qausmle"][case_id]  -> {letter: option_text}   (from test.jsonl)
    PubMedQA needs no cache: its answer space is fixed to yes/no/maybe.
    """
    cache = {"qa": {}, "qausmle": {}, "pubmedqa": {}}

    qa_path = os.path.join(input_dir, "QA_data.json")
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    c_id = str(item.get("Index"))
                    cache["qa"][c_id] = _extract_options_from_question(item.get("question", ""))
            except Exception as e:
                print(f"WARNING: could not cache QA_data.json: {e}")

    qausmle_path = os.path.join(input_dir, "test.jsonl")
    if os.path.exists(qausmle_path):
        with open(qausmle_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    opts = item.get("options", {})
                    if isinstance(opts, dict):
                        cache["qausmle"][str(idx)] = {
                            k.upper(): str(v).strip() for k, v in opts.items()
                        }
                except Exception as e:
                    print(f"WARNING: could not cache test.jsonl line {idx}: {e}")

    return cache


# --------------------------------------------------------------------------
# Step 2: Normalize a single final_answer value
# --------------------------------------------------------------------------

_PREFIX_RE = re.compile(r'^([A-Z0-9])\s*[:.\)\-]\s*', re.IGNORECASE)
_YES_RE = re.compile(r'\byes\b', re.IGNORECASE)
_NO_RE = re.compile(r'\bno\b', re.IGNORECASE)
_MAYBE_RE = re.compile(r'\bmaybe\b', re.IGNORECASE)


def _normalize_pubmedqa(ans: str) -> str:
    """
    Word-boundary matching instead of naive substring containment.
    The original `"no" in ans_lower` check false-positives on any word
    containing "no" as a substring (not, none, cannot, unknown...). Using
    \\b word boundaries fixes that. "maybe" is checked before "no"/"yes"
    because hedged answers ("it's unclear, possibly") are more often
    mis-swept into "no" by an accidental "not"/"none" substring than the
    reverse.
    """
    if _MAYBE_RE.search(ans):
        return "maybe"
    if _NO_RE.search(ans):
        return "no"
    if _YES_RE.search(ans):
        return "yes"
    return ans  # unresolved -- left as-is, caller records it


def normalize_answer(raw_ans, case_id: str, dataset: str, cache: dict, unresolved_log: list) -> str:
    """
    Standardize an answer to an option letter (A, B, C...) or, for
    PubMedQA, to yes/no/maybe. Never guesses when it isn't reasonably
    confident -- ambiguous or unmatched values are returned unchanged
    and appended to `unresolved_log` for manual review instead of being
    silently mis-mapped.
    """
    # Some agents violate the schema by wrapping final_answer in a JSON
    # array (e.g. ["B"] instead of "B"). str(["B"]) would otherwise
    # stringify to the literal "['B']" and fail every match below, so
    # unwrap it first.
    if isinstance(raw_ans, list):
        raw_ans = raw_ans[0] if len(raw_ans) == 1 else ", ".join(str(x) for x in raw_ans)

    ans = str(raw_ans).strip()
    if not ans:
        return "UNKNOWN"
    if ans == "UNKNOWN":
        return ans

    if dataset == "pubmedqa":
        return _normalize_pubmedqa(ans)

    # 1. "B. text" / "B) text" / "B: text" / "B- text" prefix
    m = _PREFIX_RE.match(ans)
    if m:
        return m.group(1).upper()

    # 2. Bare isolated letter/digit ("B")
    if len(ans) == 1 and ans.isalnum():
        return ans.upper()

    options = cache.get(dataset, {}).get(str(case_id), {})
    if not options:
        unresolved_log.append({"case_id": case_id, "dataset": dataset, "raw": ans, "reason": "no_options_cached"})
        return ans

    ans_lower = ans.lower().strip().rstrip(".")

    # 3. Exact match against an option's full text (safe, unambiguous)
    for key, val in options.items():
        if ans_lower == val.lower().strip().rstrip("."):
            return key

    # 4. The option text appears inside the answer, e.g. the model added
    #    a sentence around it. Sort candidates longest-first so a specific
    #    option ("Bowel wall biopsy") wins over an accidental short
    #    substring match before a shorter one could match spuriously.
    candidates = sorted(options.items(), key=lambda kv: -len(kv[1]))
    matches = [key for key, val in candidates if len(val) >= MIN_SUBSTRING_LEN and val.lower() in ans_lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        unresolved_log.append({
            "case_id": case_id, "dataset": dataset, "raw": ans,
            "reason": "ambiguous_multi_option_substring_match", "candidates": matches,
        })
        return ans

    # 5. The answer is (probably) a truncated/partial quote of a longer
    #    option. Only trusted when unambiguous and long enough to not be
    #    a coincidental numeric/token fragment (e.g. "87" inside "870").
    if len(ans_lower) >= MIN_SUBSTRING_LEN:
        matches = [key for key, val in candidates if ans_lower in val.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            unresolved_log.append({
                "case_id": case_id, "dataset": dataset, "raw": ans,
                "reason": "ambiguous_reverse_substring_match", "candidates": matches,
            })
            return ans

    # 6. Fuzzy fallback for near-identical wording (typos, minor rewording)
    best_key, best_ratio = None, 0.0
    for key, val in options.items():
        ratio = difflib.SequenceMatcher(None, ans_lower, val.lower()).ratio()
        if ratio > best_ratio:
            best_key, best_ratio = key, ratio
    if best_key and best_ratio >= FUZZY_MATCH_THRESHOLD:
        return best_key

    unresolved_log.append({"case_id": case_id, "dataset": dataset, "raw": ans, "reason": "no_match_found"})
    return ans


# --------------------------------------------------------------------------
# Step 3: Deterministic majority vote (fixes the `max(set(votes), key=...)`
# non-determinism in pipelines.py, which relies on Python's unordered set
# iteration and can pick a different "winner" on ties across runs/processes)
# --------------------------------------------------------------------------

def majority_vote(answers: list):
    if not answers:
        return None
    counts = Counter(answers)
    top_count = max(counts.values())
    tied = [a for a in answers if counts[a] == top_count]
    # first-occurrence order among tied options -> stable, reproducible
    for a in answers:
        if a in tied:
            return a
    return answers[0]


def recompute_aggregate(trace: dict, unresolved_log: list, cache: dict, dataset: str):
    """
    Re-derive `aggregated_answer` from the *cleaned* per-agent answers for
    architectures where that is unambiguous, instead of trusting the
    original value (which may have been computed by pipelines.py from
    raw, un-normalized text and therefore be a corrupted vote).
    """
    architecture = trace.get("architecture")
    r1, r2 = trace.get("round_1_outputs", {}), trace.get("round_2_outputs", {})

    if architecture in SINGLE_AGENT_ARCHITECTURES:
        source = r2 if r2 else r1
        if len(source) == 1:
            return next(iter(source.values())).get("final_answer")
        return trace.get("aggregated_answer")

    if architecture in VOTE_STYLE_ARCHITECTURES:
        final_round = r2 if r2 else r1
        votes = [out.get("final_answer") for out in final_round.values() if isinstance(out, dict)]
        if votes:
            return majority_vote(votes)
        return trace.get("aggregated_answer")

    if architecture == "Critic-Reviewer Board":
        # The critic's own final_answer (used as tie-breaker in
        # pipelines.py) isn't persisted anywhere in the trace, so a
        # genuine disagreement between the two revised solvers can't be
        # recomputed here. If the solvers agree post-cleaning, that's
        # unambiguous and safe to use; otherwise fall back to the
        # original (now separately cleaned) aggregated_answer and flag it.
        votes = [out.get("final_answer") for out in r2.values() if isinstance(out, dict)]
        if votes and len(set(votes)) == 1:
            return votes[0]
        unresolved_log.append({
            "case_id": trace.get("case_id"), "dataset": dataset,
            "raw": trace.get("aggregated_answer"),
            "reason": "critic_tiebreak_not_recoverable_from_trace",
        })
        return trace.get("aggregated_answer")

    return trace.get("aggregated_answer")


# --------------------------------------------------------------------------
# Step 4: Drive the whole thing over every trace file
# --------------------------------------------------------------------------

def _find_source_path(dataset: str, arch: str):
    file_name = f"traces_{dataset}_{arch}.jsonl"
    for path in (
        os.path.join("logs/execution_traces", file_name),
        os.path.join(INPUT_DIR, file_name),
        file_name,
    ):
        if os.path.exists(path):
            return path, file_name
    return None, file_name


def process_trace_files():
    print("Pre-loading source ground-truth / option data...")
    cache = build_answer_cache()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unresolved_log = []
    stats = {"files_processed": 0, "fields_changed": 0, "aggregates_recomputed": 0, "lines": 0}

    for dataset in DATASETS:
        for arch_key in ARCHITECTURES:
            source_path, file_name = _find_source_path(dataset, arch_key)
            if not source_path:
                continue

            destination_path = os.path.join(OUTPUT_DIR, file_name)
            print(f"Processing: {source_path} -> {destination_path}")

            corrected_lines = []
            file_changes = 0

            with open(source_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    stats["lines"] += 1
                    try:
                        trace = json.loads(line)
                    except Exception as e:
                        print(f"  [line {idx}] JSON parse failure: {e}")
                        corrected_lines.append(line.strip())
                        continue

                    case_id = trace.get("case_id")

                    for round_key in ("round_1_outputs", "round_2_outputs"):
                        round_data = trace.get(round_key)
                        if isinstance(round_data, dict):
                            for agent_id, agent_out in round_data.items():
                                if isinstance(agent_out, dict) and "final_answer" in agent_out:
                                    orig = agent_out["final_answer"]
                                    cleaned = normalize_answer(orig, case_id, dataset, cache, unresolved_log)
                                    if orig != cleaned:
                                        agent_out["final_answer"] = cleaned
                                        file_changes += 1

                    new_agg = recompute_aggregate(trace, unresolved_log, cache, dataset)
                    if new_agg is not None and new_agg != trace.get("aggregated_answer"):
                        trace["aggregated_answer"] = new_agg
                        file_changes += 1
                        stats["aggregates_recomputed"] += 1
                    elif "aggregated_answer" in trace:
                        # still run it through normalization even when we
                        # didn't/couldn't recompute a vote (e.g. CoT, or
                        # Critic-Reviewer Board fallback path)
                        cleaned_agg = normalize_answer(trace["aggregated_answer"], case_id, dataset, cache, unresolved_log)
                        if cleaned_agg != trace["aggregated_answer"]:
                            trace["aggregated_answer"] = cleaned_agg
                            file_changes += 1

                    corrected_lines.append(json.dumps(trace))

            with open(destination_path, "w", encoding="utf-8") as out_f:
                for line in corrected_lines:
                    out_f.write(line + "\n")

            print(f"  Done. {file_changes} field(s) changed in this file.")
            stats["files_processed"] += 1
            stats["fields_changed"] += file_changes

    report_path = os.path.join(OUTPUT_DIR, "_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "unresolved": unresolved_log}, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Files processed:        {stats['files_processed']}")
    print(f"Trace lines processed:  {stats['lines']}")
    print(f"Fields changed:         {stats['fields_changed']}")
    print(f"Aggregates recomputed:  {stats['aggregates_recomputed']}")
    print(f"Unresolved (needs eyes):{len(unresolved_log)}")
    print(f"Audit report written to: {report_path}")


if __name__ == "__main__":
    process_trace_files()
