# Issues and Recommended Improvements — 7/22

## Current Project Stage

The repository is currently in the **problem-identification prototype** stage. It can generate and store observable outputs from six medical multi-agent architectures, and it contains prototype consistency, hallucination, contradiction, sycophancy, and error-propagation detectors. It has not yet produced a validated problem-identification result, and the proposed self-evolving intervention has not been implemented.

Current status:

| Component | Status |
|---|---|
| Research question and hypotheses | Defined |
| Six baseline architectures | Implemented |
| Public rationale and communication traces | Implemented |
| Answer normalization | Offline prototype only |
| Consistency measurement | Lexical prototype only |
| Hallucination/contradiction detection | Partial and not yet validated |
| Complete comparable experiment matrix | Not complete |
| Validated problem-identification conclusion | Not complete |
| Self-evolving method | Not started |

Existing cleaned artifacts cover 18 dataset–architecture files and 13,653 trace rows, but these are not all unique cases and should be treated as exploratory. In particular, the current qausmle specialized-board evaluator reports about 67.9% accuracy while its detector output labels about 91% of cases as wrong, demonstrating that normalization and detector correctness are not yet aligned.

The immediate goal is to produce a leakage-free, reproducible diagnosis of whether answer consensus masks semantic rationale misalignment, hallucination, contradiction, or error propagation. Only after this diagnosis is frozen should the project implement and evaluate a self-evolving solution.

## 1. QA Answer Leakage

### Current issue

`load_qa_dataset()` places `Scoring_Points` in `MedicalCase.evidence_context`, and `_build_case_prompt()` sends that field to every tested agent as `GROUNDING EVIDENCE/ABSTRACT`. The scoring points often contain the correct diagnosis and answer rationale.

The detector also treats the same leaked material as source evidence. Consequently, existing QA accuracy, agreement, rationale-similarity, grounding, contradiction, and hallucination results are contaminated.

### Recommended improvement

- Remove `Scoring_Points` from all tested-agent prompts.
- Add a separate `reference_rationale` or evaluation-only field if the scoring points are retained for analysis.
- Ensure the detector does not silently treat the evaluation-only rationale as evidence shown to the model.
- Regenerate all QA traces, detector outputs, metrics, tables, and figures after the fix.
- Continue providing PubMedQA abstracts because they are legitimate task evidence rather than grading rubrics.

## 2. Rationale and Evidence Similarity

### Current issue

`reasoning_alignment()` is pairwise token-set Jaccard over extracted claims. It measures lexical overlap, not logical consistency. The evaluator's `evidence_overlap` is also exact-string Jaccard over `cited_evidence` items.

Both implementations return `1.0` when fewer than two comparable agents are present. This creates artificial perfect scores for single-agent CoT and for workflow stages that are not actually peers.

### Recommended improvement

- Use whole-rationale embedding cosine similarity as the primary semantic-similarity metric.
- Add sentence-level embedding matching as a robustness analysis.
- Use the public `reasoning` field; do not request hidden chain-of-thought.
- Use a biomedical model such as `FremyCompany/BioLORD-2023-M`; optionally compare against `BAAI/bge-m3`.
- Retain Jaccard only as a lexical-overlap baseline.
- Keep contradiction detection separate because high embedding similarity does not imply logical consistency.
- Store metrics with fewer than two comparable outputs as `null`/`not_applicable`, not `1.0`.

Suggested names:

- `lexical_rationale_overlap`
- `whole_rationale_semantic_similarity`
- `sentence_level_semantic_similarity`
- `semantic_evidence_overlap`

The broken CARA path does not need to remain a primary similarity metric if embeddings are used. An LLM judge can instead be reserved for hallucination and contradiction auditing.

## 3. Threshold Sensitivity

### Current issue

The previously discussed rule `reasoning_alignment < 0.6` is **not present in the current detector**. The current `consensus_illusion_flag` is triggered when answer agreement increases while lexical rationale alignment decreases.

Other exploratory cutoffs are still inconsistent: lexical grounding uses `0.5`, the detector summary uses hallucination rate `> 0.5`, and one visualization uses `>= 0.15` for “high hallucination.” These labels should not be treated as validated clinical thresholds.

### Recommended improvement

- Use continuous similarity and hallucination scores in the primary analysis.
- Report the paired difference between answer agreement and semantic rationale similarity with bootstrap confidence intervals.
- For a simple secondary analysis, report semantic-similarity thresholds `0.60`, `0.70`, `0.80`, and `0.90`.
- Freeze the chosen threshold grid before inspecting architecture-level outcomes, and label it as sensitivity analysis rather than a preregistered standard.
- Either justify hallucination-category cutoffs with a manual validation set or avoid categorical labels such as “high.”

## 4. Observable Communication, Tokens, Cost, and Runtime

### Current issue

The current trace stores observable agent outputs and total tokens/cost, but not the exact messages or per-call metadata. The LLM client receives prompt and completion token counts, but the pipelines sum them and discard the split. Detector/judge token usage is also discarded.

The hard-coded pricing table silently applies GPT-4-level fallback prices to unknown models, which can produce incorrect cost estimates.

### Recommended improvement

Record the following for every tested-model and judge-model call:

- Provider, model identifier/version, temperature, and seed when supported
- Prompt/template version and exact messages
- Architecture, agent role, stage/round, and call index
- Prompt, completion, reasoning, cached, and total tokens when the provider exposes them
- Price-table version/source and estimated cost; use `null` when pricing is unknown
- Latency, retry count, HTTP/API error, parsing error, and final call status

For every architecture, report:

- Calls, tokens, cost, and latency per case
- Tokens and cost per correct answer
- Accuracy or grounding gain per additional 1,000 tokens
- Resource use relative to the single-agent CoT baseline

No additional hidden-thinking logging is required. The missing object is reproducible **communication and call metadata**, not private chain-of-thought.

## 5. Answer Normalization, Aggregation, and Failure Handling

### Current issue

`fix_trace_answers.py` contains a useful dataset-specific normalizer, but it is an offline cleanup step and is not used by the live pipeline or evaluator. Online voting still groups raw strings, while correctness is evaluated using case-sensitive `startswith()`. Thus `A`, `Option A`, and `A. diagnosis` can be voted or scored inconsistently; `A and B` can be incorrectly accepted as answer `A`.

The cleanup script also uses a different tie policy from the online aggregator and contains a stale assumption that the critic output is not persisted. Cleaned aggregates therefore may not reproduce the live aggregation policy.

API or JSON failures are converted to ordinary `UNKNOWN` outputs with zero tokens, so infrastructure failures can be counted as model errors and bypass the outer retry logic. Valid JSON is not validated against the required schema.

### Recommended improvement

- Move one canonical, dataset-specific normalizer into the live path and reuse it for voting, final aggregation, correctness evaluation, cleaning, and detector checks.
- Require exactly one valid MCQ option or one of `yes`/`no`/`maybe`.
- Use the same deterministic tie policy online and offline.
- Update the critic cleanup path to use the stored `Skeptical_Reviewer` output.
- Validate required fields, types, confidence range, and answer space before accepting an output.
- Record retryable API failures, terminal API failures, parse/schema failures, and genuine model answers as separate statuses.
- Exclude infrastructure failures from the primary accuracy denominator and report their rate separately.

## 6. Hallucination and Contradiction Validation

### Current issue

Current detector files are not directly comparable. Some QA files were produced by the older lexical detector without a `detection_mode`, while the specialized-board and symmetric-debate files were produced with the LLM detector. PubMedQA detector outputs are absent, and the qausmle symmetric-debate detector file contains only two cases.

The lexical detector checks only `cited_evidence` claims with word containment, whereas the LLM detector reviews the full transcript. Their hallucination rates therefore have different meanings. Judge model, temperature, prompt version, pass count, tokens, and cost are not saved in each result row. No manual reference set currently validates either detector.

### Recommended improvement

- After fixing QA leakage and normalization, rerun one fixed detector configuration over the complete dataset × architecture matrix.
- Do not compare lexical and LLM-detector rates in the same architecture ranking.
- Persist detector mode and full judge configuration with every output.
- Manually annotate at least 50 stratified transcripts spanning datasets and architectures.
- Report precision, recall/F1 when feasible, and inter-annotator agreement when two annotators are available.
- Use `LLM-identified hallucination` or `lexically ungrounded claim` unless the finding has been manually confirmed.

## 7. Architecture-Aware and Resource-Fair Comparison

### Current issue

The architectures use different numbers of model calls: CoT uses 1, independent vote 3, symmetric debate 6, role-specialist board 6, critic-reviewer 5, and the extractor–solver workflow 2. Their shared `round_1_outputs`/`round_2_outputs` representation also gives metrics different meanings across architectures.

CoT and vote have no second round; workflow compares different stages rather than peer revisions; and critic round 1 includes a reviewer that is absent from round 2. Agreement, evidence overlap, and revision utility currently use `1.0` or `0` defaults in cases where a metric is actually not applicable. The independent-vote baseline also uses different personas, so it is a persona-diverse ensemble rather than identical-prompt self-consistency.

### Recommended improvement

- Define an applicability table before aggregation and use `not_applicable` instead of artificial defaults.
- Report natural-budget comparisons with all resource use disclosed.
- Add a budget-matched single-agent self-consistency baseline for the main multi-agent comparisons.
- Either run independent vote with identical prompts and independent samples, or rename it `persona-diverse vote`.
- Rename the current workflow `two-stage extractor–solver workflow`; do not expand it unless the proposal requires that additional scope.

## 8. Runnable and Reproducible Evaluation Entry Points

### Current issue

Several deterministic code issues prevent a reliable clean rerun:

- `main.py`, `run_evaluation.py`, and `run_detectors.py` import `load_qausmle_dataset`, but that function does not exist.
- `data/test.jsonl`, expected by the qausmle path, is absent from the repository.
- `main.py` writes `traces_new_<dataset>_<method>.jsonl`, but the evaluator only discovers `traces_<dataset>_<method>.jsonl`.
- The current QA path in `main.py` runs only nine hard-coded cases rather than the full dataset.
- `--dataset` uses `nargs='+'` but has a string default, causing inconsistent list/string behavior and malformed output directories.
- CARA expects the wrong return arity/type from `call_free_form_llm`, references a nonexistent pricing attribute, returns keys different from those expected by both callers, and only compares the first two agents.
- `run_evaluation.py --cara` is effectively enabled by default because its default is the non-empty string `"True"`; deterministic CARA errors can then enter indefinite retry.
- `run_detectors.py` imports `ollama` unconditionally, although it is not listed in `requirements.txt` and is unnecessary for OpenRouter or lexical runs.
- Embedding dependencies and a pinned embedding-model revision have not yet been added.

### Recommended improvement

- Repair one canonical end-to-end entry point and add a small smoke test before spending API budget.
- Implement or remove the qausmle loader and include a dataset manifest with file hashes and case counts.
- Make trace naming and evaluator discovery use the same convention.
- Remove hard-coded evaluation slices from the main experiment; expose a separate `--limit`/`--case-ids` debug option.
- Make dataset arguments consistently list-valued and validate them at startup.
- Disable or remove CARA until its interface is repaired; never indefinitely retry deterministic schema/interface errors.
- Make optional backends lazy imports and pin all experiment dependencies and model revisions.

## Recommended Priority

### Required now

1. Remove QA answer leakage and separate evaluation-only references.
2. Repair the canonical runner, dataset loaders, trace naming, and CARA default behavior.
3. Integrate answer normalization, schema validation, and structured failure handling.
4. Add per-call model, prompt, token, cost, latency, and error logging.
5. Replace Jaccard as the primary rationale-similarity metric and mark inapplicable metrics correctly.

### Required later

6. Regenerate QA traces and rerun one detector configuration over the complete experiment matrix.
7. Run threshold sensitivity analysis and manually validate hallucination/contradiction detection.
8. Report both natural-budget and budget-matched comparisons.

The defensible claim should remain limited to observable outputs: final-answer agreement may exceed semantic agreement between generated rationales, and consensus does not guarantee evidence grounding or factual reliability. The current stored results are preliminary and cannot yet establish that claim because of QA leakage, incomparable detector configurations, invalid metric defaults, and rerun failures.
