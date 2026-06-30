# Technical Review of the Six-Architecture Baseline

## Executive Summary

The repository implements six architecture-style baselines:

- `cot`: single-agent chain-of-thought style answering
- `vote`: independent multi-agent voting
- `symmetric_debate`: symmetric debate with two rounds
- `specialized_board`: role-specialist board with two rounds
- `critic`: critic-reviewer workflow
- `workflow`: evidence-extraction followed by solving

These are valid candidates for architecture-level comparison, but the current implementation is not yet stable, reproducible, or fully auditable. The main issues are unstable aggregation, incomplete trace persistence, weak failure handling, inconsistent metric semantics across architectures, and non-deterministic LLM behavior.

The code can produce preliminary experimental results, but it should not yet be treated as a reliable six-architecture baseline for rigorous evaluation.

## Baseline Definition Issues

### Unequal Resource Budgets

The six baselines do not use comparable computational budgets.

| Method | Architecture | LLM Calls per Case |
|---|---:|---:|
| `cot` | Single-agent CoT | 1 |
| `vote` | Three independent agents | 3 |
| `symmetric_debate` | Three agents, two rounds | 6 |
| `specialized_board` | Three specialists, two rounds | 6 |
| `critic` | Two solvers, one critic, two solver revisions | 5 |
| `workflow` | Extractor then solver | 2 |

This is acceptable for comparing architecture families, but the paper or report must explicitly disclose the call budget, token budget, model, temperature, and aggregation policy for each method. Otherwise, accuracy comparisons are not resource-fair.

### Architecture Semantics Are Mixed

The `ExecutionTrace` schema assumes a two-round agent process:

```python
round_1_outputs
round_2_outputs
aggregated_answer
```

This works reasonably for `symmetric_debate` and `specialized_board`, but it does not naturally fit `cot`, `vote`, `critic`, or `workflow`.

For example:

- `cot` has only one agent and no second round.
- `vote` has only independent first-round agents.
- `critic` has a critic output that is not preserved in the trace.
- `workflow` places the extractor in round 1 and the solver in round 2, even though they are different pipeline stages rather than the same agents revising answers.

This makes downstream metrics hard to interpret consistently.

## Output Stability Issues

### No Retry Logic for LLM Calls

Location: `src/llm_client.py`

The LLM client catches all exceptions and returns an `UNKNOWN` answer:

```python
except Exception as e:
    print(f"API Error: {e}")
    return {"final_answer": "UNKNOWN", "confidence": 0.0, "reasoning": f"Error Fallback: {str(e)}"}, 0
```

This is problematic because API failures, rate limits, network timeouts, and JSON parse failures are treated as normal model outputs. These failed calls then enter the accuracy calculation and can be mistaken for model errors.

Recommended changes:

- Add retry with exponential backoff.
- Record structured error states.
- Separate failed cases from model-incorrect cases.
- Persist raw errors in the trace.

### JSON Parsing Is Not Validated Against a Schema

Location: `src/llm_client.py`

The code only calls `json.loads(raw_text)`. It does not validate that required fields exist or have the expected types.

Expected fields include:

- `final_answer`
- `confidence`
- `diagnosis_or_hypothesis`
- `reasoning`
- `cited_evidence`
- `missing_evidence`
- `safety_concerns`
- `revision_note`

If the model returns valid JSON with missing or malformed fields, the downstream metrics may silently become invalid.

Recommended changes:

- Add schema validation.
- Validate `confidence` as a numeric value in `[0, 1]`.
- Validate `cited_evidence`, `missing_evidence`, and `safety_concerns` as lists.
- Mark malformed model outputs as invalid rather than silently evaluating them.

### Tie-Breaking Is Non-Deterministic

Location: `src/pipelines.py`

Several aggregation paths use:

```python
max(set(votes), key=votes.count)
```

This appears in independent voting, symmetric debate, and role-specialist board aggregation.

The issue is that `set(votes)` is unordered. If there is a tie, the selected answer can be arbitrary. This makes the final answer non-deterministic under tied votes.

Affected methods:

- `vote`
- `symmetric_debate`
- `specialized_board`

Recommended changes:

- Implement an explicit deterministic tie-breaker.
- Possible policies:
  - select the answer with highest average confidence;
  - use a judge model;
  - return a `TIE` status;
  - use a fixed answer ordering as a final fallback.

### Temperature Defaults Are Not Deterministic

Location: `src/llm_client.py`

The default model temperature is `0.4`.

For baseline experiments, this reduces reproducibility. Multiple runs can produce different answers, rationales, evidence lists, and votes.

Recommended changes:

- Use `temperature=0.0` as the default for baseline runs.
- Log the temperature for every model call.
- Persist model configuration in the output trace.

## Trace and Auditability Issues

### Full Execution Traces Are Not Persisted

Location: `main.py`

The main loop computes `trace = run_pipeline(case)`, but only evaluation metrics are saved to Excel. The full `ExecutionTrace` is not written to disk.

As a result, the following information is lost after execution:

- each agent's raw output;
- each agent's final answer;
- each agent's rationale;
- cited evidence;
- missing evidence;
- safety concerns;
- round-1 and round-2 outputs;
- workflow extractor outputs;
- critic outputs;
- raw model responses;
- prompts;
- error states.

This is the largest auditability gap in the current implementation.

Recommended changes:

- Save one JSONL trace file per method:

```text
logs/traces_{method}_{timestamp}.jsonl
```

- Save traces regardless of whether `--eval` is enabled.
- Include parsed outputs, raw responses, prompts, token usage, and errors.

### Running Without `--eval` Produces No Results

Location: `main.py`

The default value of `--eval` is false. However, the code still calls the LLM pipelines. If `--eval` is not set, no metrics or traces are saved.

This means the default execution can spend API calls without producing durable experimental artifacts.

Recommended changes:

- Always save traces.
- Use `--eval` only to control whether additional evaluator or judge calls are made.

### Critic Output Is Not Saved in the Trace

Location: `src/pipelines.py`

The critic architecture calls a critic:

```python
critic_out, c_tokens = self.client.call_agent(...)
```

But `critic_out` is not included in the returned `ExecutionTrace`.

This is especially problematic because when the solvers disagree, the final aggregated answer can come from the critic. In that case, the code records the critic's final decision but not the critic's full rationale.

Recommended changes:

- Add critic output to a separate trace field, such as `judge_outputs` or `meta_outputs`.
- Alternatively, include the critic under a named key in the trace.
- Ensure final aggregation records which component selected the final answer.

### Workflow Trace Semantics Are Misleading

Location: `src/pipelines.py`

The workflow architecture returns:

```python
round_1_outputs = {"Extractor": extraction}
round_2_outputs = {"Solver": final_out}
```

This makes the extractor and solver look like two rounds of the same agent process, but they are actually different workflow stages. Metrics such as agreement shift and revision utility do not have the same interpretation for this architecture.

Recommended changes:

- Represent workflow steps as ordered pipeline stages.
- Do not force workflow outputs into round-1 and round-2 agent semantics.
- Make the evaluator architecture-aware.

## Evaluation Metric Issues

### Answer Matching Is Too Fragile

Location: `src/evaluators.py`

The code checks correctness with:

```python
trace.aggregated_answer.strip().startswith(gold_label)
```

This creates both false negatives and false positives.

Examples:

- `gold_label = "yes"` and `model_answer = "Yes"` is marked incorrect.
- `gold_label = "A"` and `model_answer = "A and B"` is marked correct.
- `gold_label = "A"` and `model_answer = "option A"` is marked incorrect.

Recommended changes:

- Implement dataset-specific answer normalization.
- For multiple-choice QA, extract exactly one of `A`, `B`, `C`, or `D`.
- For PubMedQA, extract exactly one of `yes`, `no`, or `maybe`.
- Compare normalized labels rather than raw strings.

### CARA Only Compares the First Two Agents

Location: `src/evaluators.py`

The CARA evaluation selects only the first two entries from `trace.round_1_outputs`:

```python
agent_ids = list(trace.round_1_outputs.keys())
agent_a_id = agent_ids[0]
agent_b_id = agent_ids[1]
```

This is not representative for multi-agent architectures.

For example:

- `vote` has three agents, but only two are compared.
- `symmetric_debate` has three agents, but only two round-1 outputs are compared.
- `specialized_board` has three specialists, but only two are compared.

Recommended changes:

- Compute all pairwise CARA scores.
- Report mean, standard deviation, and number of pairs.
- Make CARA optional for architectures where it is not meaningful.

### CARA Is Not Meaningful for All Architectures

For `cot`, there is only one agent, so CARA is not applicable.

For `workflow`, comparing extractor reasoning against solver reasoning is not the same as comparing two agents solving the same task. The metric does not have a clean interpretation.

Recommended changes:

- Use `not_applicable` rather than `None` for metrics that do not apply.
- Report metric applicability by architecture.

### Revision Utility Only Applies to Some Architectures

Location: `src/evaluators.py`

Revision utility is only computed when round-1 and round-2 outputs have the same agent IDs:

```python
if trace.round_2_outputs and set(trace.round_1_outputs.keys()) == set(trace.round_2_outputs.keys()):
```

This means it applies to debate-style architectures but not to `cot`, `vote`, or `workflow`.

Recommended changes:

- Report revision utility only for architectures with true revision rounds.
- Do not treat missing revision utility as zero.
- Separate `not_applicable` from measured zero utility.

### Evidence Overlap Is Based on Raw String Equality

Location: `src/evaluators.py`

Evidence overlap is computed as Jaccard similarity over raw strings in `cited_evidence`.

This is fragile because the same clinical fact can be phrased differently by different agents.

Recommended changes:

- Use numbered evidence spans from the input context.
- Ask models to cite evidence IDs rather than free-form evidence text.
- Alternatively, normalize evidence strings or use semantic matching.

## Cost Accounting Issues

### Input and Output Tokens Are Not Separated

Location: `src/llm_client.py` and `src/pipelines.py`

The code uses `response.usage.total_tokens`, then multiplies it by a single output token price:

```python
cost = total_tokens * self.client.output_token_cost
```

This is inaccurate because prompt tokens and completion tokens usually have different prices.

Recommended changes:

- Store `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- Compute input and output costs separately.
- If exact model pricing is unavailable, report tokens only and avoid claiming precise cost.

### Model Pricing Is Hard-Coded

Location: `src/llm_client.py`

The cost is hard-coded:

```python
self.output_token_cost = 15.00 / 1000000
```

This does not adapt when the user changes `--model` or `--judge_model`.

Recommended changes:

- Add a pricing table keyed by model name.
- Log unknown model prices as unavailable.
- Avoid mixing different model prices under one constant.

## Environment and API Issues

### Missing API Key Breaks Module Import

Location: `src/llm_client.py`

The OpenRouter client is initialized at import time:

```python
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OR_CLIENT = openai.OpenAI(...)
```

If the environment variable is missing, even local data loading can fail because `src.evaluators` imports `MedicalLLMClient`.

Recommended changes:

- Delay OpenAI/OpenRouter client initialization until the first LLM call.
- Raise a clear error only when an LLM call is attempted.

### OpenRouter JSON Mode Compatibility Is Not Checked

The code assumes every selected model supports:

```python
response_format={ "type": "json_object" }
```

This may not be true for all OpenRouter models.

Recommended changes:

- Detect unsupported `response_format` failures.
- Maintain a model capability configuration.
- Add a fallback parser if strict JSON mode is not available.

## Prompting and Reasoning Trace Issues

### Final Answer Format Is Not Strict Enough

The schema asks for:

```json
"final_answer": "Your final chosen option"
```

But different datasets require different answer spaces:

- multiple-choice QA: `A`, `B`, `C`, `D`
- PubMedQA: `yes`, `no`, `maybe`

The current prompt does not enforce these exact formats, so the model can return answers such as:

- `A. IGF1`
- `The answer is A`
- `Yes`
- `yes, because...`

This affects voting, agreement, and correctness.

Recommended changes:

- Add dataset-specific answer instructions to the prompt.
- Require `final_answer` to be exactly one valid label.
- Validate the label after parsing.

### The Stored Reasoning Is Not Hidden Chain-of-Thought

The field `reasoning` should be interpreted as a rationale, explanation, or reasoning summary. It should not be described as the model's complete hidden chain-of-thought.

Recommended wording:

- rationale
- explanation trace
- reasoning summary
- evidence-grounded justification

Avoid claiming that the system records full internal model reasoning.

## Code Organization Issues

### Colab Export Files Are Still Present

Files:

- `Medagent.py`
- `medagent_py.py`

These files appear to be exported Colab notebooks. They depend on `google.colab`, include `drive.mount`, duplicate logic from `src/`, and may confuse users about the intended entry point.

Recommended changes:

- Move them to an `archive/` directory.
- Or delete them if they are no longer needed.
- Document that `main.py` is the supported entry point.

### No README or Experiment Protocol

The repository does not currently document:

- required environment variables;
- how to run all six architectures;
- what files are produced;
- how traces and metrics should be interpreted;
- which metrics apply to which architecture;
- model and token budget assumptions.

Recommended changes:

- Add a README with a reproducible run command.
- Document the six architectures and their call budgets.
- Document output artifacts.

### No Tests

There are no tests for core baseline behavior.

Recommended tests:

- answer normalization;
- deterministic vote aggregation;
- trace serialization;
- failed LLM calls;
- malformed JSON outputs;
- dataset loading;
- architecture call counts;
- metric applicability by architecture.

## Priority Fix List

### P0: Required for a Stable Baseline

1. Persist full traces for every case and architecture.
2. Add retry and structured error handling for LLM calls.
3. Replace non-deterministic tie-breaking.
4. Implement dataset-specific answer normalization.
5. Delay LLM client initialization until actual API calls.

### P1: Required for Reproducibility and Auditability

1. Save prompts, raw responses, parsed responses, token usage, and model config.
2. Record critic outputs in the trace.
3. Redesign workflow traces as pipeline stages rather than fake rounds.
4. Make metrics architecture-aware.
5. Report resource budgets per architecture.

### P2: Experimental Quality Improvements

1. Add schema validation for model outputs.
2. Separate prompt and completion token costs.
3. Improve CARA to use all relevant agent pairs.
4. Archive or remove Colab-exported scripts.
5. Add README and baseline tests.

## Final Assessment

The code is a reasonable prototype for comparing six medical reasoning architectures, but it currently lacks the controls needed for a stable, reproducible, and auditable baseline.

Before using it for serious experimental claims, the implementation should guarantee that:

- every model call is traceable;
- every intermediate agent output is persisted;
- failures are explicitly recorded;
- aggregation is deterministic;
- answer parsing is deterministic;
- metrics are only reported where they are semantically valid;
- prompts, configuration, and token usage are saved for reproduction.
