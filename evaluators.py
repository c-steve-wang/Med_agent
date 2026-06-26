import json
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from src.datastructures import MedicalCase, ExecutionTrace
from src.llm_client import MedicalLLMClient

def load_qa_dataset(path='data/QA_data.json'):
    with open(path, 'r',encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for item in data:
        gold_label = item.get('answer', ' ')[0]
        cases.append(MedicalCase(
            case_id=str(item.get('Index')),
            dataset_name="Drive_QA",
            case_text=item.get('question'),
            options=None,
            evidence_context=" ".join(item.get('Scoring_Points', [])),
            gold_label=gold_label
        ))
    return cases

def load_pubmedqa_dataset(questions_path='data/ori_pqal.json', ground_truth_path='data/test_ground_truth.json'):
    with open(questions_path, 'r', encoding="utf-8") as f_q:
        questions_data = json.load(f_q)
    with open(ground_truth_path, 'r', encoding="utf-8") as f_gt:
        ground_truth_data = json.load(f_gt)
    
    cases = []
    for q_id, q_info in questions_data.items():
        gold_label = ground_truth_data.get(q_id)
        if gold_label is None: continue
        cases.append(MedicalCase(
            case_id=str(q_id),
            dataset_name="PubMedQA",
            case_text=q_info.get('QUESTION'),
            options=["yes","no","maybe"],
            evidence_context=" ".join(q_info.get('CONTEXTS', [])),
            gold_label=gold_label
        ))
    return cases

class MedicalAuditEvaluator:
    """
    Implementation of Core Metrics for Multi-Agent Clinical Analysis.
    """

    # CARA LLM Constants
    _CARA_SYSTEM_PROMPT = """You are an expert evaluator assessing the reasoning alignment between two AI medical agents.\n\nYou will be given:\n1. A medical question with answer choices\n2. The reasoning traces from two agents (they may or may not have selected the same final answer)\n\nYour task: Rate how well the two agents' reasoning processes align with each other on a scale from 1 to 5.\n\nScoring rubric:\n- 5 (Perfect alignment): Both agents follow essentially the same reasoning path, cite the same medical facts, and reach the answer through identical logic.\n- 4 (Strong alignment): Both agents share the core reasoning pathway with minor differences in emphasis or supporting details.\n- 3 (Moderate alignment): The agents share some reasoning elements but take partially different paths. Some overlap, some divergence.\n- 2 (Weak alignment): The agents reason through substantially different paths. Different medical facts, different logical chains, minimal overlap.\n- 1 (No alignment / Contradictory): The agents' reasoning contradicts each other, or they rely on completely incompatible explanations.\n\nYour response must be a JSON object with a 'SCORE' field (integer 1-5) and a 'JUSTIFICATION' field (1-2 sentences). Do not include any other text or markdown formatting.\nExample:\n{\n  "SCORE": 3,\n  "JUSTIFICATION": "Both agents considered patient history but diverged on treatment options."
}"""

    _CARA_USER_TEMPLATE_AGREE = """## Medical Question\n{question_text}\n\n## Gold Answer\n{gold_answer}\n\n## Agents' Shared Answer\n{majority_answer}\n\n## Agent A Reasoning ({agent_a_id})\n{agent_a_reasoning}\n\n## Agent B Reasoning ({agent_b_id})\n{agent_b_reasoning}\n\nRate the reasoning alignment between Agent A and Agent B (1-5):"""

    _CARA_USER_TEMPLATE_DISAGREE = """## Medical Question\n{question_text}\n\n## Gold Answer\n{gold_answer}\n\n## Agent A Answer: {answer_a}\n## Agent A Reasoning ({agent_a_id})\n{agent_a_reasoning}\n\n## Agent B Answer: {answer_b}\n## Agent B Reasoning ({agent_b_id})\n{agent_b_reasoning}\n\nRate the reasoning alignment between Agent A and Agent B (1-5):"""

    MAX_RETRIES = 3
    RETRY_DELAY = 5 # seconds
    NUM_CARA_RUNS = 3 # score each pair N times, take mean

    def __init__(self, judge_model: str = "openai/gpt-4o-mini", temperature: float = 0.0):
        self.results = []
        # Use the dynamically passed judge_model and temperature values
        self.llm_client_for_cara = MedicalLLMClient(
            model_name=judge_model, 
            temperature=temperature
        ) 
    def _parse_cara_response(self, text: str) -> tuple:
        """Parse SCORE and JUSTIFICATION from LLM response."""
        try:
            parsed_json = json.loads(text)
            score = parsed_json.get('SCORE')
            justification = parsed_json.get('JUSTIFICATION', '')
            if score is None or not isinstance(score, int) or score < 1 or score > 5:
                return None, f"Failed to parse valid score from JSON: {text}"
            return score, justification
        except json.JSONDecodeError:
            return None, f"Failed to decode JSON from response: {text[:200]}"

    def _build_cara_user_prompt(self, case: MedicalCase, agent_a_id: str, agent_a_output: Dict[str, Any], agent_b_id: str, agent_b_output: Dict[str, Any]) -> str:
        agent_a_reasoning_steps = agent_a_output.get('reasoning', '').split('\n') if agent_a_output.get('reasoning') else []
        agent_b_reasoning_steps = agent_b_output.get('reasoning', '').split('\n') if agent_b_output.get('reasoning') else []

        agent_a_reasoning_formatted = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(agent_a_reasoning_steps))
        agent_b_reasoning_formatted = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(agent_b_reasoning_steps))

        is_disagree = agent_a_output.get('final_answer') != agent_b_output.get('final_answer')

        if is_disagree:
            return self._CARA_USER_TEMPLATE_DISAGREE.format(
                question_text=case.case_text,
                gold_answer=case.gold_label,
                answer_a=agent_a_output.get('final_answer'),
                agent_a_id=agent_a_id,
                agent_a_reasoning=agent_a_reasoning_formatted,
                answer_b=agent_b_output.get('final_answer'),
                agent_b_id=agent_b_id,
                agent_b_reasoning=agent_b_reasoning_formatted,
            )
        else:
            return self._CARA_USER_TEMPLATE_AGREE.format(
                question_text=case.case_text,
                gold_answer=case.gold_label,
                majority_answer=agent_a_output.get('final_answer'), # Or agent_b_output.get('final_answer'), as they agree
                agent_a_id=agent_a_id,
                agent_a_reasoning=agent_a_reasoning_formatted,
                agent_b_id=agent_b_id,
                agent_b_reasoning=agent_b_reasoning_formatted,
            )

    def _call_cara_llm_once(self, case: MedicalCase, agent_a_id: str, agent_a_output: Dict[str, Any], agent_b_id: str, agent_b_output: Dict[str, Any]) -> dict:
        user_prompt = self._build_cara_user_prompt(case, agent_a_id, agent_a_output, agent_b_id, agent_b_output)

        for attempt in range(self.MAX_RETRIES):
            try:
                text, tokens_used = self.llm_client_for_cara.call_free_form_llm(self._CARA_SYSTEM_PROMPT, user_prompt)
                score, justification = self._parse_cara_response(text)

                if score is None:
                    # log.warning(f"  CARA: parse error (attempt {attempt+1}): {justification}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY)
                        continue
                    return {"score": None, "justification": justification, "raw": text, "usage": {"tokens": tokens_used}, "error": "parse_failed"}

                return {"score": score, "justification": justification, "usage": {"tokens": tokens_used}, "error": None}

            except Exception as e:
                # log.warning(f"  CARA: API error (attempt {attempt+1}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    return {"score": None, "justification": "", "usage": {"tokens": tokens_used}, "error": str(e)}

    def run_cara_llm_evaluation(self, case: MedicalCase, trace: ExecutionTrace) -> dict:
        # CARA LLM is designed for comparing two agents. We'll take the first two agents
        # from Round 1 outputs if available.
        agent_ids = list(trace.round_1_outputs.keys())
        if len(agent_ids) < 2:
            # If less than 2 agents, CARA LLM comparison is not applicable.
            return {"cara_llm_score": None, "cara_llm_sd": None, "cara_llm_cost": 0, "n_cara_runs": 0}

        agent_a_id = agent_ids[0]
        agent_b_id = agent_ids[1]
        agent_a_output = trace.round_1_outputs[agent_a_id]
        agent_b_output = trace.round_1_outputs[agent_b_id]

        scores = []
        total_cara_tokens = 0

        for run_idx in range(self.NUM_CARA_RUNS):
            res = self._call_cara_llm_once(case, agent_a_id, agent_a_output, agent_b_id, agent_b_output)
            if res["score"] is not None:
                scores.append(res["score"])
            total_cara_tokens += res["usage"].get("tokens", 0)

        if scores:
            mean_score = np.mean(scores)
            sd_score = np.std(scores) if len(scores) > 1 else 0.0
            cost = total_cara_tokens * self.llm_client_for_cara.output_token_cost
            return {
                "cara_llm_score": round(mean_score, 4),
                "cara_llm_sd": round(sd_score, 4),
                "cara_llm_cost": cost,
                "n_cara_runs": len(scores)
            }
        else:
            return {"cara_llm_score": None, "cara_llm_sd": None, "cara_llm_cost": total_cara_tokens * self.llm_client_for_cara.output_token_cost, "n_cara_runs": 0}

    def calculate_metrics(self, trace: ExecutionTrace, gold_label: str):
        """Processes a single trace to compute metrics."""
        # 1. Final Accuracy
        is_correct = trace.aggregated_answer.strip().startswith(gold_label)

        # 2. Answer Agreement (Round 1 vs Round 2)
        r1_answers = [out.get('final_answer', '') for out in trace.round_1_outputs.values()]
        r2_answers = [out.get('final_answer', '') for out in trace.round_2_outputs.values()]

        def get_agreement(answers):
            if not answers: return 1.0
            pairs = 0
            matches = 0
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    pairs += 1
                    if answers[i] == answers[j]: matches += 1
            return matches / pairs if pairs > 0 else 1.0

        agreement_r1 = get_agreement(r1_answers)
        agreement_r2 = get_agreement(r2_answers) if r2_answers else agreement_r1

        # 3. Evidence Overlap (Simple F1 between agents)
        all_evidence = [set(out.get('cited_evidence', [])) for out in trace.round_1_outputs.values()]
        # Simplified: Average pairwise Jaccard similarity as overlap proxy
        overlaps = []
        for i in range(len(all_evidence)):
            for j in range(i + 1, len(all_evidence)):
                union = len(all_evidence[i] | all_evidence[j])
                intersection = len(all_evidence[i] & all_evidence[j])
                overlaps.append(intersection / union if union > 0 else 1.0)
        avg_evidence_overlap = np.mean(overlaps) if overlaps else 1.0

        # 4. Revision Utility
        utility = 0
        # Only calculate if there are round_2_outputs and the set of agent IDs in both rounds are the same
        if trace.round_2_outputs and set(trace.round_1_outputs.keys()) == set(trace.round_2_outputs.keys()):
            for agent_id in trace.round_1_outputs:
                r1_correct = trace.round_1_outputs[agent_id].get('final_answer', '').strip().startswith(gold_label)
                r2_correct = trace.round_2_outputs[agent_id].get('final_answer', '').strip().startswith(gold_label)
                if not r1_correct and r2_correct: utility += 1
                if r1_correct and not r2_correct: utility -= 1

        # 5. Confidence Calibration
        confidences = [out.get('confidence', 0) for out in trace.round_1_outputs.values()]
        avg_conf = np.mean(confidences)

        # 6. CARA LLM Evaluation (Reasoning Alignment)
        # Note: The cara_llm_score, sd, and cost are now being passed in from the orchestrator loop.
        # This method's purpose is to calculate *metrics from the trace itself*, not to run LLM calls.

        metrics = {
            "case_id": trace.case_id,
            "correct": is_correct,
            "agreement_r1": agreement_r1,
            "agreement_r2": agreement_r2,
            "evidence_overlap": avg_evidence_overlap,
            "revision_utility": utility,
            "avg_confidence": avg_conf,
            "cara_llm_score": None, # These will be overwritten by the orchestrator after this call.
            "cara_llm_sd": None,
            "cara_llm_cost": 0,
            "cost": trace.estimated_cost
        }
        self.results.append(metrics)
        return metrics

    def get_summary(self):
        df = pd.DataFrame(self.results)
        summary = {
            "Mean Accuracy": df['correct'].mean(),
            "Consensus Shift": df['agreement_r2'].mean() - df['agreement_r1'].mean(),
            "Avg Evidence Overlap": df['evidence_overlap'].mean(),
            "Total Revision Utility": df['revision_utility'].sum(),
            "Mean CARA LLM Score": df['cara_llm_score'].mean() if 'cara_llm_score' in df.columns else None,
            "Total CARA LLM Cost": df['cara_llm_cost'].sum() if 'cara_llm_cost' in df.columns else 0,
            "Total Cost": df['cost'].sum()
        }
        return summary