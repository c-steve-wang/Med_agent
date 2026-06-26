from src.datastructures import MedicalCase, ExecutionTrace
from src.llm_client import MedicalLLMClient
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


class AuditPipelineOrchestrator:
    def __init__(self, client: MedicalLLMClient):
        self.client = client

    def _build_case_prompt(self, case: MedicalCase) -> str:
        prompt = f"### CLINICAL CASE CONTEXT\n{case.case_text}\n\n"
        if case.evidence_context:
            prompt += f"### GROUNDING EVIDENCE/ABSTRACT\n{case.evidence_context}\n\n"
        if case.options:
            prompt += f"### OPTIONS\n"
            if isinstance(case.options, dict):
                for key, val in case.options.items():
                    prompt += f"{key}: {val}\n"
            elif isinstance(case.options, list):
                for option_val in case.options:
                    prompt += f"- {option_val}\n"
            else: # Fallback for unexpected types
                prompt += f"{case.options}\n"
        return prompt

    def run_single_agent_cot(self, case: MedicalCase) -> ExecutionTrace:
        user_prompt = self._build_case_prompt(case)
        user_prompt += "\nPerform a systematic clinical evaluation. Detail your reasoning and commit to a final answer."
        parsed_json, tokens = self.client.call_agent(user_prompt, agent_role="Single-Agent CoT")
        cost = tokens * self.client.output_token_cost
        return ExecutionTrace(case.case_id, "Single-Agent CoT", {"Agent_Primary": parsed_json}, {}, parsed_json.get("final_answer", "UNKNOWN"), tokens, cost)

    def independent_vote(self, case: MedicalCase) -> ExecutionTrace:
        agent_ids = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
        results = {}
        total_tokens = 0 # Initialize total_tokens
        base_case_prompt = self._build_case_prompt(case)
        r1_prompt= base_case_prompt + "\nPerform a systematic clinical evaluation. Detail your reasoning and commit to a final answer."

        for agent in agent_ids:
            out, tokens = self.client.call_agent(r1_prompt, agent_role=f"Symmetric Debater ({agent})")
            results[agent] = out
            total_tokens += tokens
        votes = [results[a].get("final_answer") for a in agent_ids]
        final_vote= max(set(votes), key=votes.count)
        cost= total_tokens*self.client.output_token_cost
        return ExecutionTrace(case.case_id, "Independent Vote", results, {}, final_vote, total_tokens, cost)

    def run_symmetric_debate(self, case: MedicalCase) -> ExecutionTrace:
        agent_ids = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
        base_case_prompt = self._build_case_prompt(case)
        total_tokens = 0
        r1_results, r2_results = {}, {}

        r1_prompt = base_case_prompt + "\nAnalyze the case independently."
        for agent in agent_ids:
            out, tokens = self.client.call_agent(r1_prompt, agent_role=f"Symmetric Debater ({agent})")
            r1_results[agent] = out
            total_tokens += tokens

        peer_insights = "### ANONYMOUS PEER RATIONALES\n"
        for p_id, p_data in r1_results.items():
            peer_insights += f"- Answer: {p_data.get('final_answer')}. Rationale: {p_data.get('reasoning')}\n"

        for agent in agent_ids:
            r2_prompt = base_case_prompt + f"\n{peer_insights}\nCritique peer perspectives and finalize your answer."
            out, tokens = self.client.call_agent(r2_prompt, agent_role=f"Symmetric Debater ({agent})")
            r2_results[agent] = out
            total_tokens += tokens

        r2_votes = [r2_results[a].get("final_answer") for a in agent_ids]
        final_agg = max(set(r2_votes), key=r2_votes.count)
        cost = total_tokens * self.client.output_token_cost
        return ExecutionTrace(case.case_id, "Symmetric Debate", r1_results, r2_results, final_agg, total_tokens, cost)

    def run_role_specialist_board(self, case: MedicalCase) -> ExecutionTrace:
        roles = {
            "Agent_Diagnostician": "Diagnostician (focus on identifying the primary condition)",
            "Agent_Evidence": "Evidence/Pathophysiology Reviewer (focus on clinical mechanisms and grounding)",
            "Agent_Treatment": "Treatment/Safety Specialist (focus on management and contraindications)"
        }
        base_prompt = self._build_case_prompt(case)
        r1_results, r2_results = {}, {}
        total_tokens = 0

        # Round 1: Initial specialized analysis
        for agent_id, role_desc in roles.items():
            out, tokens = self.client.call_agent(base_prompt + f"\nAnalyze from your specific perspective: {role_desc}", agent_role=role_desc)
            r1_results[agent_id] = out
            total_tokens += tokens

        # Round 2: Final revision after seeing peers
        peer_context = "\n".join([f"- {role}: {data.get('reasoning')}" for role, data in r1_results.items()])
        final_prompt = base_prompt + f"\n### PEER SPECIALIST INSIGHTS\n{peer_context}\nFinalize the diagnosis and treatment plan."

        for agent_id, role_desc in roles.items():
            out, tokens = self.client.call_agent(final_prompt, agent_role=role_desc)
            r2_results[agent_id] = out
            total_tokens += tokens

        votes = [r2_results[a].get("final_answer") for a in roles]
        final_agg = max(set(votes), key=votes.count)
        return ExecutionTrace(case.case_id, "Role-Specialist Board", r1_results, r2_results, final_agg, total_tokens, total_tokens * self.client.output_token_cost)

    def run_critic_reviewer_board(self, case: MedicalCase) -> ExecutionTrace:
        solvers = ["Solver_A", "Solver_B"]
        base_prompt = self._build_case_prompt(case)
        total_tokens = 0
        r1_results, r2_results = {}, {}

        # Round 1: Initial Solvers
        for s_id in solvers:
            out, tokens = self.client.call_agent(base_prompt, agent_role="Clinical Solver")
            r1_results[s_id] = out
            total_tokens += tokens

        # Critic Analysis
        critic_prompt = base_prompt + "\n### PROPOSED SOLUTIONS\n" + "\n".join([f"- {s}: {r1_results[s].get('reasoning')}" for s in solvers])
        critic_prompt += "\nIdentify contradictions, unsupported claims, or missing evidence."
        critic_out, c_tokens = self.client.call_agent(critic_prompt, agent_role="Skeptical Clinical Reviewer")
        total_tokens += c_tokens

        # Round 2: Revisions based on critique
        revision_prompt = base_prompt + f"\n### CRITIC REVIEW\n{critic_out.get('reasoning')}\nRevise your initial answer."
        for s_id in solvers:
            out, tokens = self.client.call_agent(revision_prompt, agent_role="Clinical Solver (Revised)")
            r2_results[s_id] = out
            total_tokens += tokens

        # Reviewer breaks tie or selects best
        final_votes = [r2_results[s].get("final_answer") for s in solvers]
        final_agg = final_votes[0] if len(set(final_votes)) == 1 else critic_out.get("final_answer", final_votes[0])
        return ExecutionTrace(case.case_id, "Critic-Reviewer Board", r1_results, r2_results, final_agg, total_tokens, total_tokens * self.client.output_token_cost)

    def run_workflow_orchestrator(self, case: MedicalCase) -> ExecutionTrace:
        base_prompt = self._build_case_prompt(case)
        total_tokens = 0

        # Step 1: Evidence Extraction
        extractor_prompt = base_prompt + "\nExtract only relevant clinical evidence and facts from the text."
        extraction, t1 = self.client.call_agent(extractor_prompt, agent_role="Evidence Extractor")
        total_tokens += t1

        # Step 2: Diagnostic Solving
        solver_prompt = f"### EXTRACTED EVIDENCE\n{extraction.get('cited_evidence')}\n{base_prompt}\nSolve the case."
        final_out, t2 = self.client.call_agent(solver_prompt, agent_role="Diagnostic Solver")
        total_tokens += t2

        return ExecutionTrace(case.case_id, "Workflow-Orchestrator", {"Extractor": extraction}, {"Solver": final_out}, final_out.get("final_answer"), total_tokens, total_tokens * self.client.output_token_cost)

def save_trace_to_jsonl(trace: ExecutionTrace, output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trace)) + "\n")