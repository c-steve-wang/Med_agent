from src.datastructures import MedicalCase, ExecutionTrace
from src.llm_client import MedicalLLMClient
import json
from typing import List, Dict, Any, Optional

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
            else:
                prompt += f"{case.options}\n"
        return prompt

    def _aggregate_votes(self, agent_outputs: Dict[str, Any]) -> str:
        votes = {}
        confidences = {}
        for agent_id, output in agent_outputs.items():
            if agent_id == "Skeptical_Reviewer":
                continue
            ans = str(output.get("final_answer", "UNKNOWN")).strip().upper()
            conf = float(output.get("confidence", 0.0))
            votes[ans] = votes.get(ans, 0) + 1
            confidences[ans] = confidences.get(ans, 0.0) + conf
        
        if not votes:
            return "UNKNOWN"
        max_votes = max(votes.values())
        candidates = [ans for ans, count in votes.items() if count == max_votes]
        
        if len(candidates) == 1:
            return candidates[0]
        
        candidates.sort(key=lambda x: (-confidences.get(x, 0.0), x))
        return candidates[0]

    def run_single_agent_cot(self, case: MedicalCase) -> ExecutionTrace:
        user_prompt = self._build_case_prompt(case)
        user_prompt += "\nPerform a systematic clinical evaluation. Detail your reasoning and commit to a final answer."
        parsed_json, p_tokens, c_tokens = self.client.call_agent(user_prompt, agent_role="Single-Agent CoT")
        cost = self.client.calculate_cost(p_tokens, c_tokens)
        return ExecutionTrace(case.case_id, "Single-Agent CoT", {"Agent_Primary": parsed_json}, {}, parsed_json.get("final_answer", "UNKNOWN"), p_tokens + c_tokens, cost)

    def independent_vote(self, case: MedicalCase) -> ExecutionTrace:
        agent_ids = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
        results = {}
        total_p_tokens, total_c_tokens = 0, 0
        base_case_prompt = self._build_case_prompt(case)
        
        personas = {
            "Agent_Alpha": "\nApproach this case prioritizing high-probability epidemiological occurrences and common guidelines.",
            "Agent_Beta": "\nApproach this case looking carefully for rare clinical secondary indications or atypical symptom expressions.",
            "Agent_Gamma": "\nApproach this case strictly from first-principles pathophysiology and metabolic pathway mechanisms."
        }

        for agent in agent_ids:
            r1_prompt = base_case_prompt + "\nPerform a systematic clinical evaluation. Detail your reasoning and commit to a final answer." + personas[agent]
            out, p_tokens, c_tokens = self.client.call_agent(r1_prompt, agent_role=f"Independent Voter ({agent})")
            results[agent] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens
            
        final_vote = self._aggregate_votes(results)
        cost = self.client.calculate_cost(total_p_tokens, total_c_tokens)
        return ExecutionTrace(case.case_id, "Independent Vote", results, {}, final_vote, total_p_tokens + total_c_tokens, cost)

    def run_symmetric_debate(self, case: MedicalCase) -> ExecutionTrace:
        agent_ids = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
        base_case_prompt = self._build_case_prompt(case)
        total_p_tokens, total_c_tokens = 0, 0
        r1_results, r2_results = {}, {}

        r1_prompt = base_case_prompt + "\nAnalyze the case independently."
        for agent in agent_ids:
            out, p_tokens, c_tokens = self.client.call_agent(r1_prompt, agent_role=f"Symmetric Debater ({agent})")
            r1_results[agent] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        for agent in agent_ids:
            peer_insights = "### ANONYMOUS PEER RATIONALES\n"
            for p_id, p_data in r1_results.items():
                if p_id == agent:
                    continue
                peer_insights += f"- Answer: {p_data.get('final_answer')}. Rationale: {p_data.get('reasoning')}\n"

            r2_prompt = base_case_prompt + f"\n{peer_insights}\nCritique peer perspectives and finalize your answer."
            out, p_tokens, c_tokens = self.client.call_agent(r2_prompt, agent_role=f"Symmetric Debater ({agent})")
            r2_results[agent] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        final_agg = self._aggregate_votes(r2_results)
        cost = self.client.calculate_cost(total_p_tokens, total_c_tokens)
        return ExecutionTrace(case.case_id, "Symmetric Debate", r1_results, r2_results, final_agg, total_p_tokens + total_c_tokens, cost)

    def run_role_specialist_board(self, case: MedicalCase) -> ExecutionTrace:
        roles = {
            "Agent_Diagnostician": "Diagnostician (focus on identifying the primary condition)",
            "Agent_Evidence": "Evidence/Pathophysiology Reviewer (focus on clinical mechanisms and grounding)",
            "Agent_Treatment": "Treatment/Safety Specialist (focus on management and contraindications)"
        }
        base_prompt = self._build_case_prompt(case)
        r1_results, r2_results = {}, {}
        total_p_tokens, total_c_tokens = 0, 0

        for agent_id, role_desc in roles.items():
            out, p_tokens, c_tokens = self.client.call_agent(base_prompt + f"\nAnalyze from your specific perspective: {role_desc}", agent_role=role_desc)
            r1_results[agent_id] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        peer_context = "\n".join([f"- {role}: {data.get('reasoning')}" for role, data in r1_results.items()])
        final_prompt = base_prompt + f"\n### PEER SPECIALIST INSIGHTS\n{peer_context}\nFinalize the diagnosis and treatment plan."

        for agent_id, role_desc in roles.items():
            out, p_tokens, c_tokens = self.client.call_agent(final_prompt, agent_role=role_desc)
            r2_results[agent_id] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        final_agg = self._aggregate_votes(r2_results)
        cost = self.client.calculate_cost(total_p_tokens, total_c_tokens)
        return ExecutionTrace(case.case_id, "Role-Specialist Board", r1_results, r2_results, final_agg, total_p_tokens + total_c_tokens, cost)

    def run_critic_reviewer_board(self, case: MedicalCase) -> ExecutionTrace:
        solvers = ["Solver_A", "Solver_B"]
        base_prompt = self._build_case_prompt(case)
        total_p_tokens, total_c_tokens = 0, 0
        r1_results, r2_results = {}, {}

        for s_id in solvers:
            out, p_tokens, c_tokens = self.client.call_agent(base_prompt, agent_role="Clinical Solver")
            r1_results[s_id] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        critic_prompt = base_prompt + "\n### PROPOSED SOLUTIONS\n" + "\n".join([f"- {s}: {r1_results[s].get('reasoning')}" for s in solvers])
        critic_prompt += "\nIdentify contradictions, unsupported claims, or missing evidence."
        critic_out, cp_tokens, cc_tokens = self.client.call_agent(critic_prompt, agent_role="Skeptical Clinical Reviewer")
        total_p_tokens += cp_tokens
        total_c_tokens += cc_tokens

        r1_results["Skeptical_Reviewer"] = critic_out

        # Round 2 Solver Revisions
        revision_prompt = base_prompt + f"\n### CRITIC REVIEW\n{critic_out.get('reasoning')}\nRevise your initial answer."
        for s_id in solvers:
            out, p_tokens, c_tokens = self.client.call_agent(revision_prompt, agent_role="Clinical Solver (Revised)")
            r2_results[s_id] = out
            total_p_tokens += p_tokens
            total_c_tokens += c_tokens

        final_votes = [r2_results[s].get("final_answer") for s in solvers]
        final_agg = final_votes[0] if len(set(final_votes)) == 1 else critic_out.get("final_answer", final_votes[0])
        cost = self.client.calculate_cost(total_p_tokens, total_c_tokens)
        return ExecutionTrace(case.case_id, "Critic-Reviewer Board", r1_results, r2_results, final_agg, total_p_tokens + total_c_tokens, cost)

    def run_workflow_orchestrator(self, case: MedicalCase) -> ExecutionTrace:
        base_prompt = self._build_case_prompt(case)
        total_p_tokens, total_c_tokens = 0, 0

        extractor_prompt = base_prompt + "\nExtract only relevant clinical evidence and facts from the text."
        extraction, p1, c1 = self.client.call_agent(extractor_prompt, agent_role="Evidence Extractor")
        total_p_tokens += p1
        total_c_tokens += c1

        solver_prompt = f"### EXTRACTED EVIDENCE\n{extraction.get('cited_evidence')}\n{base_prompt}\nSolve the case."
        final_out, p2, c2 = self.client.call_agent(solver_prompt, agent_role="Diagnostic Solver")
        total_p_tokens += p2
        total_c_tokens += c2

        cost = self.client.calculate_cost(total_p_tokens, total_c_tokens)
        return ExecutionTrace(case.case_id, "Workflow-Orchestrator", {"Extractor": extraction}, {"Solver": final_out}, final_out.get("final_answer"), total_p_tokens + total_c_tokens, cost)