import argparse
import os
import json
import datetime
import pandas as pd
import numpy as np

from src.llm_client import MedicalLLMClient
from src.pipelines import AuditPipelineOrchestrator
from src.evaluators import load_qa_dataset, load_pubmedqa_dataset, MedicalAuditEvaluator
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Automating Expert-Level Medical Reasoning Evaluation")
    
    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini', help='Target Model ID')
    parser.add_argument('--methods', type=str, nargs='+', default=['cot'], help='List of multi-agent architectures to run (e.g., cot vote)')
    parser.add_argument('--eval', action='store_true', help='Execute evaluation pipeline')
    parser.add_argument('--judge_method', type=str, default='llm-explain-with-reference', help='Judge style parameter')
    parser.add_argument('--judge_model', type=str, default='openai/gpt-4o-mini', help='Evaluation model engine')
    parser.add_argument('--judge_kwargs', type=str, default='{"temperature":0.0}', help='JSON string config for judge configuration')
    parser.add_argument('--dataset', type=str, choices=['qa', 'pubmedqa'], default='qa', help='Target evaluation dataset')

    args = parser.parse_args()
    judge_config = json.loads(args.judge_kwargs)

    print(f"Initializing Evaluation Process | Target Model: {args.model} | Dataset: {args.dataset}")
    print(f"Methods scheduled to run: {', '.join(args.methods)}")
    
    # Load dataset based on target flag
    if args.dataset == 'qa':
        cases = load_qa_dataset()  # Restricted slices for safe evaluation loops
    else:
        cases = load_pubmedqa_dataset()

    client = MedicalLLMClient(model_name=args.model)
    orchestrator = AuditPipelineOrchestrator(client)

    # Dictionary to hold separate evaluators for each architecture method run
    all_summaries = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    #specialized_board critic workflow
    method_mapping = {
        'cot': orchestrator.run_single_agent_cot,
        'vote': orchestrator.independent_vote,
        'symmetric_debate': orchestrator.run_symmetric_debate,
        'specialized_board': orchestrator.run_role_specialist_board,
        'critic' : orchestrator.run_critic_reviewer_board,
        'workflow' : orchestrator.run_workflow_orchestrator
        # You can add 'debate', 'board', etc. here once fully declared in your pipelines.py
    }

    # LOOP through each method specified in the terminal command
    for method_name in args.methods:
        if method_name not in method_mapping:
            print(f"⚠️ Warning: Method '{method_name}' is not recognized or mapped. Skipping.")
            continue
            
        print(f"\n--- 🚀 Running Pipeline Architecture: {method_name} ---")
        run_pipeline = method_mapping[method_name]
        evaluator = MedicalAuditEvaluator(judge_model=args.judge_model, temperature=judge_config.get('temperature', 0.0))
        detailed_results = []

        for case in cases:
            print(f"[{method_name.upper()}] Processing Case {case.case_id}...")
            trace = run_pipeline(case)
            
            if args.eval:
                metrics = evaluator.calculate_metrics(trace, case.gold_label)
                # Keep track of which method produced this metric row
                cara_res = evaluator.run_cara_llm_evaluation(case, trace)
                
                #Overwrite the default None/0 placeholders with actual scores
                metrics["cara_llm_score"] = cara_res["cara_llm_score"]
                metrics["cara_llm_sd"] = cara_res["cara_llm_sd"]
                metrics["cara_llm_cost"] = cara_res["cara_llm_cost"]
                metrics["method_architecture"] = method_name
                detailed_results.append(metrics)

        if args.eval:
            # Store the aggregate summary for our final multi-summary JSON
            all_summaries[method_name] = evaluator.get_summary()
            
            # Save the individual detailed Excel workbook sheet just for this method run
            df_results = pd.DataFrame(detailed_results)
            df_results.to_excel(f"logs/eval_result_{method_name}_{timestamp}.xlsx", index=False)

    # Save all calculated summary insights into a unified run log
    if args.eval and all_summaries:
        with open(f"logs/eval_summary_multi_{timestamp}.json", "w") as f:
            json.dump(all_summaries, f, indent=2, cls=NpEncoder)
        print(f"\n All methods successfully executed! Global summaries saved to: logs/eval_summary_multi_{timestamp}.json")

if __name__ == "__main__":
    main()