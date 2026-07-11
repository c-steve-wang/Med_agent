import argparse
import os
import json
import datetime
from dataclasses import asdict
import pandas as pd
import numpy as np

from src.llm_client import MedicalLLMClient
from src.pipelines import AuditPipelineOrchestrator
from src.evaluators import load_qa_dataset, load_pubmedqa_dataset, load_qausmle_dataset, MedicalAuditEvaluator
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

import os
import json
import datetime
import time
import random
import argparse
import pandas as pd
from dataclasses import asdict

def retry_with_exponential_backoff(func, *args, max_retries=500, base_delay=2, **kwargs):
    """
    Executes a function and retries with exponential backoff + jitter 
    if an exception (like a network timeout or connection drop) is raised.
    """
    retries = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1 
            if retries >= max_retries:
                print(f"\n❌ Permanent failure after {max_retries} attempts: {e}")
                raise e
            
            # Calculate exponential backoff delay with a slight random jitter to prevent thundering herds
            delay = base_delay  + random.uniform(0, 1)
            print(f"\n⚠️ Connection issue or API error encountered: {e}")
            print(f"🔄 Retrying {func.__name__} in {delay:.2f} minutes... (Attempt {retries}/{max_retries})")
            time.sleep(delay*95)


def main():

    
    parser = argparse.ArgumentParser(description="Automating Expert-Level Medical Reasoning Evaluation")
    
    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini', help='Target Model ID')
    parser.add_argument('--methods', type=str, nargs='+', default=['cot'], help='List of multi-agent architectures to run (e.g., cot vote)')
    parser.add_argument('--eval', action='store_true', help='Execute evaluation pipeline')
    parser.add_argument('--judge_method', type=str, default='llm-explain-with-reference', help='Judge style parameter')
    parser.add_argument('--judge_model', type=str, default='openai/gpt-4o-mini', help='Evaluation model engine')
    parser.add_argument('--judge_kwargs', type=str, default='{"temperature":0.0}', help='JSON string config for judge configuration')
    parser.add_argument('--dataset', type=str, nargs='+', choices=['qa', 'pubmedqa', 'qausmle'], default='qa', help='Target evaluation dataset')

    args = parser.parse_args()
    judge_config = json.loads(args.judge_kwargs)

    print(f"Initializing Evaluation Process | Target Model: {args.model} | Dataset: {args.dataset}")
    print(f"Methods scheduled to run: {', '.join(args.methods)}")
    
    # Load dataset based on target flag
    if args.dataset == 'qa':
        cases = load_qa_dataset()  # Restricted slices for safe evaluation loops
    elif args.dataset == 'pubmedqa':
        cases = load_pubmedqa_dataset()
    elif args.dataset == 'qausmle':
        cases = load_qausmle_dataset()  # Placeholder for future custom dataset loader

    client = MedicalLLMClient(model_name=args.model)
    orchestrator = AuditPipelineOrchestrator(client)

    all_summaries = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    
    method_mapping = {
        'cot': orchestrator.run_single_agent_cot,
        'vote': orchestrator.independent_vote,
        'symmetric_debate': orchestrator.run_symmetric_debate,
        'specialized_board': orchestrator.run_role_specialist_board,
        'critic': orchestrator.run_critic_reviewer_board,
        'workflow': orchestrator.run_workflow_orchestrator
    }

    dataset_mapping ={

        'qa': load_qa_dataset,
        'pubmedqa': load_pubmedqa_dataset,  
        'qausmle': load_qausmle_dataset
    }

    # LOOP through each method specified in the terminal command
    for dataset_name in args.dataset:
        if(dataset_name == "qa"):
            cases= dataset_mapping[dataset_name]()[113:120] + dataset_mapping[dataset_name]()[492:494]  # Restricted slices for safe evaluation loops
        else:
            cases= dataset_mapping[dataset_name]()
        for method_name in args.methods:
            if method_name not in method_mapping:
                print(f"Warning: Method '{method_name}' is not recognized or mapped. Skipping.")
                continue
                
            print(f"\n--- Running Pipeline Architecture: {method_name} ---")
            run_pipeline = method_mapping[method_name]
            evaluator = MedicalAuditEvaluator(judge_model=args.judge_model, temperature=judge_config.get('temperature', 0.0))
            detailed_results = []
            
            os.makedirs('logs/execution_traces', exist_ok=True)
            jsonl_path = f"logs/execution_traces/traces_new_{dataset_name}_{method_name}.jsonl"
            
            for case in cases:
                print(f"[{method_name.upper()}] Processing Case {case.case_id}...")
                
                # Protect the primary Multi-Agent execution pipeline from network failure
                trace = retry_with_exponential_backoff(run_pipeline, case)
                
                try:
                    with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                        trace_dict = asdict(trace)
                        f_jsonl.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")
                    print(f" Trace appended to master log: {jsonl_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save execution trace JSONL row: {e}")
                
                # =============================================================
                if args.eval:
                    metrics = evaluator.calculate_metrics(trace, case.gold_label)
                    
                    # Protect the LLM-as-a-judge CARA evaluation from network failure
                    cara_res = retry_with_exponential_backoff(evaluator.run_cara_llm_evaluation, case, trace)
                    
                    # Overwrite placeholders using the updated pairwise key outputs
                    metrics["cara_pairwise_results"] = cara_res["pairwise_results"]
                    metrics["cara_llm_cost"] = cara_res["total_cara_cost"]
                    metrics["method_architecture"] = method_name
                    
                    detailed_results.append(metrics)
            
            if args.eval:
                # Store the aggregate summary for our final multi-summary JSON
                all_summaries[method_name] = evaluator.get_summary()
                
                # Save individual detailed Excel workbook sheet for this method run
                df_results = pd.DataFrame(detailed_results)
                os.makedirs(f"logs/{args.dataset}", exist_ok=True)
                df_results.to_excel(f"logs/{args.dataset}/eval_result_{method_name}_{timestamp}.xlsx", index=False)
        
        # Save all calculated summary insights into a unified run log
        if args.eval and all_summaries:
            os.makedirs(f"logs/{args.dataset}", exist_ok=True)
            with open(f"logs/{args.dataset}/eval_summary_multi_{timestamp}.json", "w") as f:
                json.dump(all_summaries, f, indent=2, cls=NpEncoder)
            print(f"\n All methods successfully executed! Global summaries saved to: logs/{args.dataset}/eval_summary_multi_{timestamp}.json")
        """
        """
        """
        print("\n--- 📊 Aggregating Results Across All Executed Methods ---")
        methods= {'cot', 'vote', 'symmetric_debate', 'specialized_board', 'critic', 'workflow'}
        all_summaries = {}
        for method_name in methods:

            # Define the exact file path that was saved for this specific method
            excel_path = f"logs/{args.dataset}/{method_name}.xlsx"
            
            if os.path.exists(excel_path):
                try:
                    # Read the excel sheet back into a DataFrame
                    df_method = pd.read_excel(excel_path)
                    
                    if not df_method.empty:
                        # Re-calculate the custom summary metrics for this specific sheet
                        all_summaries[method_name] = {
                            "Mean Accuracy": float(df_method['correct'].mean()) if 'correct' in df_method.columns else 0.0,
                            "Consensus Shift": float(df_method['agreement_r2'].mean() - df_method['agreement_r1'].mean()) if ('agreement_r2' in df_method.columns and 'agreement_r1' in df_method.columns) else 0.0,
                            "Avg Evidence Overlap": float(df_method['evidence_overlap'].mean()) if 'evidence_overlap' in df_method.columns else 0.0,
                            "Total Revision Utility": int(df_method['revision_utility'].sum()) if 'revision_utility' in df_method.columns else 0,
                            "Mean CARA LLM Score": float(df_method['cara_llm_score'].dropna().mean()) if ('cara_llm_score' in df_method.columns and not df_method['cara_llm_score'].dropna().empty) else None,
                            "Total CARA LLM Cost": float(df_method['cara_llm_cost'].sum()) if 'cara_llm_cost' in df_method.columns else 0.0,
                            "Total Cost": float(df_method['cost'].sum()) if 'cost' in df_method.columns else 0.0
                        }
                except Exception as e:
                    print(f"⚠️ Warning: Could not process file for {method_name}: {e}")
    
    # Save the structured multi-summary into a single JSON file
    if all_summaries:
        summary_path = f"logs/{args.dataset}/eval_summary_multi_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"✅ Master summary compiled and saved to: {summary_path}")
    """
   
if __name__ == "__main__":
    main()