import os
import json
import time
import random
import openai
from dotenv import load_dotenv
from typing import Dict, Any, Tuple

load_dotenv()

_OR_CLIENT = None

def get_or_client():
    global _OR_CLIENT
    if _OR_CLIENT is None:
        api_key = os.getenv('OPENROUTER_API_KEY')
        _OR_CLIENT = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key if api_key else "DUMMY_KEY",
        )
    return _OR_CLIENT

def call_real_llm(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o-mini", temperature: float = 0.4):
    max_retries = 5
    client = get_or_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/your-username/medical-agent-audit",
                    "X-Title": "Medical Agent Audit",
                },
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=temperature
            )
            raw_text = response.choices[0].message.content
            p_tokens = response.usage.prompt_tokens
            c_tokens = response.usage.completion_tokens
            return json.loads(raw_text), p_tokens, c_tokens
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))

class MedicalLLMClient:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0.4):
        self.model_name = model_name
        self.temperature = temperature

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing_table = {
            "openai/gpt-4o-mini": {"input": 0.15 / 1000000, "output": 0.60 / 1000000},
            "openai/gpt-4o": {"input": 2.50 / 1000000, "output": 10.00 / 1000000},
            "deepseek/deepseek-r1": {"input": 0.55 / 1000000, "output": 2.19 / 1000000},
        }
        prices = pricing_table.get(self.model_name, {"input": 2.50 / 1000000, "output": 10.00 / 1000000})
        return (prompt_tokens * prices["input"]) + (completion_tokens * prices["output"])

    def _get_system_prompt(self) -> str:
        return """You are an expert, board-certified clinical AI assistant.
Your analysis must be evidence-grounded, rigorous, and highly attentive to patient safety.
You MUST respond strictly with a valid JSON object matching the requested schema."""

    def _get_schema_instructions(self) -> str:
        return """
Output MUST strictly follow this JSON structure:
{
  "final_answer": "Your final chosen option ["A","B","C","D",etc...] or ["yes","no","maybe"]",
  "confidence": 0.85,
  "diagnosis_or_hypothesis": ["Primary Diagnosis"],
  "reasoning": "Clinical rationale...",
  "cited_evidence": ["Fact from case"],
  "missing_evidence": ["Required data"],
  "safety_concerns": ["Risks"],
  "revision_note": null
}"""

    def call_agent(self, user_prompt: str, agent_role: str = "Generalist Clinical Agent") -> Tuple[Dict[str, Any], int, int]:
        full_system_prompt = f"{self._get_system_prompt()}\nYour Role: {agent_role}\n{self._get_schema_instructions()}"
        try:
            parsed_output, p_tokens, c_tokens = call_real_llm(full_system_prompt, user_prompt, model=self.model_name, temperature=self.temperature)
            return parsed_output, p_tokens, c_tokens
        except Exception as e:
            print(f"API Error: {e}")
            return {"final_answer": "UNKNOWN", "confidence": 0.0, "reasoning": f"Error Fallback: {str(e)}"}, 0, 0

    def call_free_form_llm(self, system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], int, int]:
        try:
            parsed_json, p_tokens, c_tokens = call_real_llm(system_prompt, user_prompt, model=self.model_name, temperature=self.temperature)
            return parsed_json, p_tokens, c_tokens
        except Exception as e:
            print(f"Free-form API Error: {e}")
            return {"error": "ERROR_LLM_CALL"}, 0, 0
