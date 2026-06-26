import os
import json
import openai
from dotenv import load_dotenv
from src.datastructures import Dict, Any

load_dotenv() # Loads API keys from a local .env file

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OR_CLIENT = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def call_real_llm(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o-mini", temperature: float = 0.4):
    response = OR_CLIENT.chat.completions.create(
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
    usage = response.usage.total_tokens
    return json.loads(raw_text), usage

class MedicalLLMClient:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0.4):
        self.model_name = model_name
        self.temperature = temperature
        self.output_token_cost = 15.00 / 1000000

    def _get_system_prompt(self) -> str:
        return """You are an expert, board-certified clinical AI assistant.
Your analysis must be evidence-grounded, rigorous, and highly attentive to patient safety.
You MUST respond strictly with a valid JSON object matching the requested schema."""

    def _get_schema_instructions(self) -> str:
        return """
Output MUST strictly follow this JSON structure:
{
  "final_answer": "Your final chosen option",
  "confidence": 0.85,
  "diagnosis_or_hypothesis": ["Primary Diagnosis"],
  "reasoning": "Clinical rationale...",
  "cited_evidence": ["Fact from case"],
  "missing_evidence": ["Required data"],
  "safety_concerns": ["Risks"],
  "revision_note": null
}"""

    def call_agent(self, user_prompt: str, agent_role: str = "Generalist Clinical Agent") -> tuple[Dict[str, Any], int]:
        full_system_prompt = f"{self._get_system_prompt()}\nYour Role: {agent_role}\n{self._get_schema_instructions()}"
        try:
            parsed_output, tokens = call_real_llm(full_system_prompt, user_prompt, model=self.model_name, temperature=self.temperature)
            return parsed_output, tokens
        except Exception as e:
            print(f"API Error: {e}")
            return {"final_answer": "UNKNOWN", "confidence": 0.0, "reasoning": f"Error Fallback: {str(e)}"}, 0

    def call_free_form_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        try:
            # Reusing the underlying utility with text format if necessary, or dumping json
            parsed_json, tokens = call_real_llm(system_prompt, user_prompt, model=self.model_name, temperature=self.temperature)
            return json.dumps(parsed_json), tokens
        except Exception as e:
            print(f"Free-form API Error: {e}")
            return "ERROR_LLM_CALL", 0