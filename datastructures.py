import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np # Import numpy

# ==========================================
# 1. DATA STRUCTURES & SCHEMAS
# ==========================================

@dataclass
class MedicalCase:
    case_id: str
    dataset_name: str
    case_text: str
    options: Optional[Dict[str, str]]  # e.g., {"A": "...", "B": "..."}
    evidence_context: Optional[str]    # e.g., PubMed Abstract
    gold_label: str

@dataclass
class AgentOutput:
    final_answer: str
    confidence: float
    diagnosis_or_hypothesis: List[str]
    reasoning: str
    cited_evidence: List[str]
    missing_evidence: List[str]
    safety_concerns: List[str]
    revision_note: Optional[str] = None

@dataclass
class ExecutionTrace:
    case_id: str
    architecture: str
    round_1_outputs: Dict[str, Any]  # Maps agent_id -> parsed schema
    round_2_outputs: Dict[str, Any]  # Maps agent_id -> parsed schema (Empty for CoT)
    aggregated_answer: str
    total_tokens_used: int
    estimated_cost: float
