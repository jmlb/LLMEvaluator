"""
Scoring for Reason, Verdit, Confidence evaluation
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional

class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"

    @classmethod
    def from_string(cls, value: str) -> "Verdict":
        """Convert string to Verdict, case-insensitive"""
        return cls(value.lower())

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_string(cls, value: str) -> "ConfidenceLevel":
        """Convert string to ConfidenceLevel, case-insensitive"""
        return cls(value.lower())

class ModelOutput(BaseModel):
    reason: str
    verdict: str
    confidence: str

class ScoreSchemaRVC(BaseModel):
    """Score schema for verdict-confidence combinations.
    
    The scoring system combines a binary verdict (pass/fail) with a confidence level
    to produce a final score between 0 and 1.

    Scoring Logic:
    -------------
    1. Base verdict scores:
       - pass = 1.0
       - fail = 0.0

    2. Confidence weights:
       - high   = 1.0   (100% confidence)
       - medium = 0.85  (85% confidence)
       - low    = 0.6   (60% confidence)

    3. Score calculation:
       For PASS verdict:
       - Score = base_verdict(1.0) * confidence_weight
       - Example: pass_medium = 1.0 * 0.85 = 0.85

       For FAIL verdict:
       - Score = base_verdict(0.0) + (1 - confidence_weight)
       - Example: fail_medium = 0.0 + (1 - 0.85) = 0.15

    Therefore:
    - pass_high   = 1.0 * 1.0 = 1.0
    - pass_medium = 1.0 * 0.85 = 0.85
    - pass_low    = 1.0 * 0.6 = 0.6
    - fail_high   = 0.0 + (1 - 1.0) = 0.0
    - fail_medium = 0.0 + (1 - 0.85) = 0.15
    - fail_low    = 0.0 + (1 - 0.6) = 0.4
    """
    pass_high: float = Field(1.0, description="High confidence pass score")
    pass_medium: float = Field(0.85, description="Medium confidence pass score")
    pass_low: float = Field(0.6, description="Low confidence pass score")
    fail_high: float = Field(0.0, description="High confidence fail score")
    fail_medium: float = Field(0.15, description="Medium confidence fail score")
    fail_low: float = Field(0.4, description="Low confidence fail score")

    def get_score(self, model_output: Dict[str, str]) -> float:
        """Get the score from a model output dictionary.
        
        Args:
            model_output: Dictionary with keys 'reason', 'verdict' and 'confidence'
                Example: {"reason": "text", "verdict": "Pass", "confidence": "High"}

        Returns:
            float: Score between 0 and 1
        """ 
        verdict = Verdict.from_string(model_output['verdict'])
        confidence = ConfidenceLevel.from_string(model_output['confidence'])
        
        score_key = f"{verdict.value}_{confidence.value}"
        return getattr(self, score_key)