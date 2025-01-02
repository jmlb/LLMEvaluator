import json
import re
import ast
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict, Any, Optional, List, ClassVar


"""
The docstring is very important
This did not work: LLM-as-A-Judge Single Evaluation
keep ... if we need to make a key required
"""
class EvaluationResponse(BaseModel):
    """Evaluation of the student to a single question"""
    reasoning: str = Field(
        ..., 
        description="Reasoning behind the evaluation"
    )
    
    verdict: Literal["Pass", "Fail"] = Field(
        ..., 
        description="Evaluation verdict"
    )
    
    confidence: Literal["High", "Medium", "Low"] = Field(
        ..., 
        description="Confidence level of the evaluation"
    )

    # Class variable to store parsing methods
    _parsing_methods: ClassVar[List[callable]] = []

    def to_json(self) -> Dict[str, Any]:
        """Convert the Evaluation object to a JSON dictionary."""
        try:
            return json.loads(self.model_dump_json())
        except json.JSONDecodeError:
            return {}

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: str) -> str:
        """
        Validate that the reasoning is not an empty string.
        
        Args:
            value: The reasoning text to validate
        
        Returns:
            The validated reasoning text
        
        Raises:
            ValueError: If the reasoning is empty or contains only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Reasoning cannot be empty.")
        return value

    @classmethod
    def register_parsing_method(cls, method: callable):
        """
        Register an additional parsing method.
        
        Args:
            method: A callable that takes a string and returns a dictionary
        """
        cls._parsing_methods.append(method)

    @classmethod
    def parse_raw_evaluation(
        cls, 
        raw_result: str, 
        required_keys: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse the input into a dictionary using different parsers 
        and return the result if it contains all required keys.

        Args:
            raw_result (str): The raw input string to parse.
            required_keys (list): List of keys required in the output dictionary.
                                  Defaults to ["reasoning", "verdict", "confidence"].

        Returns:
            dict: Parsed dictionary if successful and contains all required keys.
            None: If parsing fails or required keys are missing.
        """
        if required_keys is None:
            required_keys = ["reasoning", "verdict", "confidence"]

        required_keys_set = set(required_keys)

        # Define default parsing methods
        parsing_methods = [
            cls._parse_json,
            cls._extract_from_json_blob,
            cls._extract_from_unstructured_text
        ] + cls._parsing_methods

        # Try each parsing method
        for method in parsing_methods:
            try:
                parsed_result = method(raw_result)
                if parsed_result and required_keys_set.issubset(parsed_result.keys()):
                    return parsed_result
            except Exception:
                # Silently continue to next method
                continue

        # If all parsers fail
        return None

    @staticmethod
    def _parse_json(raw_result: str) -> Optional[Dict[str, Any]]:
        """Parse input as JSON."""
        try:
            return json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _extract_from_json_blob(input_string: str) -> Optional[Dict[str, Any]]:
        """Extract dictionary from a string containing a JSON-like blob."""
        # Pattern to match dictionary-like content within the string
        pattern = r'\{[^{}]*\}'

        # Find the first match
        match = re.search(pattern, input_string)

        if match:
            dict_str = match.group(0)
            try:
                # Use ast.literal_eval for safe parsing
                result_dict = ast.literal_eval(dict_str)
                
                # Normalize keys for case-insensitive lookup
                normalized_dict = {key.lower(): value for key, value in result_dict.items()}
                
                # Extract desired keys
                extracted = {
                    "reasoning": normalized_dict.get("reasoning"),
                    "verdict": None,
                    "confidence": None
                }

                # Validate verdict
                for v in ['Pass', 'Fail']:
                    if v.lower() in str(normalized_dict.get("verdict", '')).lower():
                        extracted["verdict"] = v
                        break

                # Validate confidence
                for c in ['Low', 'Medium', 'High']:
                    if c.lower() in str(normalized_dict.get("confidence", '')).lower():
                        extracted["confidence"] = c
                        break

                return extracted

            except (SyntaxError, ValueError):
                return None

        return None

    @staticmethod
    def _extract_from_unstructured_text(input_string: str) -> Optional[Dict[str, Any]]:
        """
        Parse the evaluation text into a dictionary with structured extraction logic.
        """
        if not input_string or not isinstance(input_string, str):
            return None

        input_string_lower = input_string.lower()
        result = {}

        def clean_text(text):
            """Clean extracted text by removing leading bullets/numbers and unwanted characters."""
            text = re.sub(r'^[\d.\-•*]+\s*', '', text, flags=re.MULTILINE).strip()
            return text.replace('"', "")

        def extract_section(pattern, input_text):
            """Extract text between sections based on the given pattern."""
            match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None

        # Extract reasoning
        reasoning = extract_section(r'reasoning:(.*?)verdict:', input_string_lower)
        if reasoning:
            reasoning = clean_text(reasoning)
            if reasoning.startswith(":"):
                reasoning = reasoning[1:].strip()
            result["reasoning"] = reasoning

        # Extract verdict
        verdict = extract_section(r'verdict:(.*?)confidence:', input_string_lower)
        if verdict:
            verdict = clean_text(verdict)
            # Validate verdict
            valid_verdicts = {'pass', 'fail'}
            for v in valid_verdicts:
                if v in verdict:
                    result["verdict"] = v.capitalize()
                    break

        # Extract confidence
        confidence = extract_section(r'confidence:(.*)', input_string_lower)
        if confidence:
            confidence = clean_text(confidence)
            # Validate confidence
            valid_confidence_levels = {'low', 'medium', 'high'}
            for c in valid_confidence_levels:
                if c in confidence:
                    result["confidence"] = c.capitalize()
                    break

        return result if len(result) == 3 else None

    class Config:
        """Pydantic configuration for the model."""
        extra = 'forbid'
        frozen = True