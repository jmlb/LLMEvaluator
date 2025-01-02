
import re
import ast
import json


def extract_evaluation_from_json_blob(input_string):
    """
    Extract a dictionary from a string and fetch specific keys case-insensitively.

    Args:
        input_string (str): Input string containing a dictionary.

    Returns:
        dict: Dictionary containing extracted key-value pairs for 'reasoning', 'verdict', and 'confidence', or None if no dictionary is found.
    """

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
                "verdict": normalized_dict.get("verdict"),
                "confidence": normalized_dict.get("confidence")
            }

            success = False
            for v in ['Pass', 'Fail']:
                if v.lower() in extracted['verdict'].lower():
                    extracted["verdict"] = v.capitalize()
                    success = True
                    break
            if not success:
                extracted["verdict"] = None

            success = False
            for c in ['Low', 'Medium', 'High']:
                if c.lower() in extracted['confidence'].lower():
                    extracted["confidence"] = c.capitalize()
                    success = True
                    break
            if not success:
                extracted["confidence"] = None

            return extracted

        except (SyntaxError, ValueError) as e:
            print(f"Error parsing dictionary: {e}")
            return {}

    # Return None if no match is found
    return {}


def extract_evaluation_from_unstructured_text(input_string):
    """
    Parse the evaluation text into a dictionary with structured extraction logic.

    Args:
        input_string (str): Evaluation text with sections (e.g., Reasoning, Verdict, Confidence).
    
    Returns:
        dict: Parsed evaluation dictionary with keys: 'reasoning', 'verdict', 'confidence'.
    """
    if not input_string or not isinstance(input_string, str):
        raise ValueError("Input must be a non-empty string.")

    input_string_lower = input_string.lower()
    result = {}

    def clean_text(text):
        """
        Clean extracted text by removing leading bullets/numbers and unwanted characters.
        """
        text = re.sub(r'^[\d.\-•*]+\s*', '', text, flags=re.MULTILINE).strip()
        return text.replace('"', "")

    def extract_section(pattern, input_text):
        """
        Extract text between sections based on the given pattern.
        """
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

    return result


def parse_raw_evaluation(raw_result, required_keys=None):
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

    # Try JSON parsing
    try:
        parsed_result = json.loads(raw_result)
        if required_keys_set.issubset(parsed_result.keys()):
            return parsed_result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting with custom parser
    try:
        parsed_result = extract_evaluation_from_json_blob(raw_result)
        if required_keys_set.issubset(parsed_result.keys()):
            return parsed_result
    except Exception as e:
        # Log or handle the exception if necessary
        pass

    # Add additional parsers if needed in the future
    try:
        parsed_result = extract_evaluation_from_unstructured_text(raw_result)
        if required_keys_set.issubset(parsed_result.keys()):
            return parsed_result
    except Exception as e:
        # Log or handle the exception if necessary
        pass

    # If all parsers fail
    return None