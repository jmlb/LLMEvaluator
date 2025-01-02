import asyncio
from termcolor import colored

from schemas.config import LLMConfig
from llm.agent import LLMAgent
from schemas.evaluation import EvaluationResponse
from schemas.scoring_rvc import ScoreSchemaRVC


system_prompt_template = """
You are an objective evaluator assigned to assess responses based on provided criteria. Your role is to:
1. Analyze responses against given evaluation criteria
2. Support judgments with evidence from the response
3. Provide clear verdicts with confidence levels

Your evaluation must be systematic and explicit.
"""


user_prompt_template = """
Answer the following assessment question by comparing Instruction to Student and Student Response for Evaluation.
Analyze the 2 texts step by step.

### **Assessment Question:**
{assessment_question}

### **Instruction to Student**
{student_instruction}

### **Student Response for Evaluation:**
{student_response}

### **Your Task**
Answer the assessment question following the evaluation guidelines above.

### **Output Format**
Example format:
{{
    "reasoning": "Explanation and reasoning to answer the question",
    "verdict": "Pass/Fail",
    "confidence": "High/Medium/Low"
}}
"""

student_role = """
    You are a product expert tasked with generating **customer-friendly
    product descriptions** based on provided technical specifications. \nMaintain a
    **professional and engaging tone** to attract potential customers.\n
    
    ### **Important Guidelines**:
        1. **No Modifications**: The description must use the exact technical specifications provided above without alterations.
        2. **No Omissions**: Include all the technical specifications provided. Do not leave any specification out.
        3. **Professional Tone**: Maintain a professional and engaging tone suitable for attracting customers.
        4. **No Extraneous Information**: Focus solely on the provided specifications and avoid adding details not explicitly mentioned.

    Provide only the product description without any introductory or concluding sentences.
"""

student_instruction = """
    Generate a concise product description in paragraph format for the following product specifications: 
    4K Ultra HD 65-inch TV, HDR10+, Dolby Vision, Smart TV with built-in Alexa, 3 HDMI ports, Wi-Fi enabled, Energy Star certified
"""

student_response = """
Experience stunning visuals and seamless smart home connectivity with
      our 4K Ultra HD 65-inch TV. Equipped with Dolby Vision for breathtaking
      contrast and color, this exceptional display brings you closer to the action
      on your favorite movies and sports. Stream your favorite content using built-in
      Alexa, allowing voice commands and effortless navigation of popular services
      like Netflix and Amazon Prime Video. With three HDMI ports, multiple devices
      can be connected at once, and Wi-Fi enables seamless streaming from anywhere
      in your home. Certified by Energy Star, this eco-friendly TV is also designed
      to reduce energy consumption, ensuring you enjoy great entertainment while being
      kind to the environment.
"""

assessment_question = """
Find all technical specifications in the instruction to the student and compare them with the student response.

### **Evaluation Criteria**
1. **Exact Matches Only**: The student response must include every technical specification from the instruction exactly as written, including specific terminology, capitalization, numbers, and units. 
2. **No Implications or Inferences**: Do not consider subjective or implied matches.
3. **No Additional Details**: The student response must not include any specifications or details that are not explicitly listed in the instruction.

### **Verdict Criteria**
- If any technical specification from the instruction is missing, not exactly matched, or if additional specifications are included, the verdict is "Fail."
- If all specifications are included as written, with no additional details, the verdict is "Pass."
"""

# user_prompt = user_prompt_template.format(
#     assessment_question=assessment_question,
#     student_response=student_response,
#     student_role=student_role,
#     student_instruction=student_instruction
# )


async def main(config):
    # Create config
    # Prepare user prompt
    user_prompt = user_prompt_template.format(
        assessment_question=assessment_question,
        student_response=student_response,
        student_role=student_role,
        student_instruction=student_instruction
    )

    result_type = str  #EvaluationResponse
    # Create agent
    judge_agent = LLMAgent(config, result_type=result_type)  #

    try:
        # Generate response with result type
        response = await judge_agent.generate(
            prompt=user_prompt,
            system_prompt=system_prompt_template
        )
        print(response.data)
        
        # Basic usage
        result = EvaluationResponse.parse_raw_evaluation(response.data)
        
        print("Evaluation Response:")
        print(result, type(result))

        # Convert to score
        score_schema = ScoreSchemaRVC()
        score = score_schema.get_score(result)
        print(f"Score: {score}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        result = None

    return result


# Run the async main function
if __name__ == "__main__":
    # "mixtral:latest"
    # "mistral:7b-instruct"
    # "avcodes/flowaicom-flow-judge:q4"
    # "mistral-nemo:12b-instruct-2407-q8_0"
    llm_config = LLMConfig(
        name= "smallthinker", #"nexusraven:latest", #"mistral:7b-instruct-v0.2-q8_0", 
        base_url="http://localhost:11434/v1",
        platform="ollama",
        api_key="api_token",
        max_tokens=500,
        retries=100,
        temperature=0
    )

    asyncio.run(main(llm_config))
