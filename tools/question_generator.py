
"""
The QuestionsGenerator agent uses the LLM to generate questions based on a provided topic.

Functionality:
- Takes a topic as input.
- Constructs a prompt for the LLM to generate questions.
- Returns the list of generated questions.
"""
import sys
import json
import asyncio

from textwrap import dedent
from termcolor import colored

# Or add to the end of the path
sys.path.append('../')

from schemas.config import LLMConfig
from llm.agent import LLMAgent


async def main(model_config, system_prompt, user_prompt):
    # Create agent
    question_crafter_agent = LLMAgent(model_config, result_type=str)  #

    print(colored(f"\n[INFO] **Specs of Agent**:\n{vars(question_crafter_agent)}\n", "blue"))

    try:
        # Generate response with result type
        response = await question_crafter_agent.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        print(response.data)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        response = None

    return response


# Run the async main function
if __name__ == "__main__":


    models = {
              "mistral7b": "mistral:7b-instruct", 
              "mistral_nemo": "mistral-nemo:12b-instruct-2407-q8_0",
              "mistral7b02": "mistral:7b-instruct-v0.2-q8_0",
              "llama3.1-8b": "llama3.1:8b-instruct-q8_0",
              "falcon3-10b": "falcon3:10b"
    }

    llm_config = LLMConfig(
        name= models["mistral_nemo"],
        base_url="http://localhost:11434/v1",
        platform="ollama",
        api_key="api_token",
        max_tokens=500,
        retries=100,
        temperature=0
    )


    from tools.question_prompts import  QUESTIONNAIRE_SYSTEM_PROMPT, QUESTIONNAIRE_USER_PROMPT
    asyncio.run(
        main(
            llm_config,
            QUESTIONNAIRE_SYSTEM_PROMPT,
            QUESTIONNAIRE_USER_PROMPT
            )
    )

