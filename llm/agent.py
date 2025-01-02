import os
import json
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager
from pydantic import BaseModel, ValidationError

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIModel

from schemas.config import LLMConfig


class LLMAgentFactory:
    """
    Factory class for creating LLM agents across different platforms.
    """
    @staticmethod
    def create_model(config: LLMConfig):
        """
        Create a model instance based on the platform.
        
        Args:
            config: LLM configuration
        
        Returns:
            Configured AI model instance
        """
        # Validate configuration
        if not isinstance(config, LLMConfig):
            raise ValueError(f"Invalid config type: {type(config)}")

        # Handle API key from environment if needed
        api_key = config.api_key
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not found")

        # Platform-specific model creation
        platform = config.platform.lower()
        
        if platform == 'ollama':
            return OllamaModel(
                model_name=config.name, 
                base_url=config.base_url
            )
        
        elif platform == 'openai':
            return OpenAIModel(
                model_name=config.name,
                openai_client=OpenAIModel.create_client(
                    base_url=config.base_url,
                    api_key=api_key
                )
            )
        
        else:
            raise ValueError(f"Unsupported platform: {platform}")


class LLMAgent:
    """
    A wrapper class for interacting with LLMs using pydantic_ai.
    Manages model connection and provides a consistent inference interface.
    """

    def __init__(
        self, 
        config: LLMConfig, 
        result_type: Optional[type[BaseModel]] = None
    ):
        """
        Initialize LLM agent with configuration and optional result type.
        
        Args:
            config: LLM configuration
            result_type: Optional Pydantic model to validate output
        """
        self.config = config
        self.result_type = result_type
        self._model = None
        self._agent = None

    @asynccontextmanager
    async def __call__(self):
        """
        Async context manager for model lifecycle management.
        
        Yields:
            Configured Agent instance
        """
        try:
            # Create model instance using factory method
            self._model = LLMAgentFactory.create_model(self.config)
            temp = 0 if not hasattr(self.config, 'temperature') else self.config.temperature
            # Create agent with result type
            self._agent = Agent(
                self._model, 
                result_type=self.result_type,
                retries=self.config.retries or 1,
                model_settings={'temperature': temp,
                                'max_tokens': self.config.max_tokens or 100,
                                }
            )
            self._agent.timeout = self.config.timeout or 60
            yield self._agent
            
        except Exception as e:
            print(f"Error creating agent: {e}")
            raise
        finally:
            # Cleanup 
            self._model = None
            self._agent = None

    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Union[Dict[str, Any], BaseModel, str]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The main user prompt
            system_prompt: Optional system prompt to set context
            
        Returns:
            Generated response, potentially validated against result_type
        """
        # If a system_prompt is provided, modify the agent's system prompt
        async with self() as agent:
            try:
                if system_prompt:
                    agent.system_prompt = system_prompt
                
                # Generate response
                response = await agent.run(prompt)
                return response
            
            except Exception as e:
                print(f"Error during generation: {e}")
                return {}
