from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


class LLMConfig(BaseModel):
    """Configuration for language models."""
    
    # Required fields
    name: str = Field(..., description="Name/identifier of the model to use")
    
    # Optional fields with defaults
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the API endpoint"
    )
    platform: Literal["openai", "ollama"] = Field(
        default="ollama",
        description="Platform type (openai or ollama)"
    )
    api_key: str = Field(
        default="-",
        description="API key for authentication. Use '-' for local models."
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens in the response"
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 = deterministic, 1.0 = creative)",
        ge=0.0,  # greater than or equal to 0
        le=1.0   # less than or equal to 1
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        gt=0  # greater than 0
    )
    retries: int = Field(
        default=1,
        description="Number of retries",
        gt=0  # greater than 0
    )


    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate model name is non-empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @validator('base_url')
    def validate_base_url(cls, v: str) -> str:
        """Validate base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip('/')  # Remove trailing slashes

    class Config:
        """Pydantic configuration."""
        frozen = True  # Make the class immutable
        extra = "forbid"  # Prevent extra fields