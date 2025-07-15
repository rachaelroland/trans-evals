import os
from typing import Optional, List, Dict, Any
import logging
import requests
import json
import aiohttp
import asyncio

from .base import BaseModel
from ..datasets.base import DatasetExample

logger = logging.getLogger(__name__)


class OpenRouterModel(BaseModel):
    """OpenRouter API model wrapper for accessing multiple models through one API."""
    
    # Popular models available on OpenRouter (updated based on actual availability)
    AVAILABLE_MODELS = {
        # OpenAI
        "gpt-4": "openai/o4-mini",  # Using o4-mini as GPT-4 alternative
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k": "openai/gpt-3.5-turbo-16k",
        
        # Anthropic
        "claude-opus-4": "anthropic/claude-opus-4",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        
        # Google
        "gemini-flash": "google/gemini-2.5-flash",
        "gemini-pro": "google/gemini-2.5-pro",
        
        # Meta
        "llama-70b": "meta-llama/llama-3.3-70b-instruct:free",
        "llama-4-scout": "meta-llama/llama-4-scout",
        
        # Mistral
        "mistral-small": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral-small-free": "mistralai/mistral-small-3.2-24b-instruct:free",
        
        # Shorter aliases
        "gpt3": "openai/gpt-3.5-turbo-0613",
        "claude": "anthropic/claude-sonnet-4",
        "gemini": "google/gemini-2.5-flash",
        "llama": "meta-llama/llama-3.3-70b-instruct:free"
    }
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        # Map friendly names to OpenRouter model IDs
        if model_name in self.AVAILABLE_MODELS:
            openrouter_model = self.AVAILABLE_MODELS[model_name]
        else:
            # Assume it's already a full OpenRouter model ID
            openrouter_model = model_name
            
        super().__init__(name=openrouter_model, **kwargs)
        self.model_name = openrouter_model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/rachaelroland/trans-evals",
            "X-Title": "trans-evals"
        }
        
    def generate(
        self,
        example: DatasetExample,
        max_length: Optional[int] = 150,
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> str:
        """Generate text using OpenRouter API."""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": example.text}
                    ],
                    "max_tokens": max_length,
                    "temperature": temperature,
                    **kwargs
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return ""
            
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {data}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating with OpenRouter: {e}")
            return ""
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity.
        Note: Most models via OpenRouter don't provide perplexity directly.
        """
        logger.warning("Perplexity computation not available via OpenRouter API")
        return 0.0
    
    def predict_multiple_choice(
        self,
        question: str,
        choices: List[str],
        context: Optional[str] = None
    ) -> int:
        """Predict answer for multiple choice question."""
        # Format the prompt
        prompt = ""
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += f"Question: {question}\n\n"
        prompt += "Choose the best answer:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nAnswer with just the letter (A, B, C, etc.):"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are answering multiple choice questions. Respond with only the letter of your choice."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code}")
                return 0
            
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                answer = data["choices"][0]["message"]["content"].strip().upper()
                
                # Convert letter to index
                if answer and answer[0] in 'ABCDEFGHIJ':
                    return ord(answer[0]) - ord('A')
                else:
                    logger.warning(f"Invalid answer format: {answer}")
                    return 0
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error in multiple choice prediction: {e}")
            return 0
    
    async def generate_async(
        self,
        prompt: str,
        max_length: Optional[int] = 150,
        temperature: Optional[float] = 0.7,
        **kwargs
    ) -> str:
        """Async version of generate for string prompts."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_length,
                        "temperature": temperature,
                        **kwargs
                    }
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"OpenRouter API error: {response.status} - {text}")
                        return ""
                    
                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"No choices in response: {data}")
                        return ""
                        
            except Exception as e:
                logger.error(f"Error in async generation: {e}")
                return ""
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List all available models with their OpenRouter IDs."""
        return cls.AVAILABLE_MODELS
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model from OpenRouter."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code == 200:
                models = response.json().get("data", [])
                for model in models:
                    if model.get("id") == self.model_name:
                        return model
            
            return {"id": self.model_name, "name": self.model_name}
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"id": self.model_name, "name": self.model_name}