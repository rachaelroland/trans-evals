from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
import asyncio
from ..datasets.base import DatasetExample, EvaluationType


class BaseModel(ABC):
    """Abstract base class for language models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        
    @abstractmethod
    def generate(
        self,
        example: DatasetExample,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on the example.
        
        Args:
            example: Dataset example to generate from
            max_length: Maximum length of generation
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    async def generate_async(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Async version of generate for string prompts.
        Default implementation uses sync version in executor.
        """
        loop = asyncio.get_event_loop()
        # Create a mock example for the sync method
        from ..datasets.base import Example
        example = Example(
            id="temp",
            prompt=prompt,
            metadata={}
        )
        return await loop.run_in_executor(
            None,
            self.generate,
            example,
            max_length,
            temperature
        )
    
    @abstractmethod
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        pass
    
    @abstractmethod
    def predict_multiple_choice(
        self,
        question: str,
        choices: List[str],
        context: Optional[str] = None
    ) -> int:
        """
        Predict answer for multiple choice question.
        
        Args:
            question: The question
            choices: List of answer choices
            context: Optional context
            
        Returns:
            Index of predicted choice
        """
        pass
    
    def process_example(self, example: DatasetExample, **kwargs) -> Union[str, int, float]:
        """Process example based on its evaluation type."""
        if example.evaluation_type == EvaluationType.GENERATION:
            return self.generate(example, **kwargs)
        elif example.evaluation_type == EvaluationType.MULTIPLE_CHOICE:
            if not example.choices:
                raise ValueError("Multiple choice example must have choices")
            return self.predict_multiple_choice(
                example.text,
                example.choices,
                context=example.metadata.get("context") if example.metadata else None
            )
        elif example.evaluation_type == EvaluationType.PERPLEXITY:
            return self.compute_perplexity(example.text)
        else:
            raise ValueError(f"Unsupported evaluation type: {example.evaluation_type}")