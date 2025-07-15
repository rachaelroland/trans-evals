from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class BiasType(Enum):
    MISGENDERING = "misgendering"
    TOXICITY = "toxicity"
    STEREOTYPE = "stereotype"
    SENTIMENT = "sentiment"
    COREFERENCE = "coreference"
    COUNTERFACTUAL = "counterfactual"


class EvaluationType(Enum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    MULTIPLE_CHOICE = "multiple_choice"
    COREFERENCE = "coreference"
    PERPLEXITY = "perplexity"


@dataclass
class DatasetExample:
    """Base class for dataset examples."""
    id: str
    text: str
    bias_type: BiasType
    evaluation_type: EvaluationType
    target_identity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # For counterfactual examples
    counterfactual_text: Optional[str] = None
    counterfactual_identity: Optional[str] = None
    
    # For multiple choice
    choices: Optional[List[str]] = None
    correct_answer: Optional[Union[str, int]] = None
    
    # For stereotype evaluation
    stereotype_label: Optional[str] = None  # stereotype, anti-stereotype, unrelated
    
    # For pronoun/coreference
    pronoun: Optional[str] = None
    referent: Optional[str] = None


class BaseDataset(ABC):
    """Abstract base class for bias evaluation datasets."""
    
    def __init__(self, name: str):
        self.name = name
        self.examples: List[DatasetExample] = []
        
    @abstractmethod
    def load(self) -> None:
        """Load the dataset."""
        pass
    
    def filter_by_bias_type(self, bias_type: BiasType) -> List[DatasetExample]:
        """Filter examples by bias type."""
        return [ex for ex in self.examples if ex.bias_type == bias_type]
    
    def filter_by_identity(self, identity: str) -> List[DatasetExample]:
        """Filter examples by target identity."""
        return [ex for ex in self.examples if ex.target_identity and identity.lower() in ex.target_identity.lower()]
    
    def get_trans_specific_examples(self) -> List[DatasetExample]:
        """Get examples specifically related to trans identities."""
        trans_keywords = ["trans", "transgender", "trans woman", "trans man", "non-binary", "nonbinary", "enby", "they/them"]
        return [
            ex for ex in self.examples 
            if ex.target_identity and any(keyword in ex.target_identity.lower() for keyword in trans_keywords)
        ]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> DatasetExample:
        return self.examples[idx]
    
    def __iter__(self):
        return iter(self.examples)