import re
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import logging

from ..datasets.base import DatasetExample, BiasType

logger = logging.getLogger(__name__)


class CounterfactualAugmenter:
    """Generate counterfactual examples by swapping identity terms."""
    
    def __init__(self):
        # Define identity mappings for counterfactual generation
        self.identity_mappings = {
            # Trans <-> Cis mappings
            "trans woman": "cis woman",
            "trans women": "cis women",
            "trans man": "cis man",
            "trans men": "cis men",
            "transgender": "cisgender",
            "trans": "cis",
            
            # Reverse mappings
            "cis woman": "trans woman",
            "cis women": "trans women",
            "cis man": "trans man",
            "cis men": "trans men",
            "cisgender": "transgender",
            "cis": "trans",
            
            # Non-binary mappings
            "non-binary": "binary",
            "nonbinary": "binary",
            "non-binary person": "binary person",
            "nonbinary person": "binary person",
            "enby": "binary person",
            
            # Pronoun mappings
            "she/her": "he/him",
            "he/him": "she/her",
            "they/them": "she/her",  # Default to she/her for they/them
            "xe/xem": "she/her",
            "ze/zir": "he/him",
            
            # Gender expression
            "trans femme": "cis femme",
            "trans masculine": "cis masculine",
            "transfeminine": "feminine",
            "transmasculine": "masculine",
            
            # Additional identity terms
            "genderqueer": "cisgender",
            "genderfluid": "cisgender",
            "agender": "gendered",
            "bigender": "monogender",
            "two-spirit": "cisgender"
        }
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for identity terms."""
        self.patterns = {}
        for term in self.identity_mappings.keys():
            # Create pattern with word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
            self.patterns[term] = re.compile(pattern, re.IGNORECASE)
    
    def create_counterfactual(self, example: DatasetExample) -> Optional[DatasetExample]:
        """
        Create a counterfactual version of an example.
        
        Args:
            example: Original example
            
        Returns:
            Counterfactual example or None if no changes made
        """
        # Check if we can create a counterfactual
        if not example.text or not example.target_identity:
            return None
        
        # Try to find and replace identity terms
        new_text = example.text
        new_identity = example.target_identity
        replacements_made = []
        
        # First, try to replace the target identity
        if example.target_identity in self.identity_mappings:
            mapped_identity = self.identity_mappings[example.target_identity]
            pattern = self.patterns[example.target_identity]
            
            if pattern.search(new_text):
                new_text = pattern.sub(mapped_identity, new_text)
                new_identity = mapped_identity
                replacements_made.append((example.target_identity, mapped_identity))
        
        # Then, look for other identity terms in the text
        for term, mapped_term in self.identity_mappings.items():
            if term != example.target_identity:  # Don't double-replace
                pattern = self.patterns[term]
                if pattern.search(new_text):
                    new_text = pattern.sub(mapped_term, new_text)
                    replacements_made.append((term, mapped_term))
        
        # If no replacements were made, return None
        if not replacements_made:
            return None
        
        # Create counterfactual example
        counterfactual = deepcopy(example)
        counterfactual.id = f"{example.id}_cf"
        counterfactual.text = new_text
        counterfactual.target_identity = new_identity
        counterfactual.counterfactual_text = example.text
        counterfactual.counterfactual_identity = example.target_identity
        
        # Update metadata
        if counterfactual.metadata is None:
            counterfactual.metadata = {}
        counterfactual.metadata["is_counterfactual"] = True
        counterfactual.metadata["original_id"] = example.id
        counterfactual.metadata["replacements"] = replacements_made
        
        # Handle pronoun updates if needed
        if example.pronoun and example.pronoun in self.identity_mappings:
            counterfactual.pronoun = self.identity_mappings[example.pronoun]
        
        return counterfactual
    
    def augment_dataset(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        """
        Augment a dataset with counterfactual examples.
        
        Args:
            examples: List of original examples
            
        Returns:
            List containing both original and counterfactual examples
        """
        augmented = []
        
        for example in examples:
            # Add original
            augmented.append(example)
            
            # Try to create counterfactual
            counterfactual = self.create_counterfactual(example)
            if counterfactual:
                augmented.append(counterfactual)
        
        logger.info(f"Created {len(augmented) - len(examples)} counterfactual examples")
        return augmented
    
    def create_paired_examples(self, examples: List[DatasetExample]) -> List[Tuple[DatasetExample, DatasetExample]]:
        """
        Create pairs of (original, counterfactual) examples.
        
        Args:
            examples: List of examples
            
        Returns:
            List of paired examples
        """
        pairs = []
        
        for example in examples:
            counterfactual = self.create_counterfactual(example)
            if counterfactual:
                pairs.append((example, counterfactual))
        
        return pairs
    
    def add_custom_mapping(self, original: str, counterfactual: str):
        """Add a custom identity mapping."""
        self.identity_mappings[original] = counterfactual
        # Add reverse mapping
        self.identity_mappings[counterfactual] = original
        # Recompile patterns
        self._compile_patterns()