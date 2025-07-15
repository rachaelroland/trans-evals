from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import itertools
import random
from copy import deepcopy

from ..datasets.base import DatasetExample, BiasType, EvaluationType


@dataclass
class Template:
    """Template for generating bias test examples."""
    template_string: str
    variables: Dict[str, List[str]]
    bias_type: BiasType
    evaluation_type: EvaluationType
    metadata: Optional[Dict[str, Any]] = None
    
    def generate_examples(self) -> List[DatasetExample]:
        """Generate all possible examples from this template."""
        examples = []
        
        # Get all variable names
        var_names = list(self.variables.keys())
        
        # Get all combinations of variable values
        var_values = [self.variables[var] for var in var_names]
        combinations = list(itertools.product(*var_values))
        
        for idx, combo in enumerate(combinations):
            # Create variable substitution dict
            substitutions = dict(zip(var_names, combo))
            
            # Generate text
            text = self.template_string.format(**substitutions)
            
            # Extract target identity if present
            target_identity = substitutions.get("identity", substitutions.get("descriptor", ""))
            
            # Create example
            example = DatasetExample(
                id=f"template_{id(self)}_{idx}",
                text=text,
                bias_type=self.bias_type,
                evaluation_type=self.evaluation_type,
                target_identity=target_identity,
                metadata={
                    "template": self.template_string,
                    "substitutions": substitutions,
                    **(self.metadata or {})
                }
            )
            examples.append(example)
        
        return examples


class TemplateGenerator:
    """Generator for creating bias evaluation examples from templates."""
    
    def __init__(self):
        self.templates: List[Template] = []
        
    def add_template(self, template: Template) -> None:
        """Add a template to the generator."""
        self.templates.append(template)
        
    def add_templates(self, templates: List[Template]) -> None:
        """Add multiple templates."""
        self.templates.extend(templates)
        
    def generate_all_examples(self) -> List[DatasetExample]:
        """Generate examples from all templates."""
        all_examples = []
        for template in self.templates:
            all_examples.extend(template.generate_examples())
        return all_examples
    
    def generate_filtered_examples(
        self, 
        bias_types: Optional[List[BiasType]] = None,
        evaluation_types: Optional[List[EvaluationType]] = None,
        filter_func: Optional[Callable[[DatasetExample], bool]] = None
    ) -> List[DatasetExample]:
        """Generate examples with filtering."""
        examples = self.generate_all_examples()
        
        # Filter by bias type
        if bias_types:
            examples = [ex for ex in examples if ex.bias_type in bias_types]
            
        # Filter by evaluation type
        if evaluation_types:
            examples = [ex for ex in examples if ex.evaluation_type in evaluation_types]
            
        # Apply custom filter
        if filter_func:
            examples = [ex for ex in examples if filter_func(ex)]
            
        return examples
    
    def sample_examples(self, n: int, seed: Optional[int] = None) -> List[DatasetExample]:
        """Sample n examples randomly."""
        if seed is not None:
            random.seed(seed)
            
        all_examples = self.generate_all_examples()
        n = min(n, len(all_examples))
        return random.sample(all_examples, n)
    
    def create_counterfactual_template(
        self,
        template: Template,
        identity_mapping: Dict[str, str]
    ) -> Template:
        """Create a counterfactual version of a template."""
        new_template = deepcopy(template)
        
        # Update variables with counterfactual identities
        if "identity" in new_template.variables:
            new_identities = []
            for identity in new_template.variables["identity"]:
                new_identities.append(identity_mapping.get(identity, identity))
            new_template.variables["identity"] = new_identities
            
        # Mark as counterfactual
        if new_template.metadata is None:
            new_template.metadata = {}
        new_template.metadata["is_counterfactual"] = True
        new_template.metadata["original_template_id"] = id(template)
        
        return new_template