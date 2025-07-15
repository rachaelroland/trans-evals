# trans-evals User Guide

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Template Generation](#template-generation)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Integration](#model-integration)
- [Counterfactual Analysis](#counterfactual-analysis)
- [Best Practices](#best-practices)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rachaelroland/trans-evals.git
cd trans-evals
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
from trans_evals import load_dataset, BiasEvaluator, OpenAIModel

# Load a dataset
dataset = load_dataset("tango")

# Initialize model and evaluator
model = OpenAIModel(model_name="gpt-4")
evaluator = BiasEvaluator(model=model)

# Run evaluation
results = evaluator.evaluate(dataset)

# Generate report
results.generate_report("bias_report.html")
```

## Datasets

### Available Datasets

1. **TANGO** - Trans And Nonbinary Gender-Oriented dataset
   - Focus: Misgendering and harmful responses to gender disclosure
   - Example: Testing correct pronoun usage

2. **HolisticBias** - Large-scale template dataset
   - Focus: Broad bias evaluation with trans-specific terms
   - Example: Perplexity-based bias detection

3. **CrowS-Pairs** - Stereotype pairs dataset
   - Focus: Stereotypical vs anti-stereotypical sentences
   - Example: Comparing model preferences for stereotypes

4. **BBQ** - Bias Benchmark for Question Answering
   - Focus: Multiple choice questions revealing bias
   - Example: Testing assumptions about trans individuals

### Loading Datasets

```python
# Load specific dataset
tango = load_dataset("tango")

# Filter HolisticBias by identity terms
holistic = load_dataset(
    "holistic_bias", 
    identity_terms=["trans woman", "non-binary"]
)

# Get trans-specific examples from any dataset
trans_examples = dataset.get_trans_specific_examples()
```

## Template Generation

### Creating Custom Templates

```python
from trans_evals import TemplateGenerator, Template
from trans_evals.datasets.base import BiasType, EvaluationType

# Create a custom template
template = Template(
    template_string="{name} is a {identity} who {action}",
    variables={
        "name": ["Alex", "Sam", "Jordan"],
        "identity": ["trans woman", "non-binary person"],
        "action": ["leads the team", "teaches students"]
    },
    bias_type=BiasType.STEREOTYPE,
    evaluation_type=EvaluationType.GENERATION
)

# Generate examples
generator = TemplateGenerator()
generator.add_template(template)
examples = generator.generate_all_examples()
```

### Using Pre-built Templates

```python
from trans_evals import TransSpecificTemplates

# Get all trans-specific templates
generator = TransSpecificTemplates.get_all_templates()

# Generate pronoun consistency tests
pronoun_templates = TransSpecificTemplates.get_pronoun_templates()

# Generate disclosure response tests
disclosure_templates = TransSpecificTemplates.get_disclosure_templates()
```

## Evaluation Metrics

### Available Metrics

1. **Misgendering Detection**
   - Detects incorrect pronoun usage
   - Requires: expected_pronouns in metadata

2. **Toxicity Scoring**
   - Measures harmful or offensive content
   - Uses: Transformer-based toxicity models

3. **Regard Analysis**
   - Measures positive/neutral/negative regard
   - Scores: 1.0 (positive), 0.5 (neutral), 0.0 (negative)

4. **Sentiment Analysis**
   - Analyzes emotional tone
   - Uses: VADER sentiment analyzer

### Custom Metrics

```python
from trans_evals.evaluation.metrics import BaseMetric, MetricResult

class CustomMetric(BaseMetric):
    def compute(self, text: str, metadata: Optional[Dict] = None) -> MetricResult:
        # Your metric logic here
        score = compute_score(text)
        return MetricResult(
            metric_name="custom",
            score=score,
            details={"custom_info": "value"}
        )

# Add to evaluator
evaluator.metrics.add_metric("custom", CustomMetric())
```

## Model Integration

### Using OpenRouter (Recommended)

OpenRouter provides access to multiple models through a single API:

```python
from trans_evals import OpenRouterModel

# Use friendly names
model = OpenRouterModel(model_name="gpt-4")
model = OpenRouterModel(model_name="claude-3-opus")
model = OpenRouterModel(model_name="mistral-7b")

# Or use full model IDs
model = OpenRouterModel(model_name="anthropic/claude-3-opus")

# List available models
models = OpenRouterModel.list_available_models()
```

Available models include:
- OpenAI: gpt-4, gpt-3.5-turbo
- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
- Meta: llama-2-70b, llama-2-13b
- Mistral: mistral-7b, mixtral-8x7b
- And many more...

### Direct API Integration

1. **OpenAI Models**
```python
from trans_evals import OpenAIModel
model = OpenAIModel(model_name="gpt-4", temperature=0.7)
```

2. **Anthropic Models**
```python
from trans_evals import AnthropicModel
model = AnthropicModel(model_name="claude-3-opus-20240229")
```

3. **HuggingFace Models**
```python
from trans_evals import HuggingFaceModel
model = HuggingFaceModel(model_name="microsoft/DialoGPT-medium")
```

### Custom Model Integration

```python
from trans_evals.models.base import BaseModel

class CustomModel(BaseModel):
    def generate(self, example, max_length=150, **kwargs):
        # Your generation logic
        return generated_text
    
    def compute_perplexity(self, text):
        # Your perplexity logic
        return perplexity_score
    
    def predict_multiple_choice(self, question, choices, context=None):
        # Your prediction logic
        return choice_index
```

## Counterfactual Analysis

### Creating Counterfactuals

```python
from trans_evals.augmentation import CounterfactualAugmenter

augmenter = CounterfactualAugmenter()

# Augment entire dataset
augmented_examples = augmenter.augment_dataset(examples)

# Create paired examples for comparison
pairs = augmenter.create_paired_examples(examples)

for original, counterfactual in pairs:
    print(f"Original: {original.text}")
    print(f"Counterfactual: {counterfactual.text}")
```

### Custom Identity Mappings

```python
# Add custom mappings
augmenter.add_custom_mapping("demiboy", "boy")
augmenter.add_custom_mapping("demigirl", "girl")
```

## Best Practices

### 1. Ethical Considerations
- Always use this framework for improving model fairness
- Be mindful of the impact on trans and non-binary communities
- Consider consulting with community members

### 2. Evaluation Design
- Use multiple datasets for comprehensive evaluation
- Include both generated templates and real-world examples
- Always include counterfactual analysis for fair comparison

### 3. Metric Selection
- Use misgendering detection for pronoun-heavy content
- Apply toxicity scoring for disclosure scenarios
- Combine multiple metrics for holistic evaluation

### 4. Result Interpretation
- Look for patterns across different identities
- Compare scores between original and counterfactual examples
- Consider statistical significance with larger sample sizes

### 5. Reporting
- Always include demographic breakdowns
- Highlight areas of concern
- Provide actionable recommendations

## Advanced Usage

### Batch Evaluation

```python
# Evaluate multiple models
models = [
    OpenAIModel("gpt-3.5-turbo"),
    OpenAIModel("gpt-4"),
    AnthropicModel("claude-3-opus-20240229")
]

results = []
for model in models:
    evaluator = BiasEvaluator(model=model)
    result = evaluator.evaluate(dataset)
    results.append(result)
```

### Custom Evaluation Pipeline

```python
# Create custom pipeline
def custom_evaluation_pipeline(dataset, model):
    # 1. Filter examples
    trans_examples = dataset.filter_by_identity("trans woman")
    
    # 2. Add counterfactuals
    augmenter = CounterfactualAugmenter()
    all_examples = augmenter.augment_dataset(trans_examples)
    
    # 3. Evaluate with specific metrics
    evaluator = BiasEvaluator(model=model)
    results = evaluator.evaluate(
        all_examples,
        metrics_to_use=["misgendering", "toxicity"]
    )
    
    # 4. Custom analysis
    df = results.to_dataframe()
    trans_scores = df[df['target_identity'] == 'trans woman']['toxicity_score']
    cis_scores = df[df['target_identity'] == 'cis woman']['toxicity_score']
    
    print(f"Trans toxicity avg: {trans_scores.mean():.3f}")
    print(f"Cis toxicity avg: {cis_scores.mean():.3f}")
    
    return results
```

## Troubleshooting

### Common Issues

1. **Model API errors**
   - Check API keys in .env file
   - Verify internet connection
   - Check rate limits

2. **Memory issues with large datasets**
   - Use `max_examples` parameter
   - Process in batches
   - Use smaller models for testing

3. **Metric computation errors**
   - Ensure required metadata is provided
   - Check model dependencies are installed
   - Review error logs for details

### Getting Help

- GitHub Issues: https://github.com/rachaelroland/trans-evals/issues
- Documentation: See docs/ directory
- Examples: See examples/ directory