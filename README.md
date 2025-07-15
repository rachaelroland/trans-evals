# trans-evals

A comprehensive framework for evaluating language model bias toward trans and non-binary individuals.

## Overview

trans-evals provides tools and datasets for systematically evaluating how language models handle content related to trans and non-binary identities. The framework focuses on:

- **Misgendering detection**: Evaluating correct pronoun usage
- **Bias measurement**: Comparing model outputs for trans vs cis individuals
- **Toxicity analysis**: Detecting harmful responses to gender disclosures
- **Stereotype evaluation**: Identifying biased assumptions and stereotypes

## Features

- Integration with existing bias datasets (TANGO, HolisticBias, CrowS-Pairs, BBQ)
- Template-based test generation for trans-specific evaluations
- Counterfactual data augmentation for systematic bias testing
- Multiple evaluation metrics (toxicity, regard, sentiment)
- Support for multiple LLMs via OpenRouter API (OpenAI, Anthropic, Google, Meta, Mistral, and more)
- Comprehensive visualization and reporting tools

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e .
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .
```

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys:
   ```
   # Required for model access
   OPENROUTER_API_KEY=your_openrouter_key_here
   
   # Optional for HuggingFace datasets
   HF_TOKEN=your_huggingface_token_here
   ```

   Get your OpenRouter API key from: https://openrouter.ai/keys

## Quick Start

```python
from trans_evals import BiasEvaluator, load_dataset, OpenRouterModel

# Load a dataset
dataset = load_dataset("tango")

# Initialize model (via OpenRouter)
model = OpenRouterModel(model_name="gpt-3.5-turbo")  # or "claude", "gemini", "llama", etc.

# Initialize evaluator
evaluator = BiasEvaluator(model=model)

# Run evaluation
results = evaluator.evaluate(dataset)

# Generate report
results.generate_report("bias_report.html")
```

### Available Models

Popular model aliases via OpenRouter:
- `"gpt-3.5-turbo"` - OpenAI GPT-3.5
- `"gpt-4"` - OpenAI GPT-4 equivalent
- `"claude"` - Anthropic Claude
- `"gemini"` - Google Gemini
- `"llama"` - Meta Llama (free tier)
- `"mistral-small"` - Mistral AI

See all available models:
```python
from trans_evals import OpenRouterModel
print(OpenRouterModel.list_available_models())
```

## Supported Datasets

- **TANGO**: Trans and Non-binary Gender-Oriented dataset
- **HolisticBias**: Large-scale template dataset with trans-specific terms
- **CrowS-Pairs**: Stereotype pairs including gender identity
- **BBQ**: Bias Benchmark for Question Answering
- **Custom**: Create your own evaluation templates

## Examples

### Running a Simple Test

```bash
# Activate environment
source .venv/bin/activate

# Run basic example
python examples/basic_usage.py

# Run OpenRouter-specific examples
python examples/openrouter_example.py
```

### Template-Based Test Generation

```python
from trans_evals import TransSpecificTemplates

# Generate pronoun consistency tests
templates = TransSpecificTemplates.get_pronoun_templates()
examples = templates[0].generate_examples()

# Generate all trans-specific tests
generator = TransSpecificTemplates.get_all_templates()
all_examples = generator.generate_all_examples()
```

### Counterfactual Analysis

```python
from trans_evals.augmentation import CounterfactualAugmenter

# Create counterfactual pairs
augmenter = CounterfactualAugmenter()
pairs = augmenter.create_paired_examples(dataset.examples)

# Compare trans vs cis treatment
for original, counterfactual in pairs:
    # Evaluate both versions and compare scores
    pass
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Comprehensive usage documentation
- [API Reference](docs/API.md) - Detailed API documentation (coming soon)
- [Examples](examples/) - Example scripts and notebooks
- [Notebooks](notebooks/quickstart.ipynb) - Interactive Jupyter notebook tutorial

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

MIT License

## Citation

If you use trans-evals in your research, please cite:

```bibtex
@software{trans-evals,
  author = {Roland, Rachael},
  title = {trans-evals: Evaluating LLM Bias Toward Trans and Non-Binary Individuals},
  year = {2025},
  url = {https://github.com/rachaelroland/trans-evals}
}
```