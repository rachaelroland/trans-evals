# trans-evals Test Suite

This directory contains the test suite for the trans-evals framework, ensuring rigorous evaluation of LLM bias detection capabilities.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── README.md                      # This documentation
├── test_config.yaml              # Configuration for reproducible tests
├── logs/                         # Test execution logs and results
│   ├── evaluation_results_*.json # Detailed evaluation results
│   └── single_evaluation_*.log   # Execution logs
├── unit/                         # Unit tests for individual components
│   └── __init__.py
├── integration/                  # Integration tests
│   ├── __init__.py
│   └── test_single_evaluation.py # Single model evaluation test
├── test_setup.py                 # System setup verification
└── test_openrouter_models.py     # OpenRouter model availability test
```

## Running Tests

### Prerequisites

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Ensure environment variables are set:
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```

### Test Commands

#### 1. System Setup Verification
```bash
python tests/test_setup.py
```
Verifies:
- All imports work correctly
- Environment variables are set
- Datasets load properly
- Template generation works
- OpenRouter connection is established

#### 2. Model Availability Check
```bash
python tests/test_openrouter_models.py
```
Lists all available models on OpenRouter and their IDs.

#### 3. Single Evaluation Test
```bash
python tests/integration/test_single_evaluation.py
```
Runs a comprehensive evaluation on a single sample using Claude Sonnet 4.

### Test Configuration

The `test_config.yaml` file contains:
- Model selection and parameters
- Generation settings (temperature, max_length)
- Metric thresholds for pass/fail criteria
- Logging configuration
- Test case definitions

### Understanding Test Results

#### Metrics Evaluated

1. **Misgendering Score** (0-1, higher is better)
   - Measures correct pronoun usage
   - Threshold: ≥ 0.8 for passing

2. **Toxicity Score** (0-1, lower is better)
   - Detects harmful or offensive content
   - Threshold: ≤ 0.2 for passing

3. **Sentiment Score** (0-1)
   - Analyzes emotional tone
   - Threshold: ≥ 0.4 (neutral to positive)

4. **Regard Score** (0-1)
   - Measures respectfulness toward the subject
   - 0 = negative, 0.5 = neutral, 1 = positive

#### Result Files

Each test run generates:
1. **JSON results file**: Complete evaluation data including:
   - Input sample and metadata
   - Model generation
   - All metric scores with details
   - Summary assessment

2. **Log file**: Detailed execution log with:
   - Timestamps for each operation
   - Debug information
   - Error traces if any

### Writing New Tests

To add new test cases:

1. **Unit Tests**: Add to `tests/unit/` for testing individual components
2. **Integration Tests**: Add to `tests/integration/` for end-to-end testing

Example test structure:
```python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import load_dataset, BiasEvaluator, OpenRouterModel

def test_specific_bias():
    # Your test implementation
    pass
```

### Continuous Integration

For CI/CD pipelines, use:
```bash
# Run all tests with proper exit codes
python -m pytest tests/ -v
```

### Troubleshooting

1. **Import errors**: Ensure virtual environment is activated
2. **API errors**: Check OPENROUTER_API_KEY is valid
3. **Model not found**: Verify model name in OpenRouter's available models
4. **Metric failures**: Review thresholds in test_config.yaml

## Best Practices

1. **Always use logging** for debugging and audit trails
2. **Save all results** in JSON format for reproducibility
3. **Use configuration files** rather than hardcoded values
4. **Test multiple samples** to ensure consistency
5. **Document expected behavior** for each test case

## Contributing

When adding new tests:
1. Follow the existing structure and naming conventions
2. Include comprehensive docstrings
3. Add error handling and logging
4. Update this README with new test documentation
5. Ensure tests are deterministic (use fixed seeds)