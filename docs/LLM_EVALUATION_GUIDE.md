# LLM-Based Evaluation Guide

## Overview

The trans-evals framework now supports LLM-based evaluation metrics using Claude Sonnet 4 (or other LLMs) to replace traditional NLP models like VADER, ToxicityScorer, and RegardScorer. This provides more nuanced, context-aware evaluation of bias toward trans and non-binary individuals.

## Key Benefits

1. **Context-Aware Analysis**: LLMs understand context and nuance better than traditional classifiers
2. **Trans-Specific Evaluation**: Prompts are designed with trans-specific considerations
3. **Detailed Explanations**: Each metric comes with explanations of the scoring
4. **Unified Evaluation**: Single model handles multiple metrics instead of managing multiple models
5. **Database Persistence**: All results stored in SQLite database for analysis
6. **No Model Downloads**: No need to download and manage multiple NLP models

## Architecture

### Components

1. **LLM Metrics** (`src/evaluation/llm_metrics.py`)
   - `LLMToxicityScorer`: Detects toxicity with trans-specific awareness
   - `LLMRegardScorer`: Evaluates regard/bias with detailed indicators
   - `LLMSentimentAnalyzer`: Analyzes sentiment from trans reader perspective
   - `LLMStereotypeDetector`: Identifies trans-specific stereotypes

2. **Database Models** (`src/database/models.py`)
   - `EvaluationRun`: Top-level evaluation session
   - `ModelEvaluation`: Results for a specific model
   - `ExampleEvaluation`: Results for individual examples
   - `MetricResult`: Detailed metric scores and explanations

3. **LLM Evaluator** (`src/evaluation/llm_evaluator.py`)
   - Orchestrates evaluation process
   - Handles async evaluation for efficiency
   - Calculates aggregate metrics
   - Performs statistical analysis

## Usage

### Basic Example

```python
from trans_evals import load_dataset, OpenRouterModel
from trans_evals.evaluation.llm_evaluator import LLMBiasEvaluator

# Initialize model to evaluate
model = OpenRouterModel("gpt-3.5-turbo")

# Load dataset
dataset = load_dataset("tango")

# Create LLM evaluator
evaluator = LLMBiasEvaluator(
    model=model,
    evaluation_model="anthropic/claude-3.5-sonnet",
    db_url="sqlite:///trans_evals.db"
)

# Run evaluation
results = evaluator.evaluate(dataset)

print(f"Overall bias score: {results['aggregate_metrics']['overall_bias_score']:.3f}")
```

### Custom Metrics

```python
# Select specific metrics
evaluator = LLMBiasEvaluator(
    model=model,
    metrics_to_use=["toxicity", "stereotypes", "misgendering"]
)
```

### Database Queries

```python
from trans_evals.database import EvaluationPersistence

# Initialize persistence
persistence = EvaluationPersistence("sqlite:///trans_evals.db")

# Get evaluation results
results = persistence.get_evaluation_results(
    model_name="gpt-3.5-turbo",
    dataset_name="tango"
)

# Get detailed example results
details = persistence.get_example_details(
    model_evaluation_id=1,
    example_id="tango_001"
)
```

## Metrics Explained

### 1. Toxicity (LLMToxicityScorer)
- **Score**: 0.0 (non-toxic) to 1.0 (highly toxic)
- **Categories**: hate_speech, harassment, threats, dehumanization, microaggressions, misgendering, deadnaming
- **Trans-Specific**: Detects subtle transphobia and microaggressions

### 2. Regard (LLMRegardScorer)
- **Score**: 0.0 (negative) to 1.0 (positive), 0.5 is neutral
- **Bias Indicators**: stereotyping, othering, patronizing, erasure, fetishization, pathologization
- **Analysis**: How the text treats trans/non-binary individuals

### 3. Sentiment (LLMSentimentAnalyzer)
- **Score**: 0.0 (very negative) to 1.0 (very positive)
- **Emotional Tone**: supportive, dismissive, affirming, invalidating, respectful
- **Trans Perspective**: Considers impact on trans/non-binary readers

### 4. Stereotypes (LLMStereotypeDetector)
- **Score**: 0.0 (no stereotypes) to 1.0 (heavy stereotyping)
- **Common Stereotypes**: appearance/"passing", medicalization, deception, trauma, fetishization
- **Severity**: Low, medium, or high for each stereotype found

### 5. Misgendering (Rule-based)
- **Score**: 0.0 (consistent misgendering) to 1.0 (correct pronouns)
- **Details**: Tracks pronoun usage and identifies misgendering instances
- **Note**: Kept as rule-based for precision

## Database Schema

### Tables

1. **evaluation_runs**: Top-level evaluation sessions
2. **model_evaluations**: Model-specific results
3. **example_evaluations**: Individual example results
4. **metric_results**: Detailed metric scores
5. **dataset_info**: Dataset metadata
6. **model_info**: Model tracking

### Example Query

```sql
-- Get average toxicity scores by model
SELECT 
    me.model_name,
    AVG(mr.score) as avg_toxicity
FROM model_evaluations me
JOIN example_evaluations ee ON ee.model_evaluation_id = me.id
JOIN metric_results mr ON mr.example_evaluation_id = ee.id
WHERE mr.metric_name = 'toxicity'
GROUP BY me.model_name;
```

## Migration from Traditional Metrics

### Before (Traditional)
```python
from trans_evals import BiasEvaluator
evaluator = BiasEvaluator(model)  # Uses VADER, toxic-bert, etc.
```

### After (LLM-based)
```python
from trans_evals.evaluation.llm_evaluator import LLMBiasEvaluator
evaluator = LLMBiasEvaluator(model)  # Uses Claude for evaluation
```

## Configuration

### Environment Variables
```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional
TRANS_EVALS_DB_URL=sqlite:///custom_path.db
EVALUATION_MODEL=anthropic/claude-3.5-sonnet
```

### Evaluation Models
- `anthropic/claude-3.5-sonnet` (recommended)
- `anthropic/claude-3-opus`
- `openai/gpt-4-turbo`
- Any model supporting JSON responses

## Performance Considerations

1. **Async Evaluation**: Processes multiple examples concurrently
2. **Batch Size**: Default 5 concurrent evaluations
3. **Caching**: Database stores all results for reuse
4. **Cost**: LLM evaluation costs more than traditional models but provides better insights

## Best Practices

1. **Start Small**: Test with subset before full dataset
2. **Monitor Costs**: LLM API calls can add up
3. **Use Database**: Query stored results instead of re-evaluating
4. **Explain Results**: Use the explanation field for insights
5. **Compare Models**: Evaluate multiple models on same dataset

## Troubleshooting

### Common Issues

1. **API Key Missing**: Set OPENROUTER_API_KEY environment variable
2. **Database Locked**: Close other connections to SQLite database
3. **Rate Limits**: Reduce batch_size if hitting API limits
4. **Memory Issues**: Process dataset in chunks

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Multi-LLM Consensus**: Use multiple LLMs for evaluation
2. **Custom Prompts**: User-defined evaluation criteria
3. **Web Dashboard**: Interactive results visualization
4. **Export Formats**: CSV, JSON, HTML reports
5. **Streaming Evaluation**: Real-time result updates

## References

- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [Claude Model Guide](https://docs.anthropic.com/claude/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)