# Evaluating Hermes 3 Dataset with Trans-Evals

## Overview

The trans-evals framework can be used to evaluate any dataset for trans-inclusive language bias. This guide shows how to analyze the Hermes 3 dataset for potential issues related to trans and non-binary representation.

## What Gets Evaluated

The evaluation process:

1. **Scans the dataset** for conversations containing gender-related topics
2. **Extracts relevant samples** using keyword matching (pronouns, identity terms, etc.)
3. **Analyzes each sample** using Claude Sonnet 4 for:
   - **Toxicity**: Harmful language, discrimination, microaggressions
   - **Regard**: Respectful vs. disrespectful treatment
   - **Sentiment**: Positive vs. negative emotional tone
   - **Stereotypes**: Biased assumptions and harmful stereotypes
4. **Generates a report** with aggregate statistics and recommendations

## Quick Start

```python
from hermes3_dataset_evaluation import Hermes3DatasetEvaluator
import asyncio

async def evaluate():
    evaluator = Hermes3DatasetEvaluator(
        dataset_path="path/to/hermes3.json",
        sample_size=100,
        evaluation_model="anthropic/claude-3.5-sonnet"
    )
    await evaluator.run_evaluation()

asyncio.run(evaluate())
```

## Command Line Usage

```bash
# Basic evaluation
python hermes3_dataset_evaluation.py /path/to/hermes3.json

# With options
python hermes3_dataset_evaluation.py /path/to/hermes3.json \
    --sample-size 200 \
    --evaluation-model anthropic/claude-3.5-sonnet \
    --db-url sqlite:///my_evaluation.db
```

## Supported Formats

The evaluator supports multiple dataset formats:

- **JSON**: Single file with array of conversations
- **JSONL**: Line-delimited JSON (one conversation per line)
- **Parquet**: Compressed columnar format (requires pyarrow)

## Understanding Results

### Metrics Explained

1. **Toxicity Score** (0-1, lower is better)
   - < 0.1: Excellent (minimal harmful content)
   - 0.1-0.3: Concerning (some problematic content)
   - > 0.3: High toxicity (significant issues)

2. **Regard Score** (0-1, higher is better)
   - > 0.7: Positive regard (respectful treatment)
   - 0.4-0.7: Neutral regard
   - < 0.4: Negative regard (disrespectful/harmful)

3. **Sentiment Score** (0-1, higher is better)
   - > 0.7: Positive sentiment
   - 0.3-0.7: Neutral sentiment
   - < 0.3: Negative sentiment

4. **Stereotype Score** (0-1, lower is better)
   - < 0.1: Minimal stereotyping
   - 0.1-0.3: Some stereotypes present
   - > 0.3: Significant stereotyping

### Report Contents

The generated report includes:

```markdown
# Hermes 3 Dataset Evaluation Report
## Executive Summary
- Total samples analyzed
- Key findings summary

## Aggregate Statistics
- Mean scores for each metric
- Distribution of problematic content
- Percentage of samples with issues

## Issues Identified
- High toxicity samples
- Low regard samples
- Negative sentiment samples
- Stereotyping samples

## Recommendations
- Content review suggestions
- Bias mitigation strategies
- Dataset improvement ideas
```

## Database Storage

All detailed results are stored in SQLite database:

```python
from trans_evals.database import EvaluationPersistence

# Query results
persistence = EvaluationPersistence("sqlite:///hermes3_evaluation.db")

# Get all evaluations
results = persistence.get_evaluation_results(
    dataset_name="hermes3_dataset"
)

# Get specific example details
examples = persistence.get_example_details(
    model_evaluation_id=1
)
```

## Keyword Detection

The evaluator looks for these keywords to identify relevant samples:

### Pronouns
- Traditional: he/him, she/her, they/them
- Neo-pronouns: xe/xem, ze/zir, ey/em, fae/faer

### Identity Terms
- trans, transgender, non-binary, nonbinary
- genderqueer, genderfluid, agender
- transman, transwoman, mtf, ftm

### Context Terms
- gender identity, transition, coming out
- discrimination, inclusive, respect
- bathroom, sports, healthcare

## Cost Considerations

- Each sample evaluation uses ~1-2K tokens
- 100 samples ≈ 150K tokens total
- Estimated cost: ~$2-3 per 100 samples with Claude Sonnet

## Best Practices

1. **Start Small**: Test with 10-20 samples first
2. **Review Keywords**: Adjust keyword list for your use case
3. **Sample Randomly**: Use consistent random seed for reproducibility
4. **Check Results**: Manually review flagged samples
5. **Iterate**: Refine evaluation based on initial findings

## Interpreting Findings

### Common Issues in Datasets

1. **Misgendering**: Incorrect pronoun usage
2. **Deadnaming**: Using previous names
3. **Stereotypes**: Reducing to medical transition focus
4. **Erasure**: Ignoring non-binary identities
5. **Fetishization**: Inappropriate sexualization

### Taking Action

Based on evaluation results:

1. **Remove**: Highly toxic or harmful content
2. **Modify**: Correct misgendering and stereotypes
3. **Balance**: Add positive, respectful examples
4. **Document**: Note known biases for users
5. **Monitor**: Re-evaluate after changes

## Example Output

```
Found 234 potentially relevant samples out of 50000
Evaluating 100 samples with anthropic/claude-3.5-sonnet...

AGGREGATE STATISTICS:
- Toxicity: Mean 0.042 (4 problematic samples)
- Regard: Mean 0.623 (12 low regard samples)
- Sentiment: Mean 0.701 (8 negative samples)
- Stereotypes: Mean 0.156 (15 samples with stereotypes)

RECOMMENDATIONS:
1. Review and remove 4 highly toxic samples
2. Improve representation in 15 stereotyping samples
3. Add more positive trans narratives to balance dataset
```

## Extending the Evaluation

The framework is extensible:

```python
class CustomHermes3Evaluator(Hermes3DatasetEvaluator):
    def extract_trans_relevant_samples(self, data):
        # Add custom logic for sample selection
        pass
    
    def generate_report(self, results):
        # Customize report format
        pass
```

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See main trans-evals docs
- Community: Contribute improvements via PR