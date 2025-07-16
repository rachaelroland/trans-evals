# Professional Dataset Evaluation with Trans-Evals

## Overview

The trans-evals framework provides professional-grade dataset evaluation for trans-inclusive language bias, designed to meet the standards of experienced AI researchers and dataset maintainers. This guide covers both basic and professional evaluation approaches for datasets like Hermes 3.

## Why Professional Evaluation Matters

Traditional keyword-based approaches miss crucial context and nuance. The professional evaluator addresses these limitations:

### Beyond Keywords
- **LLM-based relevance scoring** instead of simple keyword matching
- **Contextual understanding** of how gender/identity topics are discussed
- **Implicit bias detection** that keywords cannot capture

### Statistical Rigor
- **Stratified sampling** ensures representative evaluation
- **Confidence intervals** and margin of error calculations
- **Statistical power analysis** for validity
- **Multiple validity checks** (representativeness, internal consistency, convergent validity)

### Actionable Insights
- **Weighted multi-criteria assessment** based on impact
- **Severity classifications** for prioritization
- **Specific examples and quotes** for each issue found
- **Risk assessment** for training suitability

## Professional Evaluation Methodology

### Robust Sampling Strategy

Unlike simple keyword matching, the professional evaluator uses:

1. **LLM-based relevance scoring** - Each sample is analyzed for relevance using Claude Sonnet 4
2. **Stratified sampling** - Ensures representation across relevance levels and content types
3. **Statistical validation** - Calculates margin of error and statistical power
4. **Multi-criteria assessment** - Five professionally-defined evaluation criteria

### Evaluation Criteria

1. **Representation Quality** (Weight: 2.0)
   - Accuracy and respectfulness of gender identity representation
   - Presence of diverse identities and experiences

2. **Harmful Content** (Weight: 3.0)
   - Toxic, discriminatory, or harmful language
   - Microaggressions and implicit bias

3. **Stereotype Prevalence** (Weight: 2.0)
   - Frequency and severity of stereotypical representations
   - Common tropes (medicalization, tragedy narratives, etc.)

4. **Language Inclusivity** (Weight: 1.5)
   - Use of inclusive vs. exclusionary language patterns
   - Correct pronoun usage and identity terminology

5. **Contextual Appropriateness** (Weight: 1.5)
   - Appropriate handling of gender/identity in context
   - Relevance vs. gratuitous inclusion

## Quick Start

### Professional Evaluation (Recommended)

```python
from trans_evals import ProfessionalDatasetEvaluator
import asyncio

async def evaluate():
    evaluator = ProfessionalDatasetEvaluator(
        dataset_path="path/to/hermes3.json",
        sample_size=500,
        confidence_level=0.95,
        evaluation_model="anthropic/claude-3.5-sonnet"
    )
    report, results = await evaluator.run_professional_evaluation()
    print(report)

asyncio.run(evaluate())
```

### Basic Evaluation (Simple Keyword-Based)

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

### Professional Evaluation

```bash
# Standard professional evaluation
python professional_hermes3_evaluation.py /path/to/hermes3.json

# High-confidence evaluation with larger sample
python professional_hermes3_evaluation.py /path/to/hermes3.json \
    --sample-size 1000 \
    --confidence 0.99 \
    --output-dir results/

# Analyze existing results
python professional_hermes3_evaluation.py \
    --analyze results/hermes3_professional_eval.db

# Compare two evaluations
python professional_hermes3_evaluation.py \
    --compare eval1.json eval2.json
```

### Basic Evaluation

```bash
# Simple keyword-based evaluation
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

### Professional Report Format

The professional evaluation generates a publication-quality report with:

```markdown
# Professional Dataset Evaluation Report

## Executive Summary
- Overall bias score with interpretation
- Key findings and risk assessment
- Suitability for training recommendation

## Methodology
- Sampling strategy (stratified, LLM-based relevance)
- Sample size and confidence intervals
- Statistical power analysis
- Evaluation criteria and weights

## Detailed Findings
1. Issue Distribution
   - Tabulated by criterion with percentages
   - Statistical significance indicators
   
2. Severity Analysis
   - Distribution of issue severity levels
   - Specific examples and quotes
   
3. Statistical Validation
   - Sample representativeness assessment
   - Internal consistency (Cronbach's α)
   - Convergent validity measures

## Recommendations
- Priority actions ranked by frequency
- Risk assessment (Low/Moderate/High)
- Specific remediation strategies

## Technical Appendix
- Criteria weight justifications
- Methodological limitations
- Reproducibility information
```

### Basic Report Format

The basic evaluation provides:

```markdown
# Dataset Evaluation Report
## Executive Summary
- Sample size and coverage
- Key metrics summary

## Aggregate Statistics
- Mean scores for each metric
- Problematic sample counts

## Issues Identified
- Lists by issue type
- Example IDs for review

## Recommendations
- General improvement suggestions
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