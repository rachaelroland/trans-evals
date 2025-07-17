# Hermes 3 Dataset Evaluation Quick Start

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install datasets huggingface-hub
   ```

2. **Set API key**:
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```

## Run Professional Evaluation

### Option 1: Download and Evaluate (Recommended)

```bash
# This will download Hermes 3 and evaluate it
python examples/download_and_eval_hermes3.py

# For higher confidence evaluation
python examples/download_and_eval_hermes3.py \
    --sample-size 1000 \
    --confidence 0.99
```

### Option 2: Evaluate Existing Dataset

If you already have the dataset downloaded:

```bash
python examples/professional_hermes3_evaluation.py /path/to/hermes3.json
```

## What Happens

1. **Downloads** the official Hermes 3 dataset from HuggingFace
2. **Analyzes** the entire dataset structure and characteristics
3. **Performs LLM-based relevance scoring** on all samples
4. **Stratified sampling** across relevance levels
5. **Multi-criteria evaluation** of selected samples
6. **Generates professional report** with:
   - Statistical validation
   - Severity classifications
   - Specific examples
   - Actionable recommendations

## Output Files

After evaluation, you'll find:

```
hermes3_evaluation_results/
├── hermes3_professional_eval.db          # SQLite database with all results
├── hermes3_evaluation_metadata_*.json    # Evaluation metadata
└── professional_evaluation_*.md          # Detailed report
```

## Understanding Results

### Overall Bias Score
- **0.0 - 0.2**: Low bias, suitable for training
- **0.2 - 0.5**: Moderate bias, remediation recommended
- **0.5 - 1.0**: High bias, significant remediation required

### Criteria Breakdown
Each criterion is assessed as:
- **Good** ✅: Meets professional standards
- **Acceptable** ⚠️: Some concerns, but manageable
- **Concerning** ❌: Requires attention

## Example Output

```
════════════════════════════════════════════════════════════════
EVALUATION COMPLETE
════════════════════════════════════════════════════════════════

Dataset: Hermes 3 (NousResearch)
Total samples in dataset: 1,234,567
Samples evaluated: 500
Overall bias score: 0.234

Criteria Assessment:
✅ Representation Quality: 0.742 (good)
⚠️ Harmful Content: 0.089 (acceptable)
✅ Stereotype Prevalence: 0.156 (good)
✅ Language Inclusivity: 0.823 (good)
⚠️ Contextual Appropriateness: 0.651 (acceptable)

Issue Distribution:
  - Harmful Content: 23 samples
  - Stereotype Prevalence: 18 samples
  - Language Inclusivity: 12 samples
```

## Cost Estimate

- 500 samples ≈ 750K tokens ≈ $11-15
- 1000 samples ≈ 1.5M tokens ≈ $22-30

## Next Steps

1. **Review the report** for detailed findings
2. **Query the database** for specific examples:
   ```python
   from trans_evals.database import EvaluationPersistence
   
   db = EvaluationPersistence("sqlite:///hermes3_evaluation_results/hermes3_professional_eval.db")
   results = db.get_evaluation_results()
   ```

3. **Share findings** with dataset maintainers using the professional report

## Support

- GitHub Issues: https://github.com/rachaelroland/trans-evals
- Documentation: See `docs/HERMES3_EVALUATION.md` for detailed guide