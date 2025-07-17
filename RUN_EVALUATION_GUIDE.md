# Running the Hermes 3 Evaluation

## Step 1: Install Dependencies

First, install all required packages:

```bash
cd /Users/rachael/Documents/projects/rachaelroland/trans_evals

# Install all requirements
pip install -r requirements.txt

# Or just the essentials for dataset evaluation
pip install datasets huggingface-hub pyarrow sqlalchemy aiohttp pandas tqdm
```

## Step 2: Set Your API Key

Make sure your OpenRouter API key is set:

```bash
export OPENROUTER_API_KEY=your_actual_key_here
```

## Step 3: Test Your Setup

Run the test script to verify everything is working:

```bash
python3 test_setup.py
```

You should see all green checkmarks ✓

## Step 4: Run the Evaluation

You have several options:

### Quick Test (100 samples, ~$2-3)
```bash
./run_hermes3_eval.sh --quick
```

### Standard Evaluation (500 samples, ~$11-15)
```bash
./run_hermes3_eval.sh
```

### Thorough Evaluation (1000 samples, ~$22-30)
```bash
./run_hermes3_eval.sh --thorough
```

### Custom Parameters
```bash
python3 examples/download_and_eval_hermes3.py \
    --sample-size 250 \
    --confidence 0.90
```

## What Happens During Evaluation

1. **Download Phase** (~5-10 minutes)
   - Downloads Hermes 3 dataset from HuggingFace
   - Converts to JSON format
   - Analyzes dataset structure

2. **Relevance Scoring** (~10-20 minutes)
   - Each sample is scored for relevance using Claude
   - No simple keyword matching
   - Identifies implicit bias patterns

3. **Stratified Sampling**
   - Selects samples across relevance levels
   - Ensures statistical representativeness

4. **Multi-Criteria Evaluation** (~20-40 minutes)
   - Evaluates each sample on 5 criteria
   - Collects specific examples and quotes
   - Generates severity classifications

5. **Report Generation**
   - Statistical validation
   - Professional markdown report
   - Database with all results

## Expected Output

After completion, you'll see:

```
════════════════════════════════════════════════════════════════
EVALUATION COMPLETE
════════════════════════════════════════════════════════════════

Dataset: Hermes 3 (NousResearch)
Total samples in dataset: X,XXX,XXX
Samples evaluated: 500
Overall bias score: X.XXX

Criteria Assessment:
✅ Representation Quality: X.XXX (good/acceptable/concerning)
✅ Harmful Content: X.XXX (good/acceptable/concerning)
✅ Stereotype Prevalence: X.XXX (good/acceptable/concerning)
✅ Language Inclusivity: X.XXX (good/acceptable/concerning)
✅ Contextual Appropriateness: X.XXX (good/acceptable/concerning)

Issue Distribution:
  - [Issue type]: XX samples
  - [Issue type]: XX samples
```

## Viewing Results

### 1. Read the Professional Report
```bash
cat hermes3_evaluation_results/professional_evaluation_*.md
```

### 2. Open in Markdown Viewer
The report is formatted for professional presentation.

### 3. Query the Database
```python
from trans_evals.database import EvaluationPersistence

db = EvaluationPersistence("sqlite:///hermes3_evaluation_results/hermes3_professional_eval.db")

# Get summary
results = db.get_evaluation_results()
for r in results:
    print(f"Overall bias: {r['metrics']['overall_bias_score']:.3f}")

# Get specific examples
examples = db.get_example_details(model_evaluation_id=1)
for ex in examples[:5]:
    print(f"Example {ex['example_id']}:")
    print(f"  Metrics: {ex['metrics']}")
```

## Troubleshooting

### "No module named 'X'"
Install missing module: `pip install X`

### "API key not set"
```bash
export OPENROUTER_API_KEY=your_key_here
```

### "Rate limit exceeded"
Reduce parallel workers in the script or wait before retrying.

### "Out of memory"
Use smaller sample size or process in batches.

## Next Steps

1. **Review the report** carefully
2. **Share with Teknium/NousResearch** - The report is designed for professional presentation
3. **Query specific issues** in the database
4. **Compare with other datasets** using the same methodology

## Cost Breakdown

- Relevance scoring: ~0.5K tokens/sample
- Full evaluation: ~1-2K tokens/sample
- Total: ~1.5-2.5K tokens/sample
- At Claude 3.5 Sonnet rates: ~$0.022-0.030/sample

## Support

If you encounter issues:
1. Check `test_setup.py` output
2. Review error messages carefully
3. Open an issue on GitHub with details