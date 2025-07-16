"""
Simple example of evaluating Hermes 3 dataset with trans-evals framework.
"""

import asyncio
from hermes3_dataset_evaluation import Hermes3DatasetEvaluator


async def evaluate_hermes3():
    """Example of evaluating Hermes 3 dataset."""
    
    # Path to your Hermes 3 dataset
    # Supports: .json, .jsonl, or .parquet formats
    dataset_path = "path/to/hermes3_dataset.json"  # UPDATE THIS PATH
    
    # Initialize evaluator
    evaluator = Hermes3DatasetEvaluator(
        dataset_path=dataset_path,
        sample_size=50,  # Start with smaller sample for testing
        evaluation_model="anthropic/claude-3.5-sonnet",
        db_url="sqlite:///hermes3_evaluation.db"
    )
    
    # Run evaluation
    report = await evaluator.run_evaluation()
    
    # The report is saved automatically, but you can also process it
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(report)


def analyze_results():
    """Example of analyzing stored results."""
    from trans_evals.database import EvaluationPersistence
    
    # Connect to the database
    persistence = EvaluationPersistence("sqlite:///hermes3_evaluation.db")
    
    # Get evaluation results
    results = persistence.get_evaluation_results(
        dataset_name="hermes3_dataset",
        limit=10
    )
    
    print("\n" + "="*60)
    print("STORED EVALUATION RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\nRun ID: {result['run_id']}")
        print(f"Dataset: {result['dataset']}")
        print(f"Metrics: {result['metrics']}")
        print(f"Sample Size: {result['sample_size']}")


if __name__ == "__main__":
    print("""
Hermes 3 Dataset Evaluation for Trans-Inclusive Language
========================================================

This script will:
1. Load samples from the Hermes 3 dataset
2. Identify conversations related to gender/pronouns/trans topics
3. Evaluate them for bias, toxicity, stereotypes, and regard
4. Generate a comprehensive report

Make sure to:
1. Update the dataset_path variable with your Hermes 3 file location
2. Set OPENROUTER_API_KEY environment variable
3. Have sufficient API credits for Claude evaluations

The evaluation uses Claude Sonnet 4 to analyze each sample for:
- Toxicity and harmful content
- Regard (respectful vs. disrespectful treatment)
- Sentiment (positive vs. negative tone)
- Stereotypes and biased assumptions

Results are stored in a SQLite database for further analysis.
""")
    
    # Run the evaluation
    asyncio.run(evaluate_hermes3())
    
    # Optionally analyze stored results
    # analyze_results()