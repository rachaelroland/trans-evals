"""
Example of using the new LLM-based evaluation system.
This replaces VADER, ToxicityScorer, and RegardScorer with Claude Sonnet 4.
"""

import asyncio
from trans_evals import load_dataset, OpenRouterModel
from trans_evals.evaluation.llm_evaluator import LLMBiasEvaluator
from trans_evals.database import EvaluationPersistence
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    """Run LLM-based evaluation example."""
    
    # Initialize the model to evaluate
    model = OpenRouterModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Load dataset
    print("Loading TANGO dataset...")
    dataset = load_dataset("tango")
    
    # Use a smaller subset for this example
    examples = dataset.examples[:10]
    
    # Initialize LLM evaluator with Claude Sonnet 4
    print("\nInitializing LLM-based evaluator...")
    evaluator = LLMBiasEvaluator(
        model=model,
        evaluation_model="anthropic/claude-3.5-sonnet",  # Uses Claude for evaluation
        db_url="sqlite:///trans_evals_llm.db",  # Separate DB for LLM evaluations
        metrics_to_use=["misgendering", "toxicity", "regard", "sentiment", "stereotypes"]
    )
    
    # Run evaluation
    print(f"\nEvaluating {model.model_name} using LLM-based metrics...")
    results = evaluator.evaluate(
        examples,
        dataset_name="tango_subset",
        dataset_version="1.0",
        run_metadata={
            "evaluation_type": "llm_based",
            "evaluation_model": "claude-3.5-sonnet",
            "example_count": len(examples)
        }
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Run ID: {results['run_id']}")
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Sample Size: {results['sample_size']}")
    
    print("\nAGGREGATE METRICS:")
    for metric, value in results['aggregate_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    # Retrieve detailed results from database
    persistence = EvaluationPersistence("sqlite:///trans_evals_llm.db")
    
    print("\n" + "="*60)
    print("EXAMPLE DETAILED RESULTS")
    print("="*60)
    
    # Get evaluation results
    eval_results = persistence.get_evaluation_results(run_id=results['run_id'])
    if eval_results:
        model_eval_id = eval_results[0]['model_evaluation_id'] if 'model_evaluation_id' in eval_results[0] else None
        
        if model_eval_id:
            # Get details for first few examples
            example_details = persistence.get_example_details(model_eval_id)[:3]
            
            for i, detail in enumerate(example_details, 1):
                print(f"\nExample {i}:")
                print(f"Prompt: {detail['prompt'][:100]}...")
                print(f"Response: {detail['response'][:100]}...")
                print("Metrics:")
                
                for metric_name, metric_data in detail['metrics'].items():
                    print(f"  {metric_name}:")
                    print(f"    Score: {metric_data['score']:.3f}")
                    if metric_data.get('explanation'):
                        print(f"    Explanation: {metric_data['explanation'][:100]}...")
    
    print("\n" + "="*60)
    print("ADVANTAGES OF LLM-BASED EVALUATION:")
    print("="*60)
    print("1. More nuanced understanding of context and meaning")
    print("2. Better detection of subtle biases and microaggressions")
    print("3. Detailed explanations for each metric score")
    print("4. Trans-specific evaluation criteria built into prompts")
    print("5. All results stored in database for analysis")
    print("6. Can detect intersectional biases")
    print("7. No need to install/manage multiple NLP models")


def compare_with_traditional():
    """Show comparison between traditional and LLM-based metrics."""
    print("\n" + "="*60)
    print("COMPARISON: Traditional vs LLM-based Metrics")
    print("="*60)
    
    comparisons = [
        {
            "metric": "Toxicity Detection",
            "traditional": "BERT-based classifier (unitary/toxic-bert)",
            "llm_based": "Claude analyzes toxicity with trans-specific awareness",
            "advantages": "Detects subtle transphobia, microaggressions, context-aware"
        },
        {
            "metric": "Sentiment Analysis", 
            "traditional": "VADER (rule-based sentiment)",
            "llm_based": "Claude evaluates sentiment considering trans reader perspective",
            "advantages": "Understands affirming vs invalidating language, emotional tone"
        },
        {
            "metric": "Regard/Bias",
            "traditional": "Regard classifier (sasha/regardv3)",
            "llm_based": "Claude assesses regard with detailed bias indicators",
            "advantages": "Identifies stereotyping, othering, erasure, fetishization"
        },
        {
            "metric": "Stereotypes",
            "traditional": "Not available in original system",
            "llm_based": "Claude detects trans-specific stereotypes and assumptions",
            "advantages": "Comprehensive stereotype detection with severity ratings"
        }
    ]
    
    for comp in comparisons:
        print(f"\n{comp['metric']}:")
        print(f"  Traditional: {comp['traditional']}")
        print(f"  LLM-based: {comp['llm_based']}")
        print(f"  Advantages: {comp['advantages']}")


if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(main())
    
    # Show comparison
    compare_with_traditional()
    
    print("\n✨ LLM-based evaluation complete! Check trans_evals_llm.db for full results.")