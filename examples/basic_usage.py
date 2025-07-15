"""
Basic usage example for trans-evals framework.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from our package
from src import (
    load_dataset,
    BiasEvaluator,
    TemplateGenerator,
    TransSpecificTemplates
)
from src.models import OpenRouterModel
from src.augmentation import CounterfactualAugmenter


def example_dataset_loading():
    """Example of loading existing datasets."""
    print("=== Loading Datasets ===")
    
    # Load individual datasets
    tango = load_dataset("tango")
    print(f"TANGO dataset: {len(tango)} examples")
    
    holistic = load_dataset("holistic_bias", identity_terms=["trans woman", "non-binary"])
    print(f"HolisticBias dataset (filtered): {len(holistic)} examples")
    
    # Get trans-specific examples
    trans_examples = holistic.get_trans_specific_examples()
    print(f"Trans-specific examples: {len(trans_examples)}")
    
    return tango


def example_template_generation():
    """Example of generating test cases from templates."""
    print("\n=== Template Generation ===")
    
    # Get all trans-specific templates
    generator = TransSpecificTemplates.get_all_templates()
    
    # Generate examples
    all_examples = generator.generate_all_examples()
    print(f"Total generated examples: {len(all_examples)}")
    
    # Generate specific types
    pronoun_examples = generator.generate_filtered_examples(
        bias_types=[BiasType.MISGENDERING]
    )
    print(f"Misgendering test examples: {len(pronoun_examples)}")
    
    # Sample a few examples
    sampled = generator.sample_examples(5, seed=42)
    print("\nSample generated examples:")
    for ex in sampled[:3]:
        print(f"- {ex.text}")
    
    return generator


def example_counterfactual_augmentation():
    """Example of counterfactual data augmentation."""
    print("\n=== Counterfactual Augmentation ===")
    
    augmenter = CounterfactualAugmenter()
    
    # Create a simple example
    from src.datasets.base import DatasetExample, BiasType, EvaluationType
    
    original = DatasetExample(
        id="test_1",
        text="The trans woman engineer presented her innovative design to the team.",
        bias_type=BiasType.SENTIMENT,
        evaluation_type=EvaluationType.GENERATION,
        target_identity="trans woman"
    )
    
    # Create counterfactual
    counterfactual = augmenter.create_counterfactual(original)
    
    if counterfactual:
        print(f"Original: {original.text}")
        print(f"Counterfactual: {counterfactual.text}")
        print(f"Replacements: {counterfactual.metadata['replacements']}")


def example_evaluation():
    """Example of running bias evaluation."""
    print("\n=== Bias Evaluation ===")
    
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Note: Set OPENROUTER_API_KEY to run actual evaluation")
        print("Using mock evaluation for demonstration")
        return
    
    # Show available models
    print("\nAvailable models via OpenRouter:")
    for friendly_name, model_id in OpenRouterModel.list_available_models().items():
        print(f"  {friendly_name}: {model_id}")
    
    # Initialize model with OpenRouter
    # You can use friendly names like "gpt-4", "claude-3-opus", "mistral-7b", etc.
    model = OpenRouterModel(model_name="gpt-3.5-turbo")
    print(f"\nUsing model: {model.model_name}")
    
    # Load a small dataset
    dataset = load_dataset("tango")
    
    # Initialize evaluator
    evaluator = BiasEvaluator(model=model)
    
    # Run evaluation on a few examples
    results = evaluator.evaluate(
        dataset,
        max_examples=5,  # Limit for demo
        metrics_to_use=["misgendering", "toxicity", "sentiment"]
    )
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Total examples evaluated: {results.summary_stats['total_examples']}")
    print(f"Overall metrics:")
    for metric, stats in results.summary_stats['overall_metrics'].items():
        print(f"  {metric}: {stats['average']:.3f}")
    
    # Save results
    results.save_csv("results/demo_results.csv")
    results.generate_report("results/demo_report.html")
    print("\nResults saved to results/ directory")


def main():
    """Run all examples."""
    # Create directories
    os.makedirs("examples", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run examples
    example_dataset_loading()
    example_template_generation()
    example_counterfactual_augmentation()
    example_evaluation()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Fix import for BiasType
    from src.datasets.base import BiasType
    main()