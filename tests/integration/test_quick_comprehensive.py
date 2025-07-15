"""
Quick comprehensive evaluation test to verify the framework works.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from src.datasets.comprehensive_loader import ComprehensiveDatasetLoader


def test_dataset_loading():
    """Test that the comprehensive dataset loader works."""
    print("Testing comprehensive dataset loading...")
    
    loader = ComprehensiveDatasetLoader(include_synthetic=True)
    all_examples = loader.load_all_datasets()
    
    print(f"Total examples loaded: {len(all_examples)}")
    
    if len(all_examples) > 0:
        print("\nFirst few examples:")
        for i, ex in enumerate(all_examples[:3]):
            print(f"{i+1}. ID: {ex.id}")
            print(f"   Text: {ex.text[:100]}...")
            print(f"   Bias Type: {ex.bias_type.value}")
            print(f"   Identity: {ex.target_identity}")
            print()
    
    # Test balanced sampling
    sample = loader.get_balanced_sample(n=20, seed=42)
    print(f"Balanced sample size: {len(sample)}")
    
    # Count by bias type in sample
    bias_counts = {}
    for ex in sample:
        bias_type = ex.bias_type.value
        bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
    
    print("\nSample distribution by bias type:")
    for bias_type, count in sorted(bias_counts.items()):
        print(f"  {bias_type}: {count}")
    
    return sample


def test_model_evaluation(sample):
    """Test evaluation on a small sample."""
    print("\nTesting model evaluation...")
    
    from src import BiasEvaluator, OpenRouterModel
    
    try:
        # Test with one model on a small sample
        model = OpenRouterModel(model_name="anthropic/claude-sonnet-4")
        evaluator = BiasEvaluator(model=model)
        
        # Create test dataset
        test_dataset = type('TestDataset', (), {
            'name': 'quick_test',
            'examples': sample[:3],  # Just 3 examples for speed
            '__len__': lambda self: 3,
            '__getitem__': lambda self, idx: sample[idx],
            '__iter__': lambda self: iter(sample[:3])
        })()
        
        print("Running evaluation on 3 examples...")
        results = evaluator.evaluate(
            test_dataset,
            metrics_to_use=["misgendering", "toxicity"]
        )
        
        print(f"Evaluation completed. Results for {len(results.examples)} examples:")
        for i, (example, prediction, metrics) in enumerate(
            zip(results.examples, results.predictions, results.metric_results)
        ):
            print(f"\nExample {i+1}:")
            print(f"  Text: {example.text[:80]}...")
            print(f"  Prediction: {prediction[:80]}...")
            for metric_name, metric_result in metrics.items():
                print(f"  {metric_name}: {metric_result.score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return False


def main():
    """Run quick comprehensive test."""
    print("=" * 60)
    print("QUICK COMPREHENSIVE EVALUATION TEST")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        return 1
    
    try:
        # Test dataset loading
        sample = test_dataset_loading()
        
        if len(sample) == 0:
            print("No examples loaded - cannot proceed with evaluation test")
            return 1
        
        # Test model evaluation
        success = test_model_evaluation(sample)
        
        if success:
            print("\n" + "=" * 60)
            print("QUICK TEST COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("Ready to run full comprehensive evaluation!")
        else:
            print("\n" + "=" * 60)
            print("EVALUATION TEST FAILED")
            print("=" * 60)
            return 1
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())