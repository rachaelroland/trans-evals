"""
Simple test script to verify trans-evals setup and OpenRouter integration.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all modules can be imported."""
    print("1. Testing imports...")
    try:
        from src import (
            load_dataset,
            BiasEvaluator,
            OpenRouterModel,
            TemplateGenerator,
            TransSpecificTemplates
        )
        from src.augmentation import CounterfactualAugmenter
        from src.datasets.base import BiasType, EvaluationType
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_environment():
    """Test that required environment variables are set."""
    print("\n2. Testing environment variables...")
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    if openrouter_key:
        print(f"✓ OPENROUTER_API_KEY is set (length: {len(openrouter_key)})")
    else:
        print("✗ OPENROUTER_API_KEY is not set")
        
    if hf_token:
        print(f"✓ HF_TOKEN is set (length: {len(hf_token)})")
    else:
        print("✗ HF_TOKEN is not set")
        
    return bool(openrouter_key)


def test_datasets():
    """Test dataset loading."""
    print("\n3. Testing dataset loading...")
    try:
        from src import load_dataset
        
        # Test TANGO dataset
        tango = load_dataset("tango")
        print(f"✓ TANGO dataset loaded: {len(tango)} examples")
        
        # Show first example
        if len(tango) > 0:
            first = tango[0]
            print(f"  Example: {first.text[:60]}...")
            print(f"  Bias type: {first.bias_type.value}")
            
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False


def test_template_generation():
    """Test template generation."""
    print("\n4. Testing template generation...")
    try:
        from src import TransSpecificTemplates
        
        # Get pronoun templates
        templates = TransSpecificTemplates.get_pronoun_templates()
        print(f"✓ Generated {len(templates)} pronoun templates")
        
        # Generate examples from first template
        if templates:
            examples = templates[0].generate_examples()
            print(f"✓ Generated {len(examples)} examples from first template")
            if examples:
                print(f"  Example: {examples[0].text}")
                
        return True
    except Exception as e:
        print(f"✗ Template generation error: {e}")
        return False


def test_counterfactual():
    """Test counterfactual augmentation."""
    print("\n5. Testing counterfactual augmentation...")
    try:
        from src.augmentation import CounterfactualAugmenter
        from src.datasets.base import DatasetExample, BiasType, EvaluationType
        
        augmenter = CounterfactualAugmenter()
        
        # Create test example
        original = DatasetExample(
            id="test",
            text="The trans woman CEO led the meeting with confidence.",
            bias_type=BiasType.SENTIMENT,
            evaluation_type=EvaluationType.GENERATION,
            target_identity="trans woman"
        )
        
        # Create counterfactual
        counterfactual = augmenter.create_counterfactual(original)
        
        if counterfactual:
            print("✓ Counterfactual created successfully")
            print(f"  Original: {original.text}")
            print(f"  Counterfactual: {counterfactual.text}")
            print(f"  Replacements: {counterfactual.metadata['replacements']}")
            return True
        else:
            print("✗ Failed to create counterfactual")
            return False
            
    except Exception as e:
        print(f"✗ Counterfactual error: {e}")
        return False


def test_openrouter_connection():
    """Test OpenRouter API connection."""
    print("\n6. Testing OpenRouter connection...")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠ Skipping OpenRouter test - API key not set")
        return None
        
    try:
        from src.models import OpenRouterModel
        
        # List available models
        models = OpenRouterModel.list_available_models()
        print(f"✓ Available models: {len(models)}")
        for name in list(models.keys())[:5]:
            print(f"  - {name}: {models[name]}")
            
        # Try to initialize a model
        model = OpenRouterModel(model_name="gpt-3.5-turbo")
        print(f"✓ Initialized model: {model.model_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenRouter connection error: {e}")
        return False


def test_simple_generation():
    """Test a simple text generation."""
    print("\n7. Testing simple generation with OpenRouter...")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠ Skipping generation test - API key not set")
        return None
        
    try:
        from src.models import OpenRouterModel
        from src.datasets.base import DatasetExample, BiasType, EvaluationType
        
        # Create a simple test
        model = OpenRouterModel(model_name="gpt-3.5-turbo")
        
        example = DatasetExample(
            id="test",
            text="Complete this sentence: The weather today is",
            bias_type=BiasType.SENTIMENT,
            evaluation_type=EvaluationType.GENERATION,
            target_identity=None
        )
        
        # Generate
        print("  Generating text...")
        result = model.generate(example, max_length=20, temperature=0.5)
        
        if result:
            print("✓ Generation successful")
            print(f"  Input: {example.text}")
            print(f"  Output: {result}")
            return True
        else:
            print("✗ Generation returned empty result")
            return False
            
    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TRANS-EVALS SETUP TEST")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Environment": test_environment(),
        "Datasets": test_datasets(),
        "Templates": test_template_generation(),
        "Counterfactual": test_counterfactual(),
        "OpenRouter Connection": test_openrouter_connection(),
        "Generation": test_simple_generation()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠ SKIPPED"
        print(f"{test_name:<25} {status}")
    
    # Overall status
    failures = [k for k, v in results.items() if v is False]
    if not failures:
        print("\n✅ All tests passed! The trans-evals framework is ready to use.")
    else:
        print(f"\n❌ {len(failures)} test(s) failed: {', '.join(failures)}")
        print("Please check the error messages above.")
    
    return len(failures) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)