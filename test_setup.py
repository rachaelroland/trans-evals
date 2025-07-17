#!/usr/bin/env python3
"""Test that everything is set up correctly for Hermes 3 evaluation."""

import os
import sys
from pathlib import Path

def check_setup():
    """Check if everything is configured correctly."""
    
    print("🔍 Checking Trans-Evals setup...\n")
    
    # Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("  ⚠️  Warning: Python 3.8+ recommended")
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"✓ OPENROUTER_API_KEY is set ({len(api_key)} characters)")
    else:
        print("❌ OPENROUTER_API_KEY is not set")
        print("   Please run: export OPENROUTER_API_KEY=your_key_here")
        return False
    
    # Check imports
    required_packages = [
        ("datasets", "HuggingFace datasets"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("sqlalchemy", "SQLAlchemy"),
        ("aiohttp", "aiohttp"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm")
    ]
    
    print("\n📦 Checking required packages:")
    all_present = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} - run: pip install {package}")
            all_present = False
    
    # Check trans-evals modules
    print("\n🔧 Checking trans-evals modules:")
    try:
        sys.path.append(str(Path(__file__).parent))
        from src.dataset_evaluation import ProfessionalDatasetEvaluator
        print("  ✓ ProfessionalDatasetEvaluator")
        
        from src.evaluation.llm_metrics import LLMBiasMetrics
        print("  ✓ LLMBiasMetrics")
        
        from src.database import EvaluationPersistence
        print("  ✓ EvaluationPersistence")
    except ImportError as e:
        print(f"  ❌ Module import error: {e}")
        all_present = False
    
    # Check directories
    print("\n📁 Creating directories:")
    dirs = ["data/hermes3", "hermes3_evaluation_results"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print("\n" + "="*50)
    if all_present and api_key:
        print("✅ Everything is set up correctly!")
        print("\nYou can now run:")
        print("  ./run_hermes3_eval.sh")
        print("\nOr for a quick test (100 samples):")
        print("  ./run_hermes3_eval.sh --quick")
        print("\nOr for thorough evaluation (1000 samples):")
        print("  ./run_hermes3_eval.sh --thorough")
        return True
    else:
        print("❌ Some issues need to be fixed before running evaluation")
        return False
    

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)