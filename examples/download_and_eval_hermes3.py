"""
Download and evaluate the official Hermes 3 dataset from HuggingFace.

This script downloads the NousResearch/Hermes-3-Dataset and performs
a professional evaluation for trans-inclusive language bias.
"""

import os
import asyncio
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from huggingface_hub import snapshot_download
from src.dataset_evaluation import ProfessionalDatasetEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_hermes3_dataset(cache_dir: str = "data/hermes3") -> str:
    """
    Download Hermes 3 dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        Path to the downloaded dataset file
    """
    logger.info("Downloading Hermes 3 dataset from HuggingFace...")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset using HuggingFace datasets library
        dataset = load_dataset(
            "NousResearch/Hermes-3-Dataset",
            split="train",
            cache_dir=cache_dir
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        
        # Save as JSON for evaluation
        output_path = cache_path / "hermes3_dataset.json"
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            data.append(item)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        
        # Alternative: Try to get the parquet files directly
        logger.info("Trying alternative download method...")
        
        try:
            # Download the dataset files
            snapshot_download(
                repo_id="NousResearch/Hermes-3-Dataset",
                repo_type="dataset",
                local_dir=cache_dir,
                allow_patterns=["*.parquet", "*.json", "*.jsonl"]
            )
            
            # Look for data files
            for pattern in ["*.parquet", "*.json", "*.jsonl"]:
                files = list(cache_path.glob(f"**/{pattern}"))
                if files:
                    logger.info(f"Found data file: {files[0]}")
                    return str(files[0])
            
            raise FileNotFoundError("No data files found in downloaded dataset")
            
        except Exception as e2:
            logger.error(f"Alternative download also failed: {e2}")
            raise


async def evaluate_hermes3(
    dataset_path: str,
    sample_size: int = 500,
    confidence_level: float = 0.95,
    output_dir: str = "hermes3_evaluation_results"
):
    """
    Perform professional evaluation of Hermes 3 dataset.
    
    Args:
        dataset_path: Path to dataset file
        sample_size: Number of samples to evaluate
        confidence_level: Statistical confidence level
        output_dir: Directory for results
    """
    logger.info("=" * 60)
    logger.info("HERMES 3 DATASET PROFESSIONAL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Confidence level: {confidence_level * 100}%")
    logger.info("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize professional evaluator
    evaluator = ProfessionalDatasetEvaluator(
        dataset_path=dataset_path,
        sample_size=sample_size,
        confidence_level=confidence_level,
        evaluation_model="anthropic/claude-3.5-sonnet",
        db_url=f"sqlite:///{output_path}/hermes3_professional_eval.db",
        random_seed=42,
        parallel_workers=5
    )
    
    # Run evaluation
    logger.info("\nStarting professional evaluation...")
    report, full_results = await evaluator.run_professional_evaluation()
    
    # Save additional metadata
    metadata = {
        "dataset": {
            "name": "NousResearch/Hermes-3-Dataset",
            "source": "https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset",
            "maintainer": "NousResearch (Teknium)",
            "description": "Hermes 3 - A large-scale conversational dataset"
        },
        "evaluation": {
            "framework": "trans-evals v1.0",
            "date": datetime.now().isoformat(),
            "sample_size": sample_size,
            "confidence_level": confidence_level,
            "methodology": "Professional multi-criteria evaluation with LLM-based analysis"
        },
        "results_summary": {
            "overall_bias_score": full_results["evaluation_results"]["evaluation_summary"]["overall_metrics"]["overall_bias_score"],
            "total_samples_analyzed": full_results["dataset_stats"]["total_samples"],
            "relevant_samples_found": len(full_results["selected_samples"]),
            "report_path": str(full_results["report_path"]),
            "database_path": full_results["database_path"]
        }
    }
    
    # Save metadata
    metadata_path = output_path / f"hermes3_evaluation_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nDataset: Hermes 3 (NousResearch)")
    print(f"Total samples in dataset: {full_results['dataset_stats']['total_samples']:,}")
    print(f"Samples evaluated: {len(full_results['selected_samples'])}")
    print(f"Overall bias score: {metadata['results_summary']['overall_bias_score']:.3f}")
    print(f"\nReport: {full_results['report_path']}")
    print(f"Metadata: {metadata_path}")
    print(f"Database: {full_results['database_path']}")
    
    # Print key findings
    eval_summary = full_results["evaluation_results"]["evaluation_summary"]
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print("\nCriteria Assessment:")
    for criterion, data in eval_summary['criteria_breakdown'].items():
        assessment = data['assessment']
        emoji = "✅" if assessment == "good" else "⚠️" if assessment == "acceptable" else "❌"
        print(f"{emoji} {criterion.replace('_', ' ').title()}: {data['mean']:.3f} ({assessment})")
    
    print("\nIssue Distribution:")
    if eval_summary['issue_summary']:
        for issue, count in sorted(eval_summary['issue_summary'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {issue.replace('_', ' ').title()}: {count} samples")
    else:
        print("  No significant issues found")
    
    print("\n" + "="*60)
    
    return report, metadata


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and evaluate Hermes 3 dataset for trans-inclusive language bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Download the Hermes 3 dataset from HuggingFace
2. Perform professional evaluation using trans-evals framework
3. Generate a detailed report suitable for dataset maintainers

The evaluation uses:
- LLM-based relevance scoring (not simple keywords)
- Stratified sampling with statistical validation
- Multi-criteria assessment with professional standards
- Publication-quality reporting

Example:
  python download_and_eval_hermes3.py --sample-size 1000 --confidence 0.99
"""
    )
    
    parser.add_argument(
        "--dataset-path",
        help="Path to existing dataset file (skip download)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Statistical confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/hermes3",
        help="Directory to cache dataset (default: data/hermes3)"
    )
    parser.add_argument(
        "--output-dir",
        default="hermes3_evaluation_results",
        help="Output directory for results (default: hermes3_evaluation_results)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if dataset already exists in cache"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY=your_key_here")
        return
    
    # Get dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Check if dataset already exists
        cache_path = Path(args.cache_dir)
        existing_file = cache_path / "hermes3_dataset.json"
        
        if existing_file.exists() and args.skip_download:
            logger.info(f"Using existing dataset: {existing_file}")
            dataset_path = str(existing_file)
        else:
            # Download dataset
            dataset_path = download_hermes3_dataset(args.cache_dir)
    
    # Run evaluation
    await evaluate_hermes3(
        dataset_path=dataset_path,
        sample_size=args.sample_size,
        confidence_level=args.confidence,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Print header
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          Hermes 3 Dataset Professional Evaluation             ║
║                                                               ║
║  Framework: Trans-Evals v1.0                                  ║
║  Methodology: LLM-based multi-criteria analysis               ║
║  Purpose: Identify potential bias in conversational data      ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    asyncio.run(main())