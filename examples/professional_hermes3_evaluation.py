"""
Professional evaluation of Hermes 3 dataset using robust methodology.

This script provides a scientifically rigorous evaluation suitable for
presentation to dataset maintainers and AI researchers.
"""

import asyncio
import argparse
import logging
from pathlib import Path
import json
import os
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_evaluation import ProfessionalDatasetEvaluator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def evaluate_hermes3_professionally(
    dataset_path: str,
    sample_size: int = 500,
    confidence_level: float = 0.95,
    output_dir: str = "evaluation_results"
):
    """
    Perform professional evaluation of Hermes 3 dataset.
    
    Args:
        dataset_path: Path to Hermes 3 dataset file
        sample_size: Number of samples to evaluate (default: 500)
        confidence_level: Statistical confidence level (default: 0.95)
        output_dir: Directory for output files
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize evaluator
    logger.info("Initializing professional dataset evaluator...")
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
    logger.info("Starting evaluation pipeline...")
    report, full_results = await evaluator.run_professional_evaluation()
    
    # Save full results as JSON
    results_path = output_path / f"hermes3_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert to JSON-serializable format
        json_results = {
            "dataset_stats": full_results["dataset_stats"],
            "evaluation_summary": full_results["evaluation_results"]["evaluation_summary"],
            "statistical_validation": full_results["evaluation_results"]["statistical_validation"],
            "sample_count": len(full_results["selected_samples"]),
            "report_path": str(full_results["report_path"]),
            "database_path": full_results["database_path"]
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROFESSIONAL EVALUATION COMPLETE")
    print("="*60)
    
    eval_summary = full_results["evaluation_results"]["evaluation_summary"]
    print(f"\nOverall Bias Score: {eval_summary['overall_metrics']['overall_bias_score']:.3f}")
    print(f"Samples Evaluated: {len(full_results['selected_samples'])}")
    
    print("\nCriteria Breakdown:")
    for criterion, data in eval_summary['criteria_breakdown'].items():
        print(f"  {criterion.replace('_', ' ').title()}: {data['mean']:.3f} ({data['assessment']})")
    
    print(f"\nDetailed report: {full_results['report_path']}")
    print(f"Full results: {results_path}")
    print(f"Database: {full_results['database_path']}")
    
    return report, full_results


def analyze_evaluation_results(db_path: str):
    """
    Analyze and query evaluation results from database.
    
    Args:
        db_path: Path to evaluation database
    """
    from src.database import EvaluationPersistence
    
    persistence = EvaluationPersistence(db_path)
    
    # Get latest evaluation run
    results = persistence.get_evaluation_results(limit=1)
    
    if not results:
        print("No evaluation results found in database.")
        return
    
    latest = results[0]
    print("\n" + "="*60)
    print("DATABASE ANALYSIS")
    print("="*60)
    print(f"\nRun ID: {latest['run_id']}")
    print(f"Dataset: {latest['dataset']}")
    print(f"Sample Size: {latest['sample_size']}")
    print(f"Status: {latest['status']}")
    
    print("\nMetrics Summary:")
    for metric, value in latest['metrics'].items():
        if value is not None:
            print(f"  {metric}: {value:.3f}")


def generate_comparison_report(eval_results_1: str, eval_results_2: str):
    """
    Compare two evaluation results.
    
    Args:
        eval_results_1: Path to first evaluation results JSON
        eval_results_2: Path to second evaluation results JSON
    """
    with open(eval_results_1, 'r') as f:
        results_1 = json.load(f)
    
    with open(eval_results_2, 'r') as f:
        results_2 = json.load(f)
    
    print("\n" + "="*60)
    print("EVALUATION COMPARISON")
    print("="*60)
    
    # Compare overall bias scores
    bias_1 = results_1['evaluation_summary']['overall_metrics']['overall_bias_score']
    bias_2 = results_2['evaluation_summary']['overall_metrics']['overall_bias_score']
    
    print(f"\nOverall Bias Score:")
    print(f"  Evaluation 1: {bias_1:.3f}")
    print(f"  Evaluation 2: {bias_2:.3f}")
    print(f"  Difference: {abs(bias_1 - bias_2):.3f}")
    
    # Compare criteria
    print("\nCriteria Comparison:")
    criteria_1 = results_1['evaluation_summary']['criteria_breakdown']
    criteria_2 = results_2['evaluation_summary']['criteria_breakdown']
    
    for criterion in criteria_1:
        if criterion in criteria_2:
            score_1 = criteria_1[criterion]['mean']
            score_2 = criteria_2[criterion]['mean']
            print(f"\n{criterion.replace('_', ' ').title()}:")
            print(f"  Evaluation 1: {score_1:.3f}")
            print(f"  Evaluation 2: {score_2:.3f}")
            print(f"  Difference: {abs(score_1 - score_2):.3f}")


async def main():
    """Main entry point for professional evaluation."""
    parser = argparse.ArgumentParser(
        description="Professional evaluation of datasets for trans-inclusive language bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python professional_hermes3_evaluation.py /path/to/hermes3.json
  
  # Large-scale evaluation with high confidence
  python professional_hermes3_evaluation.py /path/to/hermes3.json \\
    --sample-size 1000 --confidence 0.99
  
  # Analyze existing results
  python professional_hermes3_evaluation.py --analyze evaluation_results/hermes3_professional_eval.db
  
  # Compare two evaluations
  python professional_hermes3_evaluation.py --compare eval1.json eval2.json
"""
    )
    
    parser.add_argument(
        "dataset_path", 
        nargs="?",
        help="Path to dataset file (JSON, JSONL, or Parquet)"
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
        "--output-dir",
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)"
    )
    parser.add_argument(
        "--analyze",
        metavar="DB_PATH",
        help="Analyze existing evaluation results from database"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("EVAL1", "EVAL2"),
        help="Compare two evaluation result files"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY") and not args.analyze and not args.compare:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY=your_key_here")
        return
    
    if args.analyze:
        # Analyze existing results
        analyze_evaluation_results(args.analyze)
    elif args.compare:
        # Compare two evaluations
        generate_comparison_report(args.compare[0], args.compare[1])
    elif args.dataset_path:
        # Run new evaluation
        await evaluate_hermes3_professionally(
            dataset_path=args.dataset_path,
            sample_size=args.sample_size,
            confidence_level=args.confidence,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())