"""
Statistical evaluation with 50+ examples from comprehensive dataset.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from src import BiasEvaluator, OpenRouterModel
from src.datasets.comprehensive_loader import ComprehensiveDatasetLoader


@dataclass
class StatisticalEvalConfig:
    """Configuration for statistical evaluation."""
    models: List[str] = field(default_factory=lambda: [
        "anthropic/claude-sonnet-4",
        "moonshotai/kimi-k2",
        "meta-llama/llama-3.3-70b-instruct:free"
    ])
    sample_size: int = 50  # Manageable size for testing
    temperature: float = 0.3
    metrics: List[str] = field(default_factory=lambda: [
        "misgendering", "toxicity", "sentiment", "regard"
    ])
    output_dir: Path = field(default_factory=lambda: Path("tests/results"))
    random_seed: int = 42


class StatisticalEvaluation:
    """Statistical evaluation framework."""
    
    def __init__(self, config: StatisticalEvalConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment
        load_dotenv()
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("statistical_eval")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_dataset(self):
        """Load balanced dataset sample."""
        self.logger.info(f"Loading {self.config.sample_size} examples...")
        
        loader = ComprehensiveDatasetLoader(include_synthetic=True)
        all_examples = loader.load_all_datasets()
        
        # Get balanced sample
        sample = loader.get_balanced_sample(
            n=self.config.sample_size,
            seed=self.config.random_seed
        )
        
        self.logger.info(f"Loaded {len(sample)} examples from {len(all_examples)} total")
        
        # Log sample distribution
        bias_counts = {}
        identity_counts = {}
        
        for ex in sample:
            # Count bias types
            bias_type = ex.bias_type.value
            bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
            
            # Count identities
            if ex.target_identity:
                identity = ex.target_identity.lower()
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
        
        self.logger.info(f"Bias type distribution: {bias_counts}")
        self.logger.info(f"Top identities: {dict(list(sorted(identity_counts.items(), key=lambda x: x[1], reverse=True))[:5])}")
        
        return sample
    
    def evaluate_model(self, model_name: str, test_batch: List) -> Dict[str, Any]:
        """Evaluate a single model."""
        self.logger.info(f"Evaluating {model_name}...")
        
        try:
            # Initialize model
            model = OpenRouterModel(model_name=model_name)
            evaluator = BiasEvaluator(model=model)
            
            # Create dataset object
            test_dataset = type('TestDataset', (), {
                'name': f'statistical_test_{model_name}',
                'examples': test_batch,
                '__len__': lambda self: len(test_batch),
                '__getitem__': lambda self, idx: test_batch[idx],
                '__iter__': lambda self: iter(test_batch)
            })()
            
            # Run evaluation
            results = evaluator.evaluate(
                test_dataset,
                metrics_to_use=self.config.metrics
            )
            
            # Process results
            model_results = {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "examples": [],
                "summary": {}
            }
            
            # Collect results
            for example, prediction, metrics in zip(results.examples, results.predictions, results.metric_results):
                model_results["examples"].append({
                    "id": example.id,
                    "bias_type": example.bias_type.value,
                    "target_identity": example.target_identity,
                    "metrics": {
                        name: metric.score for name, metric in metrics.items()
                    }
                })
            
            # Calculate summary statistics
            model_results["summary"] = self._calculate_summary_stats(model_results["examples"])
            
            self.logger.info(f"Completed {model_name}: {len(model_results['examples'])} examples")
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {e}")
            raise
    
    def _calculate_summary_stats(self, examples: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        summary = {"metrics": {}}
        
        for metric in self.config.metrics:
            scores = [ex["metrics"][metric] for ex in examples if metric in ex["metrics"]]
            if scores:
                summary["metrics"][metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "median": np.median(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores)
                }
        
        return summary
    
    def perform_statistical_tests(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical comparisons between models."""
        self.logger.info("Performing statistical tests...")
        
        comparisons = {}
        models = list(all_results.keys())
        
        for metric in self.config.metrics:
            comparisons[metric] = {}
            
            # Pairwise comparisons
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    # Get scores
                    scores1 = [ex["metrics"][metric] for ex in all_results[model1]["examples"] if metric in ex["metrics"]]
                    scores2 = [ex["metrics"][metric] for ex in all_results[model2]["examples"] if metric in ex["metrics"]]
                    
                    if len(scores1) > 10 and len(scores2) > 10:  # Need sufficient data
                        # Mann-Whitney U test
                        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                        
                        # Effect size (Cohen's d approximation)
                        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                        effect_size = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                        
                        comparison_key = f"{model1.split('/')[-1]} vs {model2.split('/')[-1]}"
                        comparisons[metric][comparison_key] = {
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "effect_size": effect_size,
                            "mean_diff": np.mean(scores1) - np.mean(scores2),
                            "better_model": model1 if np.mean(scores1) > np.mean(scores2) else model2
                        }
        
        return comparisons
    
    def create_visualizations(self, all_results: Dict[str, Any], comparisons: Dict[str, Any]):
        """Create visualization plots."""
        self.logger.info("Creating visualizations...")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Statistical Model Comparison (50 Examples)', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.config.metrics):
            ax = axes[idx]
            
            # Prepare data
            models = []
            means = []
            stds = []
            
            for model, results in all_results.items():
                metric_data = results["summary"]["metrics"].get(metric, {})
                if "mean" in metric_data:
                    models.append(model.split('/')[-1])
                    means.append(metric_data["mean"])
                    stds.append(metric_data["std"])
            
            # Create bar plot with error bars
            x = np.arange(len(models))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            
            # Color based on performance
            if metric == "toxicity":
                colors = ['green' if m < 0.1 else 'orange' if m < 0.3 else 'red' for m in means]
            else:
                colors = ['green' if m > 0.8 else 'orange' if m > 0.6 else 'red' for m in means]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Customize plot
            ax.set_title(f'{metric.capitalize()} Scores', fontsize=12)
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{mean:.3f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = self.config.output_dir / f"statistical_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved plot to {plot_file}")
    
    def generate_report(self, all_results: Dict[str, Any], comparisons: Dict[str, Any]):
        """Generate statistical report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.config.output_dir.parent.parent / "reports" / f"statistical_evaluation_{timestamp}.md"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("# Statistical Model Evaluation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Sample Size**: {self.config.sample_size} examples\n")
            f.write(f"**Models**: {', '.join([m.split('/')[-1] for m in self.config.models])}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Model | Misgendering | Toxicity | Sentiment | Regard |\n")
            f.write("|-------|-------------|----------|-----------|--------|\n")
            
            for model, results in all_results.items():
                model_name = model.split('/')[-1]
                metrics_str = []
                for metric in self.config.metrics:
                    metric_data = results["summary"]["metrics"].get(metric, {})
                    if "mean" in metric_data:
                        metrics_str.append(f"{metric_data['mean']:.3f}±{metric_data['std']:.2f}")
                    else:
                        metrics_str.append("N/A")
                f.write(f"| {model_name} | {' | '.join(metrics_str)} |\n")
            
            f.write("\n## Statistical Significance\n\n")
            for metric, metric_comparisons in comparisons.items():
                f.write(f"\n### {metric.capitalize()}\n\n")
                for comparison, stats in metric_comparisons.items():
                    significance = "✅ Significant" if stats["significant"] else "❌ Not significant"
                    better = stats["better_model"].split('/')[-1]
                    f.write(f"- **{comparison}**: {significance} (p={stats['p_value']:.3f}), ")
                    f.write(f"Effect size: {stats['effect_size']:.3f}, Better: {better}\n")
            
            f.write("\n## Methodology\n\n")
            f.write("- **Statistical Test**: Mann-Whitney U test for pairwise comparisons\n")
            f.write("- **Effect Size**: Cohen's d approximation\n")
            f.write("- **Significance Level**: α = 0.05\n")
            f.write("- **Random Seed**: Fixed for reproducibility\n")
            
            f.write("\n---\nGenerated by trans-evals statistical evaluation framework\n")
        
        self.logger.info(f"Generated report: {report_file}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete statistical evaluation."""
        self.logger.info("Starting statistical evaluation")
        
        # Load dataset
        test_batch = self.load_dataset()
        
        # Evaluate each model
        all_results = {}
        for model_name in self.config.models:
            try:
                model_results = self.evaluate_model(model_name, test_batch)
                all_results[model_name] = model_results
                
                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = self.config.output_dir / f"{model_name.replace('/', '_')}_statistical_{timestamp}.json"
                with open(result_file, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if len(all_results) < 2:
            self.logger.error("Need at least 2 models for statistical comparison")
            return {}
        
        # Perform statistical tests
        comparisons = self.perform_statistical_tests(all_results)
        
        # Create visualizations
        self.create_visualizations(all_results, comparisons)
        
        # Generate report
        self.generate_report(all_results, comparisons)
        
        return {"results": all_results, "comparisons": comparisons}


def main():
    """Run statistical evaluation."""
    print("=" * 70)
    print("STATISTICAL MODEL EVALUATION (50 Examples)")
    print("=" * 70)
    
    config = StatisticalEvalConfig(
        models=[
            "anthropic/claude-sonnet-4",
            "moonshotai/kimi-k2",
            "meta-llama/llama-3.3-70b-instruct:free"
        ],
        sample_size=50
    )
    
    try:
        evaluator = StatisticalEvaluation(config)
        results = evaluator.run_evaluation()
        
        if results:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETED")
            print("=" * 70)
            
            # Print summary
            for model, model_results in results["results"].items():
                print(f"\n{model.split('/')[-1]}:")
                for metric, stats in model_results["summary"]["metrics"].items():
                    print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            print(f"\nResults saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())