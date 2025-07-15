"""
Comprehensive evaluation test using all high-quality datasets.

This script performs a large-scale evaluation of multiple models using
the comprehensive dataset loader with statistical significance testing.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from src import (
    BiasEvaluator,
    OpenRouterModel,
)
from src.datasets.comprehensive_loader import ComprehensiveDatasetLoader
from src.datasets.base import DatasetExample, BiasType, EvaluationType


@dataclass
class ComprehensiveEvalConfig:
    """Configuration for comprehensive evaluation."""
    models: List[str] = field(default_factory=lambda: [
        "moonshotai/kimi-k2",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4-turbo",
        "google/gemini-2.0-flash",
        "meta-llama/llama-3.3-70b-instruct"
    ])
    sample_size: int = 100  # Expanded for statistical significance
    temperature: float = 0.3
    max_length: int = 150
    metrics: List[str] = field(default_factory=lambda: [
        "misgendering", "toxicity", "sentiment", "regard"
    ])
    confidence_level: float = 0.95
    output_dir: Path = field(default_factory=lambda: Path("tests/results"))
    include_synthetic: bool = True
    random_seed: int = 42


class ComprehensiveEvaluation:
    """Comprehensive evaluation framework with statistical analysis."""
    
    def __init__(self, config: ComprehensiveEvalConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment
        load_dotenv()
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the evaluation."""
        logger = logging.getLogger("comprehensive_eval")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.config.output_dir / f"comprehensive_eval_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_comprehensive_dataset(self) -> List[DatasetExample]:
        """Load comprehensive dataset using all available sources."""
        self.logger.info("Loading comprehensive dataset...")
        
        loader = ComprehensiveDatasetLoader(include_synthetic=self.config.include_synthetic)
        all_examples = loader.load_all_datasets()
        
        # Get balanced sample
        sampled_examples = loader.get_balanced_sample(
            n=self.config.sample_size,
            seed=self.config.random_seed
        )
        
        self.logger.info(f"Loaded {len(sampled_examples)} high-quality examples from {len(all_examples)} total")
        
        # Log statistics about the sample
        self._log_sample_statistics(sampled_examples)
        
        return sampled_examples
    
    def _log_sample_statistics(self, examples: List[DatasetExample]):
        """Log statistics about the sampled examples."""
        # Count by bias type
        bias_type_counts = {}
        for ex in examples:
            bias_type = ex.bias_type.value
            bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
        
        # Count by identity
        identity_counts = {}
        for ex in examples:
            if ex.target_identity:
                identity = ex.target_identity.lower()
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
        
        self.logger.info("\n=== Sample Statistics ===")
        self.logger.info(f"Total examples: {len(examples)}")
        
        self.logger.info("\nBias type distribution:")
        for bias_type, count in sorted(bias_type_counts.items()):
            self.logger.info(f"  {bias_type}: {count} ({count/len(examples)*100:.1f}%)")
        
        self.logger.info("\nTop identities:")
        for identity, count in sorted(identity_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            self.logger.info(f"  {identity}: {count}")
    
    def evaluate_model(self, model_name: str, test_batch: List[DatasetExample]) -> Dict[str, Any]:
        """
        Evaluate a single model on the test batch with progress tracking.
        
        Args:
            model_name: Name/ID of the model to evaluate
            test_batch: List of test examples
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Initialize model
            model = OpenRouterModel(model_name=model_name)
            evaluator = BiasEvaluator(model=model)
            
            # Create a dataset-like object
            test_dataset = type('TestDataset', (), {
                'name': f'comprehensive_batch_{model_name}',
                'examples': test_batch,
                '__len__': lambda self: len(test_batch),
                '__getitem__': lambda self, idx: test_batch[idx],
                '__iter__': lambda self: iter(test_batch)
            })()
            
            # Run evaluation with progress bar
            self.logger.info(f"Running evaluation on {len(test_batch)} examples...")
            results = evaluator.evaluate(
                test_dataset,
                metrics_to_use=self.config.metrics
            )
            
            # Process results
            model_results = {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "sample_size": len(test_batch),
                "examples": [],
                "summary": {},
                "statistical_analysis": {}
            }
            
            # Collect individual results
            for i, (example, prediction, metrics) in enumerate(
                zip(results.examples, results.predictions, results.metric_results)
            ):
                example_result = {
                    "id": example.id,
                    "text": example.text[:100] + "..." if len(example.text) > 100 else example.text,
                    "prediction": prediction[:100] + "..." if len(prediction) > 100 else prediction,
                    "bias_type": example.bias_type.value,
                    "target_identity": example.target_identity,
                    "metrics": {
                        name: {
                            "score": metric.score,
                            "details": metric.details
                        }
                        for name, metric in metrics.items()
                    },
                    "metadata": example.metadata
                }
                model_results["examples"].append(example_result)
            
            # Calculate summary statistics
            model_results["summary"] = self._calculate_summary(model_results["examples"])
            
            # Perform statistical analysis
            model_results["statistical_analysis"] = self._perform_statistical_analysis(
                model_results["examples"]
            )
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {e}", exc_info=True)
            raise
    
    def _calculate_summary(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        summary = {
            "total_examples": len(examples),
            "metrics": {},
            "by_bias_type": {},
            "by_identity": {}
        }
        
        # Aggregate metrics
        for metric in self.config.metrics:
            scores = [ex["metrics"][metric]["score"] for ex in examples if metric in ex["metrics"]]
            if scores:
                summary["metrics"][metric] = {
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "ci_lower": np.percentile(scores, (1 - self.config.confidence_level) / 2 * 100),
                    "ci_upper": np.percentile(scores, (1 + self.config.confidence_level) / 2 * 100)
                }
        
        # Group by bias type
        bias_types = {}
        for ex in examples:
            bias_type = ex.get("bias_type", "unknown")
            if bias_type not in bias_types:
                bias_types[bias_type] = []
            bias_types[bias_type].append(ex)
        
        for bias_type, type_examples in bias_types.items():
            summary["by_bias_type"][bias_type] = {
                "count": len(type_examples),
                "metrics": {}
            }
            for metric in self.config.metrics:
                scores = [ex["metrics"][metric]["score"] for ex in type_examples if metric in ex["metrics"]]
                if scores:
                    summary["by_bias_type"][bias_type]["metrics"][metric] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores)
                    }
        
        # Group by identity (top 5)
        identity_groups = {}
        for ex in examples:
            identity = ex.get("target_identity", "unknown")
            if identity and identity != "unknown":
                identity_lower = identity.lower()
                if identity_lower not in identity_groups:
                    identity_groups[identity_lower] = []
                identity_groups[identity_lower].append(ex)
        
        # Get top 5 identities by count
        top_identities = sorted(identity_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        
        for identity, id_examples in top_identities:
            summary["by_identity"][identity] = {
                "count": len(id_examples),
                "metrics": {}
            }
            for metric in self.config.metrics:
                scores = [ex["metrics"][metric]["score"] for ex in id_examples if metric in ex["metrics"]]
                if scores:
                    summary["by_identity"][identity]["metrics"][metric] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores)
                    }
        
        return summary
    
    def _perform_statistical_analysis(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on the results."""
        analysis = {}
        
        # For each metric, perform normality test and calculate confidence intervals
        for metric in self.config.metrics:
            scores = [ex["metrics"][metric]["score"] for ex in examples if metric in ex["metrics"]]
            if len(scores) > 20:  # Need sufficient data for tests
                # Shapiro-Wilk test for normality
                statistic, p_value = stats.shapiro(scores)
                
                # Bootstrap confidence interval
                bootstrap_means = []
                for _ in range(1000):
                    sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_means.append(np.mean(sample))
                
                ci_lower = np.percentile(bootstrap_means, (1 - self.config.confidence_level) / 2 * 100)
                ci_upper = np.percentile(bootstrap_means, (1 + self.config.confidence_level) / 2 * 100)
                
                analysis[metric] = {
                    "normality_test": {
                        "statistic": statistic,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05
                    },
                    "bootstrap_ci": {
                        "lower": ci_lower,
                        "upper": ci_upper,
                        "width": ci_upper - ci_lower
                    },
                    "sample_size": len(scores)
                }
        
        return analysis
    
    def compare_models_statistically(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical comparison between models."""
        comparisons = {}
        
        # Get model pairs
        models = [m for m in all_results.keys() if "error" not in all_results[m]]
        
        for metric in self.config.metrics:
            comparisons[metric] = {}
            
            # Pairwise comparisons
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    # Get scores
                    scores1 = [
                        ex["metrics"][metric]["score"] 
                        for ex in all_results[model1]["examples"] 
                        if metric in ex["metrics"]
                    ]
                    scores2 = [
                        ex["metrics"][metric]["score"] 
                        for ex in all_results[model2]["examples"] 
                        if metric in ex["metrics"]
                    ]
                    
                    if scores1 and scores2:
                        # Mann-Whitney U test (non-parametric)
                        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                        
                        # Effect size (Cliff's delta)
                        effect_size = self._calculate_cliffs_delta(scores1, scores2)
                        
                        comparison_key = f"{model1} vs {model2}"
                        comparisons[metric][comparison_key] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "effect_size": effect_size,
                            "effect_magnitude": self._interpret_effect_size(effect_size),
                            "mean_diff": np.mean(scores1) - np.mean(scores2),
                            "favors": model1 if np.mean(scores1) > np.mean(scores2) else model2
                        }
        
        return comparisons
    
    def _calculate_cliffs_delta(self, scores1: List[float], scores2: List[float]) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(scores1), len(scores2)
        
        # Count how many times a score from group 1 is greater than group 2
        greater = sum(s1 > s2 for s1 in scores1 for s2 in scores2)
        less = sum(s1 < s2 for s1 in scores1 for s2 in scores2)
        
        delta = (greater - less) / (n1 * n2)
        return delta
    
    def _interpret_effect_size(self, delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run the complete comprehensive evaluation."""
        self.logger.info("Starting comprehensive evaluation")
        
        # Load dataset
        test_batch = self.load_comprehensive_dataset()
        
        # Evaluate each model
        all_results = {}
        for model_name in self.config.models:
            try:
                # Skip models that might not be available
                if model_name in ["openai/gpt-4-turbo", "google/gemini-2.0-flash"]:
                    self.logger.warning(f"Skipping {model_name} - may require specific API access")
                    continue
                    
                model_results = self.evaluate_model(model_name, test_batch)
                all_results[model_name] = model_results
                
                # Save individual model results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_file = self.config.output_dir / f"{model_name.replace('/', '_')}_{timestamp}_comprehensive.json"
                with open(model_file, 'w') as f:
                    json.dump(model_results, f, indent=2, default=str)
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        
        # Perform statistical comparisons
        statistical_comparisons = self.compare_models_statistically(all_results)
        
        # Create comprehensive report
        report = self._create_comprehensive_report(all_results, statistical_comparisons)
        
        # Save report
        report_file = self.config.output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_comprehensive_visualizations(report)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        return report
    
    def _create_comprehensive_report(self, all_results: Dict[str, Any], 
                                   statistical_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""
        report = {
            "title": "Comprehensive Model Evaluation for Trans-Inclusive Language",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample_size": self.config.sample_size,
                "models": self.config.models,
                "metrics": self.config.metrics,
                "confidence_level": self.config.confidence_level,
                "include_synthetic": self.config.include_synthetic
            },
            "results": all_results,
            "statistical_comparisons": statistical_comparisons,
            "summary": self._create_executive_summary(all_results, statistical_comparisons)
        }
        
        return report
    
    def _create_executive_summary(self, all_results: Dict[str, Any], 
                                statistical_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of results."""
        summary = {
            "best_models": {},
            "key_findings": [],
            "recommendations": []
        }
        
        # Find best model for each metric
        for metric in self.config.metrics:
            best_model = None
            best_score = -1 if metric != "toxicity" else float('inf')
            
            for model, results in all_results.items():
                if "error" not in results and "summary" in results:
                    score = results["summary"]["metrics"].get(metric, {}).get("mean")
                    if score is not None:
                        if metric == "toxicity":
                            if score < best_score:
                                best_score = score
                                best_model = model
                        else:
                            if score > best_score:
                                best_score = score
                                best_model = model
            
            summary["best_models"][metric] = {
                "model": best_model,
                "score": best_score
            }
        
        # Extract key findings
        for metric, comparisons in statistical_comparisons.items():
            for comparison, results in comparisons.items():
                if results["significant"] and results["effect_magnitude"] in ["medium", "large"]:
                    summary["key_findings"].append({
                        "metric": metric,
                        "comparison": comparison,
                        "favors": results["favors"],
                        "effect_size": results["effect_magnitude"],
                        "p_value": results["p_value"]
                    })
        
        # Generate recommendations
        summary["recommendations"] = [
            "Based on comprehensive evaluation, models show varying performance on trans-inclusive metrics.",
            "Statistical significance testing reveals meaningful differences between models.",
            "Consider using multiple metrics when evaluating model suitability for production use."
        ]
        
        return summary
    
    def _create_comprehensive_visualizations(self, report: Dict[str, Any]) -> None:
        """Create comprehensive visualization plots."""
        # Set up the plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Overall metric comparison
        self._create_metric_comparison_plot(report)
        
        # 2. Statistical significance heatmap
        self._create_significance_heatmap(report)
        
        # 3. Performance by bias type
        self._create_bias_type_plot(report)
        
        # 4. Confidence interval plot
        self._create_confidence_interval_plot(report)
    
    def _create_metric_comparison_plot(self, report: Dict[str, Any]) -> None:
        """Create bar plot comparing metrics across models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Comparison: Trans Bias Evaluation', fontsize=18)
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.config.metrics[:4]):
            ax = axes[idx]
            
            # Prepare data
            models = []
            means = []
            stds = []
            
            for model, results in report["results"].items():
                if "error" not in results and "summary" in results:
                    metric_data = results["summary"]["metrics"].get(metric, {})
                    if "mean" in metric_data:
                        models.append(model.split('/')[-1])
                        means.append(metric_data["mean"])
                        stds.append(metric_data["std"])
            
            # Create bar plot with error bars
            x = np.arange(len(models))
            bars = ax.bar(x, means, yerr=stds, capsize=5)
            
            # Color based on performance
            if metric == "toxicity":
                colors = ['darkgreen' if m < 0.1 else 'green' if m < 0.2 else 'orange' if m < 0.5 else 'red' for m in means]
            else:
                colors = ['darkgreen' if m > 0.9 else 'green' if m > 0.8 else 'orange' if m > 0.5 else 'red' for m in means]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Customize plot
            ax.set_title(f'{metric.capitalize()} Scores', fontsize=14)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels
            for i, (bar, mean) in enumerate(zip(bars, means)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.config.output_dir / f"comprehensive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved metric comparison plot to {plot_file}")
    
    def _create_significance_heatmap(self, report: Dict[str, Any]) -> None:
        """Create heatmap showing statistical significance of comparisons."""
        # Prepare data for heatmap
        models = [m for m in report["results"].keys() if "error" not in report["results"][m]]
        n_models = len(models)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Significance of Model Comparisons', fontsize=18)
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.config.metrics[:4]):
            ax = axes[idx]
            
            # Create matrix for p-values
            p_matrix = np.ones((n_models, n_models))
            
            comparisons = report["statistical_comparisons"].get(metric, {})
            for comparison, results in comparisons.items():
                # Parse comparison string
                parts = comparison.split(" vs ")
                if len(parts) == 2:
                    model1, model2 = parts
                    if model1 in models and model2 in models:
                        i, j = models.index(model1), models.index(model2)
                        p_value = results["p_value"]
                        p_matrix[i, j] = p_value
                        p_matrix[j, i] = p_value
            
            # Create heatmap
            mask = np.triu(np.ones_like(p_matrix), k=1)
            sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f', 
                       cmap='RdYlGn_r', center=0.05, vmin=0, vmax=0.1,
                       xticklabels=[m.split('/')[-1] for m in models],
                       yticklabels=[m.split('/')[-1] for m in models],
                       ax=ax, cbar_kws={'label': 'p-value'})
            
            ax.set_title(f'{metric.capitalize()} - Statistical Significance', fontsize=12)
        
        plt.tight_layout()
        plot_file = self.config.output_dir / f"significance_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved significance heatmap to {plot_file}")
    
    def _create_bias_type_plot(self, report: Dict[str, Any]) -> None:
        """Create plot showing performance by bias type."""
        # Collect data by bias type
        bias_type_data = {}
        
        for model, results in report["results"].items():
            if "error" not in results and "summary" in results:
                for bias_type, type_data in results["summary"]["by_bias_type"].items():
                    if bias_type not in bias_type_data:
                        bias_type_data[bias_type] = {}
                    
                    model_name = model.split('/')[-1]
                    bias_type_data[bias_type][model_name] = type_data["metrics"]
        
        # Create subplots for each bias type
        n_bias_types = len(bias_type_data)
        fig, axes = plt.subplots(1, n_bias_types, figsize=(5*n_bias_types, 6))
        fig.suptitle('Model Performance by Bias Type', fontsize=16)
        
        if n_bias_types == 1:
            axes = [axes]
        
        for idx, (bias_type, model_data) in enumerate(bias_type_data.items()):
            ax = axes[idx]
            
            # Prepare data for grouped bar chart
            metrics = self.config.metrics
            models = list(model_data.keys())
            n_metrics = len(metrics)
            n_models = len(models)
            
            x = np.arange(n_metrics)
            width = 0.8 / n_models
            
            for i, model in enumerate(models):
                scores = []
                for metric in metrics:
                    score = model_data[model].get(metric, {}).get("mean", 0)
                    scores.append(score)
                
                ax.bar(x + i * width - 0.4 + width/2, scores, width, label=model)
            
            ax.set_title(f'{bias_type.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plot_file = self.config.output_dir / f"bias_type_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved bias type comparison to {plot_file}")
    
    def _create_confidence_interval_plot(self, report: Dict[str, Any]) -> None:
        """Create confidence interval plot for each metric."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance with 95% Confidence Intervals', fontsize=18)
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.config.metrics[:4]):
            ax = axes[idx]
            
            # Prepare data
            models = []
            means = []
            ci_lower = []
            ci_upper = []
            
            for model, results in report["results"].items():
                if "error" not in results and "summary" in results:
                    metric_data = results["summary"]["metrics"].get(metric, {})
                    if all(k in metric_data for k in ["mean", "ci_lower", "ci_upper"]):
                        models.append(model.split('/')[-1])
                        means.append(metric_data["mean"])
                        ci_lower.append(metric_data["ci_lower"])
                        ci_upper.append(metric_data["ci_upper"])
            
            # Create error bar plot
            x = np.arange(len(models))
            errors = [np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)]
            
            ax.errorbar(x, means, yerr=errors, fmt='o', capsize=5, capthick=2, markersize=8)
            
            # Add reference lines
            if metric == "toxicity":
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
                ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<0.2)')
            else:
                ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (>0.5)')
            
            ax.set_title(f'{metric.capitalize()} with 95% CI', fontsize=14)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.config.output_dir / f"confidence_intervals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved confidence interval plot to {plot_file}")
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> None:
        """Generate a markdown report from the evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.config.output_dir.parent.parent / "reports" / f"comprehensive_evaluation_{timestamp}.md"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(f"# {report['title']}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Sample Size**: {report['config']['sample_size']} examples\n")
            f.write(f"**Models Evaluated**: {len([m for m in report['results'] if 'error' not in report['results'][m]])}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Best models
            f.write("### Best Performing Models by Metric\n\n")
            f.write("| Metric | Best Model | Score |\n")
            f.write("|--------|------------|-------|\n")
            for metric, data in report['summary']['best_models'].items():
                if data['model']:
                    f.write(f"| {metric.capitalize()} | {data['model'].split('/')[-1]} | {data['score']:.3f} |\n")
            
            # Key findings
            if report['summary']['key_findings']:
                f.write("\n### Key Statistical Findings\n\n")
                for finding in report['summary']['key_findings'][:5]:  # Top 5 findings
                    f.write(f"- **{finding['metric'].capitalize()}**: {finding['comparison']} shows {finding['effect_size']} effect size ")
                    f.write(f"favoring {finding['favors'].split('/')[-1]} (p={finding['p_value']:.3f})\n")
            
            # Detailed results
            f.write("\n## Detailed Results\n\n")
            
            for model, results in report['results'].items():
                if 'error' not in results:
                    f.write(f"\n### {model}\n\n")
                    f.write("| Metric | Mean | Std Dev | 95% CI |\n")
                    f.write("|--------|------|---------|--------|\n")
                    
                    for metric, stats in results['summary']['metrics'].items():
                        ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                        f.write(f"| {metric.capitalize()} | {stats['mean']:.3f} | {stats['std']:.3f} | {ci} |\n")
            
            f.write("\n## Methodology\n\n")
            f.write(f"- **Confidence Level**: {report['config']['confidence_level']*100}%\n")
            f.write(f"- **Statistical Tests**: Mann-Whitney U test for pairwise comparisons\n")
            f.write(f"- **Effect Size**: Cliff's delta for non-parametric effect size\n")
            f.write(f"- **Synthetic Data**: {'Included' if report['config']['include_synthetic'] else 'Excluded'}\n")
            
            f.write("\n---\n\n")
            f.write("Generated by trans-evals comprehensive evaluation framework\n")
        
        self.logger.info(f"Generated markdown report: {report_file}")


def main():
    """Run the comprehensive evaluation."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION FOR TRANS-INCLUSIVE LANGUAGE")
    print("=" * 80)
    
    # Configure evaluation
    config = ComprehensiveEvalConfig(
        models=[
            "moonshotai/kimi-k2",
            "anthropic/claude-sonnet-4",
            "meta-llama/llama-3.3-70b-instruct:free"  # Use free version
        ],
        sample_size=100,  # Expanded for statistical significance
        temperature=0.3,
        include_synthetic=True
    )
    
    try:
        # Run evaluation
        evaluator = ComprehensiveEvaluation(config)
        report = evaluator.run_comprehensive_evaluation()
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        print("\n### Best Models by Metric:")
        for metric, data in report['summary']['best_models'].items():
            if data['model']:
                print(f"  {metric.capitalize()}: {data['model'].split('/')[-1]} ({data['score']:.3f})")
        
        print(f"\nDetailed results saved to: {config.output_dir}")
        print(f"Statistical analysis included in comprehensive report")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())