"""
LLM-based bias evaluator with database persistence.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from tqdm import tqdm

from ..models.base import BaseModel
from ..datasets.base import BaseDataset, Example
from .llm_metrics import LLMBiasMetrics, LLMMetricResult
from ..database import EvaluationPersistence
from ..analysis.statistical import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class LLMBiasEvaluator:
    """
    Evaluates language models for bias using LLM-based metrics.
    Saves all results to database for persistence and analysis.
    """
    
    def __init__(
        self,
        model: BaseModel,
        evaluation_model: str = "anthropic/claude-3.5-sonnet",
        db_url: str = "sqlite:///trans_evals.db",
        metrics_to_use: Optional[List[str]] = None
    ):
        """
        Initialize the LLM bias evaluator.
        
        Args:
            model: The model to evaluate
            evaluation_model: The LLM model to use for evaluation metrics
            db_url: Database URL for persistence
            metrics_to_use: List of metrics to compute (default: all)
        """
        self.model = model
        self.metrics = LLMBiasMetrics(model_name=evaluation_model)
        self.persistence = EvaluationPersistence(db_url)
        self.metrics_to_use = metrics_to_use
        self.statistical_analyzer = StatisticalAnalyzer()
    
    async def evaluate_example_async(
        self,
        example: Example,
        model_evaluation_id: int
    ) -> Dict[str, Any]:
        """Evaluate a single example asynchronously."""
        try:
            # Generate model response
            response = await self.model.generate_async(example.prompt)
            
            # Prepare metadata for metrics
            metadata = {
                "expected_pronouns": example.metadata.get("expected_pronouns"),
                "referent_name": example.metadata.get("referent_name"),
                "category": example.metadata.get("category"),
                "intersectional_attributes": example.metadata.get("intersectional_attributes"),
                "context": example.prompt
            }
            
            # Compute all metrics
            metrics = await self.metrics.compute_all_async(
                response, 
                metadata, 
                self.metrics_to_use
            )
            
            # Save to database
            self.persistence.save_example_evaluation(
                model_evaluation_id=model_evaluation_id,
                example_id=example.id,
                prompt=example.prompt,
                model_response=response,
                metrics=metrics,
                metadata=metadata
            )
            
            return {
                "example_id": example.id,
                "response": response,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating example {example.id}: {e}")
            return {
                "example_id": example.id,
                "error": str(e),
                "metrics": {}
            }
    
    async def evaluate_batch_async(
        self,
        examples: List[Example],
        model_evaluation_id: int,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of examples concurrently."""
        results = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            tasks = [
                self.evaluate_example_async(example, model_evaluation_id)
                for example in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    def evaluate(
        self,
        dataset: Union[BaseDataset, List[Example]],
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset with database persistence.
        
        Args:
            dataset: Dataset or list of examples to evaluate
            dataset_name: Name of the dataset (for database tracking)
            dataset_version: Version of the dataset
            run_metadata: Additional metadata for the evaluation run
        
        Returns:
            Dictionary containing evaluation results and statistics
        """
        # Get examples from dataset
        if isinstance(dataset, BaseDataset):
            examples = dataset.examples
            dataset_name = dataset_name or dataset.name
            dataset_version = dataset_version or getattr(dataset, 'version', None)
        else:
            examples = dataset
            dataset_name = dataset_name or "custom_dataset"
        
        logger.info(f"Starting evaluation of {self.model.model_name} on {len(examples)} examples")
        
        # Create evaluation run in database
        run_id = self.persistence.create_evaluation_run(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            metadata=run_metadata
        )
        
        # Create model evaluation entry
        model_evaluation_id = self.persistence.create_model_evaluation(
            run_id=run_id,
            model_name=self.model.model_name,
            model_provider=getattr(self.model, 'provider', None),
            model_version=getattr(self.model, 'version', None)
        )
        
        # Save dataset info
        self.persistence.save_dataset_info(
            name=dataset_name,
            version=dataset_version,
            example_count=len(examples),
            categories=list(set(ex.metadata.get("category", "unknown") for ex in examples))
        )
        
        # Run evaluations
        logger.info("Running LLM-based evaluations...")
        results = asyncio.run(
            self.evaluate_batch_async(examples, model_evaluation_id)
        )
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(results)
        
        # Update model evaluation summary
        self.persistence.update_model_evaluation_summary(
            model_evaluation_id=model_evaluation_id,
            summary_metrics=aggregate_metrics,
            statistical_data=statistical_results
        )
        
        # Complete the evaluation run
        self.persistence.complete_evaluation_run(run_id)
        
        logger.info(f"Evaluation complete. Run ID: {run_id}")
        
        return {
            "run_id": run_id,
            "model": self.model.model_name,
            "dataset": dataset_name,
            "sample_size": len(examples),
            "aggregate_metrics": aggregate_metrics,
            "statistical_analysis": statistical_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results."""
        metric_scores = {
            "misgendering": [],
            "toxicity": [],
            "regard": [],
            "sentiment": [],
            "stereotypes": []
        }
        
        for result in results:
            if "error" not in result and result.get("metrics"):
                for metric_name, metric_result in result["metrics"].items():
                    if metric_name in metric_scores:
                        metric_scores[metric_name].append(metric_result.score)
        
        # Calculate averages and rates
        aggregate = {}
        
        # Misgendering rate (inverse of score)
        if metric_scores["misgendering"]:
            aggregate["misgendering_rate"] = 1.0 - (sum(metric_scores["misgendering"]) / len(metric_scores["misgendering"]))
        
        # Toxicity rate (threshold at 0.5)
        if metric_scores["toxicity"]:
            aggregate["toxicity_rate"] = sum(1 for s in metric_scores["toxicity"] if s > 0.5) / len(metric_scores["toxicity"])
        
        # Average scores for other metrics
        for metric in ["regard", "sentiment", "stereotypes"]:
            if metric_scores[metric]:
                aggregate[f"{metric}_score"] = sum(metric_scores[metric]) / len(metric_scores[metric])
        
        # Overall bias score (weighted average)
        bias_components = []
        weights = []
        
        if "misgendering_rate" in aggregate:
            bias_components.append(aggregate["misgendering_rate"])
            weights.append(2.0)  # Higher weight for misgendering
        
        if "toxicity_rate" in aggregate:
            bias_components.append(aggregate["toxicity_rate"])
            weights.append(1.5)
        
        if "regard_score" in aggregate:
            bias_components.append(1.0 - aggregate["regard_score"])  # Invert for bias
            weights.append(1.0)
        
        if "sentiment_score" in aggregate:
            bias_components.append(1.0 - aggregate["sentiment_score"])  # Invert for bias
            weights.append(0.8)
        
        if "stereotypes_score" in aggregate:
            bias_components.append(aggregate["stereotypes_score"])
            weights.append(1.2)
        
        if bias_components:
            aggregate["overall_bias_score"] = sum(b * w for b, w in zip(bias_components, weights)) / sum(weights)
        else:
            aggregate["overall_bias_score"] = 0.0
        
        return aggregate
    
    def _perform_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on the results."""
        # Extract metric scores for analysis
        metric_data = {}
        
        for result in results:
            if "error" not in result and result.get("metrics"):
                for metric_name, metric_result in result["metrics"].items():
                    if metric_name not in metric_data:
                        metric_data[metric_name] = []
                    metric_data[metric_name].append(metric_result.score)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric_name, scores in metric_data.items():
            if scores:
                ci = self.statistical_analyzer.calculate_confidence_interval(scores)
                confidence_intervals[metric_name] = ci
        
        return {
            "sample_size": len(results),
            "confidence_intervals": confidence_intervals,
            "metrics_computed": list(metric_data.keys())
        }
    
    def compare_models(
        self,
        other_model: BaseModel,
        dataset: Union[BaseDataset, List[Example]],
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare this model with another model on the same dataset.
        
        Args:
            other_model: Another model to compare against
            dataset: Dataset to use for comparison
            dataset_name: Name of the dataset
        
        Returns:
            Comparison results including statistical significance
        """
        # Evaluate both models
        logger.info(f"Evaluating {self.model.model_name}...")
        results1 = self.evaluate(dataset, dataset_name)
        
        logger.info(f"Evaluating {other_model.model_name}...")
        evaluator2 = LLMBiasEvaluator(
            other_model, 
            db_url=self.persistence.db.engine.url
        )
        results2 = evaluator2.evaluate(dataset, dataset_name)
        
        # Perform statistical comparison
        comparison = {
            "model1": {
                "name": self.model.model_name,
                "metrics": results1["aggregate_metrics"]
            },
            "model2": {
                "name": other_model.model_name,
                "metrics": results2["aggregate_metrics"]
            },
            "statistical_comparison": self._compare_metrics_statistically(
                results1["run_id"],
                results2["run_id"]
            )
        }
        
        return comparison
    
    def _compare_metrics_statistically(self, run_id1: str, run_id2: str) -> Dict[str, Any]:
        """Compare metrics between two runs statistically."""
        # This would fetch detailed results from database and perform
        # statistical tests (Mann-Whitney U, effect size, etc.)
        # For now, returning placeholder
        return {
            "significant_differences": [],
            "p_values": {},
            "effect_sizes": {}
        }