from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from ..datasets.base import BaseDataset, DatasetExample, BiasType
from ..models.base import BaseModel
from .metrics import BiasMetrics, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from bias evaluation."""
    dataset_name: str
    model_name: str
    timestamp: str
    examples: List[DatasetExample] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    metric_results: List[Dict[str, MetricResult]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        rows = []
        for i, (example, prediction, metrics) in enumerate(zip(self.examples, self.predictions, self.metric_results)):
            row = {
                "example_id": example.id,
                "text": example.text,
                "prediction": prediction,
                "bias_type": example.bias_type.value,
                "target_identity": example.target_identity
            }
            
            # Add metric scores
            for metric_name, result in metrics.items():
                row[f"{metric_name}_score"] = result.score
                if result.confidence is not None:
                    row[f"{metric_name}_confidence"] = result.confidence
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_report(self, output_path: str) -> None:
        """Generate HTML report of evaluation results."""
        df = self.to_dataframe()
        
        # Calculate summary statistics
        summary_html = f"""
        <html>
        <head>
            <title>Bias Evaluation Report - {self.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3ff; }}
                .warning {{ background-color: #fff3cd; }}
                .danger {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Bias Evaluation Report</h1>
            <p><strong>Dataset:</strong> {self.dataset_name}</p>
            <p><strong>Model:</strong> {self.model_name}</p>
            <p><strong>Timestamp:</strong> {self.timestamp}</p>
            <p><strong>Total Examples:</strong> {len(self.examples)}</p>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean Score</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
        """
        
        # Add metric statistics
        metric_columns = [col for col in df.columns if col.endswith('_score')]
        for metric_col in metric_columns:
            metric_name = metric_col.replace('_score', '')
            mean_score = df[metric_col].mean()
            std_score = df[metric_col].std()
            min_score = df[metric_col].min()
            max_score = df[metric_col].max()
            
            # Determine row class based on score
            row_class = ""
            if metric_name == "toxicity" and mean_score > 0.5:
                row_class = "danger"
            elif metric_name == "misgendering" and mean_score < 0.8:
                row_class = "warning"
            
            summary_html += f"""
                <tr class="{row_class}">
                    <td>{metric_name}</td>
                    <td>{mean_score:.3f}</td>
                    <td>{std_score:.3f}</td>
                    <td>{min_score:.3f}</td>
                    <td>{max_score:.3f}</td>
                </tr>
            """
        
        summary_html += """
            </table>
            
            <h2>Results by Bias Type</h2>
        """
        
        # Group by bias type
        for bias_type in df['bias_type'].unique():
            bias_df = df[df['bias_type'] == bias_type]
            summary_html += f"<h3>{bias_type}</h3>"
            summary_html += f"<p>Number of examples: {len(bias_df)}</p>"
            
            # Show metric averages for this bias type
            summary_html += "<table><tr><th>Metric</th><th>Average Score</th></tr>"
            for metric_col in metric_columns:
                metric_name = metric_col.replace('_score', '')
                avg_score = bias_df[metric_col].mean()
                summary_html += f"<tr><td>{metric_name}</td><td>{avg_score:.3f}</td></tr>"
            summary_html += "</table>"
        
        summary_html += """
            <h2>Results by Target Identity</h2>
        """
        
        # Group by target identity
        for identity in df['target_identity'].dropna().unique():
            identity_df = df[df['target_identity'] == identity]
            summary_html += f"<h3>{identity}</h3>"
            summary_html += f"<p>Number of examples: {len(identity_df)}</p>"
            
            # Show metric averages for this identity
            summary_html += "<table><tr><th>Metric</th><th>Average Score</th></tr>"
            for metric_col in metric_columns:
                metric_name = metric_col.replace('_score', '')
                avg_score = identity_df[metric_col].mean()
                summary_html += f"<tr><td>{metric_name}</td><td>{avg_score:.3f}</td></tr>"
            summary_html += "</table>"
        
        summary_html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(summary_html)
        
        logger.info(f"Report saved to {output_path}")
    
    def save_csv(self, output_path: str) -> None:
        """Save results as CSV."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


class BiasEvaluator:
    """Main evaluator for bias evaluation."""
    
    def __init__(
        self,
        model: BaseModel,
        metrics: Optional[BiasMetrics] = None,
        batch_size: int = 32
    ):
        self.model = model
        self.metrics = metrics or BiasMetrics()
        self.batch_size = batch_size
    
    def evaluate(
        self,
        dataset: BaseDataset,
        max_examples: Optional[int] = None,
        metrics_to_use: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Dataset to evaluate on
            max_examples: Maximum number of examples to evaluate
            metrics_to_use: List of metrics to compute
            
        Returns:
            EvaluationResult with all metrics
        """
        examples = list(dataset)
        if max_examples:
            examples = examples[:max_examples]
        
        result = EvaluationResult(
            dataset_name=dataset.name,
            model_name=self.model.name,
            timestamp=datetime.now().isoformat()
        )
        
        # Generate predictions
        logger.info(f"Generating predictions for {len(examples)} examples...")
        for example in tqdm(examples, desc="Generating"):
            prediction = self.model.generate(example)
            result.examples.append(example)
            result.predictions.append(prediction)
        
        # Compute metrics
        logger.info("Computing metrics...")
        for example, prediction in tqdm(
            zip(result.examples, result.predictions),
            desc="Computing metrics",
            total=len(result.examples)
        ):
            # Prepare metadata for metrics
            metadata = {
                "example": example,
                **(example.metadata or {})
            }
            
            # Add pronoun information if available
            if example.pronoun:
                metadata["expected_pronouns"] = example.pronoun
            if example.referent:
                metadata["referent_name"] = example.referent
            
            # Compute metrics
            metric_results = self.metrics.compute_all(
                prediction,
                metadata,
                metrics_to_use
            )
            result.metric_results.append(metric_results)
        
        # Compute summary statistics
        result.summary_stats = self._compute_summary_stats(result)
        
        return result
    
    def _compute_summary_stats(self, result: EvaluationResult) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        stats = {
            "total_examples": len(result.examples),
            "by_bias_type": {},
            "by_identity": {},
            "overall_metrics": {}
        }
        
        # Group by bias type
        bias_type_counts = {}
        for example in result.examples:
            bias_type = example.bias_type.value
            bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
        stats["by_bias_type"] = bias_type_counts
        
        # Group by identity
        identity_counts = {}
        for example in result.examples:
            if example.target_identity:
                identity_counts[example.target_identity] = identity_counts.get(example.target_identity, 0) + 1
        stats["by_identity"] = identity_counts
        
        # Overall metric averages
        if result.metric_results:
            metric_sums = {}
            metric_counts = {}
            
            for metrics in result.metric_results:
                for metric_name, metric_result in metrics.items():
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0
                        metric_counts[metric_name] = 0
                    metric_sums[metric_name] += metric_result.score
                    metric_counts[metric_name] += 1
            
            for metric_name in metric_sums:
                stats["overall_metrics"][metric_name] = {
                    "average": metric_sums[metric_name] / metric_counts[metric_name],
                    "count": metric_counts[metric_name]
                }
        
        return stats