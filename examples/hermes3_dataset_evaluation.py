"""
Evaluate Hermes 3 dataset for trans-inclusive language bias using trans-evals framework.

This script analyzes a sample of the Hermes 3 dataset to identify potential biases,
misgendering, stereotypes, and other issues related to trans and non-binary representation.
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from trans_evals.evaluation.llm_evaluator import LLMBiasEvaluator
from trans_evals.evaluation.llm_metrics import LLMBiasMetrics
from trans_evals.database import EvaluationPersistence
from trans_evals.datasets.base import Example
from trans_evals import OpenRouterModel


class Hermes3DatasetEvaluator:
    """Evaluates Hermes 3 dataset for trans-inclusive language bias."""
    
    def __init__(
        self,
        dataset_path: str,
        sample_size: int = 100,
        evaluation_model: str = "anthropic/claude-3.5-sonnet",
        db_url: str = "sqlite:///hermes3_trans_eval.db",
        random_seed: int = 42
    ):
        """
        Initialize Hermes 3 dataset evaluator.
        
        Args:
            dataset_path: Path to Hermes 3 dataset file (JSON, JSONL, or Parquet)
            sample_size: Number of samples to evaluate
            evaluation_model: Model to use for evaluation
            db_url: Database URL for storing results
            random_seed: Random seed for reproducible sampling
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.evaluation_model = evaluation_model
        self.db_url = db_url
        self.random_seed = random_seed
        
        # Set random seed
        random.seed(random_seed)
        
        # Initialize persistence
        self.persistence = EvaluationPersistence(db_url)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load Hermes 3 dataset from file."""
        path = Path(self.dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"Loading dataset from: {self.dataset_path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    raise ValueError("Unknown JSON format")
                    
        elif path.suffix == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
            
        elif path.suffix == '.parquet':
            try:
                df = pd.read_parquet(path)
                return df.to_dict('records')
            except ImportError:
                raise ImportError("Please install pyarrow or fastparquet to read Parquet files")
                
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def extract_trans_relevant_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract samples potentially relevant to trans/non-binary evaluation.
        
        Looks for keywords and patterns in conversations that might involve
        gender, pronouns, identity, etc.
        """
        trans_keywords = [
            # Pronouns
            'pronoun', 'pronouns', 'he/him', 'she/her', 'they/them', 
            'xe/xem', 'ze/zir', 'ey/em', 'fae/faer', 'neo-pronoun',
            
            # Identity terms
            'trans', 'transgender', 'non-binary', 'nonbinary', 'enby',
            'genderqueer', 'genderfluid', 'agender', 'bigender',
            'transman', 'transwoman', 'trans man', 'trans woman',
            'mtf', 'ftm', 'afab', 'amab',
            
            # Related topics
            'gender identity', 'gender expression', 'transition',
            'deadname', 'misgender', 'coming out', 'gender dysphoria',
            'gender affirming', 'hormone therapy', 'hrt',
            
            # Common contexts
            'bathroom', 'restroom', 'sports', 'locker room',
            'discrimination', 'inclusive', 'respect', 'identity'
        ]
        
        relevant_samples = []
        
        for item in data:
            # Check if item contains conversation text
            text_to_check = ""
            
            # Handle different data formats
            if isinstance(item, dict):
                # Common fields in conversation datasets
                for field in ['text', 'conversation', 'messages', 'prompt', 'response', 'content']:
                    if field in item:
                        if isinstance(item[field], str):
                            text_to_check += item[field].lower()
                        elif isinstance(item[field], list):
                            # Handle message lists
                            for msg in item[field]:
                                if isinstance(msg, dict) and 'content' in msg:
                                    text_to_check += msg['content'].lower()
                                elif isinstance(msg, str):
                                    text_to_check += msg.lower()
            
            # Check for keywords
            if any(keyword in text_to_check for keyword in trans_keywords):
                relevant_samples.append(item)
        
        print(f"Found {len(relevant_samples)} potentially relevant samples out of {len(data)}")
        return relevant_samples
    
    def convert_to_examples(self, samples: List[Dict[str, Any]]) -> List[Example]:
        """Convert Hermes 3 samples to trans-evals Example format."""
        examples = []
        
        for i, sample in enumerate(samples):
            # Extract conversation text
            prompt = ""
            response = ""
            
            # Handle different formats
            if 'messages' in sample and isinstance(sample['messages'], list):
                # Chat format
                conversation = sample['messages']
                # Combine all messages into a prompt for evaluation
                full_text = "\n".join([
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in conversation
                ])
                prompt = full_text
                
            elif 'prompt' in sample and 'response' in sample:
                prompt = sample['prompt']
                response = sample['response']
                
            elif 'text' in sample:
                prompt = sample['text']
                
            else:
                # Try to extract any text content
                prompt = str(sample)
            
            # Create example
            example = Example(
                id=f"hermes3_{i}",
                prompt=prompt[:1000],  # Limit length
                metadata={
                    "source": "hermes3",
                    "original_index": i,
                    "category": "dataset_review",
                    "response": response[:1000] if response else None
                }
            )
            examples.append(example)
        
        return examples
    
    async def evaluate_samples(self, examples: List[Example]) -> Dict[str, Any]:
        """Evaluate samples using LLM-based metrics."""
        print(f"\nEvaluating {len(examples)} samples with {self.evaluation_model}...")
        
        # Create a mock model for dataset evaluation
        # We're not generating responses, just evaluating existing content
        mock_model = OpenRouterModel(
            model_name="gpt-3.5-turbo",  # Placeholder
            api_key=os.getenv("OPENROUTER_API_KEY", "dummy")
        )
        
        # Initialize evaluator
        evaluator = LLMBiasEvaluator(
            model=mock_model,
            evaluation_model=self.evaluation_model,
            db_url=self.db_url,
            metrics_to_use=["toxicity", "regard", "sentiment", "stereotypes"]
        )
        
        # For dataset evaluation, we'll analyze the content directly
        run_id = self.persistence.create_evaluation_run(
            dataset_name="hermes3_dataset",
            dataset_version="1.0",
            metadata={
                "evaluation_type": "dataset_content_analysis",
                "sample_size": len(examples),
                "evaluation_model": self.evaluation_model
            }
        )
        
        model_eval_id = self.persistence.create_model_evaluation(
            run_id=run_id,
            model_name="hermes3_dataset_content",
            model_provider="dataset",
            model_version="1.0"
        )
        
        # Initialize metrics directly
        metrics = LLMBiasMetrics(model_name=self.evaluation_model)
        
        # Evaluate each example
        results = []
        for example in tqdm(examples, desc="Evaluating samples"):
            try:
                # Evaluate the conversation content directly
                text_to_evaluate = example.prompt
                if example.metadata.get('response'):
                    text_to_evaluate += f"\n\nResponse: {example.metadata['response']}"
                
                # Compute metrics
                metric_results = await metrics.compute_all_async(
                    text=text_to_evaluate,
                    metadata={
                        "source": "hermes3",
                        "context": "dataset_content_review"
                    }
                )
                
                # Save results
                self.persistence.save_example_evaluation(
                    model_evaluation_id=model_eval_id,
                    example_id=example.id,
                    prompt=example.prompt[:500],  # Truncate for storage
                    model_response=example.metadata.get('response', '')[:500],
                    metrics=metric_results,
                    metadata=example.metadata
                )
                
                results.append({
                    "example_id": example.id,
                    "metrics": metric_results
                })
                
            except Exception as e:
                print(f"Error evaluating example {example.id}: {e}")
                continue
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(results)
        
        # Update evaluation summary
        self.persistence.update_model_evaluation_summary(
            model_evaluation_id=model_eval_id,
            summary_metrics=aggregate_stats['metrics'],
            statistical_data={
                "sample_size": len(results),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        )
        
        self.persistence.complete_evaluation_run(run_id)
        
        return {
            "run_id": run_id,
            "total_samples": len(examples),
            "evaluated_samples": len(results),
            "aggregate_stats": aggregate_stats,
            "issues_found": self._identify_issues(results)
        }
    
    def _calculate_aggregate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from evaluation results."""
        metrics_data = {
            "toxicity": [],
            "regard": [],
            "sentiment": [],
            "stereotypes": []
        }
        
        for result in results:
            for metric_name, metric_result in result['metrics'].items():
                if metric_name in metrics_data:
                    metrics_data[metric_name].append(metric_result.score)
        
        # Calculate statistics
        stats = {}
        for metric_name, scores in metrics_data.items():
            if scores:
                stats[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                    "problematic_count": sum(1 for s in scores if 
                        (metric_name == "toxicity" and s > 0.1) or
                        (metric_name == "regard" and s < 0.4) or
                        (metric_name == "sentiment" and s < 0.3) or
                        (metric_name == "stereotypes" and s > 0.3)
                    )
                }
        
        return {
            "metrics": {
                "toxicity_rate": stats.get("toxicity", {}).get("mean", 0),
                "regard_score": stats.get("regard", {}).get("mean", 0.5),
                "sentiment_score": stats.get("sentiment", {}).get("mean", 0.5),
                "stereotype_score": stats.get("stereotypes", {}).get("mean", 0)
            },
            "detailed_stats": stats
        }
    
    def _identify_issues(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify specific issues found in the dataset."""
        issues = {
            "high_toxicity": [],
            "low_regard": [],
            "negative_sentiment": [],
            "stereotypes": []
        }
        
        for result in results:
            example_id = result['example_id']
            metrics = result['metrics']
            
            # Check each metric
            if 'toxicity' in metrics and metrics['toxicity'].score > 0.1:
                issues['high_toxicity'].append(example_id)
                
            if 'regard' in metrics and metrics['regard'].score < 0.4:
                issues['low_regard'].append(example_id)
                
            if 'sentiment' in metrics and metrics['sentiment'].score < 0.3:
                issues['negative_sentiment'].append(example_id)
                
            if 'stereotypes' in metrics and metrics['stereotypes'].score > 0.3:
                issues['stereotypes'].append(example_id)
        
        return issues
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a detailed evaluation report."""
        report = f"""
# Hermes 3 Dataset Evaluation Report
## Trans-Inclusive Language Analysis

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: Hermes 3
**Sample Size**: {evaluation_results['total_samples']}
**Evaluated**: {evaluation_results['evaluated_samples']}
**Evaluation Model**: {self.evaluation_model}

## Executive Summary

This evaluation analyzed samples from the Hermes 3 dataset that contain references to gender, pronouns, or trans/non-binary topics. The analysis focused on identifying potential biases, harmful content, and stereotypes.

## Aggregate Statistics

### Overall Metrics
"""
        
        stats = evaluation_results['aggregate_stats']['detailed_stats']
        
        for metric, data in stats.items():
            report += f"\n**{metric.title()}**:\n"
            report += f"- Mean Score: {data['mean']:.3f}\n"
            report += f"- Range: [{data['min']:.3f}, {data['max']:.3f}]\n"
            report += f"- Problematic Samples: {data['problematic_count']} ({data['problematic_count']/data['count']*100:.1f}%)\n"
        
        report += "\n## Issues Identified\n\n"
        
        issues = evaluation_results['issues_found']
        
        for issue_type, example_ids in issues.items():
            if example_ids:
                report += f"### {issue_type.replace('_', ' ').title()}\n"
                report += f"Found in {len(example_ids)} samples ({len(example_ids)/evaluation_results['evaluated_samples']*100:.1f}%)\n"
                report += f"Example IDs: {', '.join(example_ids[:5])}"
                if len(example_ids) > 5:
                    report += f" (and {len(example_ids) - 5} more)"
                report += "\n\n"
        
        report += """
## Recommendations

Based on this evaluation, we recommend:

1. **Content Review**: Samples identified with high toxicity or stereotypes should be reviewed and potentially removed or modified.

2. **Bias Mitigation**: The dataset shows patterns of bias that could be propagated by models trained on it. Consider:
   - Balancing representation of diverse gender identities
   - Removing or correcting misgendering examples
   - Ensuring respectful and affirming language

3. **Augmentation**: Consider adding more positive, respectful examples of trans and non-binary individuals to balance the dataset.

4. **Documentation**: Document known biases and limitations for users of the dataset.

## Detailed Results

Detailed results for each sample are stored in the database: `{}`

Use the EvaluationPersistence class to query specific examples and their metric scores.
""".format(self.db_url)
        
        return report
    
    async def run_evaluation(self) -> str:
        """Run the complete evaluation pipeline."""
        print("Starting Hermes 3 dataset evaluation...")
        
        # Load dataset
        data = self.load_dataset()
        print(f"Loaded {len(data)} total samples")
        
        # Extract relevant samples
        relevant_samples = self.extract_trans_relevant_samples(data)
        
        # Sample if needed
        if len(relevant_samples) > self.sample_size:
            relevant_samples = random.sample(relevant_samples, self.sample_size)
            print(f"Sampled {self.sample_size} examples for evaluation")
        
        # Convert to examples
        examples = self.convert_to_examples(relevant_samples)
        
        # Run evaluation
        results = await self.evaluate_samples(examples)
        
        # Generate report
        report = self.generate_report(results)
        
        # Save report
        report_path = f"hermes3_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nEvaluation complete! Report saved to: {report_path}")
        print(f"Database with detailed results: {self.db_url}")
        
        return report


async def main():
    """Main function to run Hermes 3 evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Hermes 3 dataset for trans-inclusive language bias")
    parser.add_argument("dataset_path", help="Path to Hermes 3 dataset file")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--evaluation-model", default="anthropic/claude-3.5-sonnet", 
                       help="Model to use for evaluation")
    parser.add_argument("--db-url", default="sqlite:///hermes3_trans_eval.db",
                       help="Database URL for storing results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Hermes3DatasetEvaluator(
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
        evaluation_model=args.evaluation_model,
        db_url=args.db_url,
        random_seed=args.seed
    )
    
    # Run evaluation
    await evaluator.run_evaluation()


if __name__ == "__main__":
    # For direct script execution
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hermes3_dataset_evaluation.py <dataset_path> [options]")
        print("\nExample:")
        print("  python hermes3_dataset_evaluation.py /path/to/hermes3.json --sample-size 200")
        sys.exit(1)
    
    asyncio.run(main())