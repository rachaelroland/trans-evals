"""
Persistence layer for storing evaluation results in the database.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
from contextlib import contextmanager

from sqlalchemy.orm import Session
from .models import (
    Database, EvaluationRun, ModelEvaluation, 
    ExampleEvaluation, MetricResult, DatasetInfo, ModelInfo
)
from ..evaluation.llm_metrics import LLMMetricResult

logger = logging.getLogger(__name__)


class EvaluationPersistence:
    """Handles persistence of evaluation results to database."""
    
    def __init__(self, db_url: str = "sqlite:///trans_evals.db"):
        self.db = Database(db_url)
        self.db.create_tables()
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.db.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def create_evaluation_run(
        self, 
        dataset_name: str,
        dataset_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new evaluation run and return its ID."""
        run_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            run = EvaluationRun(
                run_id=run_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                metadata=metadata or {}
            )
            session.add(run)
            session.commit()
            logger.info(f"Created evaluation run: {run_id}")
        
        return run_id
    
    def create_model_evaluation(
        self,
        run_id: str,
        model_name: str,
        model_provider: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> int:
        """Create a model evaluation entry for a run."""
        with self.get_session() as session:
            # Get the evaluation run
            run = session.query(EvaluationRun).filter_by(run_id=run_id).first()
            if not run:
                raise ValueError(f"Evaluation run {run_id} not found")
            
            model_eval = ModelEvaluation(
                evaluation_run_id=run.id,
                model_name=model_name,
                model_provider=model_provider,
                model_version=model_version
            )
            session.add(model_eval)
            session.commit()
            
            # Update model info
            self._update_model_info(session, model_name, model_provider)
            
            return model_eval.id
    
    def save_example_evaluation(
        self,
        model_evaluation_id: int,
        example_id: str,
        prompt: str,
        model_response: str,
        metrics: Dict[str, LLMMetricResult],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save evaluation results for a single example."""
        with self.get_session() as session:
            # Create example evaluation
            example_eval = ExampleEvaluation(
                model_evaluation_id=model_evaluation_id,
                example_id=example_id,
                prompt=prompt,
                model_response=model_response,
                example_category=metadata.get("category") if metadata else None,
                expected_pronouns=metadata.get("expected_pronouns") if metadata else None,
                referent_name=metadata.get("referent_name") if metadata else None,
                intersectional_attributes=metadata.get("intersectional_attributes") if metadata else None
            )
            session.add(example_eval)
            session.flush()  # Get the ID without committing
            
            # Save each metric result
            for metric_name, result in metrics.items():
                metric_result = MetricResult(
                    example_evaluation_id=example_eval.id,
                    metric_name=metric_name,
                    score=result.score,
                    confidence=result.confidence,
                    details=result.details,
                    explanation=result.explanation,
                    raw_response=result.raw_response,
                    evaluation_model=getattr(result, 'evaluation_model', 'claude-3.5-sonnet')
                )
                session.add(metric_result)
            
            session.commit()
            return example_eval.id
    
    def update_model_evaluation_summary(
        self,
        model_evaluation_id: int,
        summary_metrics: Dict[str, float],
        statistical_data: Optional[Dict[str, Any]] = None
    ):
        """Update model evaluation with aggregate metrics."""
        with self.get_session() as session:
            model_eval = session.query(ModelEvaluation).get(model_evaluation_id)
            if not model_eval:
                raise ValueError(f"Model evaluation {model_evaluation_id} not found")
            
            # Update aggregate metrics
            model_eval.overall_bias_score = summary_metrics.get("overall_bias_score")
            model_eval.misgendering_rate = summary_metrics.get("misgendering_rate")
            model_eval.toxicity_rate = summary_metrics.get("toxicity_rate")
            model_eval.regard_score = summary_metrics.get("regard_score")
            model_eval.sentiment_score = summary_metrics.get("sentiment_score")
            model_eval.stereotype_score = summary_metrics.get("stereotype_score")
            
            # Update statistical measures
            if statistical_data:
                model_eval.sample_size = statistical_data.get("sample_size")
                model_eval.confidence_interval = statistical_data.get("confidence_intervals")
                model_eval.statistical_significance = statistical_data.get("p_values")
            
            model_eval.end_time = datetime.utcnow()
            model_eval.status = 'completed'
            
            session.commit()
    
    def complete_evaluation_run(self, run_id: str, status: str = 'completed'):
        """Mark an evaluation run as complete."""
        with self.get_session() as session:
            run = session.query(EvaluationRun).filter_by(run_id=run_id).first()
            if run:
                run.end_time = datetime.utcnow()
                run.status = status
                session.commit()
    
    def save_dataset_info(
        self,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        example_count: Optional[int] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save or update dataset information."""
        with self.get_session() as session:
            dataset = session.query(DatasetInfo).filter_by(name=name).first()
            
            if dataset:
                # Update existing
                dataset.version = version or dataset.version
                dataset.description = description or dataset.description
                dataset.example_count = example_count or dataset.example_count
                dataset.categories = categories or dataset.categories
                dataset.metadata = metadata or dataset.metadata
                dataset.updated_at = datetime.utcnow()
            else:
                # Create new
                dataset = DatasetInfo(
                    name=name,
                    version=version,
                    description=description,
                    example_count=example_count,
                    categories=categories,
                    metadata=metadata or {}
                )
                session.add(dataset)
            
            session.commit()
    
    def _update_model_info(self, session: Session, model_name: str, provider: Optional[str] = None):
        """Update model information tracking."""
        model = session.query(ModelInfo).filter_by(model_name=model_name).first()
        
        if model:
            model.last_evaluated = datetime.utcnow()
            model.total_evaluations += 1
        else:
            model = ModelInfo(
                model_name=model_name,
                provider=provider,
                total_evaluations=1
            )
            session.add(model)
    
    def get_evaluation_results(
        self,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve evaluation results with filtering."""
        with self.get_session() as session:
            query = session.query(ModelEvaluation).join(EvaluationRun)
            
            if run_id:
                query = query.filter(EvaluationRun.run_id == run_id)
            if model_name:
                query = query.filter(ModelEvaluation.model_name == model_name)
            if dataset_name:
                query = query.filter(EvaluationRun.dataset_name == dataset_name)
            
            results = []
            for model_eval in query.limit(limit).all():
                result = {
                    "run_id": model_eval.evaluation_run.run_id,
                    "dataset": model_eval.evaluation_run.dataset_name,
                    "model": model_eval.model_name,
                    "metrics": {
                        "overall_bias_score": model_eval.overall_bias_score,
                        "misgendering_rate": model_eval.misgendering_rate,
                        "toxicity_rate": model_eval.toxicity_rate,
                        "regard_score": model_eval.regard_score,
                        "sentiment_score": model_eval.sentiment_score,
                        "stereotype_score": model_eval.stereotype_score
                    },
                    "sample_size": model_eval.sample_size,
                    "status": model_eval.status,
                    "timestamp": model_eval.end_time or model_eval.start_time
                }
                results.append(result)
            
            return results
    
    def get_example_details(
        self,
        model_evaluation_id: int,
        example_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get detailed results for specific examples."""
        with self.get_session() as session:
            query = session.query(ExampleEvaluation).filter_by(
                model_evaluation_id=model_evaluation_id
            )
            
            if example_id:
                query = query.filter_by(example_id=example_id)
            
            results = []
            for example in query.all():
                metrics = {}
                for metric in example.metrics:
                    metrics[metric.metric_name] = {
                        "score": metric.score,
                        "confidence": metric.confidence,
                        "explanation": metric.explanation,
                        "details": metric.details
                    }
                
                result = {
                    "example_id": example.example_id,
                    "prompt": example.prompt,
                    "response": example.model_response,
                    "category": example.example_category,
                    "metrics": metrics,
                    "metadata": {
                        "expected_pronouns": example.expected_pronouns,
                        "referent_name": example.referent_name,
                        "intersectional_attributes": example.intersectional_attributes
                    }
                }
                results.append(result)
            
            return results