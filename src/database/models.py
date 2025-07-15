"""
Database models for storing trans-evals evaluation results.
Uses SQLAlchemy for ORM with SQLite as default database.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class EvaluationRun(Base):
    """Represents a complete evaluation run across multiple models."""
    __tablename__ = 'evaluation_runs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    dataset_name = Column(String(100), nullable=False)
    dataset_version = Column(String(50))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed
    metadata = Column(JSON)  # Additional run metadata
    
    # Relationships
    model_evaluations = relationship("ModelEvaluation", back_populates="evaluation_run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<EvaluationRun(run_id={self.run_id}, dataset={self.dataset_name}, status={self.status})>"


class ModelEvaluation(Base):
    """Evaluation results for a specific model."""
    __tablename__ = 'model_evaluations'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey('evaluation_runs.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_provider = Column(String(50))  # openai, anthropic, google, etc.
    model_version = Column(String(50))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(20), default='running')
    
    # Aggregate metrics
    overall_bias_score = Column(Float)
    misgendering_rate = Column(Float)
    toxicity_rate = Column(Float)
    regard_score = Column(Float)
    sentiment_score = Column(Float)
    stereotype_score = Column(Float)
    
    # Statistical measures
    sample_size = Column(Integer)
    confidence_interval = Column(JSON)  # {metric: [lower, upper]}
    statistical_significance = Column(JSON)  # {metric: p_value}
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="model_evaluations")
    example_evaluations = relationship("ExampleEvaluation", back_populates="model_evaluation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_run', 'evaluation_run_id', 'model_name'),
    )
    
    def __repr__(self):
        return f"<ModelEvaluation(model={self.model_name}, bias_score={self.overall_bias_score})>"


class ExampleEvaluation(Base):
    """Evaluation results for a single example/prompt."""
    __tablename__ = 'example_evaluations'
    
    id = Column(Integer, primary_key=True)
    model_evaluation_id = Column(Integer, ForeignKey('model_evaluations.id'), nullable=False)
    example_id = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=False)
    model_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Example metadata
    example_category = Column(String(50))  # pronoun_consistency, identity_respect, etc.
    expected_pronouns = Column(String(50))
    referent_name = Column(String(100))
    intersectional_attributes = Column(JSON)  # race, disability, etc.
    
    # Evaluation metrics
    metrics = relationship("MetricResult", back_populates="example_evaluation", cascade="all, delete-orphan")
    
    # Relationships
    model_evaluation = relationship("ModelEvaluation", back_populates="example_evaluations")
    
    # Indexes
    __table_args__ = (
        Index('idx_example_model', 'model_evaluation_id', 'example_id'),
    )
    
    def __repr__(self):
        return f"<ExampleEvaluation(example_id={self.example_id}, category={self.example_category})>"


class MetricResult(Base):
    """Individual metric results for an example."""
    __tablename__ = 'metric_results'
    
    id = Column(Integer, primary_key=True)
    example_evaluation_id = Column(Integer, ForeignKey('example_evaluations.id'), nullable=False)
    metric_name = Column(String(50), nullable=False)  # toxicity, regard, sentiment, etc.
    score = Column(Float, nullable=False)
    confidence = Column(Float)
    
    # Detailed results from LLM
    details = Column(JSON)  # Structured details (categories, indicators, etc.)
    explanation = Column(Text)  # LLM's explanation
    raw_response = Column(Text)  # Full LLM response
    
    # Metadata
    evaluation_model = Column(String(100))  # Which model performed the evaluation (e.g., claude-3.5-sonnet)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    example_evaluation = relationship("ExampleEvaluation", back_populates="metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_example', 'example_evaluation_id', 'metric_name'),
    )
    
    def __repr__(self):
        return f"<MetricResult(metric={self.metric_name}, score={self.score})>"


class DatasetInfo(Base):
    """Information about evaluation datasets."""
    __tablename__ = 'dataset_info'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    version = Column(String(50))
    description = Column(Text)
    example_count = Column(Integer)
    categories = Column(JSON)  # List of example categories
    metadata = Column(JSON)  # Additional dataset metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DatasetInfo(name={self.name}, version={self.version})>"


class ModelInfo(Base):
    """Information about evaluated models."""
    __tablename__ = 'model_info'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), unique=True, nullable=False)
    provider = Column(String(50))
    model_type = Column(String(50))  # chat, completion, etc.
    parameters = Column(JSON)  # Model parameters used
    api_endpoint = Column(String(200))
    first_evaluated = Column(DateTime, default=datetime.utcnow)
    last_evaluated = Column(DateTime, default=datetime.utcnow)
    total_evaluations = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<ModelInfo(name={self.model_name}, provider={self.provider})>"


# Database connection and session management
class Database:
    """Database connection and session management."""
    
    def __init__(self, db_url: str = "sqlite:///trans_evals.db"):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)