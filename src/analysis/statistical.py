"""Statistical analysis utilities for evaluation results."""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on evaluation results."""
    
    def calculate_confidence_interval(
        self, 
        data: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a dataset.
        
        Args:
            data: List of numeric values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not data:
            return (0.0, 0.0)
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        
        # Use t-distribution for small samples
        if len(data) < 30:
            se = stats.sem(data_array)
            interval = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        else:
            # Use normal distribution for larger samples
            se = stats.sem(data_array)
            interval = se * stats.norm.ppf((1 + confidence) / 2.)
        
        return (mean - interval, mean + interval)
    
    def mann_whitney_test(
        self,
        group1: List[float],
        group2: List[float]
    ) -> Dict[str, float]:
        """
        Perform Mann-Whitney U test for comparing two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Dictionary with test statistics
        """
        if not group1 or not group2:
            return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative='two-sided'
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1 = len(group1)
        n2 = len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size)
        }
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Cohen's d value
        """
        if not group1 or not group2:
            return 0.0
        
        n1 = len(group1)
        n2 = len(group2)
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_sd = np.sqrt(pooled_var)
        
        if pooled_sd == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_sd
    
    def calculate_metrics_summary(
        self,
        values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate summary statistics for a metric.
        
        Args:
            values: List of metric values
            
        Returns:
            Dictionary with summary statistics
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "q1": 0.0,
                "q3": 0.0
            }
        
        values_array = np.array(values)
        
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array, ddof=1)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75))
        }