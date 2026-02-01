"""
Statistics Module
통계 분석 관련 Python 코드

This module contains statistical analysis examples including:
- Descriptive statistics
- Hypothesis testing
- Probability distributions
- Statistical inference
"""

import numpy as np
from scipy import stats as scipy_stats
import pandas as pd


def descriptive_statistics(data):
    """
    Calculate descriptive statistics for a given dataset
    
    Args:
        data: numpy array or list of numerical values
    
    Returns:
        dict: Dictionary containing mean, median, std, variance, etc.
    """
    data = np.array(data)
    
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'variance': np.var(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def t_test(sample1, sample2, alpha=0.05):
    """
    Perform independent t-test to compare two samples
    
    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level (default: 0.05)
    
    Returns:
        dict: Test results including t-statistic, p-value, and conclusion
    """
    t_stat, p_value = scipy_stats.ttest_ind(sample1, sample2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'significance_level': alpha
    }


def correlation_analysis(x, y):
    """
    Calculate correlation between two variables
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        dict: Pearson and Spearman correlation coefficients
    """
    pearson_corr, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_corr, spearman_p = scipy_stats.spearmanr(x, y)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    }


def normal_distribution_test(data):
    """
    Test if data follows normal distribution using Shapiro-Wilk test
    
    Args:
        data: Sample data
    
    Returns:
        dict: Test results
    """
    stat, p_value = scipy_stats.shapiro(data)
    
    return {
        'test_statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }


if __name__ == "__main__":
    # Example usage
    print("=== Descriptive Statistics Example ===")
    sample_data = np.random.normal(100, 15, 1000)
    desc_stats = descriptive_statistics(sample_data)
    print(f"Mean: {desc_stats['mean']:.2f}")
    print(f"Std: {desc_stats['std']:.2f}")
    print(f"Median: {desc_stats['median']:.2f}")
    
    print("\n=== T-Test Example ===")
    group1 = np.random.normal(100, 15, 100)
    group2 = np.random.normal(105, 15, 100)
    test_result = t_test(group1, group2)
    print(f"T-statistic: {test_result['t_statistic']:.4f}")
    print(f"P-value: {test_result['p_value']:.4f}")
    print(f"Reject null hypothesis: {test_result['reject_null']}")
    
    print("\n=== Correlation Analysis Example ===")
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.5, 100)
    corr_result = correlation_analysis(x, y)
    print(f"Pearson correlation: {corr_result['pearson_correlation']:.4f}")
    print(f"Spearman correlation: {corr_result['spearman_correlation']:.4f}")
    
    print("\n=== Normality Test Example ===")
    normal_data = np.random.normal(0, 1, 100)
    normality_result = normal_distribution_test(normal_data)
    print(f"Is normal: {normality_result['is_normal']}")
    print(f"P-value: {normality_result['p_value']:.4f}")
