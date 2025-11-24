"""
Utility functions for data validation and quality checks.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: List[str],
    name: str = "DataFrame"
) -> bool:
    """
    Validate that a dataframe has required columns and basic quality checks.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for logging purposes
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    try:
        logger.info(f"Validating {name}")
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError(f"{name} is empty")
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{name} missing required columns: {missing_columns}")
        
        # Check for completely null columns
        null_columns = [col for col in required_columns if df[col].isnull().all()]
        if null_columns:
            raise ValueError(f"{name} has completely null columns: {null_columns}")
        
        logger.info(f"{name} validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed for {name}: {e}")
        raise


def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'duplicates': 0,
        'unique_values': {},
        'outliers': {},
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    try:
        # Missing values
        missing = df.isnull().sum()
        quality_report['missing_values'] = {
            col: {'count': int(count), 'percentage': float(count / len(df) * 100)}
            for col, count in missing.items() if count > 0
        }
        
        # Data types
        quality_report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Duplicates
        quality_report['duplicates'] = int(df.duplicated().sum())
        
        # Unique values for categorical columns
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                quality_report['unique_values'][col] = int(df[col].nunique())
        
        # Outliers for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    quality_report['outliers'][col] = {
                        'count': int(outliers),
                        'percentage': float(outliers / len(df) * 100)
                    }
        
        logger.info(f"Data quality check completed for {len(df)} rows")
        
    except Exception as e:
        logger.error(f"Error in data quality check: {e}")
        
    return quality_report


def suggest_data_improvements(quality_report: Dict[str, any]) -> List[str]:
    """
    Suggest improvements based on data quality report.
    
    Args:
        quality_report: Data quality report from check_data_quality
        
    Returns:
        List of improvement suggestions
    """
    suggestions = []
    
    try:
        # Missing values suggestions
        for col, info in quality_report.get('missing_values', {}).items():
            if info['percentage'] > 50:
                suggestions.append(f"Column '{col}' has {info['percentage']:.1f}% missing values - consider dropping or imputing")
            elif info['percentage'] > 10:
                suggestions.append(f"Column '{col}' has {info['percentage']:.1f}% missing values - consider imputation strategy")
        
        # Duplicates suggestion
        if quality_report.get('duplicates', 0) > 0:
            suggestions.append(f"Found {quality_report['duplicates']} duplicate rows - consider deduplication")
        
        # Outliers suggestions
        for col, info in quality_report.get('outliers', {}).items():
            if info['percentage'] > 5:
                suggestions.append(f"Column '{col}' has {info['percentage']:.1f}% outliers - consider outlier treatment")
        
        # Memory usage suggestion
        memory_mb = quality_report.get('memory_usage', 0) / 1024 / 1024
        if memory_mb > 100:
            suggestions.append(f"High memory usage ({memory_mb:.1f} MB) - consider data type optimization")
        
        # High cardinality suggestion
        total_rows = quality_report.get('total_rows', 0)
        for col, unique_count in quality_report.get('unique_values', {}).items():
            if unique_count > total_rows * 0.9:
                suggestions.append(f"Column '{col}' has very high cardinality - might be an ID column")
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        
    return suggestions


def create_data_summary(df: pd.DataFrame) -> str:
    """
    Create a comprehensive data summary string.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Formatted summary string
    """
    try:
        quality_report = check_data_quality(df)
        suggestions = suggest_data_improvements(quality_report)
        
        summary = f"""
=== DATA SUMMARY ===
Shape: {quality_report['total_rows']} rows × {quality_report['total_columns']} columns
Memory Usage: {quality_report['memory_usage'] / 1024 / 1024:.2f} MB

=== MISSING VALUES ===
"""
        
        if quality_report['missing_values']:
            for col, info in quality_report['missing_values'].items():
                summary += f"{col}: {info['count']} ({info['percentage']:.1f}%)\n"
        else:
            summary += "No missing values found\n"
        
        summary += f"\n=== DUPLICATES ===\n{quality_report['duplicates']} duplicate rows\n"
        
        if quality_report['outliers']:
            summary += "\n=== OUTLIERS ===\n"
            for col, info in quality_report['outliers'].items():
                summary += f"{col}: {info['count']} ({info['percentage']:.1f}%)\n"
        
        if suggestions:
            summary += "\n=== SUGGESTIONS ===\n"
            for i, suggestion in enumerate(suggestions, 1):
                summary += f"{i}. {suggestion}\n"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating data summary: {e}")
        return f"Error creating summary: {e}"
