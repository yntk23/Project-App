"""
Data preparation module for the recommendation system.

This module handles data loading, cleaning, and transformation for the recommendation system.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from src.config import config

# Set up logger
logger = logging.getLogger(__name__)


def load_data(filepath: str, columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and preprocess data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        columns_to_drop: List of columns to drop from the dataset
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if columns_to_drop is None:
        columns_to_drop = config.data.columns_to_drop
    
    try:
        logger.info(f"Loading file: {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Drop unnecessary columns
        df = df.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Dropped columns: {[col for col in columns_to_drop if col in df.columns]}")
        
        # Data type conversions and cleaning
        df = _clean_data(df)
        
        # Log data quality information
        _log_data_quality(df)
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataframe.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    # Convert date column
    if "BSNS_DT" in df.columns:
        df["BSNS_DT"] = pd.to_datetime(df["BSNS_DT"], errors="coerce")
        logger.info(f"Converted BSNS_DT to datetime. Invalid dates: {df['BSNS_DT'].isnull().sum()}")
    
    # Convert ID columns to string only if they exist and are not already strings
    for col in ["STORE_ID", "PROD_CD"]:
        if col in df.columns and not df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            logger.info(f"Converted {col} to string type")
    
    # Handle missing values in quantity
    if "PROD_QTY" in df.columns:
        missing_qty = df["PROD_QTY"].isnull().sum()
        if missing_qty > 0:
            logger.warning(f"Found {missing_qty} missing values in PROD_QTY. Filling with 0.")
            df["PROD_QTY"] = df["PROD_QTY"].fillna(0)
        
        # Ensure quantity is positive
        negative_qty = (df["PROD_QTY"] < 0).sum()
        if negative_qty > 0:
            logger.warning(f"Found {negative_qty} negative quantities. Setting to 0.")
            df.loc[df["PROD_QTY"] < 0, "PROD_QTY"] = 0
    
    return df


def _log_data_quality(df: pd.DataFrame) -> None:
    """
    Log data quality information.
    
    Args:
        df: DataFrame to analyze
    """
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning("Missing values found:")
        for col, count in missing_values[missing_values > 0].items():
            logger.warning(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check data types
    logger.info("Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare transaction data by grouping and aggregating.
    
    Args:
        df: Raw transaction dataframe
        
    Returns:
        Grouped transaction dataframe
        
    Raises:
        KeyError: If required columns are missing
    """
    required_columns = ["STORE_ID", "BSNS_DT", "PROD_CD", "PROD_QTY"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    try:
        logger.info("Preparing transaction data...")
        
        # Group by store, date, and product, summing quantities
        grouped = df.groupby(["STORE_ID", "BSNS_DT", "PROD_CD"])["PROD_QTY"].sum().reset_index()
        
        logger.info(f"Grouped data shape: {grouped.shape}")
        logger.info(f"Unique stores: {grouped['STORE_ID'].nunique()}")
        logger.info(f"Unique products: {grouped['PROD_CD'].nunique()}")
        
        return grouped
        
    except Exception as e:
        logger.error(f"Error preparing transactions: {e}")
        raise


def create_transaction_matrix(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create a transaction matrix from transaction data.
    
    Args:
        transactions: Transaction dataframe
        
    Returns:
        Transaction matrix with stores as rows and products as columns
        
    Raises:
        ValueError: If the input data is invalid
    """
    try:
        logger.info("Creating transaction matrix...")
        
        # Clean list values if they exist (legacy data handling)
        for col in ['STORE_ID', 'PROD_CD']:
            if col in transactions.columns:
                list_mask = transactions[col].apply(lambda x: isinstance(x, list))
                if list_mask.any():
                    logger.warning(f"Found list values in {col}, extracting first element")
                    transactions.loc[list_mask, col] = transactions.loc[list_mask, col].apply(lambda x: x[0] if x else None)
        
        # Remove any rows with missing store or product IDs
        before_cleaning = len(transactions)
        transactions = transactions.dropna(subset=['STORE_ID', 'PROD_CD'])
        after_cleaning = len(transactions)
        
        if before_cleaning != after_cleaning:
            logger.warning(f"Removed {before_cleaning - after_cleaning} rows with missing IDs")
        
        if transactions.empty:
            raise ValueError("No valid transactions after cleaning")
        
        logger.info(f"Transaction data shape: {transactions.shape}")
        logger.info(f"Sample data:\n{transactions.head()}")
        
        # Create pivot table
        matrix = transactions.pivot_table(
            index='STORE_ID', 
            columns='PROD_CD', 
            values='PROD_QTY', 
            aggfunc='sum', 
            fill_value=0
        )
        
        logger.info(f"Transaction matrix shape: {matrix.shape}")
        logger.info(f"Matrix sparsity: {(matrix == 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]) * 100:.2f}%")
        
        return matrix
        
    except Exception as e:
        logger.error(f"Error creating transaction matrix: {e}")
        raise
