"""
Utility functions for prediction model artifacts and results management.
"""

import os
import logging
import json
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def save_prediction_results(
    prediction_results: Dict[str, List[Tuple[str, int]]],
    next_date: pd.Timestamp,
    output_dir: str = "output"
) -> str:
    """
    Save prediction results to CSV file organized by date.
    
    Args:
        prediction_results: Dictionary mapping store_id to list of (product_code, predicted_quantity) tuples
        next_date: The date for which predictions were made
        output_dir: Base output directory
        
    Returns:
        Path to the saved CSV file
    """
    try:
        # Create date-organized directory
        date_output_dir = os.path.join(output_dir, next_date.strftime('%Y-%m-%d'))
        os.makedirs(date_output_dir, exist_ok=True)
        
        # Create CSV file
        output_file = os.path.join(date_output_dir, f"next_day_top5_{next_date.strftime('%Y%m%d')}.csv")
        
        with open(output_file, "w") as f:
            f.write("Store_ID,Prediction_Date,Rank,Product_Code,Predicted_Quantity\n")
            for store_id, top5 in prediction_results.items():
                for rank, (prod_cd, qty) in enumerate(top5, 1):
                    f.write(f"{store_id},{next_date.strftime('%Y-%m-%d')},{rank},{prod_cd},{qty}\n")
        
        logger.info(f"Prediction results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving prediction results: {e}")
        raise


def save_prediction_metadata(
    data_path: str,
    prediction_date: pd.Timestamp,
    num_stores: int,
    num_products: int,
    output_dir: str = "output"
) -> str:
    """
    Save metadata about the prediction run.
    
    Args:
        data_path: Path to the source data file
        prediction_date: Date for which predictions were made
        num_stores: Number of stores processed
        num_products: Total number of unique products
        output_dir: Base output directory
        
    Returns:
        Path to the saved metadata file
    """
    try:
        # Create date-organized directory
        date_output_dir = os.path.join(output_dir, prediction_date.strftime('%Y-%m-%d'))
        os.makedirs(date_output_dir, exist_ok=True)
        
        metadata = {
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'data_source': data_path,
            'num_stores': num_stores,
            'num_products': num_products,
            'prediction_method': '7-day moving average',
            'top_n_products': 5,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(date_output_dir, f"metadata_{prediction_date.strftime('%Y%m%d')}.json")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Prediction metadata saved to: {metadata_file}")
        return metadata_file
        
    except Exception as e:
        logger.error(f"Error saving prediction metadata: {e}")
        raise


def load_prediction_results(prediction_date: str, output_dir: str = "output") -> Optional[pd.DataFrame]:
    """
    Load prediction results from CSV file.
    
    Args:
        prediction_date: Date string in YYYY-MM-DD format
        output_dir: Base output directory
        
    Returns:
        DataFrame with prediction results or None if not found
    """
    try:
        date_str = pd.to_datetime(prediction_date).strftime('%Y%m%d')
        file_path = os.path.join(output_dir, prediction_date, f"next_day_top5_{date_str}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Prediction file not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded prediction results from: {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading prediction results: {e}")
        return None


def list_available_predictions(output_dir: str = "output") -> List[str]:
    """
    List all available prediction dates.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        List of available prediction dates in YYYY-MM-DD format
    """
    try:
        if not os.path.exists(output_dir):
            return []
        
        predictions = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid date directory with prediction files
                try:
                    pd.to_datetime(item)  # Validate date format
                    csv_files = [f for f in os.listdir(item_path) if f.startswith('next_day_top5_') and f.endswith('.csv')]
                    if csv_files:
                        predictions.append(item)
                except:
                    continue
        
        return sorted(predictions)
        
    except Exception as e:
        logger.error(f"Error listing available predictions: {e}")
        return []
