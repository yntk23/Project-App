"""
Main entry point with Ensemble Model Selection
Path: backend/main.py
"""

import os
import sys
import argparse
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.utils.logging_config import setup_logging
from src.modeling.train_recommender import train_recommender


def main(
    data_path: Optional[str] = None, 
    log_level: str = "INFO",
    use_database: bool = False,
    db_url: Optional[str] = None,
    selected_model: str = 'ensemble'
) -> None:
    """
    Main function with model selection
    
    Args:
        selected_model: 'ensemble', 'autoencoder', 'exp_smoothing', 'linear_regression'
    """
    logger = setup_logging(log_level)
    
    try:
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), config.data_path)
        
        if not use_database:
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                sys.exit(1)
            logger.info(f"Data path: {data_path}")
        else:
            logger.info(f"Using database: {db_url}")
        
        logger.info(f"Starting prediction system with {selected_model.upper()} model")
        logger.info(f"Configuration: {config}")
        
        # Run prediction
        predictions, next_date, comparison_data = train_recommender(
            data_path=data_path,
            use_database=use_database,
            db_url=db_url,
            selected_model=selected_model
        )
        
        # Print summary
        logger.info("=== PREDICTION SUMMARY ===")
        logger.info(f"Model used: {selected_model.upper()}")
        logger.info(f"Generated predictions for {len(predictions)} stores")
        logger.info(f"Prediction date: {next_date.strftime('%Y-%m-%d')}")
        
        # Print model comparison
        if 'metrics' in comparison_data:
            logger.info("\n=== MODEL PERFORMANCE ===")
            for metric in comparison_data['metrics']:
                logger.info(f"{metric['model']}: MAE={metric['mae']:.2f}, RMSE={metric['rmse']:.2f}, MAPE={metric['mape']:.2f}%")
        
        # Sample predictions
        sample_stores = list(predictions.keys())[:3]
        logger.info("\n=== SAMPLE PREDICTIONS ===")
        for store_id in sample_stores:
            top_products = predictions[store_id]
            products_str = ", ".join([f"{prod}({qty})" for prod, qty in top_products])
            logger.info(f"Store {store_id}: {products_str}")
        
        if len(predictions) > 3:
            logger.info(f"... and {len(predictions) - 3} more stores")
        
        logger.info("\n=== OUTPUT FILES ===")
        logger.info("✅ Predictions CSV: output/YYYY-MM-DD/next_day_top5_YYYYMMDD.csv")
        logger.info("✅ Model Comparison: output/YYYY-MM-DD/model_comparison_YYYYMMDD.csv")
        logger.info("✅ Metrics JSON: output/models/ensemble_comparison.json")
        
        logger.info("\nPrediction system completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ensemble Prediction System")
    
    parser.add_argument('--data-path', type=str, help='Path to CSV file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--use-database', action='store_true', help='Use database')
    parser.add_argument('--db-url', type=str, help='Database URL')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['ensemble', 'autoencoder', 'exp_smoothing', 'linear_regression'],
                       help='Model to use for prediction')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    main(
        data_path=args.data_path,
        log_level=args.log_level,
        use_database=args.use_database,
        db_url=args.db_url,
        selected_model=args.model
    )