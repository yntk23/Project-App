"""
Main entry point for the time series prediction system.

This script initializes the system and runs the next-day prediction pipeline.
"""

import os
import sys
import argparse
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.utils.logging_config import setup_logging
from src.modeling.train_recommender import train_recommender


def main(
    data_path: Optional[str] = None, 
    log_level: str = "INFO",
    use_database: bool = False,
    db_url: Optional[str] = None
) -> None:
    """
    Main function to run the prediction system.
    
    Args:
        data_path: Path to the data file (uses config default if None)
        log_level: Logging level
        use_database: Whether to use database instead of CSV
        db_url: Database connection URL
    """
    # Set up logging
    logger = setup_logging(log_level)
    
    try:
        # Use config default if no path provided
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), config.data_path)
        
        # Validate data source
        if not use_database:
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                sys.exit(1)
            logger.info(f"Data path: {data_path}")
        else:
            logger.info(f"Using database: {db_url}")
        
        logger.info("Starting time series prediction system with Autoencoder")
        logger.info(f"Configuration: {config}")
        
        # Run the prediction pipeline
        predictions, next_date = train_recommender(
            data_path=data_path,
            use_database=use_database,
            db_url=db_url
        )
        
        # Print summary results
        logger.info("=== PREDICTION SUMMARY ===")
        logger.info(f"Generated predictions for {len(predictions)} stores")
        logger.info(f"Prediction date: {next_date.strftime('%Y-%m-%d')}")
        
        # Print sample predictions (only 3 stores for console)
        sample_stores = list(predictions.keys())[:3]
        for store_id in sample_stores:
            top_products = predictions[store_id]
            products_str = ", ".join([f"{prod}({qty})" for prod, qty in top_products])
            logger.info(f"Store {store_id}: {products_str}")
        
        if len(predictions) > 3:
            logger.info(f"... and {len(predictions) - 3} more stores")
        
        logger.info("=== PREDICTION DETAILS ===")
        logger.info(f"Moving average window: {config.prediction.moving_average_window} days")
        logger.info(f"Top products per store: {config.prediction.top_n_products}")
        
        logger.info("=== OUTPUT FILES ===")
        logger.info("Check the 'output/YYYY-MM-DD/' directory for:")
        logger.info("• CSV files organized by date for easy analysis and data import")
        logger.info("• All 27 stores' predictions are included")
        
        logger.info("Time series prediction system completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Time Series Product Prediction System")
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to the data CSV file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    parser.add_argument('--use-database', action='store_true', help='Use database instead of CSV')
    parser.add_argument('--db-url', type=str, help='Database URL (e.g., postgresql://user:pass@host/db)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    main(
        data_path=args.data_path,
        log_level=args.log_level,
        use_database=args.use_database,
        db_url=args.db_url
    )
