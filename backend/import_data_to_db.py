"""
Script to import CSV data into database.

Usage:
    # PostgreSQL
    python import_data_to_db.py --csv data/sales.csv --db postgresql://user:password@localhost/sales_db
    
    # SQLite
    python import_data_to_db.py --csv data/sales.csv --db sqlite:///sales_data.db
"""

import sys
import os
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.db_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Import CSV data to database')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--db', required=True, help='Database URL')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for import')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Connecting to database: {args.db}")
        db_manager = DatabaseManager(args.db)
        
        logger.info(f"Importing data from: {args.csv}")
        total_rows = db_manager.import_csv_to_database(args.csv, args.batch_size)
        
        logger.info(f"✓ Successfully imported {total_rows} rows")
        
        # Verify import
        df = db_manager.load_sales_data()
        logger.info(f"✓ Verification: {len(df)} rows in database")
        logger.info(f"  - Stores: {df['STORE_ID'].nunique()}")
        logger.info(f"  - Products: {df['PROD_CD'].nunique()}")
        logger.info(f"  - Date range: {df['BSNS_DT'].min()} to {df['BSNS_DT'].max()}")
        
        db_manager.close()
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()