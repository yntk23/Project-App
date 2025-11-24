"""
Database manager for sales data and predictions.

Supports PostgreSQL and SQLite backends.
"""

import logging
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, Float, DateTime, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations for sales data."""
    
    def __init__(self, db_url: str):
        """
        Initialize database manager.
        
        Args:
            db_url: Database URL (e.g., 'postgresql://user:pass@localhost/dbname' or 'sqlite:///data.db')
        """
        self.db_url = db_url
        self.engine = None
        self.metadata = MetaData()
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.engine = create_engine(self.db_url, echo=False)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info(f"Connected to database: {self.db_url.split('@')[-1] if '@' in self.db_url else self.db_url}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            # Sales data table
            sales_table = Table(
                'sales_data',
                self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('store_id', String(50), nullable=False, index=True),
                Column('prod_cd', String(50), nullable=False, index=True),
                Column('prod_qty', Integer, nullable=False),
                Column('prod_amt', Float),
                Column('bsns_dt', Date, nullable=False, index=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # Predictions table
            predictions_table = Table(
                'predictions',
                self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('store_id', String(50), nullable=False, index=True),
                Column('prediction_date', Date, nullable=False, index=True),
                Column('rank', Integer, nullable=False),
                Column('product_code', String(50), nullable=False),
                Column('predicted_quantity', Integer, nullable=False),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # Model metadata table
            model_metadata_table = Table(
                'model_metadata',
                self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('prediction_date', Date, nullable=False, index=True),
                Column('model_type', String(50)),
                Column('num_stores', Integer),
                Column('num_products', Integer),
                Column('window_size', Integer),
                Column('encoding_dim', Integer),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def load_sales_data(
        self,
        store_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load sales data from database.
        
        Args:
            store_id: Filter by store ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with sales data
        """
        try:
            query = "SELECT store_id, prod_cd, prod_qty, prod_amt, bsns_dt FROM sales_data WHERE 1=1"
            params = {}
            
            if store_id:
                query += " AND store_id = :store_id"
                params['store_id'] = store_id
            
            if start_date:
                query += " AND bsns_dt >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND bsns_dt <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY bsns_dt, store_id, prod_cd"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            
            # Rename columns to match existing code
            df = df.rename(columns={
                'store_id': 'STORE_ID',
                'prod_cd': 'PROD_CD',
                'prod_qty': 'PROD_QTY',
                'prod_amt': 'PROD_AMT',
                'bsns_dt': 'BSNS_DT'
            })
            
            logger.info(f"Loaded {len(df)} rows from database")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to load sales data: {e}")
            raise
    
    def import_csv_to_database(self, csv_path: str, batch_size: int = 10000) -> int:
        """
        Import CSV data to database.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Number of rows to insert per batch
            
        Returns:
            Number of rows imported
        """
        try:
            logger.info(f"Importing data from {csv_path}")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Clean and prepare data
            df = df[['STORE_ID', 'PROD_CD', 'PROD_QTY', 'PROD_AMT', 'BSNS_DT']].copy()
            df['BSNS_DT'] = pd.to_datetime(df['BSNS_DT']).dt.date
            df.columns = df.columns.str.lower()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['store_id', 'prod_cd', 'bsns_dt'])
            
            # Insert in batches
            total_rows = 0
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                batch.to_sql('sales_data', self.engine, if_exists='append', index=False)
                total_rows += len(batch)
                logger.info(f"Imported {total_rows}/{len(df)} rows")
            
            logger.info(f"Successfully imported {total_rows} rows")
            return total_rows
            
        except Exception as e:
            logger.error(f"Failed to import CSV: {e}")
            raise
    
    def save_predictions(
        self,
        predictions: Dict[str, List[Tuple[str, int]]],
        prediction_date: pd.Timestamp
    ) -> int:
        """
        Save predictions to database.
        
        Args:
            predictions: Dictionary mapping store_id to list of (product_code, quantity) tuples
            prediction_date: Date for which predictions were made
            
        Returns:
            Number of predictions saved
        """
        try:
            records = []
            current_time = datetime.now()
            
            for store_id, products in predictions.items():
                for rank, (prod_cd, qty) in enumerate(products, 1):
                    records.append({
                        'store_id': store_id,
                        'prediction_date': prediction_date.date(),
                        'rank': rank,
                        'product_code': prod_cd,
                        'predicted_quantity': qty,
                        'created_at': current_time
                    })
            
            df = pd.DataFrame(records)
            df.to_sql('predictions', self.engine, if_exists='append', index=False)
            
            logger.info(f"Saved {len(records)} predictions to database")
            return len(records)
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save predictions: {e}")
            raise
    
    def save_model_metadata(
        self,
        prediction_date: pd.Timestamp,
        model_type: str,
        num_stores: int,
        num_products: int,
        window_size: int,
        encoding_dim: int
    ) -> None:
        """Save model metadata to database."""
        try:
            metadata = {
                'prediction_date': prediction_date.date(),
                'model_type': model_type,
                'num_stores': num_stores,
                'num_products': num_products,
                'window_size': window_size,
                'encoding_dim': encoding_dim
            }
            
            df = pd.DataFrame([metadata])
            df.to_sql('model_metadata', self.engine, if_exists='append', index=False)
            
            logger.info("Saved model metadata to database")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save model metadata: {e}")
            raise
    
    def get_latest_predictions(self, store_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get the most recent predictions.
        
        Args:
            store_id: Filter by store ID
            
        Returns:
            DataFrame with latest predictions
        """
        try:
            query = """
                SELECT store_id, prediction_date, rank, product_code, predicted_quantity
                FROM predictions
                WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
            """
            
            if store_id:
                query += " AND store_id = :store_id"
                params = {'store_id': store_id}
            else:
                params = {}
            
            query += " ORDER BY store_id, rank"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            logger.info(f"Retrieved {len(df)} latest predictions")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest predictions: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")