"""
Time series prediction module using Autoencoder for next-day product recommendations.

This module uses an Autoencoder architecture where:
- Encoder compresses historical time-series data into a latent representation
- Decoder predicts next-day quantities from the latent vector
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models

from src.preprocessing.data_preparation import load_data
from src.database.db_manager import DatabaseManager
from src.config import config

# Set up logger
logger = logging.getLogger(__name__)


def create_autoencoder_model(sequence_length: int, encoding_dim: int = 16):
    """
    Create Autoencoder model for time-series prediction.
    
    Architecture:
    - Encoder: Input(sequence_length) → Dense(64) → Dense(32) → Dense(encoding_dim)
    - Decoder: Input(encoding_dim) → Dense(32) → Dense(64) → Dense(1) [next-day prediction]
    
    Args:
        sequence_length: Length of input time series (e.g., 7 days)
        encoding_dim: Dimension of latent representation
        
    Returns:
        Tuple of (full_model, encoder, decoder)
    """
    # Encoder: Compresses sequence to latent vector
    encoder_input = layers.Input(shape=(sequence_length,), name='encoder_input')
    x = layers.Dense(64, activation='relu')(encoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='latent')(x)
    
    encoder = models.Model(encoder_input, encoded, name='encoder')
    
    # Decoder: Predicts next-day quantity from latent
    decoder_input = layers.Input(shape=(encoding_dim,), name='decoder_input')
    x = layers.Dense(32, activation='relu')(decoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    decoded = layers.Dense(1, activation='linear', name='output')(x)
    
    decoder = models.Model(decoder_input, decoded, name='decoder')
    
    # Full autoencoder pipeline
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name='autoencoder')
    
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    logger.info(f"Created Autoencoder: sequence_length={sequence_length}, encoding_dim={encoding_dim}")
    logger.info(f"Encoder params: {encoder.count_params()}, Decoder params: {decoder.count_params()}")
    
    return autoencoder, encoder, decoder


def prepare_time_series_data(df: pd.DataFrame, window_size: int = 7) -> Dict:
    """
    Prepare time-series sequences for each store-product pair.
    
    เตรียมข้อมูล: ใช้ N วันย้อนหลัง เพื่อพยากรณ์ปริมาณที่ควรสั่งสำหรับวันถัดไป
    
    Args:
        df: Sales dataframe with STORE_ID, PROD_CD, BSNS_DT, PROD_QTY
        window_size: Number of historical days to use as input (default: 7)
        
    Returns:
        Dictionary: {store_id: {prod_cd: {'sequence': array, 'max_val': scalar, 'last_qty': scalar}}}
    """
    df = df.sort_values('BSNS_DT')
    sequences = {}
    
    for store_id in df['STORE_ID'].unique():
        store_df = df[df['STORE_ID'] == store_id]
        sequences[store_id] = {}
        
        for prod_cd in store_df['PROD_CD'].unique():
            prod_df = store_df[store_df['PROD_CD'] == prod_cd]
            
            # Group by date to get DAILY total (รวมปริมาณต่อวัน)
            daily_sales = prod_df.groupby('BSNS_DT')['PROD_QTY'].sum().reset_index()
            daily_sales = daily_sales.sort_values('BSNS_DT')
            
            # ต้องมีข้อมูลอย่างน้อย window_size วัน
            if len(daily_sales) >= window_size:
                # เอา window_size วันล่าสุด
                recent = daily_sales.tail(window_size)
                sequence = recent['PROD_QTY'].values
                
                # Normalize เพื่อให้ model train ได้ดี
                max_val = sequence.max() if sequence.max() > 0 else 1
                normalized_seq = sequence / max_val
                
                sequences[store_id][prod_cd] = {
                    'sequence': normalized_seq,
                    'max_val': max_val,
                    'last_qty': sequence[-1],  # ปริมาณวันล่าสุด
                    'mean_qty': sequence.mean()  # ค่าเฉลี่ย 7 วัน
                }
    
    total_sequences = sum(len(prods) for prods in sequences.values())
    logger.info(f"Prepared {total_sequences} time-series sequences from {len(sequences)} stores")
    
    return sequences


def train_autoencoder(sequences: Dict, encoding_dim: int = 16, epochs: int = 50) -> Tuple:
    """
    Train autoencoder on all store-product sequences.
    
    Args:
        sequences: Dictionary of normalized sequences
        encoding_dim: Latent dimension size
        epochs: Training epochs
        
    Returns:
        Trained (autoencoder, encoder, decoder) models
    """
    # Collect all sequences into training data
    all_sequences = []
    for store_data in sequences.values():
        for prod_data in store_data.values():
            all_sequences.append(prod_data['sequence'])
    
    if not all_sequences:
        raise ValueError("No sequences available for training")
    
    X = np.array(all_sequences)
    y = X[:, -3:].mean(axis=1, keepdims=True)
    
    window_size = X.shape[1]
    logger.info(f"Training autoencoder on {len(X)} sequences of length {window_size}")
    logger.info(f"Target: 3-day moving average for next-day prediction")
    
    # Create model
    autoencoder, encoder, decoder = create_autoencoder_model(window_size, encoding_dim)
    
    # Train: input sequence → predict next value
    history = autoencoder.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    logger.info(f"Training complete: loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
    
    return autoencoder, encoder, decoder


def train_recommender(
    data_path: str,
    save_model: bool = True,
    model_name: Optional[str] = None,
    use_database: bool = False,
    db_url: Optional[str] = None
) -> Tuple[Dict[str, List[Tuple[str, int]]], pd.Timestamp]:
    """
    Main function to predict next-day top N products and quantities using Autoencoder.
    
    Pipeline:
    1. Load sales data (from CSV or database)
    2. Prepare time-series sequences (window_size historical days)
    3. Train Autoencoder (Encoder compresses → Decoder predicts)
    4. Generate predictions for each store-product
    5. Save results to database and/or CSV
    
    Args:
        data_path: Path to the sales data CSV file
        save_model: Whether to save the trained model
        model_name: Custom name for saved model
        use_database: Whether to use database instead of CSV
        db_url: Database connection URL (e.g., 'postgresql://user:pass@host/db')
        
    Returns:
        Tuple of (prediction_results, next_date) where:
        - prediction_results: Dict mapping store_id to list of (product_code, predicted_quantity)
        - next_date: The date for which predictions were made
        
    Raises:
        Exception: If prediction fails
    """
    try:
        logger.info("Starting next-day prediction pipeline using Autoencoder")
        
        # 1. Load data
        db_manager = None
        if use_database and db_url:
            db_manager = DatabaseManager(db_url)
            df = db_manager.load_sales_data()
            logger.info("Loaded data from database")
        else:
            df = load_data(data_path)
            logger.info("Loaded data from CSV")

        # 2. Prepare time-series sequences
        df['BSNS_DT'] = pd.to_datetime(df['BSNS_DT'])
        last_date = df['BSNS_DT'].max()
        next_date = last_date + pd.Timedelta(days=1)
        
        window_size = config.prediction.moving_average_window
        encoding_dim = getattr(config.prediction, 'encoding_dim', 16)
        
        logger.info(f"Using window_size={window_size}, encoding_dim={encoding_dim}")
        sequences = prepare_time_series_data(df, window_size)
        
        # 3. Train Autoencoder
        logger.info("Training Autoencoder model")
        autoencoder, encoder, decoder = train_autoencoder(sequences, encoding_dim)

        # 4. Generate predictions for each store (BATCH MODE)
        logger.info("Generating predictions using trained Autoencoder")
        
        # Prepare all sequences for batch prediction
        all_sequences_batch = []
        sequence_mapping = []  # (store_id, prod_cd, max_val)
        
        for store_id in df['STORE_ID'].unique():
            if store_id not in sequences:
                continue
            
            for prod_cd, prod_data in sequences[store_id].items():
                all_sequences_batch.append(prod_data['sequence'])
                sequence_mapping.append((store_id, prod_cd, prod_data['max_val']))
        
        if not all_sequences_batch:
            raise ValueError("No sequences available for prediction")
        
        # Batch prediction (100x faster than loop!)
        X_batch = np.array(all_sequences_batch)
        logger.info(f"Running batch prediction on {len(X_batch)} sequences...")
        
        latent_batch = encoder.predict(X_batch, verbose=0)
        predictions_batch = decoder.predict(latent_batch, verbose=0)
        
        logger.info("Batch prediction completed, organizing results by store...")
        
        # Organize predictions by store
        store_predictions = {}
        
        for idx, (store_id, prod_cd, max_val) in enumerate(sequence_mapping):
            if store_id not in store_predictions:
                store_predictions[store_id] = {}
            
            # Denormalize prediction
            prediction_normalized = predictions_batch[idx][0]
            predicted_qty = int(np.round(prediction_normalized * max_val))
            predicted_qty = max(0, predicted_qty)  # Ensure non-negative
            
            store_predictions[store_id][prod_cd] = predicted_qty
        
        # Get top N for each store
        prediction_results = {}
        top_n = config.prediction.top_n_products
        
        for store_id, product_predictions in store_predictions.items():
            top_products = sorted(product_predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
            prediction_results[store_id] = top_products
        
        logger.info(f"Generated predictions for {len(prediction_results)} stores")

        # 5. Save results
        if use_database and db_manager:
            db_manager.save_predictions(prediction_results, next_date)
            db_manager.save_model_metadata(
                next_date,
                'Autoencoder',
                len(prediction_results),
                df['PROD_CD'].nunique(),
                window_size,
                encoding_dim
            )
            logger.info("Saved predictions to database")
        
        # Also export to CSV
        output_dir = config.output_path
        date_output_dir = os.path.join(output_dir, next_date.strftime('%Y-%m-%d'))
        os.makedirs(date_output_dir, exist_ok=True)
        
        top_n = config.prediction.top_n_products
        output_file = os.path.join(
            date_output_dir,
            f"next_day_top{top_n}_{next_date.strftime('%Y%m%d')}.csv"
        )
        
        with open(output_file, "w") as f:
            f.write("Store_ID,Prediction_Date,Rank,Product_Code,Predicted_Quantity\n")
            for store_id, top_products in prediction_results.items():
                for rank, (prod_cd, qty) in enumerate(top_products, 1):
                    f.write(f"{store_id},{next_date.strftime('%Y-%m-%d')},{rank},{prod_cd},{qty}\n")
        
        logger.info(f"Next day prediction CSV created: {output_file}")
        logger.info(f"Prediction completed for date: {next_date.strftime('%Y-%m-%d')}")
        
        if db_manager:
            db_manager.close()

        return prediction_results, next_date
        
    except Exception as e:
        logger.error(f"Error in next-day prediction: {e}")
        raise