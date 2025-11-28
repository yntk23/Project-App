"""
Enhanced Training Module with Ensemble Forecasting Models
Path: backend/src/modeling/train_recommender.py

รวม 3 วิธีพยากรณ์:
1. Autoencoder (ปรับปรุงจากเดิม)
2. Exponential Smoothing / Moving Average
3. Linear Regression on Time Series

+ Ensemble (Weighted Average of all 3 models)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import json

from src.preprocessing.data_preparation import load_data
from src.database.db_manager import DatabaseManager
from src.config import config

logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


# ==================== HELPER FUNCTIONS ====================

def save_model_comparison(all_predictions, sequence_mapping, save_dir='output/models'):
    """บันทึกผลเปรียบเทียบทุกโมเดลเป็น dict"""
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_data = {}
    for idx, (store_id, prod_cd, max_val) in enumerate(sequence_mapping):
        key = f"{store_id}_{prod_cd}"
        comparison_data[key] = {
            'store_id': store_id,
            'prod_cd': prod_cd,
            'predictions': {
                model_name: int(np.round(preds[idx][0]))
                for model_name, preds in all_predictions.items()
            }
        }
    
    with open(os.path.join(save_dir, 'predictions_detail.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"Model comparison data saved to {save_dir}")
    return comparison_data


#region AUTOENCODER 

def create_autoencoder_model(sequence_length: int, encoding_dim: int = 32):
    """Enhanced Autoencoder with better architecture"""
    encoder_input = layers.Input(shape=(sequence_length,), name='encoder_input')
    
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='latent')(x)
    
    encoder = models.Model(encoder_input, encoded, name='encoder')
    
    decoder_input = layers.Input(shape=(encoding_dim,), name='decoder_input')
    x = layers.Dense(64, activation='relu')(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(1, activation='linear', name='output')(x)
    
    decoder = models.Model(decoder_input, decoded, name='decoder')
    
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name='autoencoder')
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Enhanced Autoencoder: seq_len={sequence_length}, encoding_dim={encoding_dim}")
    return autoencoder, encoder, decoder
#endregion


# region EXPONENTIAL SMOOTHING 

class ExponentialSmoothingModel:
    """Exponential Smoothing + Moving Average Model"""
    
    def __init__(self, alpha=0.3, window=3):
        self.alpha = alpha
        self.window = window
        self.name = 'exp_smoothing'
    
    def predict_batch(self, sequences):
        """Batch prediction"""
        predictions = []
        for seq in sequences:
            # Exponential Smoothing
            s = seq[0]
            for val in seq[1:]:
                s = self.alpha * val + (1 - self.alpha) * s
            exp_pred = s
            
            # Moving Average
            ma_pred = np.mean(seq[-self.window:])
            
            # Combined
            prediction = 0.6 * exp_pred + 0.4 * ma_pred
            predictions.append(prediction)
        
        return np.array(predictions).reshape(-1, 1)
#endregion


# region LINEAR REGRESSION 

class LinearRegressionModel:
    """Linear Regression for Time Series"""
    
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.name = 'linear_regression'
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        """Train linear regression"""
        X_features = self._create_features(X_train)
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_train.ravel())
        self.is_fitted = True
    
    def predict_batch(self, sequences):
        """Batch prediction"""
        if not self.is_fitted:
            return np.mean(sequences, axis=1, keepdims=True)
        
        X_features = self._create_features(sequences)
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        return predictions.reshape(-1, 1)
    
    def _create_features(self, sequences):
        """Create features from sequences"""
        features = []
        for seq in sequences:
            feat = [
                seq[-1],
                np.mean(seq[-3:]),
                np.mean(seq),
                np.std(seq),
                seq[-1] - seq[0],
                np.max(seq),
                np.min(seq)
            ]
            features.append(feat)
        return np.array(features)
#endregion

# region ENSEMBLE SYSTEM 

class EnsembleModel:
    """Ensemble of 3 forecasting models"""
    
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {'autoencoder': 0.4, 'exp_smoothing': 0.3, 'linear_regression': 0.3}
        else:
            self.weights = weights
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, autoencoder_pred, exp_pred, linear_pred):
        """Combine predictions"""
        ensemble_pred = (
            self.weights['autoencoder'] * autoencoder_pred +
            self.weights['exp_smoothing'] * exp_pred +
            self.weights['linear_regression'] * linear_pred
        )
        return ensemble_pred
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = 0
        
        metrics = {
            'model': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
        
        logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return metrics
#endregion

# ==================== DATA PREPARATION ====================

def prepare_time_series_data(df: pd.DataFrame, window_size: int = 7) -> Dict:
    """Prepare time-series sequences"""
    df = df.sort_values('BSNS_DT')
    sequences = {}
    
    for store_id in df['STORE_ID'].unique():
        store_df = df[df['STORE_ID'] == store_id]
        sequences[store_id] = {}
        
        for prod_cd in store_df['PROD_CD'].unique():
            prod_df = store_df[store_df['PROD_CD'] == prod_cd]
            
            daily_sales = prod_df.groupby('BSNS_DT')['PROD_QTY'].sum().reset_index()
            daily_sales = daily_sales.sort_values('BSNS_DT')
            
            if len(daily_sales) >= window_size:
                recent = daily_sales.tail(window_size)
                sequence = recent['PROD_QTY'].values
                
                max_val = sequence.max() if sequence.max() > 0 else 1
                normalized_seq = sequence / max_val
                
                sequences[store_id][prod_cd] = {
                    'sequence': normalized_seq,
                    'raw_sequence': sequence,
                    'max_val': max_val,
                    'last_qty': sequence[-1],
                    'mean_qty': sequence.mean()
                }
    
    total_sequences = sum(len(prods) for prods in sequences.values())
    logger.info(f"Prepared {total_sequences} sequences from {len(sequences)} stores")
    
    return sequences


# ==================== MAIN TRAINING FUNCTION ====================

def train_recommender(
    data_path: str,
    save_model: bool = True,
    model_name: Optional[str] = None,
    use_database: bool = False,
    db_url: Optional[str] = None,
    ensemble_weights: Optional[Dict] = None,
    selected_model: str = 'ensemble'  # 'ensemble', 'autoencoder', 'exp_smoothing', 'linear_regression'
) -> Tuple[Dict[str, List[Tuple[str, int]]], pd.Timestamp, Dict]:
    """
    Main training with Ensemble
    
    Returns:
        (prediction_results, next_date, model_comparison_data)
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Starting Prediction Pipeline - Model: {selected_model.upper()}")
        logger.info("=" * 80)
        
        # 1. Load data
        db_manager = None
        if use_database and db_url:
            db_manager = DatabaseManager(db_url)
            df = db_manager.load_sales_data()
        else:
            df = load_data(data_path)

        df['BSNS_DT'] = pd.to_datetime(df['BSNS_DT'])
        last_date = df['BSNS_DT'].max()
        next_date = last_date + pd.Timedelta(days=1)
        
        window_size = config.prediction.moving_average_window
        encoding_dim = config.prediction.encoding_dim
        
        logger.info(f"Prediction date: {next_date.strftime('%Y-%m-%d')}")
        
        # 2. Prepare sequences
        sequences = prepare_time_series_data(df, window_size)
        
        # 3. Prepare training data
        all_normalized_sequences = []
        all_raw_sequences = []
        all_targets = []
        sequence_mapping = []
        
        for store_id in df['STORE_ID'].unique():
            if store_id not in sequences:
                continue
            
            for prod_cd, prod_data in sequences[store_id].items():
                all_normalized_sequences.append(prod_data['sequence'])
                all_raw_sequences.append(prod_data['raw_sequence'])
                all_targets.append(np.mean(prod_data['raw_sequence'][-3:]))
                sequence_mapping.append((store_id, prod_cd, prod_data['max_val']))
        
        X_normalized = np.array(all_normalized_sequences)
        X_raw = np.array(all_raw_sequences)
        y_true = np.array(all_targets).reshape(-1, 1)
        
        logger.info(f"Training data: {len(X_normalized)} sequences")
        
        # 4. Train all models
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)
        
        # Autoencoder
        autoencoder, encoder, decoder = create_autoencoder_model(window_size, encoding_dim)
        history = autoencoder.fit(
            X_normalized, y_true / np.array([m[2] for m in sequence_mapping]).reshape(-1, 1),
            epochs=config.model.autoencoder_epochs,
            batch_size=config.model.autoencoder_batch_size,
            validation_split=config.model.autoencoder_validation_split,
            verbose=0
        )
        logger.info(f"✅ Autoencoder trained")
        
        # Exponential Smoothing
        exp_model = ExponentialSmoothingModel(alpha=config.model.exp_smoothing_alpha, window=config.model.exp_smoothing_window)
        logger.info(f"✅ Exponential Smoothing ready")
        
        # Linear Regression
        linear_model = LinearRegressionModel(alpha=config.model.linear_regression_alpha)
        linear_model.train(X_raw, y_true)
        logger.info(f"✅ Linear Regression trained")
        
        # 5. Generate predictions
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING PREDICTIONS")
        logger.info("=" * 60)
        
        latent = encoder.predict(X_normalized, verbose=0)
        autoencoder_pred_normalized = decoder.predict(latent, verbose=0)
        autoencoder_pred = autoencoder_pred_normalized * np.array([m[2] for m in sequence_mapping]).reshape(-1, 1)
        
        exp_pred = exp_model.predict_batch(X_raw)
        linear_pred = linear_model.predict_batch(X_raw)
        
        # Ensemble
        ensemble = EnsembleModel(weights=ensemble_weights)
        ensemble_pred = ensemble.predict(autoencoder_pred, exp_pred, linear_pred)
        
        # 6. Calculate metrics
        all_metrics = []
        all_metrics.append(ensemble.calculate_metrics(y_true, autoencoder_pred, 'Autoencoder'))
        all_metrics.append(ensemble.calculate_metrics(y_true, exp_pred, 'Exponential Smoothing'))
        all_metrics.append(ensemble.calculate_metrics(y_true, linear_pred, 'Linear Regression'))
        all_metrics.append(ensemble.calculate_metrics(y_true, ensemble_pred, 'Ensemble'))
        
        # 7. Save all predictions
        all_predictions = {
            'autoencoder': autoencoder_pred,
            'exp_smoothing': exp_pred,
            'linear_regression': linear_pred,
            'ensemble': ensemble_pred
        }
        
        comparison_data = save_model_comparison(all_predictions, sequence_mapping)
        
        # 8. Select model to use
        if selected_model == 'autoencoder':
            final_pred = autoencoder_pred
            logger.info("Using: Autoencoder")
        elif selected_model == 'exp_smoothing':
            final_pred = exp_pred
            logger.info("Using: Exponential Smoothing")
        elif selected_model == 'linear_regression':
            final_pred = linear_pred
            logger.info("Using: Linear Regression")
        else:  # ensemble
            final_pred = ensemble_pred
            logger.info("Using: Ensemble")
        
        # 9. Organize predictions
        store_predictions = {}
        store_predictions_detail = {}
        
        for idx, (store_id, prod_cd, max_val) in enumerate(sequence_mapping):
            if store_id not in store_predictions:
                store_predictions[store_id] = {}
                store_predictions_detail[store_id] = {}
            
            predicted_qty = int(np.round(final_pred[idx][0]))
            predicted_qty = max(0, predicted_qty)
            
            store_predictions[store_id][prod_cd] = predicted_qty
            
            store_predictions_detail[store_id][prod_cd] = {
                'ensemble': int(np.round(ensemble_pred[idx][0])),
                'autoencoder': int(np.round(autoencoder_pred[idx][0])),
                'exp_smoothing': int(np.round(exp_pred[idx][0])),
                'linear_regression': int(np.round(linear_pred[idx][0]))
            }
        
        # 10. Get top N
        prediction_results = {}
        top_n = config.prediction.top_n_products
        
        for store_id, product_predictions in store_predictions.items():
            top_products = sorted(product_predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
            prediction_results[store_id] = top_products
        
        # 11. Save models
        if save_model:
            save_dir = 'output/models'
            os.makedirs(save_dir, exist_ok=True)
            
            autoencoder.save(os.path.join(save_dir, 'autoencoder_model.keras'))
            encoder.save(os.path.join(save_dir, 'encoder_model.keras'))
            decoder.save(os.path.join(save_dir, 'decoder_model.keras'))
            
            # Save metrics
            with open(os.path.join(save_dir, 'ensemble_comparison.json'), 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'weights': ensemble.weights,
                    'metrics': all_metrics,
                    'selected_model': selected_model
                }, f, indent=2)
            
            logger.info(f"Models saved to {save_dir}")
        
        # 12. Save to CSV
        output_dir = config.output_path
        date_output_dir = os.path.join(output_dir, next_date.strftime('%Y-%m-%d'))
        os.makedirs(date_output_dir, exist_ok=True)
        
        output_file = os.path.join(
            date_output_dir,
            f"next_day_top{top_n}_{next_date.strftime('%Y%m%d')}.csv"
        )
        
        with open(output_file, "w") as f:
            f.write("Store_ID,Prediction_Date,Rank,prod_cd,Predicted_Quantity\n")
            for store_id, top_products in prediction_results.items():
                for rank, (prod_cd, qty) in enumerate(top_products, 1):
                    f.write(f"{store_id},{next_date.strftime('%Y-%m-%d')},{rank},{prod_cd},{qty}\n")
        
        # Detailed comparison
        detail_file = os.path.join(
            date_output_dir,
            f"model_comparison_{next_date.strftime('%Y%m%d')}.csv"
        )
        
        with open(detail_file, "w") as f:
            f.write("Store_ID,prod_cd,Ensemble,Autoencoder,Exp_Smoothing,Linear_Regression\n")
            for store_id in store_predictions_detail:
                for prod_cd, preds in store_predictions_detail[store_id].items():
                    f.write(f"{store_id},{prod_cd},{preds['ensemble']},{preds['autoencoder']},{preds['exp_smoothing']},{preds['linear_regression']}\n")
        
        logger.info(f"Predictions saved: {output_file}")
        logger.info(f"Comparison saved: {detail_file}")
        
        # 13. Save to database
        if use_database and db_manager:
            db_manager.save_predictions(prediction_results, next_date)
            
            # Save model comparison to database
            db_manager.save_model_comparison(store_predictions_detail, next_date)
            logger.info("Model comparison saved to database")
            
            db_manager.save_model_metadata(
                next_date,
                f'{selected_model.title()} Model',
                len(prediction_results),
                df['PROD_CD'].nunique(),
                window_size,
                encoding_dim
            )
        
        if db_manager:
            db_manager.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION COMPLETED")
        logger.info("=" * 80)

        return prediction_results, next_date, {
            'metrics': all_metrics,
            'comparison_data': store_predictions_detail
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise