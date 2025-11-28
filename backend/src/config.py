"""
Configuration file for the time series prediction system.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PredictionConfig:
    """Configuration for time series prediction."""
    moving_average_window: int = 7
    top_n_products: int = 5
    encoding_dim: int = 32
    
    # Model selection
    available_models: List[str] = None
    default_model: str = 'ensemble'
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = ['ensemble', 'autoencoder', 'exp_smoothing', 'linear_regression']
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'autoencoder': 0.2,
                'exp_smoothing': 0.2,
                'linear_regression': 0.6
            }


@dataclass
class DataConfig:
    """Configuration for data processing."""
    columns_to_drop: List[str] = None
    
    def __post_init__(self):
        if self.columns_to_drop is None:
            self.columns_to_drop = ['UNIT_PROD_CD', 'PACK_SIZE', 'TYPE_IND', 'UPDATE_DATETIME']


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Autoencoder
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 32
    autoencoder_validation_split: float = 0.1
    
    # Exponential Smoothing
    exp_smoothing_alpha: float = 0.3
    exp_smoothing_window: int = 3
    
    # Linear Regression
    linear_regression_alpha: float = 1.0


@dataclass
class Config:
    """Main configuration class."""
    # Paths
    data_path: str = 'data/T_PROD_DAY_SALE_ORDER_202504091438.csv'
    output_path: str = 'output/'
    model_save_path: str = 'output/models/'
    
    # Database
    db_url: Optional[str] = None
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Initialize sub-configs
        self.prediction = PredictionConfig()
        self.data = DataConfig()
        self.model = ModelConfig()


# Default configuration instance
config = Config()