"""
Configuration file for the time series prediction system.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class PredictionConfig:
    """Configuration for time series prediction."""
    moving_average_window: int = 7
    top_n_products: int = 5


@dataclass
class DataConfig:
    """Configuration for data processing."""
    columns_to_drop: List[str] = None
    
    def __post_init__(self):
        if self.columns_to_drop is None:
            self.columns_to_drop = ['UNIT_PROD_CD', 'PACK_SIZE', 'TYPE_IND', 'UPDATE_DATETIME']


@dataclass
class Config:
    """Main configuration class."""
    # Paths
    data_path: str = 'data/T_PROD_DAY_SALE_ORDER_202504091438.csv'
    output_path: str = 'output/'
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize sub-configs
        self.prediction = PredictionConfig()
        self.data = DataConfig()


# Default configuration instance
config = Config()
