"""
Recommendation generation module.

This module handles finding similar stores and generating product recommendations.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

# Set up logger
logger = logging.getLogger(__name__)


def find_similar_stores(
    store_data: np.ndarray, 
    encoded_transactions: np.ndarray, 
    all_stores: List[str],
    similarity_threshold: float = None,
    top_k: int = None
) -> List[str]:
    """
    Find the most similar stores using cosine similarity.
    
    Args:
        store_data: Encoded features for the target store
        encoded_transactions: Encoded features for all stores
        all_stores: List of all store IDs
        similarity_threshold: Minimum similarity threshold (default from config)
        top_k: Number of top similar stores to return (default from config)
        
    Returns:
        List of similar store IDs
        
    Raises:
        ValueError: If input dimensions don't match
    """
    if similarity_threshold is None:
        similarity_threshold = config.recommendation.similarity_threshold
    if top_k is None:
        top_k = config.recommendation.top_similar_stores
    
    try:
        logger.debug(f"Finding similar stores with threshold {similarity_threshold} and top_k {top_k}")
        
        # Validate input dimensions
        if len(store_data.shape) == 1:
            store_data = store_data.reshape(1, -1)
        
        if store_data.shape[1] != encoded_transactions.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {store_data.shape[1]} vs {encoded_transactions.shape[1]}")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(store_data, encoded_transactions)[0]
        
        # Get indices of most similar stores (excluding self if present)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter by similarity threshold and top_k
        similar_stores = []
        for idx in similar_indices[:top_k * 2]:  # Check more stores to account for threshold filtering
            if len(similar_stores) >= top_k:
                break
            if similarities[idx] >= similarity_threshold:
                similar_stores.append(all_stores[idx])
        
        logger.debug(f"Found {len(similar_stores)} similar stores")
        if similar_stores:
            logger.debug(f"Top similarities: {[f'{similarities[similar_indices[i]]:.3f}' for i in range(min(3, len(similar_stores)))]}")
        
        return similar_stores
        
    except Exception as e:
        logger.error(f"Error finding similar stores: {e}")
        return []


def find_similar_products(
    similar_stores: List[str], 
    transaction_matrix: pd.DataFrame,
    top_n: int = None
) -> List[str]:
    """
    Find the most frequent products among similar stores.
    
    Args:
        similar_stores: List of similar store IDs
        transaction_matrix: Transaction matrix with stores as rows and products as columns
        top_n: Number of top products to recommend (default from config)
        
    Returns:
        List of recommended product IDs
        
    Raises:
        ValueError: If no similar stores are provided or found in matrix
    """
    if top_n is None:
        top_n = config.recommendation.top_n_recommendations
    
    try:
        if not similar_stores:
            logger.warning("No similar stores provided for product recommendation")
            return []
        
        # Filter stores that exist in the transaction matrix
        available_stores = [store for store in similar_stores if store in transaction_matrix.index]
        
        if not available_stores:
            logger.warning(f"None of the similar stores {similar_stores} found in transaction matrix")
            return []
        
        logger.debug(f"Using {len(available_stores)} stores for product recommendation")
        
        # Sum product quantities across similar stores
        similar_product_counts = transaction_matrix.loc[available_stores].sum(axis=0)
        
        # Get top N products
        recommended_products = similar_product_counts.sort_values(ascending=False).head(top_n).index.tolist()
        
        logger.debug(f"Generated {len(recommended_products)} product recommendations")
        
        return recommended_products
        
    except Exception as e:
        logger.error(f"Error finding similar products: {e}")
        return []


def generate_fallback_recommendations(
    transaction_matrix: pd.DataFrame,
    top_n: int = None
) -> List[str]:
    """
    Generate fallback recommendations based on overall product popularity.
    
    Args:
        transaction_matrix: Transaction matrix
        top_n: Number of recommendations to generate
        
    Returns:
        List of popular product IDs
    """
    if top_n is None:
        top_n = config.recommendation.top_n_recommendations
    
    try:
        logger.info("Generating fallback recommendations based on product popularity")
        
        # Calculate overall product popularity
        product_popularity = transaction_matrix.sum(axis=0)
        popular_products = product_popularity.sort_values(ascending=False).head(top_n).index.tolist()
        
        logger.debug(f"Generated {len(popular_products)} fallback recommendations")
        
        return popular_products
        
    except Exception as e:
        logger.error(f"Error generating fallback recommendations: {e}")
        return []


def validate_recommendations(
    recommendations: dict,
    transaction_matrix: pd.DataFrame
) -> dict:
    """
    Validate and clean recommendations.
    
    Args:
        recommendations: Dictionary of store_id -> product_list
        transaction_matrix: Transaction matrix for validation
        
    Returns:
        Cleaned recommendations dictionary
    """
    try:
        logger.info("Validating recommendations...")
        
        valid_products = set(transaction_matrix.columns)
        cleaned_recommendations = {}
        
        for store_id, products in recommendations.items():
            # Filter out invalid products
            valid_products_for_store = [p for p in products if p in valid_products]
            
            if len(valid_products_for_store) != len(products):
                logger.warning(f"Store {store_id}: Filtered out {len(products) - len(valid_products_for_store)} invalid products")
            
            cleaned_recommendations[store_id] = valid_products_for_store
        
        logger.info(f"Validated recommendations for {len(cleaned_recommendations)} stores")
        
        return cleaned_recommendations
        
    except Exception as e:
        logger.error(f"Error validating recommendations: {e}")
        return recommendations