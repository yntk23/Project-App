"""
Export utilities for recommendation results.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def export_recommendations_to_excel(
    recommendations: Dict[str, List[str]],
    evaluation_results: Dict[str, float],
    output_dir: str = "output",
    filename_prefix: str = "recommendations"
) -> str:
    """
    Export recommendations to Excel file with multiple sheets.
    
    Args:
        recommendations: Dictionary of store_id -> recommended_products
        evaluation_results: Evaluation metrics
        output_dir: Directory to save the Excel file
        filename_prefix: Prefix for the filename
        
    Returns:
        Path to the created Excel file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)
        
        logger.info(f"Creating Excel file: {excel_path}")
        
        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary (one row per store)
            summary_data = []
            for store_id, products in recommendations.items():
                summary_data.append({
                    'Store_ID': store_id,
                    'Recommendation_1': products[0] if len(products) > 0 else '',
                    'Recommendation_2': products[1] if len(products) > 1 else '',
                    'Recommendation_3': products[2] if len(products) > 2 else '',
                    'Recommendation_4': products[3] if len(products) > 3 else '',
                    'Recommendation_5': products[4] if len(products) > 4 else '',
                    'All_Recommendations': ', '.join(products),
                    'Number_of_Recommendations': len(products)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed (one row per recommendation)
            detailed_data = []
            for store_id, products in recommendations.items():
                for rank, product in enumerate(products, 1):
                    detailed_data.append({
                        'Store_ID': store_id,
                        'Recommendation_Rank': rank,
                        'prod_cd': product
                    })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
            
            # Sheet 3: Evaluation Metrics
            eval_data = []
            for metric, value in evaluation_results.items():
                eval_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': value
                })
            
            eval_df = pd.DataFrame(eval_data)
            eval_df.to_excel(writer, sheet_name='Evaluation', index=False)
            
            # Sheet 4: Store Statistics
            store_stats = []
            all_products = set()
            for products in recommendations.values():
                all_products.update(products)
            
            # Count how many times each product was recommended
            product_counts = {}
            for products in recommendations.values():
                for product in products:
                    product_counts[product] = product_counts.get(product, 0) + 1
            
            # Top recommended products
            top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
            
            for product, count in top_products:
                store_stats.append({
                    'prod_cd': product,
                    'Times_Recommended': count,
                    'Percentage_of_Stores': f"{count/len(recommendations)*100:.1f}%"
                })
            
            stats_df = pd.DataFrame(store_stats)
            stats_df.to_excel(writer, sheet_name='Product_Statistics', index=False)
        
        logger.info(f"Excel file created successfully: {excel_path}")
        logger.info(f"Sheets created: Summary, Detailed, Evaluation, Product_Statistics")
        
        return excel_path
        
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}")
        raise


def export_recommendations_to_csv(
    recommendations: Dict[str, List[str]],
    evaluation_results: Dict[str, float],
    output_dir: str = "output",
    filename_prefix: str = "recommendations",
    sales_data: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Export recommendations to CSV files organized by date.
    
    Args:
        recommendations: Dictionary of store_id -> recommended_products
        evaluation_results: Evaluation metrics
        output_dir: Directory to save the CSV files
        filename_prefix: Prefix for the filename
        sales_data: Optional sales data for calculating order quantities
        
    Returns:
        List of paths to created CSV files
    """
    try:
        # Create date-based subdirectory
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(date_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        created_files = []
        
        # Calculate recommended order quantities if sales data is available
        order_quantities = {}
        if sales_data is not None:
            order_quantities = _calculate_order_quantities(recommendations, sales_data)
        
        # Summary CSV
        summary_data = []
        for store_id, products in recommendations.items():
            product_list = []
            for product in products:
                if order_quantities and store_id in order_quantities and product in order_quantities[store_id]:
                    qty = order_quantities[store_id][product]
                    product_list.append(f"{product} (Qty: {qty})")
                else:
                    product_list.append(product)
            
            summary_data.append({
                'Store_ID': store_id,
                'Recommended_Products': ', '.join(product_list),
                'Number_of_Recommendations': len(products)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(date_output_dir, f"{filename_prefix}_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        created_files.append(summary_file)
        
        # Detailed CSV
        detailed_data = []
        for store_id, products in recommendations.items():
            for rank, product in enumerate(products, 1):
                qty = ""
                if order_quantities and store_id in order_quantities and product in order_quantities[store_id]:
                    qty = order_quantities[store_id][product]
                
                detailed_data.append({
                    'Store_ID': store_id,
                    'Recommendation_Rank': rank,
                    'prod_cd': product,
                    'Recommended_Order_Quantity': qty if qty else "N/A"
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = os.path.join(date_output_dir, f"{filename_prefix}_detailed_{timestamp}.csv")
        detailed_df.to_csv(detailed_file, index=False)
        created_files.append(detailed_file)
        
        logger.info(f"CSV files created: {created_files}")
        
        return created_files
        
    except Exception as e:
        logger.error(f"Error creating CSV files: {e}")
        raise


def print_recommendations_summary(
    recommendations: Dict[str, List[str]],
    evaluation_results: Dict[str, float],
    max_stores_to_show: int = 5
) -> None:
    """
    Print a summary of recommendations to console.
    
    Args:
        recommendations: Dictionary of store_id -> recommended_products
        evaluation_results: Evaluation metrics
        max_stores_to_show: Maximum number of stores to show in console
    """
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Stores: {len(recommendations)}")
    
    # Show sample stores
    store_items = list(recommendations.items())
    stores_to_show = min(max_stores_to_show, len(store_items))
    
    print(f"\nSample Recommendations (showing {stores_to_show} of {len(recommendations)} stores):")
    for i in range(stores_to_show):
        store_id, products = store_items[i]
        print(f"  Store {store_id}: {products}")
    
    if len(recommendations) > max_stores_to_show:
        print(f"  ... and {len(recommendations) - max_stores_to_show} more stores")
    
    # Evaluation metrics
    print(f"\nEvaluation Metrics:")
    for metric, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print(f"{'='*60}")


def _calculate_order_quantities(recommendations: Dict[str, List[str]], sales_data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Calculate recommended daily order quantities based on historical sales data.
    
    Args:
        recommendations: Dictionary of store_id -> recommended_products
        sales_data: Historical sales data with STORE_ID, PROD_CD, PROD_QTY columns
        
    Returns:
        Dictionary of {store_id: {product_id: recommended_daily_quantity}}
    """
    order_quantities = {}
    
    try:
        # Calculate average daily quantities for each product across all stores
        if 'BSNS_DT' in sales_data.columns:
            # Calculate daily averages per product across all stores
            daily_avg = sales_data.groupby(['PROD_CD', 'BSNS_DT'])['PROD_QTY'].sum().reset_index()
            product_avg_qty = daily_avg.groupby('PROD_CD')['PROD_QTY'].mean().round().astype(int)
        else:
            # Fallback: calculate overall average per product per transaction
            product_avg_qty = sales_data.groupby('PROD_CD')['PROD_QTY'].mean().round().astype(int)
        
        # For each store's recommendations, assign daily quantities
        for store_id, products in recommendations.items():
            order_quantities[store_id] = {}
            
            for product in products:
                if product in product_avg_qty.index:
                    # Use historical daily average, with minimum of 1
                    base_daily_qty = max(1, product_avg_qty[product])
                    
                    # Apply store-specific modifier based on store daily performance
                    store_sales = sales_data[sales_data['STORE_ID'] == store_id]
                    if not store_sales.empty and 'BSNS_DT' in sales_data.columns:
                        # Calculate store's daily average for this product (if exists)
                        store_product_sales = store_sales[store_sales['PROD_CD'] == product]
                        if not store_product_sales.empty:
                            store_daily_avg = store_product_sales.groupby('BSNS_DT')['PROD_QTY'].sum().mean()
                            recommended_qty = max(1, int(store_daily_avg))
                        else:
                            # Use store's overall daily performance vs global average
                            store_daily_avg = store_sales.groupby('BSNS_DT')['PROD_QTY'].sum().mean()
                            global_daily_avg = sales_data.groupby('BSNS_DT')['PROD_QTY'].sum().mean()
                            
                            if global_daily_avg > 0:
                                store_multiplier = min(2.0, max(0.5, store_daily_avg / global_daily_avg))
                                recommended_qty = max(1, int(base_daily_qty * store_multiplier))
                            else:
                                recommended_qty = base_daily_qty
                    else:
                        recommended_qty = base_daily_qty
                    
                    order_quantities[store_id][product] = recommended_qty
                else:
                    # Default daily quantity for new products
                    order_quantities[store_id][product] = 1
    
    except Exception as e:
        logger.warning(f"Error calculating daily order quantities: {e}")
        # Return empty dict on error - quantities will show as N/A
        return {}
    
    return order_quantities
