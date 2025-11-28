"""
Model Evaluation and Comparison Tool
Path: backend/src/modeling/model_evaluator.py

ใช้สำหรับ:
1. วิเคราะห์ผลลัพธ์แต่ละโมเดล
2. สร้างกราฟเปรียบเทียบ
3. คำนวณ metrics เพิ่มเติม
4. แนะนำ weights ที่เหมาะสม

Usage:
    python backend/src/modeling/model_evaluator.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """ประเมินและเปรียบเทียบโมเดล"""
    
    def __init__(self, comparison_file='output/models/ensemble_comparison.json'):
        self.comparison_file = comparison_file
        self.metrics = None
        self.load_comparison()
    
    def load_comparison(self):
        """โหลดผลการเปรียบเทียบ"""
        if os.path.exists(self.comparison_file):
            with open(self.comparison_file, 'r') as f:
                data = json.load(f)
                self.metrics = data.get('metrics', [])
                logger.info(f"Loaded comparison from {self.comparison_file}")
        else:
            logger.warning(f"Comparison file not found: {self.comparison_file}")
            self.metrics = []
    
    def print_summary(self):
        """พิมพ์สรุปผลการเปรียบเทียบ"""
        if not self.metrics:
            print("No metrics available")
            return
        
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Model':<25} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
        print("-" * 80)
        
        for m in self.metrics:
            print(f"{m['model']:<25} {m['mae']:>12.2f} {m['rmse']:>12.2f} {m['mape']:>11.2f}%")
        
        print("=" * 80)
        
        # หา best model
        best_mae = min(self.metrics, key=lambda x: x['mae'])
        best_rmse = min(self.metrics, key=lambda x: x['rmse'])
        best_mape = min(self.metrics, key=lambda x: x['mape'])
        
        print("\nBest Models:")
        print(f"  - Lowest MAE:  {best_mae['model']} ({best_mae['mae']:.2f})")
        print(f"  - Lowest RMSE: {best_rmse['model']} ({best_rmse['rmse']:.2f})")
        print(f"  - Lowest MAPE: {best_mape['model']} ({best_mape['mape']:.2f}%)")
        print("=" * 80 + "\n")
    
    def plot_comparison(self, save_path='output/models/model_comparison.png'):
        """สร้างกราฟเปรียบเทียบ"""
        if not self.metrics:
            logger.warning("No metrics to plot")
            return
        
        models = [m['model'] for m in self.metrics]
        mae_values = [m['mae'] for m in self.metrics]
        rmse_values = [m['rmse'] for m in self.metrics]
        mape_values = [m['mape'] for m in self.metrics]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE
        axes[0].bar(models, mae_values, color='skyblue')
        axes[0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MAE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # RMSE
        axes[1].bar(models, rmse_values, color='lightcoral')
        axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # MAPE
        axes[2].bar(models, mape_values, color='lightgreen')
        axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved: {save_path}")
        print(f"Plot saved: {save_path}")
    
    def suggest_weights(self):
        """แนะนำ weights สำหรับ ensemble ตาม performance"""
        if len(self.metrics) < 4:  # ต้องมีครบ 3 โมเดล + ensemble
            logger.warning("Insufficient metrics for weight suggestion")
            return None
        
        # ใช้ inverse error เป็น weight
        model_metrics = {m['model']: m for m in self.metrics if m['model'] != 'Ensemble'}
        
        # คำนวณ inverse MAE (ยิ่ง MAE ต่ำ weight ยิ่งสูง)
        inverse_errors = {name: 1 / (m['mae'] + 1) for name, m in model_metrics.items()}
        total = sum(inverse_errors.values())
        suggested_weights = {name: val / total for name, val in inverse_errors.items()}
        
        print("\n" + "=" * 80)
        print("SUGGESTED ENSEMBLE WEIGHTS (based on MAE)")
        print("=" * 80)
        
        weight_mapping = {
            'Autoencoder': 'autoencoder',
            'Exponential Smoothing': 'exp_smoothing',
            'Linear Regression': 'linear_reg'
        }
        
        formatted_weights = {}
        for name, weight in suggested_weights.items():
            key = weight_mapping.get(name, name.lower().replace(' ', '_'))
            formatted_weights[key] = round(weight, 3)
            print(f"  {name:<25}: {weight:.3f}")
        
        print("=" * 80 + "\n")
        
        return formatted_weights
    
    def analyze_predictions(self, comparison_csv):
        """วิเคราะห์ผลพยากรณ์จาก CSV"""
        if not os.path.exists(comparison_csv):
            logger.warning(f"File not found: {comparison_csv}")
            return
        
        df = pd.read_csv(comparison_csv)
        
        print("\n" + "=" * 80)
        print("PREDICTION ANALYSIS")
        print("=" * 80)
        
        print(f"\nTotal predictions: {len(df)}")
        print(f"Number of stores: {df['Store_ID'].nunique()}")
        print(f"Number of products: {df['prod_cd'].nunique()}")
        
        print("\nStatistics for each model:")
        models = ['Ensemble', 'Autoencoder', 'Exp_Smoothing', 'Linear_Regression']
        
        for model in models:
            if model in df.columns:
                print(f"\n{model}:")
                print(f"  Mean: {df[model].mean():.2f}")
                print(f"  Median: {df[model].median():.2f}")
                print(f"  Std: {df[model].std():.2f}")
                print(f"  Min: {df[model].min():.2f}")
                print(f"  Max: {df[model].max():.2f}")
        
        # เปรียบเทียบความแตกต่าง
        if all(col in df.columns for col in models):
            print("\nAverage differences from Ensemble:")
            for model in ['Autoencoder', 'Exp_Smoothing', 'Linear_Regression']:
                diff = np.abs(df['Ensemble'] - df[model]).mean()
                print(f"  {model}: {diff:.2f}")
        
        print("=" * 80 + "\n")


def main():
    """Main function สำหรับรัน standalone"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # สร้าง evaluator
    evaluator = ModelEvaluator()
    
    # แสดงสรุป
    evaluator.print_summary()
    
    # สร้างกราฟ
    evaluator.plot_comparison()
    
    # แนะนำ weights
    suggested_weights = evaluator.suggest_weights()
    
    if suggested_weights:
        print("\nTo use these weights, modify train_recommender() call:")
        print("```python")
        print("ensemble_weights = {")
        for key, val in suggested_weights.items():
            print(f"    '{key}': {val},")
        print("}")
        print("train_recommender(data_path, ensemble_weights=ensemble_weights)")
        print("```\n")
    
    # วิเคราะห์ CSV ล่าสุด
    output_dir = 'backend/output'
    if os.path.exists(output_dir):
        # หาไฟล์ล่าสุด
        all_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.startswith('model_comparison_') and file.endswith('.csv'):
                    all_files.append(os.path.join(root, file))
        
        if all_files:
            latest_file = max(all_files, key=os.path.getmtime)
            print(f"\nAnalyzing latest comparison file: {latest_file}")
            evaluator.analyze_predictions(latest_file)


if __name__ == '__main__':
    main()