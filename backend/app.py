"""
Flask Backend API with Ensemble Model Selection
Path: backend/app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import subprocess
import threading
import sys
import os
import json

app = Flask(__name__)
CORS(app)

DATABASE = 'sales_data.db'

prediction_status = {"running": False, "last_run": None, "error": None, "selected_model": "ensemble"}

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get predictions with optional model filter"""
    store_id = request.args.get('store_id')
    prod_cd = request.args.get('prod_cd')
    model_type = request.args.get('model', 'ensemble')  # ensemble, autoencoder, exp_smoothing, linear_regression
    
    if not store_id:
        return jsonify({"error": "store_id is required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if prod_cd:
        query = """
            SELECT 
                prediction_date as date, 
                prod_cd, 
                predicted_quantity as predicted_qty,
                created_at
            FROM predictions
            WHERE store_id = ? AND prod_cd = ?
            ORDER BY prediction_date DESC, rank ASC
        """
        cursor.execute(query, (store_id, prod_cd))
    else:
        query = """
            SELECT 
                prediction_date as date, 
                prod_cd, 
                predicted_quantity as predicted_qty,
                created_at
            FROM predictions
            WHERE store_id = ?
            ORDER BY prediction_date DESC, rank ASC
            LIMIT 100
        """
        cursor.execute(query, (store_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "date": row['date'],
            "prod_cd": row['prod_cd'],
            "predicted_qty": row['predicted_qty'],
            "created_at": row['created_at'],
            "model": model_type
        })
    
    return jsonify(results)

@app.route('/stores', methods=['GET'])
def get_stores():
    """Get list of all available store IDs"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT store_id 
        FROM sales_data 
        ORDER BY store_id
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    stores = [row['store_id'] for row in rows]
    return jsonify({"stores": stores, "total": len(stores)})

@app.route('/model-comparison', methods=['GET'])
def get_model_comparison():
    """Get model comparison data"""
    try:
        comparison_file = 'output/models/ensemble_comparison.json'
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"error": "No comparison data available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-predictions/<store_id>', methods=['GET'])
def get_model_predictions_detail(store_id):
    """Get detailed predictions from all models for a store"""
    try:
        # Get from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get latest prediction date for this store
        cursor.execute("""
            SELECT store_id, prod_cd, ensemble, autoencoder, exp_smoothing, linear_regression
            FROM model_comparison
            WHERE store_id = ? 
            AND prediction_date = (
                SELECT MAX(prediction_date) 
                FROM model_comparison 
                WHERE store_id = ?
            )
            ORDER BY ensemble DESC
        """, (store_id, store_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({
                "error": "No comparison data available",
                "message": "Please run prediction first",
                "store_id": store_id
            }), 404
        
        results = []
        for row in rows:
            results.append({
                'prod_cd': row['prod_cd'],
                'ensemble': int(row['ensemble']),
                'autoencoder': int(row['autoencoder']),
                'exp_smoothing': int(row['exp_smoothing']),
                'linear_regression': int(row['linear_regression'])
            })
        
        return jsonify(results)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500

@app.route('/run-prediction', methods=['POST'])
def run_prediction():
    """Trigger prediction script with model selection"""
    global prediction_status
    
    if prediction_status["running"]:
        return jsonify({
            "status": "error",
            "message": "Prediction is already running. Please wait."
        }), 429
    
    data = request.get_json() or {}
    selected_model = data.get('model', 'ensemble')  # ensemble, autoencoder, exp_smoothing, linear_regression
    
    # Validate model
    valid_models = ['ensemble', 'autoencoder', 'exp_smoothing', 'linear_regression']
    if selected_model not in valid_models:
        return jsonify({
            "status": "error",
            "message": f"Invalid model. Must be one of: {', '.join(valid_models)}"
        }), 400
    
    def run_script():
        global prediction_status
        prediction_status["running"] = True
        prediction_status["error"] = None
        prediction_status["selected_model"] = selected_model
        
        try:
            result = subprocess.run(
                [
                    sys.executable, 'main.py',
                    '--use-database',
                    '--db-url', 'sqlite:///sales_data.db',
                    '--model', selected_model
                ],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=900,
                text=True,
                capture_output=True
            )
            
            prediction_status["last_run"] = datetime.now().isoformat()
            
            if result.returncode == 0:
                prediction_status["error"] = None
            else:
                prediction_status["error"] = f"Process exited with code {result.returncode}"
                if result.stderr:
                    prediction_status["error"] += f": {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            prediction_status["error"] = "Prediction timeout after 15 minutes"
            prediction_status["last_run"] = datetime.now().isoformat()
        except Exception as e:
            prediction_status["error"] = str(e)
            prediction_status["last_run"] = datetime.now().isoformat()
        finally:
            prediction_status["running"] = False
    
    thread = threading.Thread(target=run_script)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": f"Prediction started with {selected_model} model",
        "model": selected_model
    })

@app.route('/prediction-status', methods=['GET'])
def get_prediction_status():
    """Check prediction status"""
    return jsonify(prediction_status)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Sales Prediction API Server (Ensemble Edition)")
    print("="*60)
    print(f"✅ Backend API: http://localhost:5000")
    print(f"📊 Health Check: http://localhost:5000/health")
    print(f"🏪 Stores List: http://localhost:5000/stores")
    print(f"🤖 Model Comparison: http://localhost:5000/model-comparison")
    print("="*60)
    print("🎯 Available Models:")
    print("   • Ensemble (Recommended)")
    print("   • Autoencoder")
    print("   • Exponential Smoothing")
    print("   • Linear Regression")
    print("="*60)
    print("💡 To access the web interface:")
    print("   1. Open a NEW terminal")
    print("   2. Navigate to frontend: cd frontend")
    print("   3. Run: python start_frontend.py")
    print("   4. Open browser: http://localhost:8000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)