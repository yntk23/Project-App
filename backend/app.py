from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import subprocess
import threading
import sys
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

DATABASE = 'sales_data.db'

# Track running prediction status
prediction_status = {"running": False, "last_run": None, "error": None}

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/predictions', methods=['GET'])
def get_predictions():
    store_id = request.args.get('store_id')
    product_code = request.args.get('product_code')
    
    if not store_id:
        return jsonify({"error": "store_id is required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query based on parameters - เพิ่ม created_at
    if product_code:
        query = """
            SELECT 
                prediction_date as date, 
                product_code, 
                predicted_quantity as predicted_qty,
                created_at
            FROM predictions
            WHERE store_id = ? AND product_code = ?
            ORDER BY prediction_date DESC, rank ASC
        """
        cursor.execute(query, (store_id, product_code))
    else:
        query = """
            SELECT 
                prediction_date as date, 
                product_code, 
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
    
    # Convert to list of dicts
    results = []
    for row in rows:
        results.append({
            "date": row['date'],
            "product_code": row['product_code'],
            "predicted_qty": row['predicted_qty'],
            "created_at": row['created_at']
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

@app.route('/run-prediction', methods=['POST'])
def run_prediction():
    """Trigger prediction script execution"""
    global prediction_status
    
    if prediction_status["running"]:
        return jsonify({
            "status": "error",
            "message": "Prediction is already running. Please wait."
        }), 429
    
    data = request.get_json() or {}
    store_id = data.get('store_id')
    
    def run_script():
        global prediction_status
        prediction_status["running"] = True
        prediction_status["error"] = None
        
        try:
            result = subprocess.run(
                [
                    sys.executable, 'main.py',
                    '--use-database',
                    '--db-url', 'sqlite:///sales_data.db'
                ],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=900,  # 15 minutes timeout
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
        "message": "Prediction process started in background"
    })

@app.route('/prediction-status', methods=['GET'])
def get_prediction_status():
    """Check if prediction is running"""
    return jsonify(prediction_status)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Sales Prediction API Server")
    print("="*60)
    print(f"✅ Backend API: http://localhost:5000")
    print(f"📊 Health Check: http://localhost:5000/health")
    print(f"🏪 Stores List: http://localhost:5000/stores")
    print("="*60)
    print("💡 To access the web interface:")
    print("   1. Open a NEW terminal")
    print("   2. Navigate to frontend folder: cd frontend")
    print("   3. Run: python -m http.server 8000")
    print("   4. Open browser: http://localhost:8000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)