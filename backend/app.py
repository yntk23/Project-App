from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

DATABASE = 'sales_data.db'

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
    
    # Build query based on parameters
    if product_code:
        query = """
            SELECT prediction_date as date, product_code, predicted_quantity as predicted_qty
            FROM predictions
            WHERE store_id = ? AND product_code = ?
            ORDER BY prediction_date DESC, rank ASC
        """
        cursor.execute(query, (store_id, product_code))
    else:
        query = """
            SELECT prediction_date as date, product_code, predicted_quantity as predicted_qty
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
            "predicted_qty": row['predicted_qty']
        })
    
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)