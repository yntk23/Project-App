# 📊 Sales Prediction System - Backend

ระบบพยากรณ์ยอดขายสินค้าด้วย Autoencoder Neural Network

---

## 🚀 วิธีการรัน

### **1. ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

### **2. เริ่มต้น Backend API Server**
```bash
python app.py
```

### **3. เปิด Frontend (Terminal ใหม่)**
```bash
cd ../frontend
python start_frontend.py
```

จากนั้นเปิดเบราว์เซอร์: **http://localhost:8000**

---

**Output ที่คาดหวัง:**
```
============================================================
🚀 Sales Prediction API Server
============================================================
✅ Backend API: http://localhost:5000
📊 Health Check: http://localhost:5000/health
🏪 Stores List: http://localhost:5000/stores
============================================================
💡 To access the web interface:
   1. Open a NEW terminal
   2. Navigate to frontend folder: cd frontend
   3. Run: python -m http.server 8000
   4. Open browser: http://localhost:8000
============================================================
```

## 🔌 API Endpoints

### **1. Health Check**
```bash
GET http://localhost:5000/health
```
ตรวจสอบว่า API Server ทำงานปกติหรือไม่

### **2. รายการร้านค้า**
```bash
GET http://localhost:5000/stores
```
ดึงรายการ Store ID ทั้งหมด

### **3. ดึงข้อมูลพยากรณ์**
```bash
GET http://localhost:5000/predictions?store_id=11001
GET http://localhost:5000/predictions?store_id=11001&product_code=98050138
```

### **4. รันการพยากรณ์**
```bash
POST http://localhost:5000/run-prediction
Content-Type: application/json

{}
```
เริ่มกระบวนการพยากรณ์ (ใช้เวลา 5-7 นาที)

### **5. เช็คสถานะการพยากรณ์**
```bash
GET http://localhost:5000/prediction-status
```
ตรวจสอบว่าการพยากรณ์กำลังทำงานอยู่หรือไม่

---

## 🖥️ วิธีรันผ่าน Terminal (ไม่ใช้ API)

### **รันด้วย CSV File**
```bash
python main.py --data-path data/sales_data.csv
```

### **รันด้วย Database**
```bash
python main.py --use-database --db-url sqlite:///sales_data.db
```

## 📤 Output Files

ผลลัพธ์การพยากรณ์ถูกเก็บใน:
```
output/
└── 2025-02-09/
    └── next_day_top5_20250209.csv
```

**รูปแบบไฟล์ CSV:**
```csv
Store_ID,Prediction_Date,Rank,Product_Code,Predicted_Quantity
11001,2025-02-09,1,98050138,150
11001,2025-02-09,2,98050142,120
11001,2025-02-09,3,98050156,95
...
```

---

## ⚙️ การตั้งค่า

แก้ไขได้ที่: `src/config.py`

```python
class PredictionConfig:
    moving_average_window = 7    # จำนวนวันย้อนหลังสำหรับพยากรณ์
    encoding_dim = 16            # Latent dimension ของ Autoencoder
    top_n_products = 5           # จำนวนสินค้าที่แนะนำต่อร้าน
```

---

## 📚 เทคโนโลยีที่ใช้

- **Flask** - Web Framework สำหรับ API
- **TensorFlow/Keras** - Deep Learning Framework
- **SQLite** - Database
- **Pandas** - Data Processing
- **NumPy** - Numerical Computing

---