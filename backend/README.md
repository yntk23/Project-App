# ระบบพยากรณ์ยอดขายด้วย Autoencoder
ระบบพยากรณ์ปริมาณสินค้าสำหรับวันถัดไปโดยใช้โมเดล Autoencoder และฐานข้อมูล SQLite/PostgreSQL

# คุณสมบัติ
ใช้ Autoencoder เรียนรู้รูปแบบยอดขายย้อนหลัง 7 วัน
รองรับการเก็บผลลัพธ์ทั้งใน Database และ CSV
Batch prediction สำหรับหลายร้านค้า
Logging รายละเอียดการทำงาน

# การติดตั้ง
git clone <repo-url>
cd <project-directory>
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
mkdir -p data logs output

# การใช้งาน
Import ข้อมูล
python import_data_to_db.py --csv data/sales.csv --db sqlite:///sales_data.db

# รันพยากรณ์
python main.py --use-database --db-url sqlite:///sales_data.db
# หรือ CSV
python main.py --data-path data/sales.csv

# ตรวจสอบผลลัพธ์
from src.database.db_manager import DatabaseManager
db = DatabaseManager('sqlite:///sales_data.db')
predictions = db.get_latest_predictions()
print(predictions.head())

# โครงสร้างโปรเจค
project/
├── data/        # CSV ข้อมูลต้นฉบับ
├── logs/        # Log รายวัน
├── output/      # ผลลัพธ์ CSV
├── src/         # Source code
├── main.py
├── import_data_to_db.py
├── requirements.txt
└── sales_data.db

# การกำหนดค่า

แก้ไข src/config.py สำหรับ:

moving_average_window : จำนวนวันย้อนหลัง

top_n_products : จำนวนสินค้าที่แนะนำ

encoding_dim : ขนาด latent vector

epochs : จำนวนรอบฝึก

# ผลลัพธ์

Prediction date: ล่าสุดหลังจากข้อมูล BSNS_DT

บันทึกในตาราง predictions และ model_metadata

CSV ใน output/YYYY-MM-DD/next_day_top5_YYYYMMDD.csv