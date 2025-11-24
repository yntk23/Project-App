"""
Frontend Server Starter
สคริปต์สำหรับเริ่ม HTTP server สำหรับ frontend
แสดง URL ที่ถูกต้องให้ผู้ใช้เปิดในเบราว์เซอร์
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler ที่ไม่แสดง log ทุก request"""
    def log_message(self, format, *args):
        # แสดงเฉพาะ request หลัก ไม่แสดง CSS/JS requests
        if not any(x in self.path for x in ['.css', '.js', '.png', '.jpg', '.ico']):
            print(f"📄 {self.command} {self.path}")

def main():
    # ตรวจสอบว่ามีไฟล์ index.html หรือไม่
    if not Path('index.html').exists():
        print("❌ Error: index.html not found in current directory!")
        print("💡 Please navigate to the frontend folder first:")
        print("   cd frontend")
        return
    
    Handler = CustomHTTPRequestHandler
    
    print("\n" + "="*60)
    print("🌐 Sales Prediction Frontend Server")
    print("="*60)
    print(f"✅ Frontend URL: http://localhost:{PORT}")
    print(f"📁 Serving files from: {Path.cwd()}")
    print("="*60)
    print("💡 How to use:")
    print("   1. Make sure Backend API is running (app.py)")
    print("   2. Open browser and go to: http://localhost:8000")
    print("   3. Press Ctrl+C to stop this server")
    print("="*60 + "\n")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 Server started successfully!")
            print(f"🔗 Click here: http://localhost:{PORT}\n")
            
            # เปิดเบราว์เซอร์อัตโนมัติ (optional)
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("🌐 Opening browser automatically...\n")
            except:
                pass
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 Server stopped by user")
        print("="*60 + "\n")
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"\n❌ Error: Port {PORT} is already in use!")
            print("💡 Solutions:")
            print(f"   1. Stop other server using port {PORT}")
            print(f"   2. Or change PORT in this script")
        else:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()