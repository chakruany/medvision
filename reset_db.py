import os
import shutil
from pathlib import Path

def clean_old_database():
    print("=== MedVision Database Cleaner ===")
    
    # 1. ไฟล์และโฟลเดอร์เป้าหมายที่ต้องกำจัด
    targets = [
        "data/reference",            # โฟลเดอร์เก็บรูปที่ลบฉากหลังแล้ว
        "data/reference_cache.pkl",  # ไฟล์สมอง AI (Vector Cache)
        "data/reference/metadata.csv"        # ไฟล์ฐานข้อมูลตัวอักษร
    ]
    
    for target in targets:
        path = Path(target)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)  # ลบโฟลเดอร์และไส้ในทั้งหมดทิ้ง
                    print(f"[*] ลบโฟลเดอร์สำเร็จ: {target}")
                else:
                    path.unlink()        # ลบไฟล์ทิ้ง
                    print(f"[*] ลบไฟล์แคชสำเร็จ: {target}")
            except Exception as e:
                print(f"[!] ไม่สามารถลบ {target} ได้: {e}")
        else:
            print(f"[ ] ไม่พบเป้าหมาย (สะอาดอยู่แล้ว): {target}")

    # 2. สร้างโฟลเดอร์เปล่าๆ มารองรับการรันรอบใหม่
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/raw_reference", exist_ok=True)
    
    print("-" * 35)
    print("✨ คลีนระบบเสร็จสิ้น! พื้นที่พร้อมสำหรับการสร้างฐานข้อมูลใหม่แล้ว")
    print("👉 ขั้นตอนต่อไป: นำรูปใหม่ใส่ raw_reference แล้วรัน python auto_bg_remover.py")

if __name__ == "__main__":
    clean_old_database()