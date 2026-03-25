import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal
from gui.model import MedVisionModel
from gui.view import MedVisionView
from gui.controller import MedVisionController

class BootWorker(QThread):
    finished = Signal(object)
    def run(self):
        model = MedVisionModel() 
        self.finished.emit(model)

class AppManager:
    def __init__(self):
        self.view = MedVisionView()
        self.view.show()
        
        # [THE MAGIC]: เปลี่ยนข้อความบนปุ่มให้ผู้ใช้รู้ว่าต้องรอ
        self.view.btn_upload.setEnabled(False)
        self.view.btn_upload.setText("⏳ กำลังโหลดน้ำหนัก AI...")
        
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_analyze.setText("⏳ กำลังเตรียมฐานข้อมูลเวกเตอร์...")
        
        self.view.status_label.setText("⏳ สถานะ: กำลังเตรียมระบบ (อาจใช้เวลาหลายนาทีในครั้งแรก)...")
        
        self.boot_worker = BootWorker()
        self.boot_worker.finished.connect(self.on_boot_complete)
        self.boot_worker.start()

    def on_boot_complete(self, model):
        self.controller = MedVisionController(model, self.view)
        
        # คืนค่าข้อความบนปุ่มกลับเป็นปกติ
        self.view.btn_upload.setEnabled(True)
        self.view.btn_upload.setText("📸 1. เลือกรูปภาพ (Upload Image)")
        self.view.btn_analyze.setText("🔍 3. เริ่มวิเคราะห์ (Analyze)")
        
        self.view.status_label.setText("🟢 สถานะ: ระบบ AI พร้อมใช้งาน 100%!")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # ทำให้หน้าตาแอปดูทันสมัยขึ้นใน Windows
    manager = AppManager()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()