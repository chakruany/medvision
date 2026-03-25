import time  # <--- [เพิ่ม] ไลบรารีสำหรับจับเวลา
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QThread, Signal
from gui.view import MedVisionView
from gui.model import MedVisionModel
from PySide6.QtGui import QPixmap

class AIWorker(QThread):
    finished = Signal(dict)
    
    def __init__(self, model, img_path, human_text):
        super().__init__()
        self.model = model
        self.img_path = img_path
        self.human_text = human_text
        
    def run(self):
        # รันการวิเคราะห์ใน Background Thread
        response = self.model.predict(self.img_path, self.human_text)
        self.finished.emit(response)

class MedVisionController:
    def __init__(self, model: MedVisionModel, view: MedVisionView):
        self.model = model
        self.view = view
        self.current_img = None
        self.start_time = 0.0  # <--- [เพิ่ม] ตัวแปรเก็บเวลาเริ่มต้น
        
        # เชื่อมต่อปุ่มต่างๆ เข้ากับฟังก์ชัน
        self.view.btn_upload.clicked.connect(self.upload_image)
        self.view.btn_analyze.clicked.connect(self.start_analysis)
        self.view.btn_clear.clicked.connect(self.clear_data)
        
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.view, "เลือกภาพเม็ดยา", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_img = file_path
            self.view.img_preview.setPixmap(QPixmap(file_path))
            self.view.btn_analyze.setEnabled(True)
            self.view.reset_results()
            self.view.status_label.setText("🟢 สถานะ: โหลดภาพสำเร็จ พร้อมวิเคราะห์")
            
    def start_analysis(self):
        if not self.current_img: return
        
        human_text = self.view.input_imprint.text()
        
        # ล็อกปุ่มต่างๆ ระหว่าง AI ทำงาน
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_upload.setEnabled(False)
        self.view.btn_clear.setEnabled(False)
        self.view.btn_analyze.setText("⏳ กำลังประมวลผล...") 
        self.view.status_label.setText("⏳ สถานะ: AI กำลังประมวลผล (กรุณารอสักครู่)...")
        self.view.reset_results()
        
        # ==========================================
        # ⏱️ [เริ่มจับเวลา]: ทันทีที่กดปุ่มวิเคราะห์
        # ==========================================
        self.start_time = time.time()
        
        self.worker = AIWorker(self.model, self.current_img, human_text)
        self.worker.finished.connect(self.display_results)
        self.worker.start()
        
    def display_results(self, response):
        # ==========================================
        # ⏱️ [หยุดจับเวลา]: ทันทีที่ AI ส่งผลลัพธ์กลับมา
        # ==========================================
        inference_time = time.time() - self.start_time
        
        # ปลดล็อกปุ่ม
        self.view.btn_analyze.setEnabled(True)
        self.view.btn_upload.setEnabled(True)
        self.view.btn_clear.setEnabled(True)
        self.view.btn_analyze.setText("🔍 3. เริ่มวิเคราะห์ (Analyze)") 
        
        if "error" in response:
            self.view.status_label.setText(f"❌ เกิดข้อผิดพลาด: {response['error']} (ใช้เวลา {inference_time:.2f} วินาที)")
            return
            
        # [อัปเดต]: แสดงเวลาที่ใช้ไป บนหน้าจอสถานะ
        self.view.status_label.setText(
            f"✅ วิเคราะห์เสร็จสิ้น (AI ตรวจพบอักษร: '{response['extracted_text']}') "
            f"| ⏱️ ใช้เวลาประมวลผล: {inference_time:.2f} วินาที"
        )
        
        results = response["results"]
        for i, res in enumerate(results):
            if i < len(self.view.result_widgets):
                card = self.view.result_widgets[i]
                
                # แสดงรูปด้านหน้าและหลัง
                if res['front_img']: card['img_front'].setPixmap(QPixmap(res['front_img']))
                if res['back_img']: card['img_back'].setPixmap(QPixmap(res['back_img']))
                
                card['title'].setText(f"{'⭐ ' if i==0 else ''}{res['trade_name']}")
                
                generic = res.get('generic_name', 'UNKNOWN').replace("-", " ")
                company = res.get('company', 'UNKNOWN').replace("-", " ")
                card['subtitle'].setText(f"ชื่อสามัญ: {generic} | บริษัท: {company}")
                
                card['score'].setText(f"ความมั่นใจรวม (Fusion Score): {res['fusion_score']}%")
                card['detail'].setText(f"รหัสยา: {res['drug_id']} | รูปร่าง: {res['visual_score']}% | สี: {res['color_score']}% | อักษร: {res['text_score']}%")

    def clear_data(self):
        self.current_img = None
        self.view.clear_all_inputs()
        self.view.status_label.setText("🟢 สถานะ: พร้อมใช้งาน (ล้างข้อมูลเรียบร้อย)")