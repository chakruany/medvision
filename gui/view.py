from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QGroupBox, QFrame, QDialog)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt, Signal

# =========================================================
# คลาสพิเศษสำหรับทำให้รูปภาพสามารถคลิกได้
# =========================================================
class ClickableLabel(QLabel):
    clicked = Signal(QPixmap)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() and not self.pixmap().isNull():
            self.clicked.emit(self.pixmap())
        super().mousePressEvent(event)

# =========================================================
# คลาสหลักของหน้าจอ
# =========================================================
class MedVisionView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedVision: AI Pill Screening System")
        self.resize(1150, 850)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #F4F9F4; }
            QGroupBox { font-size: 16px; font-weight: bold; color: #1E8449; border: 2px solid #A9DFBF; border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }
            QLineEdit { 
                border: 2px solid #D5DBDB; 
                border-radius: 6px; 
                padding: 8px; 
                font-size: 14px; 
                background-color: white; 
                color: black;
            }
            QLineEdit:focus { 
                border: 2px solid #2ECC71; 
                color: black;
            }
        """)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- แผงควบคุมด้านซ้าย ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        self.img_preview = QLabel("📸\nคลิกเพื่ออัปโหลดภาพเม็ดยา")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setStyleSheet("""
            border: 3px dashed #7DCEA0; border-radius: 12px; 
            background-color: white; color: #555; font-size: 16px; font-weight: bold;
        """)
        self.img_preview.setFixedSize(400, 400)
        self.img_preview.setScaledContents(True)
        
        self.btn_upload = QPushButton("📸 1. เลือกรูปภาพ (Upload Image)")
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.setStyleSheet("""
            QPushButton { background-color: #27AE60; color: white; font-weight: bold; font-size: 16px; border-radius: 8px; }
            QPushButton:hover { background-color: #2ECC71; }
            QPushButton:disabled { background-color: #A9DFBF; color: #ECF0F1; }
        """)
        
        self.input_imprint = QLineEdit()
        self.input_imprint.setPlaceholderText("⌨️ 2. (ตัวเลือกเสริม) พิมพ์ตัวอักษรบนยาเพื่อช่วย AI ...")
        
        self.btn_analyze = QPushButton("🔍 3. เริ่มวิเคราะห์ (Analyze)")
        self.btn_analyze.setMinimumHeight(60)
        self.btn_analyze.setStyleSheet("""
            QPushButton { background-color: #1E8449; color: white; font-weight: bold; font-size: 18px; border-radius: 8px; }
            QPushButton:hover { background-color: #239B56; }
            QPushButton:disabled { background-color: #A9DFBF; color: #ECF0F1; }
        """)
        self.btn_analyze.setEnabled(False)
        
        # --- [ใหม่] ปุ่มล้างข้อมูล ---
        self.btn_clear = QPushButton("🗑️ ล้างข้อมูลเริ่มใหม่ (Clear All)")
        self.btn_clear.setMinimumHeight(45)
        self.btn_clear.setStyleSheet("""
            QPushButton { background-color: #BDC3C7; color: #2C3E50; font-weight: bold; font-size: 15px; border-radius: 8px; }
            QPushButton:hover { background-color: #A6ACAF; }
        """)
        
        self.status_label = QLabel("🟢 สถานะ: รอระบบเตรียมความพร้อม...")
        self.status_label.setStyleSheet("color: #27AE60; font-weight: bold; font-size: 14px; padding-top: 10px;")
        
        left_panel.addWidget(self.img_preview)
        left_panel.addWidget(self.btn_upload)
        left_panel.addWidget(self.input_imprint)
        left_panel.addWidget(self.btn_analyze)
        left_panel.addWidget(self.btn_clear) # แทรกปุ่มตรงนี้
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()
        
        # --- แผงแสดงผลด้านขวา (Result Area) ---
        right_panel = QVBoxLayout()
        self.group_box = QGroupBox("📋 ผลลัพธ์การคัดกรองจากฐานข้อมูล (Top-3 Predictions)")
        self.result_layout = QVBoxLayout()
        self.group_box.setLayout(self.result_layout)
        
        self.result_widgets = []
        for i in range(3):
            card = self.create_result_card(i + 1)
            self.result_layout.addWidget(card['widget'])
            self.result_widgets.append(card)
            
        right_panel.addWidget(self.group_box)
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)

    def create_result_card(self, rank):
        card_widget = QFrame()
        card_widget.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #D5DBDB; border-radius: 10px; }
            QFrame:hover { border: 2px solid #2ECC71; }
        """)
        layout = QHBoxLayout(card_widget)
        
        # โซนรูปภาพ 2 ด้าน
        images_layout = QVBoxLayout()
        
        img_front = ClickableLabel("ไม่มีรูป")
        img_front.setFixedSize(100, 100)
        img_front.setStyleSheet("border: 1px solid #eee; border-radius: 6px; background-color: #fafafa;")
        img_front.setAlignment(Qt.AlignCenter)
        img_front.setScaledContents(True)
        img_front.setCursor(Qt.PointingHandCursor)
        img_front.clicked.connect(self.show_image_popup)
        lbl_front = QLabel("FRONT")
        lbl_front.setStyleSheet("color: #888; font-size: 10px; font-weight: bold; border: none;")
        lbl_front.setAlignment(Qt.AlignCenter)
        
        img_back = ClickableLabel("ไม่มีรูป")
        img_back.setFixedSize(100, 100)
        img_back.setStyleSheet("border: 1px solid #eee; border-radius: 6px; background-color: #fafafa;")
        img_back.setAlignment(Qt.AlignCenter)
        img_back.setScaledContents(True)
        img_back.setCursor(Qt.PointingHandCursor)
        img_back.clicked.connect(self.show_image_popup)
        lbl_back = QLabel("BACK")
        lbl_back.setStyleSheet("color: #888; font-size: 10px; font-weight: bold; border: none;")
        lbl_back.setAlignment(Qt.AlignCenter)
        
        front_box = QVBoxLayout(); front_box.addWidget(img_front); front_box.addWidget(lbl_front)
        back_box = QVBoxLayout(); back_box.addWidget(img_back); back_box.addWidget(lbl_back)
        
        img_pair_layout = QHBoxLayout()
        img_pair_layout.addLayout(front_box)
        img_pair_layout.addLayout(back_box)
        images_layout.addLayout(img_pair_layout)
        
        # โซนข้อมูลตัวหนังสือ
        info_layout = QVBoxLayout()
        title_label = QLabel(f"อันดับ {rank}: -")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #1E8449; border: none;")
        
        subtitle_label = QLabel("ชื่อสามัญ: - | บริษัท: -")
        subtitle_label.setStyleSheet("color: #2980B9; font-size: 13px; font-weight: bold; border: none;")
        
        score_label = QLabel("ความมั่นใจ: -")
        score_label.setStyleSheet("color: #E67E22; font-weight: bold; font-size: 14px; border: none;")
        
        detail_label = QLabel("Vision: - | Color: - | OCR: -")
        detail_label.setStyleSheet("color: #7F8C8D; font-size: 13px; border: none;")
        
        info_layout.addWidget(title_label)
        info_layout.addWidget(subtitle_label)
        info_layout.addWidget(score_label)
        info_layout.addWidget(detail_label)
        info_layout.addStretch()
        
        layout.addLayout(images_layout)
        layout.addSpacing(15)
        layout.addLayout(info_layout)
        layout.setStretch(1, 1)
        
        return {'widget': card_widget, 'img_front': img_front, 'img_back': img_back, 
                'title': title_label, 'subtitle': subtitle_label, 'score': score_label, 'detail': detail_label}

    def reset_results(self):
        for card in self.result_widgets:
            card['img_front'].clear(); card['img_front'].setText("ไม่มีรูป")
            card['img_back'].clear(); card['img_back'].setText("ไม่มีรูป")
            card['title'].setText(f"รอการวิเคราะห์...")
            card['subtitle'].setText("ชื่อสามัญ: - | บริษัท: -")
            card['score'].setText("ความมั่นใจ: -")
            card['detail'].setText("Vision: - | Color: - | OCR: -")

    # --- [ใหม่] ฟังก์ชันกวาดล้างข้อมูลบนหน้าจอทั้งหมด ---
    def clear_all_inputs(self):
        self.img_preview.clear()
        self.img_preview.setText("📸\nคลิกเพื่ออัปโหลดภาพเม็ดยา")
        self.input_imprint.clear()
        self.btn_analyze.setEnabled(False)
        self.reset_results()

    def show_image_popup(self, pixmap):
        dialog = QDialog(self)
        dialog.setWindowTitle("ภาพขยายเม็ดยา (Enlarged View)")
        dialog.setFixedSize(550, 550)
        layout = QVBoxLayout(dialog)
        lbl = QLabel()
        lbl.setPixmap(pixmap.scaled(520, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        dialog.exec()