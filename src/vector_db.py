# โมดูลจัดการฐานข้อมูล FAISS และการค้นหา

import faiss
import numpy as np
import os

class PillVectorDatabase:
    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        # ใช้ IndexFlatIP (Inner Product) ซึ่งเมื่อใช้กับ Normalized Vector จะมีค่าเท่ากับ Cosine Similarity
        self.index = faiss.IndexFlatIP(embedding_dim) 
        self.metadata = [] # เก็บชื่อไฟล์/ชื่อยาที่ตรงกับ Index
        
    def add_reference_images(self, features_list: list, labels_list: list):
        """เพิ่มข้อมูล Vector ของยาตั้งต้นเข้าฐานข้อมูล"""
        if not features_list:
            return
            
        # แปลงเป็น Numpy array รูปแบบ float32 ตามที่ FAISS ต้องการ
        embeddings = np.array(features_list).astype('float32')
        self.index.add(embeddings)
        self.metadata.extend(labels_list)
        print(f"[*] Successfully added {len(labels_list)} pills to the database.")

    def search(self, query_feature: np.ndarray, top_k: int = 3):
        """ค้นหายาที่คล้ายคลึงที่สุด"""
        query_vector = np.array([query_feature]).astype('float32')
        # D คือระยะห่าง (Similarity Score), I คือ Index ที่เจอ
        D, I = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1: # ถ้าเจอข้อมูล
                results.append({
                    "pill_name": self.metadata[idx],
                    "similarity_score": round(float(score) * 100, 2) # แปลงเป็นเปอร์เซ็นต์
                })
        return results