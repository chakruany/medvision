import logging
import re
from itertools import product
from typing import List, Dict, Any, Tuple

import cv2
import easyocr
import numpy as np
from thefuzz import fuzz

try:
    import torch
except Exception:
    torch = None

logging.getLogger("easyocr").setLevel(logging.ERROR)


class PillTextRecognizer:
    """
    OCR engine สำหรับ imprint บนเม็ดยา

    คุณสมบัติ:
    - Auto GPU/CPU
    - Multi-preprocessing OCR
    - Rotation TTA
    - Candidate aggregation
    - Confusion-aware text matching
    - Weak-text suppression แบบไม่โหดเกินไป
    """

    CONFUSION_MAP = {
        "0": ["0", "O", "D"],
        "O": ["O", "0", "D"],
        "D": ["D", "0", "O"],

        "1": ["1", "I", "L", "T"],
        "I": ["I", "1", "L", "T"],
        "L": ["L", "1", "I"],
        "T": ["T", "1", "I"],

        "5": ["5", "S"],
        "S": ["S", "5"],

        "2": ["2", "Z"],
        "Z": ["Z", "2"],

        "8": ["8", "B"],
        "B": ["B", "8"],

        "6": ["6", "G"],
        "G": ["G", "6"],

        "M": ["M", "N", "W"],
        "N": ["N", "M"],
        "W": ["W", "M"],

        "C": ["C", "G", "O"],
        "P": ["P", "R"],
        "R": ["R", "P"],
    }

    WEAK_TEXTS = {
        "", "0", "00", "000", "0000",
        "O", "OO", "OOO",
        "1", "11", "111",
        "I", "II", "III",
        "L", "LL",
        "TO", "OT", "OS", "SO", "10", "01", "IO", "OI", "LO", "OL",
        "C", "D", "G"
    }

    IGNORE_TARGETS = {"UNKNOWN", "NONE", "LOGO", "SCORE", "CROSS", ""}

    def __init__(self):
        use_gpu = False
        if torch is not None:
            try:
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False

        print(f"[*] Initializing EasyOCR Engine (Pill OCR Matcher) on: {'cuda' if use_gpu else 'cpu'}")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    # =========================================================
    # Image Enhancement
    # =========================================================
    def normalize_text(self, text: str) -> str:
        text = (text or "").upper().strip()
        text = re.sub(r"[\s\-_./\\|]+", "", text)
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def is_weak_text(self, text: str) -> bool:
        t = self.normalize_text(text)
        return (t in self.WEAK_TEXTS) or (len(t) <= 1)

    def confidence_weight(self, text: str) -> float:
        """
        ลดน้ำหนักข้อความสั้น แต่ไม่กดแรงเกินไป
        """
        t = self.normalize_text(text)

        if t in self.WEAK_TEXTS:
            return 0.0

        if len(t) <= 1:
            return 0.0
        if len(t) == 2:
            return 0.72
        if len(t) == 3:
            return 0.86
        if len(t) == 4:
            return 0.95
        return 1.0

    def canonicalize_for_confusion(self, text: str) -> str:
        """
        ทำ canonical form เพื่อช่วย matching ในกลุ่มตัวที่ OCR สับสน
        """
        t = self.normalize_text(text)
        replacements = {
            "O": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "T": "1",
            "S": "5",
            "Z": "2",
            "B": "8",
            "G": "6",
        }
        return "".join(replacements.get(ch, ch) for ch in t)

    def enhance_image_for_ocr(self, img_path: str) -> Dict[str, np.ndarray]:
        """
        คืน preprocessing หลายแบบ เพื่อให้ OCR มีโอกาสอ่าน imprint ได้มากขึ้น
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image path: {img_path}")

        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            bgr = img[:, :, :3]

            # วางบนพื้นขาวเพื่อให้ OCR อ่านร่องสลักได้ดีขึ้น
            white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
            alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
            bgr = (bgr.astype(np.float32) * alpha_f + white_bg.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
        elif len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img[:, :, :3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        denoised = cv2.bilateralFilter(clahe_img, 9, 60, 60)

        sharpen_kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

        # adaptive threshold
        th_adaptive = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )

        # otsu
        _, th_otsu = cv2.threshold(
            sharpened, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # inverse otsu
        _, th_inv = cv2.threshold(
            sharpened, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # morphology แบบเบา ๆ เพื่อช่วยตัวอักษรสลัก
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(th_adaptive, cv2.MORPH_CLOSE, kernel)

        return {
            "gray": gray,
            "clahe": clahe_img,
            "sharpened": sharpened,
            "th_adaptive": th_adaptive,
            "th_otsu": th_otsu,
            "th_inv": th_inv,
            "morph": morph,
        }

    # =========================================================
    # Candidate Generation
    # =========================================================
    def generate_variants(self, text: str, max_variants: int = 48) -> List[str]:
        text = self.normalize_text(text)
        if not text:
            return [""]

        pools = [self.CONFUSION_MAP.get(ch, [ch]) for ch in text]

        variants = []
        for combo in product(*pools):
            variants.append("".join(combo))
            if len(variants) >= max_variants:
                break

        variants.append(self.canonicalize_for_confusion(text))
        return list(set(v for v in variants if v))

    def subsequence_score(self, query: str, target: str) -> float:
        query = self.normalize_text(query)
        target = self.normalize_text(target)

        if not query or not target:
            return 0.0

        i = 0
        for ch in target:
            if i < len(query) and query[i] == ch:
                i += 1

        return i / max(len(target), 1)

    def containment_score(self, query: str, target: str) -> float:
        """
        ถ้า query อยู่ใน target หรือ target อยู่ใน query
        """
        q = self.normalize_text(query)
        t = self.normalize_text(target)

        if not q or not t:
            return 0.0

        if q in t or t in q:
            return min(len(q), len(t)) / max(len(q), len(t))

        return 0.0

    def build_ocr_candidates(self, text_lines: List[str]) -> List[str]:
        cleaned = [self.normalize_text(x) for x in text_lines if self.normalize_text(x)]
        candidates = set(cleaned)

        if cleaned:
            candidates.add("".join(cleaned))
            candidates.add("".join(reversed(cleaned)))

        if len(cleaned) >= 2:
            for i in range(len(cleaned)):
                for j in range(i + 1, len(cleaned)):
                    candidates.add(cleaned[i] + cleaned[j])
                    candidates.add(cleaned[j] + cleaned[i])

        return [c for c in candidates if c]

    # =========================================================
    # OCR Read
    # =========================================================
    def _score_candidate_quality(self, candidate: str, avg_conf: float) -> float:
        """
        ใช้เลือกว่า candidate ไหนจาก OCR ควรเชื่อที่สุด
        """
        c = self.normalize_text(candidate)
        if not c:
            return 0.0

        length_bonus = min(len(c), 6) * 0.08
        weak_penalty = 0.25 if self.is_weak_text(c) else 0.0

        mixed_bonus = 0.0
        if any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
            mixed_bonus = 0.08

        repeated_penalty = 0.0
        if len(set(c)) == 1 and len(c) >= 2:
            repeated_penalty = 0.20

        score = (
            (avg_conf * 0.62)
            + length_bonus
            + mixed_bonus
            - weak_penalty
            - repeated_penalty
        )
        return float(score)

    def _read_single_image(self, img: np.ndarray) -> List[Tuple[str, float]]:
        """
        คืน [(text, conf), ...]
        """
        try:
            result = self.reader.readtext(img, detail=1)
        except Exception:
            return []

        parsed = []
        for _, text, conf in result:
            clean = self.normalize_text(text)
            if clean:
                parsed.append((clean, float(conf)))
        return parsed

    def extract_text_details(self, img_path: str) -> Dict[str, Any]:
        """
        คืน:
        {
            "text": ข้อความหลัก,
            "lines": รายการข้อความที่ OCR เจอ,
            "candidates": ตัวเลือกข้อความรวม,
            "confidence": confidence เฉลี่ย,
            "rotation": มุมที่ดีที่สุด,
        }
        """
        try:
            processed_versions = self.enhance_image_for_ocr(img_path)

            best_result = {
                "text": "",
                "lines": [],
                "candidates": [],
                "confidence": 0.0,
                "rotation": "0",
                "preprocess": "",
            }

            rotations = [
                ("0", None),
                ("90", cv2.ROTATE_90_CLOCKWISE),
                ("180", cv2.ROTATE_180),
                ("270", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]

            for prep_name, base_img in processed_versions.items():
                for rotation_name, rotate_code in rotations:
                    if rotate_code is None:
                        rotated_img = base_img
                    else:
                        rotated_img = cv2.rotate(base_img, rotate_code)

                    parsed = self._read_single_image(rotated_img)
                    if not parsed:
                        continue

                    lines = [t for t, _ in parsed]
                    confs = [c for _, c in parsed]

                    avg_conf = float(sum(confs) / max(len(confs), 1))
                    candidates = self.build_ocr_candidates(lines)

                    if not candidates:
                        continue

                    best_candidate_this_rotation = max(
                        candidates,
                        key=lambda x: self._score_candidate_quality(x, avg_conf)
                    )

                    score_key = self._score_candidate_quality(best_candidate_this_rotation, avg_conf)
                    current_best_key = self._score_candidate_quality(best_result["text"], best_result["confidence"])

                    if score_key > current_best_key:
                        best_result = {
                            "text": best_candidate_this_rotation,
                            "lines": lines,
                            "candidates": candidates,
                            "confidence": avg_conf,
                            "rotation": rotation_name,
                            "preprocess": prep_name,
                        }

                    # early stop: ถ้าอ่านดีมากแล้วหยุดได้
                    if avg_conf >= 0.82 and len(best_candidate_this_rotation) >= 3 and not self.is_weak_text(best_candidate_this_rotation):
                        return best_result

            return best_result

        except Exception as e:
            print(f"  [-] OCR Error: {e}")
            return {
                "text": "",
                "lines": [],
                "candidates": [],
                "confidence": 0.0,
                "rotation": "0",
                "preprocess": "",
            }

    def extract_text(self, img_path: str) -> str:
        details = self.extract_text_details(img_path)
        return details.get("text", "")

    # =========================================================
    # Similarity Scoring
    # =========================================================
    def score_pair(self, query_text: str, target_text: str) -> float:
        """
        ให้คะแนนระหว่าง query กับ target หนึ่งคู่ แบบ confusion-aware
        return 0.0 - 1.0
        """
        query = self.normalize_text(query_text)
        target = self.normalize_text(target_text)

        if not query or not target:
            return 0.0

        if query == target:
            return 1.0

        query_canon = self.canonicalize_for_confusion(query)
        target_canon = self.canonicalize_for_confusion(target)

        best_score = 0.0
        variants = self.generate_variants(query, max_variants=48)
        variants.append(query_canon)

        for v in set(variants):
            exact = 100.0 if v == target else 0.0
            exact_canon = 100.0 if self.canonicalize_for_confusion(v) == target_canon else 0.0

            ratio = fuzz.ratio(v, target)
            partial = fuzz.partial_ratio(v, target)
            token_set = fuzz.token_set_ratio(v, target)
            subseq = self.subsequence_score(v, target) * 100.0
            contain = self.containment_score(v, target) * 100.0

            score = max(
                exact,
                exact_canon,
                0.34 * ratio +
                0.24 * partial +
                0.10 * token_set +
                0.16 * subseq +
                0.16 * contain
            )

            if score > best_score:
                best_score = score

        # bonus สำหรับความยาวใกล้กัน
        len_gap = abs(len(query) - len(target))
        if len_gap == 0:
            best_score += 4.0
        elif len_gap == 1:
            best_score += 2.0

        # penalty ถ้าต่างกันเยอะและข้อความสั้น
        if len(query) <= 2 and len(target) >= 4:
            best_score -= 8.0

        best_score *= self.confidence_weight(query)

        return max(0.0, min(best_score / 100.0, 1.0))

    def calculate_text_similarity(self, query_text: str, target_imprints: List[str]) -> float:
        query = self.normalize_text(query_text)
        if not query:
            return 0.0

        max_score = 0.0

        for target in target_imprints:
            target_clean = self.normalize_text(target)
            if target_clean in self.IGNORE_TARGETS:
                continue

            score = self.score_pair(query, target_clean)
            if score > max_score:
                max_score = score

        return max_score

    def calculate_text_similarity_from_metadata(self, query_text: str, metadata: Dict[str, Any]) -> float:
        """
        ให้น้ำหนัก imprint มากที่สุด
        """
        query = self.normalize_text(query_text)
        if not query:
            return 0.0

        imprint = self.normalize_text(metadata.get("imprint", ""))
        trade_name = self.normalize_text(metadata.get("trade_name", ""))
        generic_name = self.normalize_text(metadata.get("generic_name", ""))

        scores = []

        if imprint and imprint not in self.IGNORE_TARGETS:
            s = self.score_pair(query, imprint)

            # bonus ถ้า canonical form ตรงกัน
            if self.canonicalize_for_confusion(query) == self.canonicalize_for_confusion(imprint):
                s = min(1.0, s + 0.08)

            scores.append(s * 1.18)

        if trade_name and trade_name not in self.IGNORE_TARGETS:
            scores.append(self.score_pair(query, trade_name) * 0.82)

        if generic_name and generic_name not in self.IGNORE_TARGETS:
            scores.append(self.score_pair(query, generic_name) * 0.62)

        if not scores:
            return 0.0

        return max(0.0, min(max(scores), 1.0))