import os
import cv2
import numpy as np


SPECIAL_IMPRINT_VALUES = {"UNKNOWN", "NONE", "LOGO", "SCORE", "CROSS", ""}


def parse_filename_metadata(filename: str) -> dict:
    """
    รองรับรูปแบบชื่อไฟล์:
    [DrugID]_[Generic]_[Trade]_[Company]_[Imprint]_[Side]
    หรือ
    [DrugID]_[Generic]_[Trade]_[Company]_[Side]
    """
    name_without_ext = os.path.splitext(os.path.basename(filename))[0]
    parts = name_without_ext.split("_")

    if len(parts) >= 6:
        return {
            "drug_id": parts[0].strip() or "UNKNOWN",
            "generic_name": "_".join(parts[1:-4]).strip() or "UNKNOWN",
            "trade_name": parts[-4].strip() or "UNKNOWN",
            "company": parts[-3].strip() or "UNKNOWN",
            "imprint": parts[-2].strip() or "UNKNOWN",
            "side": parts[-1].strip().upper() or "UNKNOWN",
        }

    if len(parts) == 5:
        return {
            "drug_id": parts[0].strip() or "UNKNOWN",
            "generic_name": "_".join(parts[1:-3]).strip() or "UNKNOWN",
            "trade_name": parts[-3].strip() or "UNKNOWN",
            "company": parts[-2].strip() or "UNKNOWN",
            "imprint": "NONE",
            "side": parts[-1].strip().upper() or "UNKNOWN",
        }

    return {
        "drug_id": "UNKNOWN",
        "generic_name": "UNKNOWN",
        "trade_name": "UNKNOWN",
        "company": "UNKNOWN",
        "imprint": "UNKNOWN",
        "side": "UNKNOWN",
    }


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def get_bgr_and_mask(img: np.ndarray):
    """
    คืนค่า:
    - bgr: ภาพ 3 channels
    - mask: mask ของบริเวณเม็ดยา

    รองรับทั้ง RGBA, BGR และ grayscale
    """
    if img is None:
        raise ValueError("Image is None")

    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255

    elif len(img.shape) == 3 and img.shape[2] == 3:
        bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    elif len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    nonzero = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    ratio = nonzero / total if total > 0 else 0.0

    if ratio > 0.05:
        kernel_erode = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel_erode, iterations=1)
        if cv2.countNonZero(eroded) > 0:
            mask = eroded

    return bgr, mask


def safe_mean_std_color(bgr: np.ndarray, mask: np.ndarray):
    pixels = bgr[mask > 0]
    if len(pixels) == 0:
        return (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    pixels = pixels.astype(np.float32)
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)
    return mean, std


def simple_gray_world_white_balance(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    White balance เฉพาะบริเวณเม็ดยา
    ปรับแบบเบาลง เพื่อให้ทน brightness shift ดีขึ้น
    """
    img = bgr.astype(np.float32)

    if cv2.countNonZero(mask) == 0:
        return ensure_uint8(bgr)

    pixels = img[mask > 0]
    if len(pixels) == 0:
        return ensure_uint8(bgr)

    channel_means = np.mean(pixels, axis=0)  # B, G, R
    gray_mean = float(np.mean(channel_means))

    scales = gray_mean / (channel_means + 1e-6)

    # เบาลงจากเดิม เพื่อไม่ให้ภาพมืดโดนปรับสีแรงเกิน
    scales = np.clip(scales, 0.92, 1.08)

    img[:, :, 0] *= scales[0]
    img[:, :, 1] *= scales[1]
    img[:, :, 2] *= scales[2]

    return ensure_uint8(img)


def remove_highlight_from_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    ตัดจุดสะท้อนแฟลช/เงามันวาว
    ใช้ HSV + ค่า V สูง + S ต่ำ
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    highlight_mask = cv2.inRange(hsv, (0, 0, 240), (180, 50, 255))
    refined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(highlight_mask))

    if cv2.countNonZero(refined_mask) > max(20, int(cv2.countNonZero(mask) * 0.25)):
        return refined_mask

    return mask


def compute_hsv_histogram(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    ใช้ H,S เป็นหลักผ่าน histogram
    ช่วยให้ทน brightness shift ได้ดีกว่าใช้สีดิบตรง ๆ
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        mask,
        [36, 32],
        [0, 180, 0, 256]
    )

    if hist is None:
        return np.zeros((36, 32), dtype=np.float32)

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_lab_stats(bgr: np.ndarray, mask: np.ndarray):
    """
    เก็บเฉพาะ a,b เป็นหลัก และ L ไว้ช่วยนิดเดียว
    เพื่อลดผลกระทบจากภาพมืด/สว่าง
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]

    if len(pixels) == 0:
        return None

    pixels = pixels.astype(np.float32)

    l_vals = pixels[:, 0]
    a_vals = pixels[:, 1]
    b_vals = pixels[:, 2]

    return {
        "L_mean": float(np.mean(l_vals)),
        "ab_mean": np.array([np.mean(a_vals), np.mean(b_vals)], dtype=np.float32),
        "ab_std": np.array([np.std(a_vals), np.std(b_vals)], dtype=np.float32),
    }


def vector_similarity(v1: np.ndarray, v2: np.ndarray, max_dist: float) -> float:
    if v1 is None or v2 is None:
        return 0.0

    dist = np.linalg.norm(v1 - v2)
    score = 1.0 - (dist / max_dist)
    return float(max(0.0, min(1.0, score)))


def lab_similarity(stats1, stats2) -> float:
    """
    ใช้ a,b เป็นหลัก
    """
    if stats1 is None or stats2 is None:
        return 0.0

    mean_score = vector_similarity(stats1["ab_mean"], stats2["ab_mean"], max_dist=60.0)
    std_score = vector_similarity(stats1["ab_std"], stats2["ab_std"], max_dist=35.0)

    return float((0.82 * mean_score) + (0.18 * std_score))


def luminance_similarity(stats1, stats2) -> float:
    """
    ให้ L channel มีผลน้อย ๆ
    """
    if stats1 is None or stats2 is None:
        return 0.0

    dist = abs(stats1["L_mean"] - stats2["L_mean"])
    score = 1.0 - (dist / 90.0)
    return float(max(0.0, min(1.0, score)))


def bgr_mean_similarity(mean1: np.ndarray, mean2: np.ndarray) -> float:
    """
    เก็บไว้ใช้เป็นตัวช่วยเล็กน้อยเท่านั้น
    """
    if mean1 is None or mean2 is None:
        return 0.0

    dist = np.linalg.norm(mean1 - mean2)
    max_dist = np.sqrt(255**2 + 255**2 + 255**2)
    score = 1.0 - (dist / max_dist)
    return float(max(0.0, min(1.0, score)))


def calculate_color_similarity(img1_path: str, img2_path: str) -> float:
    """
    คืนค่า similarity อยู่ในช่วง 0.0 - 1.0

    อัปเดตจากเวอร์ชันล่าสุด โดยปรับให้:
    - ทนภาพมืด/สว่างมากขึ้น
    - เน้นโทนสีมากกว่าความสว่าง
    - ลดผลจาก mean BGR
    """
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            return 0.0

        bgr1, mask1 = get_bgr_and_mask(img1)
        bgr2, mask2 = get_bgr_and_mask(img2)

        # 1) white balance แบบเบากว่าเดิม
        bgr1 = simple_gray_world_white_balance(bgr1, mask1)
        bgr2 = simple_gray_world_white_balance(bgr2, mask2)

        # 2) ตัด highlight
        mask1_refined = remove_highlight_from_mask(bgr1, mask1)
        mask2_refined = remove_highlight_from_mask(bgr2, mask2)

        if cv2.countNonZero(mask1_refined) > 0:
            mask1 = mask1_refined
        if cv2.countNonZero(mask2_refined) > 0:
            mask2 = mask2_refined

        # 3) HSV histogram similarity
        hist1 = compute_hsv_histogram(bgr1, mask1)
        hist2 = compute_hsv_histogram(bgr2, mask2)
        hsv_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hsv_score = float(max(0.0, min(1.0, hsv_score)))

        # 4) LAB similarity (เน้น a,b)
        lab_stats1 = compute_lab_stats(bgr1, mask1)
        lab_stats2 = compute_lab_stats(bgr2, mask2)
        lab_score = lab_similarity(lab_stats1, lab_stats2)

        # 5) Luminance similarity ให้มีผลน้อย
        lum_score = luminance_similarity(lab_stats1, lab_stats2)

        # 6) mean BGR similarity ให้เป็นแค่ตัวช่วยเล็กมาก
        mean1, _ = safe_mean_std_color(bgr1, mask1)
        mean2, _ = safe_mean_std_color(bgr2, mask2)
        mean_bgr_score = bgr_mean_similarity(mean1, mean2)

        # 7) fusion ใหม่
        final_score = (
            (0.58 * hsv_score) +
            (0.27 * lab_score) +
            (0.10 * lum_score) +
            (0.05 * mean_bgr_score)
        )

        return float(max(0.0, min(1.0, final_score)))

    except Exception as e:
        print(f"Color Extraction Error: {e}")
        return 0.0