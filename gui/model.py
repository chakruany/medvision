import os
import glob
import re
import pickle
import hashlib
from collections import defaultdict
from PIL import Image, ImageEnhance

from src.bg_remover import AutoBackgroundRemover
from src.feature_extractor import ImageFeatureExtractor
from src.ocr_engine import PillTextRecognizer
from src.utils import calculate_color_similarity, parse_filename_metadata
from src.vector_db import PillVectorDatabase


SPECIAL_EDGE_CASES = {"NONE", "LOGO", "SCORE", "CROSS", "UNKNOWN", ""}


def normalize_text_simple(text: str) -> str:
    text = (text or "").upper().strip()
    text = re.sub(r"[\s\-_./\\|]+", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def get_reference_prior(metadata: dict) -> float:
    imprint = normalize_text_simple(metadata.get("imprint", ""))
    return -0.04 if imprint in SPECIAL_EDGE_CASES else 0.02


def predict_query_side(extracted_text: str, ocr_engine: PillTextRecognizer) -> str:
    text = normalize_text_simple(extracted_text)
    return "FRONT" if len(text) >= 3 and not ocr_engine.is_weak_text(text) else "BACK"


def get_side_bonus(predicted_side: str, metadata: dict) -> float:
    side = (metadata.get("side", "") or "").upper().strip()
    if side == predicted_side:
        return 0.05
    if side in {"FRONT", "BACK"}:
        return -0.02
    return 0.0


def get_dynamic_weights(extracted_text: str, metadata: dict, ocr_engine: PillTextRecognizer):
    query = normalize_text_simple(extracted_text)
    imprint = normalize_text_simple(metadata.get("imprint", ""))

    if ocr_engine.is_weak_text(query):
        return 0.82, 0.18, 0.00

    if imprint in SPECIAL_EDGE_CASES:
        if len(query) == 2:
            return 0.76, 0.18, 0.06
        if len(query) == 3:
            return 0.66, 0.18, 0.16
        return 0.58, 0.17, 0.25

    if len(query) == 2:
        return 0.75, 0.17, 0.08
    if len(query) == 3:
        return 0.58, 0.17, 0.25
    return 0.48, 0.17, 0.35


def apply_color_gating(visual_score: float, color_score: float) -> float:
    if visual_score < 0.55:
        return color_score * 0.35
    if visual_score < 0.60:
        return color_score * 0.50
    if visual_score < 0.65:
        return color_score * 0.70
    return color_score


def extract_query_tta_features(clean_img_path: str, extractor: ImageFeatureExtractor):
    features = []
    try:
        img = Image.open(clean_img_path).convert("RGB")

        query_imgs = [
            img,
            ImageEnhance.Brightness(img).enhance(0.6),
            ImageEnhance.Brightness(img).enhance(1.4),
            ImageEnhance.Contrast(img).enhance(0.7),
            img.rotate(90, expand=True),
            img.rotate(180, expand=True),
            img.rotate(270, expand=True),
        ]

        for i, qimg in enumerate(query_imgs):
            feat = extractor._extract_single_feature(qimg)
            if feat is not None:
                features.append(feat)
            else:
                print(f"  [-] TTA feature extraction failed for variant index {i}")

    except Exception as e:
        print(f"  [-] Query TTA extraction error: {e}")

    return features


def search_with_query_tta(vector_db: PillVectorDatabase, query_features, top_k_per_view: int = 30):
    merged = {}
    for feat in query_features:
        results = vector_db.search(feat, top_k=top_k_per_view)
        for res in results:
            pill_name = res["pill_name"]
            if pill_name not in merged or res["similarity_score"] > merged[pill_name]["similarity_score"]:
                merged[pill_name] = res
    return list(merged.values())


def compute_drug_support_counts(candidates):
    counts = defaultdict(int)
    for res in candidates:
        metadata = parse_filename_metadata(res["pill_name"])
        key = (metadata.get("drug_id", ""), metadata.get("trade_name", "").upper())
        counts[key] += 1
    return counts


def get_drug_support_bonus(metadata: dict, support_counts: dict) -> float:
    key = (metadata.get("drug_id", ""), metadata.get("trade_name", "").upper())
    count = support_counts.get(key, 0)
    if count >= 3:
        return 0.04
    if count == 2:
        return 0.025
    return 0.0


def aggregate_candidates(final_scores):
    aggregated = {}
    for item in final_scores:
        key = (item.get("drug_id", ""), item.get("trade_name", "").upper())
        if key not in aggregated or item["fusion_score"] > aggregated[key]["fusion_score"]:
            aggregated[key] = item
    return sorted(aggregated.values(), key=lambda x: x["fusion_score"], reverse=True)


def get_file_fingerprint(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_pill_images_by_id(drug_id: str, ref_dir: str) -> dict:
    images = {"FRONT": None, "BACK": None}
    if not drug_id or drug_id == "UNKNOWN":
        return images

    for filename in os.listdir(ref_dir):
        if not filename.lower().endswith(".png"):
            continue

        metadata = parse_filename_metadata(filename)
        if metadata.get("drug_id") != drug_id:
            continue

        side = metadata.get("side", "").upper()
        full_path = os.path.join(ref_dir, filename)

        if side == "FRONT":
            images["FRONT"] = full_path
        elif side == "BACK":
            images["BACK"] = full_path

    return images


def print_debug_top5(
    query_img_path: str,
    clean_img_path: str,
    primary_text: str,
    ocr_text: str,
    ocr_details: dict,
    predicted_side: str,
    query_warnings: list,
    final_scores: list
):
    print("\n" + "=" * 100)
    print("[*] MEDVISION DEBUG REPORT")
    print("=" * 100)

    print(f"[Query Image Path ] : {query_img_path}")
    print(f"[Clean Image Path ] : {clean_img_path}")
    print(f"[Primary Text     ] : {primary_text}")
    print(f"[OCR Text         ] : {ocr_text}")
    print(f"[Predicted Side   ] : {predicted_side}")

    print(f"[OCR Confidence   ] : {ocr_details.get('confidence', 0.0):.4f}")
    print(f"[OCR Rotation     ] : {ocr_details.get('rotation', '0')}")
    print(f"[OCR Preprocess   ] : {ocr_details.get('preprocess', '')}")
    print(f"[OCR Lines        ] : {ocr_details.get('lines', [])}")
    print(f"[OCR Candidates   ] : {ocr_details.get('candidates', [])[:10]}")

    if query_warnings:
        print("[Warnings         ]")
        for w in query_warnings:
            print(f"  - {w}")
    else:
        print("[Warnings         ] : None")

    print("\n" + "-" * 100)
    print("[*] TOP-5 RANKED CANDIDATES")
    print("-" * 100)

    for i, item in enumerate(final_scores[:5], start=1):
        print(f"\n#{i}")
        print(f"  pill_name           : {item.get('pill_name', 'UNKNOWN')}")
        print(f"  drug_id             : {item.get('drug_id', 'UNKNOWN')}")
        print(f"  trade_name          : {item.get('trade_name', 'UNKNOWN')}")
        print(f"  generic_name        : {item.get('generic_name', 'UNKNOWN')}")
        print(f"  company             : {item.get('company', 'UNKNOWN')}")
        print(f"  reference_imprint   : {item.get('reference_imprint', 'UNKNOWN')}")
        print(f"  reference_side      : {item.get('reference_side', 'UNKNOWN')}")
        print()

        print(f"  visual_score        : {item.get('visual_score', 0):8.2f}%")
        print(f"  raw_color_score     : {item.get('raw_color_score', 0):8.2f}%")
        print(f"  gated_color_score   : {item.get('color_score', 0):8.2f}%")
        print(f"  text_score          : {item.get('text_score', 0):8.2f}%")
        print(f"  fusion_score        : {item.get('fusion_score', 0):8.2f}%")
        print()

        print(f"  weight_visual       : {item.get('w_visual', 0):8.4f}")
        print(f"  weight_color        : {item.get('w_color', 0):8.4f}")
        print(f"  weight_ocr          : {item.get('w_ocr', 0):8.4f}")
        print()

        print(f"  reference_prior     : {item.get('reference_prior', 0):8.4f}")
        print(f"  side_bonus          : {item.get('side_bonus', 0):8.4f}")
        print(f"  support_bonus       : {item.get('support_bonus', 0):8.4f}")
        print()

        print(f"  weighted_visual     : {item.get('weighted_visual', 0):8.4f}")
        print(f"  weighted_color      : {item.get('weighted_color', 0):8.4f}")
        print(f"  weighted_text       : {item.get('weighted_text', 0):8.4f}")
        print()

        print(f"  human_text          : {item.get('human_text', '')}")
        print(f"  ocr_text            : {item.get('ocr_text', '')}")
        print(f"  human_match         : {item.get('human_match', 0):8.4f}")
        print(f"  ocr_match           : {item.get('ocr_match', 0):8.4f}")
        print(f"  imprint_used        : {item.get('imprint_read', '')}")

        if item.get("debug_reason"):
            print(f"  debug_reason        : {item.get('debug_reason')}")

        print("-" * 100)

    print("=" * 100 + "\n")


class MedVisionModel:
    def __init__(self):
        self.ocr_engine = PillTextRecognizer()
        self.extractor = ImageFeatureExtractor(model_name="efficientnet_b4")
        self.vector_db = PillVectorDatabase(embedding_dim=self.extractor.embedding_dim)
        self.bg_remover = AutoBackgroundRemover()
        self.ref_dir = "data/reference"
        self._load_database()

    def _load_database(self, force_rebuild: bool = False):
        ref_images = sorted(glob.glob(os.path.join(self.ref_dir, "*.png")))
        if not ref_images:
            print("[-] No reference images found.")
            return

        ref_signatures = {
            os.path.basename(p): get_file_fingerprint(p)
            for p in ref_images
        }

        cache_path = "data/reference_cache.pkl"

        if (not force_rebuild) and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                if (
                    cached_data.get("num_images") == len(ref_images) and
                    cached_data.get("signatures") == ref_signatures
                ):
                    self.vector_db.add_reference_images(
                        cached_data["features"],
                        cached_data["labels"]
                    )
                    print("[*] Loaded reference embeddings from cache.")
                    return
            except Exception as e:
                print(f"[-] Failed to load cache, rebuilding... {e}")

        features_list, labels_list = [], []

        for img_path in ref_images:
            filename = os.path.basename(img_path)
            try:
                augmented_features = self.extractor.extract_features(img_path, use_augmentation=True)
                for feature in augmented_features:
                    features_list.append(feature)
                    labels_list.append(filename)
            except Exception as e:
                print(f"[-] Failed to extract features for {filename}: {e}")

        self.vector_db.add_reference_images(features_list, labels_list)

        with open(cache_path, "wb") as f:
            pickle.dump({
                "num_images": len(ref_images),
                "signatures": ref_signatures,
                "features": features_list,
                "labels": labels_list
            }, f)

        print("[*] Rebuilt reference embeddings cache.")

    def predict(self, query_img_path: str, human_imprint: str = "") -> dict:
        clean_img_path = None
        try:
            clean_img_path = self.bg_remover.clean_image(query_img_path)

            query_warnings = []

            try:
                with Image.open(clean_img_path) as cleaned_img:
                    ratio = self.bg_remover.estimate_foreground_ratio(cleaned_img)
                    if ratio < 0.08:
                        query_warnings.append("ภาพเม็ดยาเล็กเกินไปหรือฉากหลังรบกวนมาก อาจทำให้ความแม่นยำลดลง")
            except Exception as e:
                print(f"  [-] Query quality check failed: {e}")

            ocr_details = self.ocr_engine.extract_text_details(clean_img_path)
            ocr_text = ocr_details.get("text", "")
            human_text = human_imprint.strip().upper()

            primary_text = human_text if human_text else ocr_text
            predicted_side = predict_query_side(primary_text, self.ocr_engine)

            query_features = extract_query_tta_features(clean_img_path, self.extractor)
            if not query_features:
                return {"error": "Failed to extract vision features."}

            candidates = sorted(
                search_with_query_tta(self.vector_db, query_features, 30),
                key=lambda x: x["similarity_score"],
                reverse=True
            )

            if not candidates:
                return {"error": "No candidate matches found in vector database."}

            support_counts = compute_drug_support_counts(candidates)

            final_scores = []
            for res in candidates:
                ref_filename = res["pill_name"]
                ref_img_path = os.path.join(self.ref_dir, ref_filename)
                metadata = parse_filename_metadata(ref_filename)

                visual_score = res["similarity_score"] / 100.0
                raw_color_score = calculate_color_similarity(clean_img_path, ref_img_path)
                color_score = apply_color_gating(visual_score, raw_color_score)

                human_match = 0.0
                ocr_match = 0.0
                debug_reason = ""

                if human_text:
                    human_match = self.ocr_engine.calculate_text_similarity_from_metadata(human_text, metadata)
                    ocr_match = self.ocr_engine.calculate_text_similarity_from_metadata(ocr_text, metadata)

                    if human_match >= 0.65:
                        if ocr_match >= 0.60:
                            text_score = 1.0
                            debug_reason = "Human imprint matched strongly and OCR agreed."
                        else:
                            if visual_score >= 0.70 and color_score >= 0.60:
                                text_score = 0.95
                                debug_reason = "Human matched strongly, OCR weak, but visual/color strongly supported."
                            else:
                                text_score = human_match * 0.80
                                debug_reason = "Human matched strongly, but visual/color not strong enough, so reduced text score."
                    else:
                        text_score = human_match
                        debug_reason = "Human text provided but did not strongly match this candidate."
                else:
                    text_score = self.ocr_engine.calculate_text_similarity_from_metadata(ocr_text, metadata)
                    ocr_match = text_score
                    debug_reason = "No human text. Used OCR-only similarity."

                w_visual, w_color, w_ocr = get_dynamic_weights(primary_text, metadata, self.ocr_engine)
                reference_prior = get_reference_prior(metadata)
                side_bonus = get_side_bonus(predicted_side, metadata)
                support_bonus = get_drug_support_bonus(metadata, support_counts)

                weighted_visual = visual_score * w_visual
                weighted_color = color_score * w_color
                weighted_text = text_score * w_ocr

                fusion_score = max(
                    0.0,
                    min(
                        1.0,
                        weighted_visual +
                        weighted_color +
                        weighted_text +
                        reference_prior +
                        side_bonus +
                        support_bonus
                    )
                )

                drug_id = metadata.get("drug_id", "")
                sides_imgs = get_pill_images_by_id(drug_id, self.ref_dir)

                final_scores.append({
                    "pill_name": ref_filename,
                    "drug_id": drug_id,
                    "trade_name": metadata.get("trade_name", "UNKNOWN"),
                    "generic_name": metadata.get("generic_name", "UNKNOWN"),
                    "company": metadata.get("company", "UNKNOWN"),
                    "reference_imprint": metadata.get("imprint", "UNKNOWN"),
                    "reference_side": metadata.get("side", "UNKNOWN"),

                    "fusion_score": round(fusion_score * 100, 2),
                    "visual_score": round(visual_score * 100, 2),
                    "raw_color_score": round(raw_color_score * 100, 2),
                    "color_score": round(color_score * 100, 2),
                    "text_score": round(text_score * 100, 2),

                    "w_visual": round(w_visual, 4),
                    "w_color": round(w_color, 4),
                    "w_ocr": round(w_ocr, 4),

                    "reference_prior": round(reference_prior, 4),
                    "side_bonus": round(side_bonus, 4),
                    "support_bonus": round(support_bonus, 4),

                    "weighted_visual": round(weighted_visual, 4),
                    "weighted_color": round(weighted_color, 4),
                    "weighted_text": round(weighted_text, 4),

                    "human_text": human_text,
                    "ocr_text": ocr_text,
                    "human_match": round(human_match, 4),
                    "ocr_match": round(ocr_match, 4),

                    "front_img": sides_imgs["FRONT"],
                    "back_img": sides_imgs["BACK"],
                    "imprint_read": primary_text,
                    "debug_reason": debug_reason
                })

            final_scores = aggregate_candidates(final_scores)
            
            # --------------------------------------------------------
            # Debug: แสดง Top 5 บน Terminal เพื่อใช้วิเคราะห์ละเอียด
            # --------------------------------------------------------
            print_debug_top5(
                query_img_path=query_img_path,
                clean_img_path=clean_img_path,
                primary_text=primary_text,
                ocr_text=ocr_text,
                ocr_details=ocr_details,
                predicted_side=predicted_side,
                query_warnings=query_warnings,
                final_scores=final_scores
            )

            
            ##############################

            return {
                "status": "success",
                "results": final_scores[:3],
                "extracted_text": primary_text,
                "warnings": query_warnings
            }

        except Exception as e:
            return {"error": str(e)}

        finally:
            if clean_img_path and os.path.exists(clean_img_path):
                try:
                    os.remove(clean_img_path)
                except Exception:
                    pass