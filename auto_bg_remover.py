import os
import csv
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm

from src.bg_remover import AutoBackgroundRemover

ort.preload_dlls()


def parse_filename(name: str) -> dict:
    """
    รูปแบบชื่อไฟล์ที่รองรับ:
    [DrugID]_[Generic]_[Trade]_[Company]_[Imprint]_[Side]
    หรือ
    [DrugID]_[Generic]_[Trade]_[Company]_[Side]   # ไม่มี imprint
    """
    parts = name.split("_")

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
        "generic_name": name.strip() or "UNKNOWN",
        "trade_name": "UNKNOWN",
        "company": "UNKNOWN",
        "imprint": "UNKNOWN",
        "side": "UNKNOWN",
    }


def process_bulk_images(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    metadata_records = []
    supported_exts = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    ]

    print(f"[*] Found {len(image_paths)} images in {input_dir}. Starting processing...")

    bg_remover = AutoBackgroundRemover()

    for img_path in tqdm(image_paths, desc="Removing Backgrounds"):
        filename = img_path.name
        name = img_path.stem
        output_path = Path(output_dir) / f"{name}.png"

        try:
            parsed = parse_filename(name)

            with Image.open(img_path) as input_img:
                removed = bg_remover.remove_background_from_image(input_img)
                final_img = bg_remover.standardize_image(removed)
                final_img.save(output_path, format="PNG")

            metadata_records.append({
                "file_name": f"{name}.png",
                "drug_id": parsed["drug_id"],
                "generic_name": parsed["generic_name"],
                "trade_name": parsed["trade_name"],
                "company": parsed["company"],
                "imprint": parsed["imprint"],
                "side": parsed["side"],
            })

        except Exception as e:
            print(f"\n[-] Error processing {filename}: {e}")

    csv_path = Path(output_dir) / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_name", "drug_id", "generic_name", "trade_name", "company", "imprint", "side"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    print("-" * 60)
    print(f"[*] Done! Processed {len(metadata_records)} images.")
    print(f"[*] Metadata saved to: {csv_path}")


if __name__ == "__main__":
    INPUT_FOLDER = "data/raw_reference"
    OUTPUT_FOLDER = "data/reference"

    os.makedirs(INPUT_FOLDER, exist_ok=True)
    process_bulk_images(INPUT_FOLDER, OUTPUT_FOLDER)