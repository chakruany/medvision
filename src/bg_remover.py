import os
from PIL import Image
from rembg import remove, new_session
import onnxruntime as ort

ort.preload_dlls()


class AutoBackgroundRemover:
    def __init__(self):
        print("[*] Initializing Auto-Background Remover Engine (U^2-Net)...")
        self.session = new_session(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def remove_background_from_image(self, input_img: Image.Image) -> Image.Image:
        if input_img.mode not in ("RGB", "RGBA"):
            input_img = input_img.convert("RGBA")

        output_img = remove(input_img, session=self.session)

        if output_img.mode != "RGBA":
            output_img = output_img.convert("RGBA")

        return output_img

    def standardize_image(
        self,
        img: Image.Image,
        output_size=(224, 224),
        padding_ratio=0.05
    ) -> Image.Image:
        """
        ทำภาพมาตรฐานเดียวกันสำหรับทั้ง reference และ query
        1) หา alpha bbox
        2) tight crop
        3) padding รอบ object
        4) วางกลางบน square canvas
        5) resize เป็น 224x224
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        bbox = img.getbbox()
        if not bbox:
            return img.resize(output_size, Image.Resampling.LANCZOS)

        cropped = img.crop(bbox)

        max_dim = max(cropped.size)
        padding = max(2, int(max_dim * padding_ratio))
        canvas_dim = max_dim + (padding * 2)

        square_img = Image.new("RGBA", (canvas_dim, canvas_dim), (0, 0, 0, 0))
        offset = (
            padding + (max_dim - cropped.size[0]) // 2,
            padding + (max_dim - cropped.size[1]) // 2
        )
        square_img.paste(cropped, offset)

        return square_img.resize(output_size, Image.Resampling.LANCZOS)

    def estimate_foreground_ratio(self, img: Image.Image) -> float:
        """
        ประเมินว่า object มีขนาดกินพื้นที่ภาพมากน้อยแค่ไหน
        ใช้เตือนกรณีภาพ query ไกลเกินไปหรือฉากหลังรบกวนมาก
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        if not bbox:
            return 0.0

        obj_w = bbox[2] - bbox[0]
        obj_h = bbox[3] - bbox[1]
        obj_area = obj_w * obj_h
        total_area = img.size[0] * img.size[1]

        return obj_area / total_area if total_area > 0 else 0.0

    def clean_image(self, img_path: str, output_path: str = "data/query/_temp_clean_query.png") -> str:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with Image.open(img_path) as input_img:
                removed = self.remove_background_from_image(input_img)
                final_img = self.standardize_image(removed)
                final_img.save(output_path, format="PNG")

            return output_path

        except Exception as e:
            print(f"  [-] Auto-Clean Failed on {os.path.basename(img_path)}: {e}")
            return img_path