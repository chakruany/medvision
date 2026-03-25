import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_b5,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
)


class ImageFeatureExtractor:
    """
    EfficientNet feature extractor สำหรับ retrieval

    รองรับ:
    - efficientnet_b0
    - efficientnet_b4
    - efficientnet_b5

    output:
    - B0 = 1280 dims
    - B4 = 1792 dims
    - B5 = 2048 dims
    """

    MODEL_CONFIGS = {
        "efficientnet_b0": {
            "builder": efficientnet_b0,
            "weights": EfficientNet_B0_Weights.DEFAULT,
            "embedding_dim": 1280,
        },
        "efficientnet_b4": {
            "builder": efficientnet_b4,
            "weights": EfficientNet_B4_Weights.DEFAULT,
            "embedding_dim": 1792,
        },
        "efficientnet_b5": {
            "builder": efficientnet_b5,
            "weights": EfficientNet_B5_Weights.DEFAULT,
            "embedding_dim": 2048,
        },
    }

    def __init__(self, model_name: str = "efficientnet_b4", use_rotation_aug: bool = True):
        self.model_name = model_name.lower().strip()
        self.use_rotation_aug = use_rotation_aug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model_name='{model_name}'. "
                f"Supported: {list(self.MODEL_CONFIGS.keys())}"
            )

        config = self.MODEL_CONFIGS[self.model_name]
        print(f"[*] Initializing Feature Extractor ({self.model_name.upper()}) with Augmentation on: {self.device}")

        base_model = config["builder"](weights=config["weights"])
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1)
        )

        self.embedding_dim = config["embedding_dim"]
        self.preprocess = config["weights"].transforms()

        self.model.to(self.device)
        self.model.eval()

    def _safe_open_image(self, img_path: str):
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [-] Image Open Error on {img_path}: {e}")
            return None

    def _get_augmented_images(self, img_path: str) -> list:
        """
        จำลองสภาพแสง + rotation
        """
        img = self._safe_open_image(img_path)
        if img is None:
            return []

        augs = [img]

        try:
            # Photometric augmentation
            augs.append(ImageEnhance.Brightness(img).enhance(0.6))   # darker
            augs.append(ImageEnhance.Brightness(img).enhance(1.4))   # brighter
            augs.append(ImageEnhance.Contrast(img).enhance(0.7))     # low contrast

            # Rotation augmentation
            if self.use_rotation_aug:
                augs.append(img.rotate(90, expand=True))
                augs.append(img.rotate(180, expand=True))
                augs.append(img.rotate(270, expand=True))

        except Exception as e:
            print(f"  [-] Augmentation Error on {img_path}: {e}")

        return augs

    def _extract_single_feature(self, img: Image.Image):
        try:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().detach().cpu().numpy().astype(np.float32)

            norm = np.linalg.norm(features)
            if norm < 1e-12:
                return None

            features = features / norm
            return features

        except Exception as e:
            print(f"  [-] Feature Extraction Error: {e}")
            return None

    def extract_features(self, image_path: str, use_augmentation: bool = False):
        """
        ถ้า use_augmentation=True  -> return list[np.ndarray]
        ถ้า use_augmentation=False -> return np.ndarray เดียว
        """
        if use_augmentation:
            images_to_process = self._get_augmented_images(image_path)
        else:
            img = self._safe_open_image(image_path)
            images_to_process = [img] if img is not None else []

        vectors = []
        for img in images_to_process:
            if img is None:
                continue

            feat = self._extract_single_feature(img)
            if feat is not None:
                vectors.append(feat)

        if not use_augmentation:
            return vectors[0] if vectors else None

        return vectors