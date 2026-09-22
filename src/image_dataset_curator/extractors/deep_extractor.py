import cv2
import numpy as np
from typing import Optional
from image_dataset_curator.extractors.base import BaseFeatureExtractor

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeepModelExtractor(BaseFeatureExtractor):
    """
    Feature extractor utilizing PyTorch models (DINOv2 or ResNet50).
    """
    def __init__(self, model_name: str = 'dinov2', device: Optional[str] = None):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch / Torchvision is required for DeepModelExtractor. "
                "Install with: uv add --optional deep"
            )
            
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if model_name.startswith('dinov2'):
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        else:
            import torchvision.models as models
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
            
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.to(self.device)
        self.model.eval()

    def extract_from_path(self, img_path: str) -> np.ndarray:
        pil_img = Image.open(img_path).convert('RGB')
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.squeeze().cpu().numpy()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
        return feat

    def extract_from_array(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.squeeze().cpu().numpy()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
        return feat
