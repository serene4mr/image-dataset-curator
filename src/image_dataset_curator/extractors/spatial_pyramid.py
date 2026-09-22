import cv2
import numpy as np
from image_dataset_curator.extractors.base import BaseFeatureExtractor


class SpatialPyramidExtractor(BaseFeatureExtractor):
    """
    Ultra-fast, zero-GPU feature extractor using Spatial Pyramid Color & Texture Histograms.
    Captures color distribution (HSV) and edge magnitude (Sobel) across spatial grid.
    """
    def __init__(self, resize_dim=(256, 160), grid_size=(2, 2)):
        self.resize_dim = resize_dim
        self.grid_size = grid_size

    def extract_from_array(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, self.resize_dim, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        
        h, w = self.resize_dim[1], self.resize_dim[0]
        rows, cols = self.grid_size
        
        features = []
        for i in range(rows):
            for j in range(cols):
                y1, y2 = i * h // rows, (i + 1) * h // rows
                x1, x2 = j * w // cols, (j + 1) * w // cols
                
                cell_hsv = hsv[y1:y2, x1:x2]
                cell_mag = mag[y1:y2, x1:x2]
                
                h_hist = cv2.calcHist([cell_hsv], [0], None, [16], [0, 180]).flatten()
                s_hist = cv2.calcHist([cell_hsv], [1], None, [8], [0, 256]).flatten()
                v_hist = cv2.calcHist([cell_hsv], [2], None, [8], [0, 256]).flatten()
                m_hist = np.histogram(cell_mag, bins=8, range=(0, 255))[0].astype(np.float32)
                
                features.extend([h_hist, s_hist, v_hist, m_hist])
                
        feat_vec = np.concatenate(features)
        norm = np.linalg.norm(feat_vec)
        if norm > 0:
            feat_vec = feat_vec / norm
        return feat_vec

    def extract_from_path(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        return self.extract_from_array(img)
