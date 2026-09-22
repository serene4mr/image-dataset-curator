from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for image feature extractors."""
    
    @abstractmethod
    def extract_from_path(self, img_path: str) -> np.ndarray:
        """Extract a 1D normalized feature embedding from an image file path."""
        pass
        
    @abstractmethod
    def extract_from_array(self, img: np.ndarray) -> np.ndarray:
        """Extract a 1D normalized feature embedding from a BGR numpy image array."""
        pass
