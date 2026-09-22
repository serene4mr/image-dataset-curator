import cv2
import numpy as np


def compute_fast_ssim(img1: np.ndarray, img2: np.ndarray, resize_dim=(256, 160)) -> float:
    """
    Computes fast Structural Similarity Index (SSIM) on downsampled grayscale images.
    Returns value in range [0.0, 1.0].
    """
    g1 = cv2.resize(img1, resize_dim, interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(img2, resize_dim, interpolation=cv2.INTER_AREA)
    
    if len(g1.shape) == 3:
        g1 = cv2.cvtColor(g1, cv2.COLOR_BGR2GRAY)
    if len(g2.shape) == 3:
        g2 = cv2.cvtColor(g2, cv2.COLOR_BGR2GRAY)
        
    g1 = g1.astype(np.float32)
    g2 = g2.astype(np.float32)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Computes Difference Hash (dHash) as a boolean 1D array.
    """
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    return (gray[:, 1:] > gray[:, :-1]).flatten()


def compute_hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """
    Computes Hamming distance between two boolean hash arrays.
    """
    return int(np.count_nonzero(hash1 != hash2))
