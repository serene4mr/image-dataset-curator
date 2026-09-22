import cv2
import numpy as np
from typing import List, Optional, Callable
from image_dataset_curator.utils.metrics import compute_fast_ssim, compute_dhash, compute_hamming_distance


def run_stage2_similarity(
    file_paths: List[str],
    metric: str = 'ssim',
    ssim_thresh: float = 0.85,
    dhash_thresh: int = 5,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> List[str]:
    """
    Stage 2: Sequential Similarity Filter
    Eliminates consecutive frame redundancy when camera is stationary or slow-moving.
    """
    if not file_paths:
        return []

    selected = [file_paths[0]]
    last_img = cv2.imread(file_paths[0])
    last_hash = compute_dhash(last_img) if metric == 'dhash' else None
    
    total = len(file_paths)
    if progress_callback:
        progress_callback(1, total, len(selected))

    for idx, path in enumerate(file_paths[1:], start=1):
        curr_img = cv2.imread(path)
        if curr_img is None:
            continue

        if metric == 'ssim':
            similarity = compute_fast_ssim(last_img, curr_img)
            # If similarity < threshold, scene has changed enough to keep
            if similarity < ssim_thresh:
                selected.append(path)
                last_img = curr_img
        else: # dhash
            curr_hash = compute_dhash(curr_img)
            hamming_dist = compute_hamming_distance(last_hash, curr_hash)
            if hamming_dist >= dhash_thresh:
                selected.append(path)
                last_hash = curr_hash
                last_img = curr_img

        if progress_callback:
            progress_callback(idx + 1, total, len(selected))

    return selected
