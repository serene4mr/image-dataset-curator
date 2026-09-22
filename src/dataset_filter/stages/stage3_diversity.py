import numpy as np
from typing import List, Optional, Callable, Tuple
from dataset_filter.extractors.base import BaseFeatureExtractor
from dataset_filter.extractors.spatial_pyramid import SpatialPyramidExtractor
from dataset_filter.extractors.deep_extractor import DeepModelExtractor, HAS_TORCH


def auto_sphere_packing(features: np.ndarray, similarity_threshold: float = 0.975) -> List[int]:
    """
    Greedy Sphere Packing / Dispersion Filtering in Feature Space.
    Automatically decides the exact number of representative frames.
    A candidate is ONLY accepted if its cosine similarity with ALL selected images is < threshold.
    """
    N, D = features.shape
    if N <= 1:
        return list(range(N))

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = features / norms

    selected_indices = [0]
    selected_feats = [feats[0]]

    for i in range(1, N):
        curr_feat = feats[i]
        sims = np.dot(selected_feats, curr_feat)
        max_sim = float(np.max(sims))

        if max_sim < similarity_threshold:
            selected_indices.append(i)
            selected_feats.append(curr_feat)

    return selected_indices


def farthest_point_sampling(features: np.ndarray, k: int) -> List[int]:
    """
    Selects exactly k samples that are maximally distant in cosine feature space.
    Guarantees maximal diversity for fixed budget target.
    """
    N, D = features.shape
    if k >= N:
        return list(range(N))

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = features / norms

    selected_indices = [0]
    min_distances = np.full(N, np.inf, dtype=np.float32)

    for _ in range(1, k):
        last_selected = feats[selected_indices[-1]]
        dist_to_last = 1.0 - np.dot(feats, last_selected)
        dist_to_last = np.maximum(dist_to_last, 0.0)

        min_distances = np.minimum(min_distances, dist_to_last)
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)

    return sorted(selected_indices)


def get_feature_extractor(model_type: str = 'auto') -> BaseFeatureExtractor:
    """Instantiate appropriate feature extractor."""
    if model_type in ['dinov2', 'resnet'] or (model_type == 'auto' and HAS_TORCH):
        try:
            m_name = 'dinov2' if model_type in ['auto', 'dinov2'] else 'resnet'
            return DeepModelExtractor(model_name=m_name)
        except Exception:
            return SpatialPyramidExtractor()
    return SpatialPyramidExtractor()


def run_stage3_diversity(
    file_paths: List[str],
    mode: str = 'auto',
    auto_similarity_thresh: float = 0.975,
    target_count: Optional[int] = None,
    model_type: str = 'auto',
    extract_progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[List[str], dict]:
    """
    Stage 3: Global Diversity & Representation Filtering.
    Supports auto-adaptive sphere packing or budget farthest point sampling.
    """
    if len(file_paths) <= 1:
        return file_paths, {"method": "none", "retained": len(file_paths)}

    extractor = get_feature_extractor(model_type=model_type)
    total = len(file_paths)
    feature_list = []

    for idx, path in enumerate(file_paths):
        feat = extractor.extract_from_path(path)
        feature_list.append(feat)
        if extract_progress_callback:
            extract_progress_callback(idx + 1, total)

    features = np.array(feature_list, dtype=np.float32)
    meta = {}

    if mode == 'auto':
        selected_indices = auto_sphere_packing(features, similarity_threshold=auto_similarity_thresh)
        meta = {
            "method": "auto_sphere_packing",
            "similarity_threshold": auto_similarity_thresh,
            "retained": len(selected_indices),
            "extractor": extractor.__class__.__name__
        }
    else: # budget mode
        if target_count is not None and len(file_paths) > target_count:
            selected_indices = farthest_point_sampling(features, target_count)
            meta = {
                "method": "farthest_point_sampling",
                "target_count": target_count,
                "retained": len(selected_indices),
                "extractor": extractor.__class__.__name__
            }
        else:
            selected_indices = list(range(len(file_paths)))
            meta = {
                "method": "pass_through",
                "retained": len(selected_indices),
                "extractor": extractor.__class__.__name__
            }

    selected_files = [file_paths[i] for i in selected_indices]
    return selected_files, meta
