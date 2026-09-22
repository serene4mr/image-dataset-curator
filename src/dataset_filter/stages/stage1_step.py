from typing import List


def run_stage1_step(file_paths: List[str], step: int = 3) -> List[str]:
    """
    Stage 1: Fixed Step (Coarse Skip-Frame Subsampling)
    Fast I/O reduction before heavy metric computations.
    """
    if step <= 1:
        return file_paths
    return file_paths[::step]
