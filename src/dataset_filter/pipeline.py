import os
import time
from typing import List, Optional, Callable, Tuple

from dataset_filter.stages.stage1_step import run_stage1_step
from dataset_filter.stages.stage2_similarity import run_stage2_similarity
from dataset_filter.stages.stage3_diversity import run_stage3_diversity


class DatasetFilteringPipeline:
    """
    Multi-Stage Dataset Filtering Pipeline for image directories.
    
    Stages:
      Stage 1: Coarse Skip-Frame Subsampling (Fast I/O reduction)
      Stage 2: Sequential Similarity Filter (SSIM/dHash - removes stationary/slow-movement frames)
      Stage 3: Global Diversity & Representation (Auto Sphere-Packing or Budget FPS)
    """
    def __init__(
        self,
        mode: str = 'auto',
        auto_sensitivity: str = 'medium',
        target_pct: Optional[float] = None,
        target_count: Optional[int] = None,
        enable_stage1: bool = True,
        stage1_step: int = 3,
        enable_stage2: bool = True,
        stage2_metric: str = 'ssim',
        stage2_ssim_thresh: float = 0.85,
        stage2_dhash_thresh: int = 5,
        enable_stage3: bool = True,
        stage3_model: str = 'auto',
    ):
        self.mode = mode
        self.auto_sensitivity = auto_sensitivity
        self.target_pct = target_pct
        self.target_count = target_count

        self.enable_stage1 = enable_stage1
        self.stage1_step = stage1_step

        self.enable_stage2 = enable_stage2
        self.stage2_metric = stage2_metric
        self.stage2_ssim_thresh = stage2_ssim_thresh
        self.stage2_dhash_thresh = stage2_dhash_thresh

        self.enable_stage3 = enable_stage3
        self.stage3_model = stage3_model

        # Map sensitivity to threshold
        self.sim_threshold_map = {'high': 0.985, 'medium': 0.975, 'low': 0.950}

    def run(
        self,
        input_files: List[str],
        stage_callback: Optional[Callable[[str, int, float], None]] = None,
        progress_callbacks: Optional[dict] = None
    ) -> Tuple[List[str], dict]:
        """
        Executes pipeline over a list of existing image file paths on disk.
        Returns: (selected_file_paths, pipeline_statistics)
        """
        total_raw = len(input_files)
        current_pool = list(input_files)
        start_time = time.time()

        computed_target = None
        if self.mode == 'budget':
            if self.target_pct is not None:
                computed_target = max(1, int(round(total_raw * (self.target_pct / 100.0))))
            elif self.target_count is not None:
                computed_target = min(total_raw, max(1, self.target_count))

        stats = {
            "total_raw_frames": total_raw,
            "mode": self.mode,
            "auto_sensitivity": self.auto_sensitivity if self.mode == 'auto' else None,
            "target_budget": computed_target,
            "stages": {}
        }

        # Stage 1: Fixed Step (Coarse Skip-Frame Sampling)
        t0 = time.time()
        if self.enable_stage1 and self.stage1_step > 1:
            current_pool = run_stage1_step(current_pool, step=self.stage1_step)
            t_stage1 = time.time() - t0
            stats["stages"]["stage1"] = {
                "step": self.stage1_step,
                "retained": len(current_pool),
                "duration_sec": round(t_stage1, 3)
            }
            if stage_callback:
                stage_callback("Stage 1 (Fixed Step)", len(current_pool), t_stage1)

        # Stage 2: Sequential Similarity (SSIM / dHash)
        t0 = time.time()
        if self.enable_stage2 and len(current_pool) > 1:
            cb = progress_callbacks.get("stage2") if progress_callbacks else None
            current_pool = run_stage2_similarity(
                current_pool,
                metric=self.stage2_metric,
                ssim_thresh=self.stage2_ssim_thresh,
                dhash_thresh=self.stage2_dhash_thresh,
                progress_callback=cb
            )
            t_stage2 = time.time() - t0
            stats["stages"]["stage2"] = {
                "metric": self.stage2_metric,
                "retained": len(current_pool),
                "duration_sec": round(t_stage2, 3)
            }
            if stage_callback:
                stage_callback("Stage 2 (Sequential Similarity)", len(current_pool), t_stage2)

        # Stage 3: Global Diversity & Representation
        t0 = time.time()
        if self.enable_stage3 and len(current_pool) > 1:
            cb = progress_callbacks.get("stage3") if progress_callbacks else None
            auto_thresh = self.sim_threshold_map.get(self.auto_sensitivity, 0.975)
            current_pool, s3_meta = run_stage3_diversity(
                current_pool,
                mode=self.mode,
                auto_similarity_thresh=auto_thresh,
                target_count=computed_target,
                model_type=self.stage3_model,
                extract_progress_callback=cb
            )
            t_stage3 = time.time() - t0
            s3_meta["duration_sec"] = round(t_stage3, 3)
            stats["stages"]["stage3"] = s3_meta
            if stage_callback:
                stage_callback("Stage 3 (Global Diversity)", len(current_pool), t_stage3)

        total_time = time.time() - start_time
        stats["total_time_sec"] = round(total_time, 2)
        stats["final_retained"] = len(current_pool)
        stats["reduction_ratio_pct"] = round(100.0 * (1.0 - len(current_pool) / total_raw), 2)
        stats["retention_ratio_pct"] = round(100.0 * (len(current_pool) / total_raw), 2)

        return current_pool, stats
