import os
import tempfile
import cv2
import numpy as np
import pytest
from image_dataset_curator.pipeline import DatasetFilteringPipeline
from image_dataset_curator.stages.stage1_step import run_stage1_step
from image_dataset_curator.stages.stage2_similarity import run_stage2_similarity
from image_dataset_curator.stages.stage3_diversity import auto_sphere_packing, farthest_point_sampling
from image_dataset_curator.utils.metrics import compute_fast_ssim, compute_dhash


def test_metrics():
    # Test identical images
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 128, dtype=np.uint8)
    ssim_val = compute_fast_ssim(img1, img2)
    assert ssim_val > 0.99

    # Test different images
    img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dhash1 = compute_dhash(img1)
    dhash3 = compute_dhash(img3)
    assert len(dhash1) == 64
    assert len(dhash3) == 64


def test_stage1_step():
    files = [f"frame_{i:04d}.png" for i in range(10)]
    retained = run_stage1_step(files, step=3)
    assert retained == ["frame_0000.png", "frame_0003.png", "frame_0006.png", "frame_0009.png"]


def test_stage3_algorithms():
    np.random.seed(42)
    features = np.random.randn(20, 32).astype(np.float32)
    
    # Test FPS
    fps_indices = farthest_point_sampling(features, k=5)
    assert len(fps_indices) == 5
    assert len(set(fps_indices)) == 5

    # Test Sphere Packing
    auto_indices = auto_sphere_packing(features, similarity_threshold=0.5)
    assert len(auto_indices) > 0
    assert len(auto_indices) <= 20


def test_full_pipeline_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)

        # Create 10 dummy images
        image_paths = []
        for i in range(10):
            p = os.path.join(input_dir, f"frame_{i:04d}.png")
            # Alternate between black and noise
            if i < 5:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(p, img)
            image_paths.append(p)

        pipeline = DatasetFilteringPipeline(mode='auto', enable_stage1=False)
        selected_files, stats = pipeline.run(image_paths)

        assert len(selected_files) > 0
        assert len(selected_files) <= len(image_paths)
        assert stats["total_raw_frames"] == 10
