# 🚀 Image Dataset Curator (`image-dataset-curator`)

> **A High-Performance Multi-Stage Image Dataset Deduplication and Diversity Curation Pipeline for Computer Vision & Semantic Segmentation.**

Fully managed with **[uv](https://github.com/astral-sh/uv)** — the ultra-fast Python package manager.

---

## 📌 Key Features

* **Multi-Stage Curation Pipeline:**
  * **Stage 1 (Coarse Fixed-Step Sampling):** Rapidly downsamples dense video streams to reduce initial disk I/O.
  * **Stage 2 (Sequential Similarity Filtering):** Utilizes fast Structural Similarity (SSIM) or Difference Hash (dHash) to eliminate consecutive redundant frames when the camera is stationary or moving slowly.
  * **Stage 3 (Global Diversity & Novelty Selection):** Extracts normalized feature representations (Spatial Color/Texture Pyramid or PyTorch DINOv2 / ResNet) and applies **Sphere-Packing Dispersion / Farthest Point Sampling** to eliminate global scene redundancy.
* **Auto-Adaptive Mode (`--mode auto`):**
  * Automatically discovers the natural core dataset size based on scene variance and feature novelty without requiring hardcoded percentages or counts.
* **Budget Mode (`--mode budget`):**
  * Retains an exact target number of images (`--target-count 800` / `-k 800`) or target percentage (`--target-pct 5` / `-p 5`).
* **Flexible Export Actions (`--action`):**
  * `copy`: Exports selected frames into a clean output directory for annotation.
  * `symlink`: Creates symbolic links for instant export with zero disk storage overhead.
  * `list`: Exports `selected_frames.txt` and `filter_report.json` metadata without duplicating files.
* **Automatic RGB + Depth Synchronization:**
  * Automatically detects and pairs corresponding depth maps (`depth/frame_XXXXXX_depth.npy` or `.png`) from RealSense/stereo capture sessions.

---

## 🛠 Installation & Setup with UV

### 1. Environment Installation
Run the following command inside the repository root:
```bash
uv sync
```

*(Optional: If you want to use DINOv2 or PyTorch deep backbones)*:
```bash
uv sync --extra deep
```

---

## 📖 Usage Guide

### 1. Auto-Adaptive Mode (Recommended)
Automatically analyzes visual variance and extracts the non-redundant core keyframes:

```bash
# Auto-curate and copy to output directory
uv run image-dataset-curator -i /path/to/raw_rgb -o /path/to/curated_rgb --mode auto

# Or use the short alias:
uv run img-curator -i /path/to/raw_rgb -o /path/to/curated_rgb --mode auto

# Configure auto-sensitivity: 'high' (~8-12%), 'medium' (~4-7%), 'low' (~1-3%)
uv run img-curator -i /path/to/raw_rgb -o /path/to/curated_rgb --mode auto --auto-sensitivity high
```

---

### 2. Budget Mode (Target Percentage or Fixed Count)
```bash
# Retain exactly 5% of the raw dataset (~650 images from 13k frames)
uv run img-curator -i /path/to/raw_rgb -o /path/to/curated_rgb -p 5

# Retain exactly 800 most diverse frames
uv run img-curator -i /path/to/raw_rgb -o /path/to/curated_rgb -k 800
```

---

### 3. Zero-Disk Overhead with Symlinks
Save storage space by creating symbolic links instead of copying gigabytes of images:
```bash
uv run img-curator -i /path/to/raw_rgb -o /path/to/curated_rgb --mode auto --action symlink
```

---

## 📊 CLI Parameters Reference

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `-i`, `--input-dir` | Path to raw image input directory | *(Required)* |
| `-o`, `--output-dir` | Path to destination directory | *(Required)* |
| `--mode` | Curation mode: `auto` or `budget` | `auto` |
| `--auto-sensitivity` | Sensitivity in auto mode: `high`, `medium`, `low` | `medium` |
| `-p`, `--target-pct` | Target percentage of dataset to retain (e.g. `5` for 5%) | `None` |
| `-k`, `--target-count` | Exact number of images to retain (e.g. `800`) | `None` |
| `--action` | Export action: `copy`, `symlink`, `list` | `copy` |
| `--stage1-step` | Subsampling step size for Stage 1 (e.g. `3` takes 1 in 3 frames) | `3` |
| `--stage2-metric` | Temporal similarity metric: `ssim` or `dhash` | `ssim` |
| `--stage2-ssim-thresh` | SSIM threshold for Stage 2 | `0.85` |
| `--stage3-model` | Feature representation for Stage 3: `auto`, `dinov2`, `spatial_hist` | `auto` |

---

## 🏗 Repository Structure

```
image-dataset-curator/
├── pyproject.toml              # UV project configuration and dependencies
├── uv.lock                     # Deterministic environment lockfile
├── README.md                   # Documentation and user guide
├── src/
│   └── image_dataset_curator/
│       ├── __init__.py
│       ├── cli.py              # Rich terminal CLI interface
│       ├── pipeline.py         # Multi-stage pipeline coordinator
│       ├── stages/             # Pipeline filtering stages
│       │   ├── stage1_step.py        # Stage 1: Coarse skip-frame subsampling
│       │   ├── stage2_similarity.py  # Stage 2: Sequential SSIM / dHash
│       │   └── stage3_diversity.py   # Stage 3: Global diversity & Sphere-Packing
│       ├── extractors/         # Feature extraction backbones
│       │   ├── base.py               # Abstract feature extractor interface
│       │   ├── spatial_pyramid.py    # Spatial Color/Edge Histogram (Zero-GPU)
│       │   └── deep_extractor.py     # PyTorch DINOv2 / ResNet backbone
│       └── utils/              # Metrics and I/O utilities
│           ├── io.py                 # Export & RGB+Depth synchronization
│           └── metrics.py            # SSIM and dHash computations
```
