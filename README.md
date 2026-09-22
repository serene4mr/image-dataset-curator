# 🚀 Dataset Filter (Multi-Stage Dataset Deduplication & Diversity Pipeline)

> **Công cụ lọc trùng lặp và chọn lọc đa dạng hình ảnh đa tầng (Multi-Stage Data Curation Pipeline) cho bài toán Thị giác máy tính & Semantic Segmentation.**

Được quản lý hoàn toàn bằng **[uv](https://github.com/astral-sh/uv)** — Package manager nhanh nhất cho Python hiện nay.

---

## 📌 Tính Năng Nổi Bật

* **Pipeline 3 tầng tối ưu (Multi-Stage Pipeline):**
  * **Stage 1 (Coarse Step):** Lọc thô theo bước nhảy để giảm tải đọc đĩa I/O ngay lập tức.
  * **Stage 2 (Sequential SSIM / dHash):** So sánh các frame liên tiếp để loại bỏ triệt để lúc camera đứng yên hoặc di chuyển chậm.
  * **Stage 3 (Global Diversity & Novelty):** Trích xuất vector đặc trưng (DINOv2 / ResNet / Spatial Pyramid) và sử dụng thuật toán **Sphere Packing / Farthest Point Sampling** để loại bỏ trùng lặp toàn cục.
* **Chế độ Tự Động Thích Ứng (`--mode auto`):**
  * Tự động xác định số lượng frame tối ưu dựa trên biến thiên bối cảnh thực tế (không cần người dùng đoán mò con số %).
* **Chế độ Chỉ Định Ngân Sách (`--mode budget`):**
  * Cho phép chọn chính xác số lượng ảnh (`--target-count 800`) hoặc phần trăm dataset (`--target-pct 5`).
* **Linh hoạt đầu ra (`--action`):**
  * `copy`: Copy ảnh đã lọc sang thư mục mới để gán nhãn.
  * `symlink`: Tạo liên kết tượng trưng (zero-disk overhead, không tốn thêm dung lượng).
  * `list`: Chỉ xuất danh sách file `selected_frames.txt` và `filter_report.json`.

---

## 🛠 Cài Đặt & Khởi Chạy với UV

### 1. Cài đặt môi trường
Chỉ cần chạy lệnh sau trong thư mục repo:
```bash
uv sync
```

*(Tùy chọn: Nếu muốn dùng DINOv2 / PyTorch Deep Models)*:
```bash
uv sync --extra deep
```

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Chế độ Tự Động (Khuyên dùng)
App sẽ tự động phân tích và trích xuất tập frame cốt lõi không trùng lặp:

```bash
# Tự động lọc và copy sang thư mục mới
uv run dataset-filter -i /path/to/raw_rgb -o /path/to/filtered_rgb --mode auto

# Tùy chỉnh độ nhạy: 'high' (~8-12%), 'medium' (~4-7%), 'low' (~1-3%)
uv run dataset-filter -i /path/to/raw_rgb -o /path/to/filtered_rgb --mode auto --auto-sensitivity high
```

---

### 2. Chế độ Chỉ Định Mục Tiêu (% hoặc Số Lượng Cố Định)
```bash
# Lọc lấy chính xác 5% dataset (~650 ảnh)
uv run dataset-filter -i /path/to/raw_rgb -o /path/to/filtered_rgb -p 5

# Lọc lấy đúng 800 ảnh đa dạng nhất
uv run dataset-filter -i /path/to/raw_rgb -o /path/to/filtered_rgb -k 800
```

---

### 3. Tiết Kiệm Dung Lượng Ổ Cứng với Symlink
Thay vì nhân bản hàng GB ảnh, tạo symlink liên kết đến ảnh gốc:
```bash
uv run dataset-filter -i /path/to/raw_rgb -o /path/to/filtered_rgb --mode auto --action symlink
```

---

## 📊 Bảng Tham Số CLI

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `-i`, `--input-dir` | Đường dẫn thư mục ảnh đầu vào | *(Bắt buộc)* |
| `-o`, `--output-dir` | Đường dẫn thư mục xuất kết quả | *(Bắt buộc)* |
| `--mode` | Chế độ lọc: `auto` hoặc `budget` | `auto` |
| `--auto-sensitivity` | Độ nhạy chế độ auto: `high`, `medium`, `low` | `medium` |
| `-p`, `--target-pct` | Phần trăm ảnh cần giữ lại (vd: `5` cho 5%) | `None` |
| `-k`, `--target-count` | Số lượng ảnh chính xác cần giữ lại (vd: `800`) | `None` |
| `--action` | Cách xuất ảnh: `copy`, `symlink`, `list` | `copy` |
| `--stage1-step` | Bước nhảy Stage 1 (vd: `3` là lấy 1 trong 3 frame) | `3` |
| `--stage2-metric` | Thuật toán Stage 2: `ssim` hoặc `dhash` | `ssim` |
| `--stage3-model` | Mô hình trích xuất Stage 3: `auto`, `dinov2`, `spatial_hist` | `auto` |

---

## 🏗 Cấu Trúc Dự Án

```
dataset-filter/
├── pyproject.toml              # Cấu hình dự án & dependencies (UV)
├── README.md                   # Tài liệu hướng dẫn
├── src/
│   └── dataset_filter/
│       ├── __init__.py
│       ├── cli.py              # Giao diện dòng lệnh Rich CLI
│       ├── pipeline.py         # Bộ điều phối Pipeline 3 tầng
│       ├── stages/             # Các tầng lọc dữ liệu
│       │   ├── stage1_step.py
│       │   ├── stage2_similarity.py
│       │   └── stage3_diversity.py
│       ├── extractors/         # Trích xuất vector đặc trưng
│       │   ├── base.py
│       │   ├── spatial_pyramid.py
│       │   └── deep_extractor.py
│       └── utils/              # Tiện ích đo lường & I/O
│           ├── io.py
│           └── metrics.py
```
