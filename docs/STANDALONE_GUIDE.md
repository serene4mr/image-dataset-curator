# 📦 Standalone Binary User Guide (`image-dataset-curator`)

> **A complete step-by-step guide on how to download, set up, and use the standalone prebuilt binaries on Windows, Linux, and macOS without installing Python, UV, or any external dependencies.**

---

## 🎯 Who is this guide for?
* **Data Annotators & Labeling Teams:** Curate raw video datasets before importing to CVAT, Label Studio, or Roboflow.
* **Field Operators & Robotics Engineers:** Quickly deduplicate camera sessions on field laptops.
* **Users without Python:** Run the tool with zero setup, zero virtual environments, and zero dependency conflicts.

---

## 📥 1. Download the Prebuilt Binary

Visit the **[GitHub Releases Page](https://github.com/serene4mr/image-dataset-curator/releases/latest)** and download the executable matching your operating system:

| Platform | Architecture | File to Download |
| :--- | :--- | :--- |
| **Windows** | 64-bit (x86_64) | `image-dataset-curator-windows-x86_64.exe` |
| **Linux** | 64-bit (x86_64) | `image-dataset-curator-linux-x86_64` |
| **macOS** | Apple Silicon (M1/M2/M3/M4) | `image-dataset-curator-macos-arm64` |

---

## 🚀 2. OS-Specific Setup & Running Instructions

### 🪟 Windows Setup (Command Prompt or PowerShell)

1. Move the downloaded `image-dataset-curator-windows-x86_64.exe` into a convenient folder (or rename it to `img-curator.exe` for convenience).
2. Open **Command Prompt (`cmd`)** or **PowerShell**:
   * Press `Win + R`, type `cmd`, and hit Enter.
   * Or hold `Shift` + Right Click in the folder where the `.exe` is located and select **"Open in Terminal"**.
3. Run the curator on your image folder:
   ```cmd
   image-dataset-curator-windows-x86_64.exe -i "C:\Users\username\Pictures\raw_images" -o "C:\Users\username\Pictures\curated_images" --mode auto
   ```

> **Tip (Windows SmartScreen):** If Windows shows *"Windows protected your PC"* on first run, click **More info** $\rightarrow$ **Run anyway**.

---

### 🐧 Linux Setup (Ubuntu / Debian / Fedora / Arch)

1. Open a terminal in the folder where the binary was downloaded.
2. Grant execution permissions:
   ```bash
   chmod +x image-dataset-curator-linux-x86_64
   ```
3. *(Optional)* Move to system PATH to run from any directory:
   ```bash
   sudo mv image-dataset-curator-linux-x86_64 /usr/local/bin/img-curator
   ```
4. Run the curation command:
   ```bash
   # If running from current folder:
   ./image-dataset-curator-linux-x86_64 -i /path/to/raw_images -o /path/to/curated_images --mode auto

   # Or if installed to PATH:
   img-curator -i /path/to/raw_images -o /path/to/curated_images --mode auto
   ```

---

### 🍎 macOS Setup (Apple Silicon & Intel)

1. Open **Terminal** in the folder containing the downloaded binary.
2. Grant execution permissions:
   ```bash
   chmod +x image-dataset-curator-macos-arm64
   ```
3. *(If macOS blocks the binary due to Gatekeeper quarantine)*:
   ```bash
   xattr -d com.apple.quarantine image-dataset-curator-macos-arm64
   ```
   *(Or navigate to **System Settings $\rightarrow$ Privacy & Security** and click **"Allow Anyway"**)*.
4. Run the tool:
   ```bash
   ./image-dataset-curator-macos-arm64 -i /path/to/raw_images -o /path/to/curated_images --mode auto
   ```

---

## 💡 3. Common Practical Use Cases

### Case 1: Automatic Curation (Auto-Adaptive Mode)
The algorithm automatically measures visual variance across all frames and removes duplicates without needing to guess target numbers:

```bash
# Default balanced curation (~4-7% core frames retained)
./img-curator -i /data/camera_stream -o /data/curated_dataset --mode auto

# High sensitivity (~8-12% retained - keeps finer lighting & angle shifts)
./img-curator -i /data/camera_stream -o /data/curated_dataset --mode auto --auto-sensitivity high

# Low sensitivity (~1-3% retained - aggressive deduplication for distinctly new scenes only)
./img-curator -i /data/camera_stream -o /data/curated_dataset --mode auto --auto-sensitivity low
```

---

### Case 2: Curation by Fixed Count (Budget Mode)
Force the pipeline to select exactly $K$ most representative images across the visual feature space:
```bash
./img-curator -i /data/camera_stream -o /data/curated_dataset -k 500
```

---

### Case 3: Curation by Target Percentage
Keep a specific percentage of the total raw video frames:
```bash
./img-curator -i /data/camera_stream -o /data/curated_dataset -p 10
```

---

### Case 4: Zero Disk Storage Overhead (Symlinks)
Instead of copying gigabytes of images, create symbolic links pointing to the original raw files:
```bash
./img-curator -i /data/camera_stream -o /data/curated_dataset --mode auto --action symlink
```

---

## 📂 4. Understanding Output Files

When finished, the destination folder (`-o /path/to/curated_dataset`) will contain:

```
curated_dataset/
├── frame_000000.png          # High-diversity curated images directly in destination
├── frame_000015.png
├── frame_000042.png
├── ...
├── selected_frames.txt       # List of selected filenames (one per line)
└── filter_report.json        # Execution metrics and reduction statistics
```

* **Image files:** Ready to drag-and-drop into CVAT / Label Studio / Roboflow.
* **`selected_frames.txt`:** Plain text list of selected filenames for scripting with other sensor data (IMU, GPS, LiDAR).
* **`filter_report.json`:** Comprehensive JSON summary with duration, reduction percentage, and stage retention logs.

---

## 🛠 5. Building the Binary Yourself from Source

If you want to compile the standalone binary yourself on your own system:

```bash
# 1. Clone repo and install uv
git clone https://github.com/serene4mr/image-dataset-curator.git
cd image-dataset-curator
uv sync --dev

# 2. Build standalone onefile executable with PyInstaller
uv run pyinstaller \
  --name image-dataset-curator \
  --onefile \
  --collect-all image_dataset_curator \
  src/image_dataset_curator/cli.py

# 3. Your executable is ready in dist/
./dist/image-dataset-curator -i /path/to/raw_images -o /path/to/curated_images
```
