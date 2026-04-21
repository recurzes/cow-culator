# Cow-culator

**Cow-culator** is a high-efficiency frame extraction utility designed for the **HERD-VISION** cattle gait analysis pipeline. It precisely calculates and extracts the optimal number of frames from raw video data to ensure high-quality training sets for YOLOv8 Pose models.

## Core Functionalities

* **Automated Environment Provisioning:** Automatically detects system-level dependencies. If `ffmpeg` is missing, the script identifies the host OS (Arch, Debian/Ubuntu, macOS, or Windows) and handles the installation process.
* **Temporal Sampling:** Samples video at **2 frames per second (FPS)**. This specific frequency is optimized for bovine gait analysis, capturing distinct stride phases while preventing dataset redundancy and model overfitting.
* **Automated Namespace Management:** Iterates through a target directory and organizes output into dedicated sub-folders named after the source video, ensuring a structured workflow for CVAT or Roboflow imports.
* **Vision-Grade Output:** Exports frames with a high-quality JPEG scale to preserve critical anatomical landmarks such as the hock, stifle, and spine curvature.

## Installation

1. Clone the repository and navigate to the project root.
2. Initialize your environment:
   ```bash
   pip install -e .
   ```

## Usage

Execute the script by passing the directory containing your raw video clips as an argument:
`python cow_culator.py ./path/to/video_data`

## Dataset Structure

The script transforms raw video files into an organized image hierarchy:
```
video_data/
├── cow_001_sideview.mp4
├── cow_001_sideview/       <-- Generated Dataset
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
└── cow_002_backview.mp4
```
