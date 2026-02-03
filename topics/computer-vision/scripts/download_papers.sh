#!/bin/bash
# Download all research papers from arXiv

set -e

echo "Downloading Computer Vision papers..."

# Phase 1: Foundations
curl -L "https://arxiv.org/pdf/1506.02640.pdf" -o papers/01_foundations/yolov1_2016.pdf
echo "✓ YOLOv1 (2016)"

curl -L "https://arxiv.org/pdf/1612.08242.pdf" -o papers/01_foundations/yolov2_2017.pdf
echo "✓ YOLOv2 (2017)"

# Phase 2: Maturity
curl -L "https://arxiv.org/pdf/1804.02767.pdf" -o papers/02_maturity/yolov3_2018.pdf
echo "✓ YOLOv3 (2018)"

curl -L "https://arxiv.org/pdf/2004.10934.pdf" -o papers/02_maturity/yolov4_2020.pdf
echo "✓ YOLOv4 (2020)"

# Phase 3: Industrial
curl -L "https://arxiv.org/pdf/2209.02976.pdf" -o papers/03_industrial/yolov6_2022.pdf
echo "✓ YOLOv6 (2022)"

curl -L "https://arxiv.org/pdf/2207.02696.pdf" -o papers/03_industrial/yolov7_2022.pdf
echo "✓ YOLOv7 (2022)"

# Phase 4: Frontier
curl -L "https://arxiv.org/pdf/2402.13616.pdf" -o papers/04_frontier/yolov9_2024.pdf
echo "✓ YOLOv9 (2024)"

curl -L "https://arxiv.org/pdf/2405.14458.pdf" -o papers/04_frontier/yolov10_2024.pdf
echo "✓ YOLOv10 (2024)"

curl -L "https://arxiv.org/pdf/2502.12524.pdf" -o papers/04_frontier/yolov12_2025.pdf
echo "✓ YOLOv12 (2025)"

# Phase 5: Defect Detection
curl -L "https://arxiv.org/pdf/1505.04597.pdf" -o papers/05_defect_detection/unet_2015.pdf
echo "✓ U-Net (2015)"

curl -L "https://arxiv.org/pdf/1907.08218.pdf" -o papers/05_defect_detection/mvtec_ad_2019.pdf
echo "✓ MVTec AD (2019)"

curl -L "https://arxiv.org/pdf/2106.08265.pdf" -o papers/05_defect_detection/patchcore_2022.pdf
echo "✓ PatchCore (2022)"

curl -L "https://arxiv.org/pdf/2502.07592.pdf" -o papers/05_defect_detection/glass_defect_yolo.pdf
echo "✓ Glass Defect YOLO (2025)"

echo ""
echo "All papers downloaded successfully! (13 PDFs)"
echo "Note: YOLOv5, YOLOv8, YOLOv11, YOLO26 have no official papers — see notes files."
