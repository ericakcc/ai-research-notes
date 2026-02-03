# Computer Vision 實作指南

按順序跟著做，每個步驟都有明確的「完成標準」。

---

## 環境準備

```bash
# 1. 在專案根目錄同步環境
make setup

# 2. 安裝 CV 相關套件
uv add torch torchvision
uv add ultralytics          # YOLOv5/v8/v11/YOLO26 統一框架
uv add anomalib             # PatchCore 等異常偵測
uv add opencv-python-headless matplotlib seaborn
uv add --dev jupyter

# 3. 下載所有論文 (13 篇 PDF)
cd topics/computer-vision
make papers
```

---

## Phase 1: YOLO From Scratch

**目標**：用純 PyTorch 寫出 YOLOv1，理解 single-stage detection 的核心機制。

**工作目錄**：`experiments/01_yolo_from_scratch/`

### 資料集

```python
# Pascal VOC 2007 — torchvision 內建，不需手動下載
import torchvision
dataset = torchvision.datasets.VOCDetection(
    root="../../datasets", year="2007", download=True
)
```

### Step 1-1: YOLOv1 模型

在 `src/` 下建立以下檔案：

| 檔案 | 內容 |
|------|------|
| `model.py` | YOLOv1 網路架構（24 conv layers + 2 FC layers） |
| `loss.py` | YOLO loss = λ_coord × 定位損失 + 信心損失 + 分類損失 |
| `utils.py` | IoU 計算、NMS、mAP 計算、bbox 轉換 |
| `dataset.py` | VOC 資料載入 + grid cell 標籤編碼 |

**關鍵實作重點**：
- 輸入：448×448 影像
- 輸出：S×S×(B×5+C) tensor，其中 S=7, B=2, C=20 → 7×7×30
- 每個 cell 預測 B 個 bbox (x, y, w, h, confidence) + C 個 class probabilities
- Loss 中 `λ_coord=5`, `λ_noobj=0.5` 平衡正負樣本

### Step 1-2: 訓練與視覺化

```python
# 在 notebooks/ 下建立 01_yolov1_training.ipynb
# 訓練 ~50 epochs on VOC 2007
# 目標 mAP@50 不需要很高，重點是跑通整個 pipeline
```

### Step 1-3: YOLOv2 改進

在 v1 基礎上加入：
1. 所有 conv 層後加 BatchNorm
2. 用 k-means 在 VOC 上聚類出 5 個 anchor boxes
3. Passthrough layer（高解析度特徵拼接）

**完成標準**：
- [ ] YOLOv1 能在 VOC 上跑出偵測結果（視覺化 bbox 到圖上）
- [ ] YOLOv2 加入 anchor boxes 後 recall 有明顯提升
- [ ] 能口述解釋 grid-based detection + loss function 設計

**參考論文**：`papers/01_foundations/` 下的 YOLOv1、YOLOv2

---

## Phase 2: Backbone & Neck 演化

**目標**：理解從 Darknet-19 → CSPDarknet-53 的演化，以及 FPN/PANet 多尺度偵測。

**工作目錄**：`experiments/02_yolo_darknet_fpn/`

### Step 2-1: Darknet-53 Backbone

```python
# src/backbone.py
# 實作 Darknet-53：52 conv + 1 FC
# 核心：residual block (1×1 conv → 3×3 conv + shortcut)
# 與 Phase 1 的 Darknet-19 對比：更深但有殘差連結不會梯度消失
```

### Step 2-2: FPN + PANet Neck

```python
# src/neck.py
# FPN：top-down pathway（大特徵圖 + 上採樣的小特徵圖）
# PANet：再加 bottom-up pathway（雙向特徵融合）
# 輸出 3 個尺度的特徵圖：大物件 / 中物件 / 小物件
```

### Step 2-3: CSP 結構 + 資料增強

```python
# src/csp.py — Cross Stage Partial connection
# 將特徵圖分兩半：一半走 dense block，一半直接 concat
# 減少計算量同時維持梯度流

# src/augmentation.py — Mosaic augmentation
# 4 張圖拼成 1 張，一次看到 4 種場景 + 4 倍 bbox 密度
```

### Step 2-4: Ultralytics YOLOv5 Baseline

```python
from ultralytics import YOLO
model = YOLO("yolov5n.pt")
results = model.train(data="coco128.yaml", epochs=50)
# COCO128 內建於 ultralytics，自動下載
# 用這個當 baseline 跟自己寫的 Darknet-53+FPN 做對比
```

**完成標準**：
- [ ] Darknet-53 + FPN 能產出 3 個尺度的特徵圖
- [ ] Mosaic augmentation 實作完成，視覺化 4-in-1 拼接結果
- [ ] 能口述 FPN vs PANet 差異、CSP 為什麼能減少計算量

**參考論文**：`papers/02_maturity/` 下的 YOLOv3、YOLOv4

---

## Phase 3: 現代 YOLO + 工業缺陷偵測

**目標**：用 YOLOv8 在真實工業資料上訓練，並完成模型導出流程。

**工作目錄**：`experiments/03_modern_yolo_training/`

### 資料集

```bash
# NEU Surface Defect Dataset (~20MB, 6 類鋼材缺陷)
# 下載連結：http://faculty.neu.edu.cn/songkechen/en/zdylm/263265/list/
# 放到 datasets/NEU-DET/
# 6 類：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
```

### Step 3-1: YOLOv8 訓練

```python
from ultralytics import YOLO

# 載入預訓練模型
model = YOLO("yolov8n.pt")

# 訓練 — 需要先準備 NEU dataset YAML 設定檔
results = model.train(
    data="neu-det.yaml",  # 自己寫的 dataset config
    epochs=100,
    imgsz=640,
    batch=16,
)

# 比較不同 model size
for size in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
    model = YOLO(size)
    model.train(data="neu-det.yaml", epochs=100)
```

### Step 3-2: Anchor-Free 理解

在 notebook 中分析：
- YOLOv8 的 TaskAlignedAssigner 如何分配正負樣本
- 對比 v5（anchor-based）vs v8（anchor-free）在 NEU 上的表現
- Decoupled head 分離分類/回歸的效果

### Step 3-3: 模型導出

```python
model = YOLO("runs/detect/train/weights/best.pt")

# PyTorch → ONNX
model.export(format="onnx")

# ONNX → TensorRT (需要 NVIDIA GPU)
model.export(format="engine", half=True)  # FP16

# Benchmark
from ultralytics.utils.benchmarks import benchmark
benchmark(model="best.pt", data="neu-det.yaml", imgsz=640)
```

**完成標準**：
- [ ] YOLOv8 在 NEU 上 mAP@50 > 0.7
- [ ] 完成 n/s/m 三種 size 的 speed-accuracy tradeoff 比較表
- [ ] 成功導出 ONNX 模型並量測推理速度
- [ ] 能口述 anchor-free vs anchor-based 的差異

**參考論文**：`papers/03_industrial/` 下的 YOLOv6、YOLOv7

---

## Phase 4: 最新 YOLO 跨版本評估

**目標**：跑 v8 ~ YOLO26 的 benchmark，建立完整比較表。

**工作目錄**：`experiments/04_yolo_latest_eval/`

### Step 4-1: 各版本快速評估

```python
from ultralytics import YOLO

# Ultralytics 支援的版本可以直接用
versions = {
    "v8n": "yolov8n.pt",
    "v11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}

for name, weights in versions.items():
    model = YOLO(weights)
    metrics = model.val(data="neu-det.yaml")
    print(f"{name}: mAP50={metrics.box.map50:.3f}, mAP50-95={metrics.box.map:.3f}")
```

```python
# YOLOv9 — 需要用官方 repo
# git clone https://github.com/WongKinYiu/yolov9

# YOLOv10 — 需要用官方 repo
# git clone https://github.com/THU-MIG/yolov10

# YOLOv12 — 需要用官方 repo
# git clone https://github.com/sunsmarterjie/yolov12
```

### Step 4-2: 跨版本 Benchmark 表

在 notebook 中建立統一評估：

```python
# 在 NEU 或 MVTec AD 子集上統一測試
# 記錄：mAP@50, mAP@50:95, 推理延遲 (ms), 參數量 (M), FLOPs (G)
# 填入 notes/yolo_evolution_comparison.md 的 Performance Trends 表格
```

### Step 4-3: 關鍵技術筆記

閱讀 + 在 `notes/` 對應目錄寫筆記：
- YOLOv9：PGI 如何解決資訊瓶頸
- YOLOv10：NMS-free 的 consistent dual assignments
- YOLOv12：Area Attention (A2) 為何能在偵測中用 attention
- YOLO26：MuSGD optimizer、native NMS-free

**完成標準**：
- [ ] 跨版本 benchmark 表完成（至少 v8/v11/YOLO26 三個版本）
- [ ] 能口述每個版本的核心創新（一句話總結）
- [ ] `notes/yolo_evolution_comparison.md` 的 Performance Trends 填完

**參考論文**：`papers/04_frontier/` 下的 YOLOv9、YOLOv10、YOLOv12

---

## Phase 5: 工業缺陷偵測 Pipeline

**目標**：組合 YOLO + U-Net + PatchCore 建立完整的缺陷偵測系統。

**工作目錄**：`experiments/05_defect_detection_pipeline/`

### 資料集

```bash
# MVTec AD (~5GB) — 15 類工業異常偵測資料
# 下載：https://www.mvtec.com/company/research/datasets/mvtec-ad
# 放到 datasets/MVTec-AD/

# DAGM 2007 (~500MB) — 合成工業紋理
# 下載：https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection
# 放到 datasets/DAGM-2007/
```

### Step 5-1: U-Net From Scratch

```python
# src/unet.py
# Encoder: 4 次 (conv3×3 + conv3×3 + MaxPool2×2)，通道 64→128→256→512
# Bottleneck: conv3×3 + conv3×3，通道 1024
# Decoder: 4 次 (UpConv2×2 + concat skip + conv3×3 + conv3×3)
# 最後 1×1 conv 輸出 segmentation mask

# src/dice_loss.py
# Dice Loss = 1 - (2 * intersection) / (union)
# 比 CrossEntropy 更適合前景/背景不平衡的分割任務
```

訓練目標：
- 用 MVTec AD 的 texture 類別（carpet, leather, grid）
- 輸入：defect image → 輸出：pixel-level defect mask
- Dice coefficient > 0.8

### Step 5-2: PatchCore 異常偵測

```python
# 用 anomalib 框架
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTec(category="bottle", image_size=(256, 256))
model = Patchcore()
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# PatchCore 只需要正常樣本訓練（無監督）
# 用 pre-trained backbone 提取 patch features → memory bank → 推理時比對距離

# 跑完 15 個類別，記錄 AUROC
for category in ["bottle", "cable", "capsule", "carpet", "grid",
                  "hazelnut", "leather", "metal_nut", "pill", "screw",
                  "tile", "toothbrush", "transistor", "wood", "zipper"]:
    datamodule = MVTec(category=category, image_size=(256, 256))
    engine.fit(model=Patchcore(), datamodule=datamodule)
```

### Step 5-3: Hybrid Pipeline

組合三個模型的完整流程：

```
輸入影像
  │
  ├─→ PatchCore 異常偵測（粗篩：正常 or 異常？）
  │     │
  │     └─→ 異常 → YOLO 缺陷分類（這是什麼缺陷？）
  │                  │
  │                  └─→ U-Net 分割（缺陷的精確輪廓和面積）
  │
  └─→ 正常 → Pass
```

```python
# src/pipeline.py
# 1. PatchCore.predict(image) → anomaly_score
# 2. if anomaly_score > threshold:
#        yolo_results = YOLO.predict(image) → defect class + bbox
#        for bbox in yolo_results:
#            crop = image[bbox]
#            mask = UNet.predict(crop) → pixel-level segmentation
#            area = mask.sum() * pixel_size  → 缺陷面積 (mm²)
```

### Step 5-4: 部署優化

```python
# FP16/INT8 量化
model.export(format="engine", half=True)   # FP16
model.export(format="engine", int8=True)   # INT8

# 吞吐量測試
import time
images = load_test_images(100)
start = time.perf_counter()
for img in images:
    pipeline.predict(img)
elapsed = time.perf_counter() - start
print(f"Throughput: {100/elapsed:.1f} images/sec")
```

**完成標準**：
- [ ] U-Net 在 MVTec texture 類別上 Dice > 0.8
- [ ] PatchCore 在 MVTec 15 類的平均 AUROC > 0.95
- [ ] Hybrid pipeline 能跑通：輸入一張圖 → 輸出缺陷類型 + 分割 mask + 面積
- [ ] 量化後推理速度有明顯提升（記錄 FP32 vs FP16 對比）
- [ ] 能口述為何用 hybrid pipeline（PatchCore 無監督粗篩 + YOLO 有監督分類 + U-Net 精確分割）

**參考論文**：`papers/05_defect_detection/` 下的 U-Net、MVTec AD、PatchCore、Glass Defect YOLO

---

## 快速回想路線（如果時間緊迫）

如果這週只能集中火力，建議優先做這三件事：

1. **Phase 1 的 Step 1-1 只做 loss + NMS**（~半天）— 確保能手寫 IoU、NMS、YOLO loss
2. **Phase 3 完整做完**（~1-2 天）— YOLOv8 + NEU 缺陷訓練 + 導出，最直接的面試素材
3. **Phase 5 的 Step 5-1 + 5-2**（~1-2 天）— U-Net from scratch + PatchCore 跑 MVTec

這三塊涵蓋：手寫能力 + 框架使用 + 工業場景，面試最常問的三個維度。
