# Paper Summary: YOLO9000: Better, Faster, Stronger (YOLOv2)

**Authors**: Joseph Redmon, Ali Farhadi
**arXiv**: [1612.08242](https://arxiv.org/abs/1612.08242)
**Year**: 2016 (CVPR 2017)

## Core Contribution

三大主軸：**(1) Better** — 系統性改進 YOLOv1 的精度（+15 mAP）；**(2) Faster** — 提出 Darknet-19 輕量骨幹（5.58B FLOPs vs VGG-16 30.69B）；**(3) Stronger** — 提出 WordTree 層級分類，聯合訓練實現 YOLO9000 可偵測 9000+ 類別。

## Method — Better（YOLOv2 改進清單）

### 逐步改進及 mAP 變化（VOC 2007）

| 改進 | 累積 mAP | 增量 |
|------|---------|------|
| YOLOv1 baseline | 63.4% | — |
| + Batch Normalization | 65.8% | +2.4 |
| + High Resolution Classifier | 69.5% | +3.7 |
| + Convolutional（移除 FC） | 69.5% | +0 |
| + Anchor Boxes | 69.2% | -0.3（recall 81%→88%）|
| + New Network (Darknet-19) | 69.6% | +0.4 |
| + Dimension Clusters | — | — |
| + Direct Location Prediction | 74.4% | +4.8 |
| + Passthrough（fine-grained features）| 75.4% | +1.0 |
| + Multi-Scale Training | 76.8% | +1.4 |
| + Hi-Res Detector (544×544) | **78.6%** | +1.8 |

注：切換到全卷積 + anchor boxes 降低了 mAP 但增加了 recall（+7%）；新 backbone 減少 33% 計算量。

### 關鍵改進詳解

**1. Batch Normalization（+2% mAP）**
- 所有 conv 層加 BN → 可移除 dropout，不會過擬合
- 同時加速收斂 + 正則化

**2. High Resolution Classifier（+4% mAP）**
- YOLOv1：ImageNet 預訓練 224×224 → 偵測直接跳 448×448（同時切換任務 + 解析度）
- YOLOv2：先在 **448×448** 上 fine-tune 分類 **10 epochs**（LR=10⁻³）→ 再訓練偵測
- 讓網路先適應高解析度 filters，避免同時學兩件事
- Fine-tune 後分類精度：top-1 76.5%, top-5 93.3%

**3. Anchor Boxes（mAP 微降 -0.3%，recall 大增 +7%）**
- 移除 FC 層，改為全卷積 + anchor boxes
- 輸入從 448→**416**（確保 feature map 為奇數 **13×13**，中心 cell 在正中央）
  - 為何要奇數？大物件常佔據影像中心，中心 cell 可直接負責，而非 4 個相鄰 cell
  - 416 = 13 × 32（下採樣倍數）
- 預測量從 98 boxes → **1000+ boxes**
- 解耦 class prediction 和 spatial location：每個 anchor box 獨立預測 class + objectness

**4. Dimension Clusters（核心創新）**
- 不用手工 anchor，改用 **k-means clustering**（k=5）在訓練集 bounding boxes 上找 priors
- 距離度量：`d(box, centroid) = 1 - IOU(box, centroid)`
  - 不用歐氏距離，因為歐氏距離偏向大 box（大 box 絕對距離天然更大）
  - IOU 是 **scale-invariant**，大小 box 同等對待

| Box Generation | # Priors | Avg IOU |
|---------------|----------|---------|
| Cluster SSE | 5 | 58.7 |
| **Cluster IOU** | **5** | **61.0** |
| Anchor Boxes (Faster R-CNN) | 9 | 60.9 |
| **Cluster IOU** | **9** | **67.2** |

- 5 個 IOU-based clusters 就匹敵 9 個手工 anchors
- VOC 和 COCO 的 centroids 都偏 **thinner/taller**，COCO 有更大 size variation

**5. Direct Location Prediction（+5% mAP，最大單項改進）**
- **問題**：RPN 的 offset 公式 `x = (tx * wa) - xa` 不受約束 → 任何 anchor 可跑到影像任何位置 → 訓練初期極度不穩定
- **解法**：預測相對於 grid cell 的 **sigmoid 偏移**，約束中心在 cell 內
  - `bx = σ(tx) + cx`（cx: cell 左上角 x 偏移）
  - `by = σ(ty) + cy`
  - `bw = pw × e^tw`（pw: anchor prior 寬度）
  - `bh = ph × e^th`
  - `Pr(object) × IOU(b, object) = σ(to)`（objectness 也用 sigmoid）
- σ() 約束到 0~1 → parametrization 更容易學習，訓練穩定

**6. Passthrough Layer（+1% mAP）**
- 13×13 feature map 對大物件足夠，但小物件需要 finer grained features
- **Stacking 機制**：將 26×26×512 的早期 feature map 每 2×2 spatial 區域 stack 到 channel
  - 26×26×512 → **(26/2)×(26/2)×(512×4)** = 13×13×2048
  - 類似 ResNet 的 identity mapping 概念
- 與原始 13×13 feature map **拼接**（concatenate） → detector 在擴展的 feature map 上運行
- 意義：**FPN（Feature Pyramid Network）的前身概念**

**7. Multi-Scale Training**
- 每 **10 batches** 隨機切換輸入尺寸：{320, 352, 384, 416, 448, 480, 512, 544, 576, 608}
- 全部是 32 的倍數（下採樣倍數），確保 feature map 為整數
- 同一模型、同一權重可在不同解析度推理 → speed/accuracy tradeoff
- 288×288: 90+ FPS | 416×416: 67 FPS, 76.8 mAP | 544×544: 40 FPS, 78.6 mAP

## Method — Faster（Darknet-19）

### 完整架構（Table 6）

| Layer | Filters | Size/Stride | Output |
|-------|---------|-------------|--------|
| Conv | 32 | 3×3 | 224×224 |
| Maxpool | — | 2×2/2 | 112×112 |
| Conv | 64 | 3×3 | 112×112 |
| Maxpool | — | 2×2/2 | 56×56 |
| Conv | 128 | 3×3 | 56×56 |
| Conv | 64 | **1×1** | 56×56 |
| Conv | 128 | 3×3 | 56×56 |
| Maxpool | — | 2×2/2 | 28×28 |
| Conv | 256 | 3×3 | 28×28 |
| Conv | 128 | **1×1** | 28×28 |
| Conv | 256 | 3×3 | 28×28 |
| Maxpool | — | 2×2/2 | 14×14 |
| Conv | 512 | 3×3 | 14×14 |
| Conv | 256 | **1×1** | 14×14 |
| Conv | 512 | 3×3 | 14×14 |
| Conv | 256 | **1×1** | 14×14 |
| Conv | 512 | 3×3 | 14×14 |
| Maxpool | — | 2×2/2 | 7×7 |
| Conv | 1024 | 3×3 | 7×7 |
| Conv | 512 | **1×1** | 7×7 |
| Conv | 1024 | 3×3 | 7×7 |
| Conv | 512 | **1×1** | 7×7 |
| Conv | 1024 | 3×3 | 7×7 |
| Conv | 1000 | 1×1 | 7×7 |
| Avgpool | — | Global | 1000 |
| Softmax | — | — | — |

**設計特點**：
- 通道倍增：每次 maxpool 後通道數 ×2（32→64→128→256→512→1024）
- 1×1 bottleneck：壓縮通道數後再用 3×3 擴展（減少計算量，來自 NIN 思想）
- Global average pooling：取代 FC 層（來自 NIN/GoogLeNet）
- **5.58B FLOPs**（VGG-16: 30.69B, YOLOv1: 8.52B）
- ImageNet top-1: 72.9%, top-5: 91.2%

### 偵測適配
- 移除最後 **1×1 conv + global avgpool + softmax**
- 加入 **3 個 3×3×1024 conv** + **1 個 1×1 conv**（輸出 = 5 boxes × (5 coords + 20 classes) = 125 filters for VOC）
- Passthrough layer 連接 final 3×3×512 層 → second-to-last conv 層

### Training Hyperparameters

**Classification training**:
- 160 epochs, SGD, starting LR=0.1, polynomial decay (power=4)
- Weight decay: 0.0005, momentum: 0.9
- Data augmentation: random crops, rotations, hue/saturation/exposure shifts
- 448 fine-tune: 10 epochs, LR=10⁻³

**Detection training**:
- 160 epochs, starting LR=10⁻³
- LR schedule: divide by 10 at **epoch 60** and **epoch 90**
- Weight decay: 0.0005, momentum: 0.9
- Data augmentation: random crops, color shifting（同 YOLO/SSD）

## Method — Stronger（YOLO9000 & WordTree）

### WordTree 層級分類

**構建演算法**：
1. 找出 ImageNet visual nouns 在 WordNet DAG 中到根節點 ("physical object") 的所有路徑
2. 先加入**只有一條路徑**的 synsets
3. 逐步處理剩餘概念，選擇**使樹增長最少**的路徑（shortest path heuristic）
4. 最終 WordTree1k：1000 類 → **1369 節點**

**推理方式**：
- 每個節點預測其子節點的**條件機率**，用 **sibling softmax**（同一父節點下的子節點互斥）
- 絕對機率 = 沿路徑相乘：`Pr(Norfolk terrier) = Pr(Norfolk terrier|terrier) × Pr(terrier|hunting dog) × ... × Pr(animal|physical object)`
- 偵測時：用 objectness predictor 給出 `Pr(physical object)` 的值，沿樹往下走到信心低於閾值為止
- 精度僅掉 ~1%（71.9% vs 72.9% top-1）
- **優雅降級**：不確定具體品種時仍能高信心預測 "dog"

### Joint Training（Detection + Classification）

**Dataset 構建**：
- COCO 偵測 + ImageNet top 9000 分類 + ImageNet detection challenge 額外類別
- 最終 WordTree：**9418 classes**
- COCO oversampling，使 ImageNet 僅比 COCO 大 **4:1**

**架構調整**：YOLOv2 base，但只用 **3 priors**（非 5）以限制輸出大小

**反傳策略**：
- **偵測圖片**：完整 YOLOv2 loss 反傳
- **分類圖片**：
  1. 找到預測該類別**最高機率**的 bounding box
  2. 假設該 box 與 GT 重疊 **≥ 0.3 IOU** → 反傳 objectness loss
  3. 只反傳標籤**對應層級及以上**的分類 loss（若標籤是 "dog"，不懲罰 "German Shepherd" vs "Golden Retriever" 的區分）

### YOLO9000 Results

- 整體 mAP：**19.7%**（ImageNet 200 類）
- 156 個從未見過偵測標註的類別：**16.0 mAP**（優於 DPM）

| Best Classes | AP | Worst Classes | AP |
|-------------|-----|--------------|-----|
| armadillo | 61.7 | diaper | 0.0 |
| tiger | 61.0 | horizontal bar | 0.0 |
| koala bear | 54.3 | rubber eraser | 0.0 |
| fox | 52.1 | sunglasses | 0.0 |
| red panda | 50.7 | swimming trunks | 0.0 |

- **動物表現好**：COCO 的動物 objectness 泛化良好
- **衣物/器材表現差**：COCO 只有 "person" bbox，沒有衣物/配件的標註

## Key Results

### VOC 2007

| Model | mAP | FPS |
|-------|-----|-----|
| Fast R-CNN | 70.0% | 0.5 |
| Faster R-CNN VGG-16 | 73.2% | 7 |
| Faster R-CNN ResNet | 76.4% | 5 |
| YOLO | 63.4% | 45 |
| SSD300 | 74.3% | 46 |
| SSD512 | 76.8% | 19 |
| YOLOv2 288×288 | 69.0% | 91 |
| YOLOv2 416×416 | 76.8% | 67 |
| **YOLOv2 544×544** | **78.6%** | 40 |

每個 YOLOv2 entry 是**同一模型同一權重**，僅改變推理解析度。

### VOC 2012
- YOLOv2 544: 73.4 mAP（SSD512: 74.9, ResNet: 73.8 — 相當水平，但 YOLOv2 快 2-10×）
- 弱項：bottle (51.8), plant (49.1), chair (52.1)（小物件）
- 強項：cat (90.6), dog (89.3), horse (82.5)

### COCO test-dev 2015

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AP_S | AP_M | AP_L |
|-------|-------------|---------|----------|------|------|------|
| SSD512 | **26.8** | **46.5** | **27.8** | **9.0** | **28.9** | **41.9** |
| YOLOv2 | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| Faster R-CNN | 24.2 | 45.3 | 23.5 | 7.7 | 26.4 | 37.1 |

- **小物件 (AP_S=5.0)** 顯著弱於 SSD512 (9.0)
- **嚴格 IOU (mAP@0.75=19.2)** 落後 SSD512 (27.8) → **精確定位仍有差距**
- 在 mAP@0.5 上與 SSD/Faster R-CNN 相當 (44.0 vs 46.5/45.3)

## Limitations

- COCO 上精確定位 (mAP@0.75) 仍顯著落後 SSD512
- 小物件偵測 (AP_S=5.0) 不如 SSD 的多尺度 feature maps 方案
- YOLO9000 對衣物/器材等類別效果差（COCO 缺乏相關 bbox 標註）
- Passthrough layer 僅融合 1 個額外尺度，不如 FPN 的完整金字塔

## YOLOv1 → v2 核心差異速查

| 特性 | YOLOv1 | YOLOv2 |
|------|--------|--------|
| Backbone | 24 conv + 2 FC | Darknet-19（19 conv, 全 conv） |
| BN | 無 | 全部 conv 層 |
| Anchor | 無（直接回歸） | Dimension Clusters (k=5) |
| Box 預測 | 直接座標 (x,y,w,h) | σ(tx)+cx, pw×e^tw |
| 解析度 | 固定 448 | Multi-scale {320..608} |
| Feature fusion | 無 | Passthrough layer (26×26→13×13) |
| 類別數 | ~20 (VOC) | 9000+ (WordTree) |
| Boxes/image | 98 (7×7×2) | 845 (13×13×5) |
| Recall | 81% | 88% |
| mAP (VOC07) | 63.4% | 78.6% |
| FPS | 45 | 40~91 |

## Relevance（面試重點）

- **「v2 相比 v1 最重要的改進是什麼？」**：Direct Location Prediction (+5% mAP) 解決 v1 最大弱點 localization error。Batch Norm + High-Res Classifier + Multi-Scale Training 合計 +15 mAP
- **「為什麼用 k-means 而不是手工 anchor？」**：data-driven priors 更符合實際資料分佈，5 個 IOU-based cluster 就匹敵 9 個手工 anchor。泛化到新領域（如瑕疵檢測）時需重新跑 k-means
- **「Direct location prediction 解決了什麼問題？」**：RPN 的 unconstrained offset 讓 anchor 可跑到任意位置，訓練不穩定。sigmoid 約束到 cell 範圍內 (0~1) 更易學習
- **「Passthrough layer 的作用？」**：將高解析度 feature（26×26）與低解析度 feature（13×13）融合，改善小物件偵測。是 FPN (v3) 的前身概念
- **「WordTree 是什麼？」**：將 WordNet DAG 簡化為樹，每層做 sibling softmax，聯合 detection + classification 訓練。可在未見過 bbox 標註的類別上做偵測
- **「v2 的 recall 上限 vs v1？」**：v1 僅 98 boxes (7²×2)，v2 有 845 boxes (13²×5)。recall 從 81% 提升到 88%
- **「如何改進 v2 的 loss function？」**：(1) Focal Loss 取代 SSE for classification（處理不平衡），(2) GIoU/DIoU/CIoU Loss 取代 MSE for bbox（直接優化偵測指標），(3) 加 auxiliary segmentation task 提升表徵。這些在 v4+ 逐步引入
- **「YOLO 如何處理遮擋？」**：全局推理有助於部分遮擋，但嚴重遮擋仍會失敗（IOU < 0.5 被 NMS 移除）

> **→ 前置閱讀**：YOLOv1 基礎，見 [yolov1_summary.md](./yolov1_summary.md)
> **→ 演化分析**：v1→v2 因果關係 + 技術種子追蹤，見 [yolo_v1v2_evolution.md](./yolo_v1v2_evolution.md)
