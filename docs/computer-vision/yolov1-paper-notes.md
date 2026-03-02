# Paper Summary: You Only Look Once (YOLOv1)

**Authors**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
**arXiv**: [1506.02640](https://arxiv.org/abs/1506.02640)
**Year**: 2016 (CVPR 2016)

## Core Contribution

將物件偵測重新定義為**單一回歸問題**（single regression problem），從影像像素直接預測 bounding box 座標和類別機率。一個神經網路、一次前向傳播完成所有偵測，打破了傳統多階段 pipeline（如 R-CNN: region proposal → classification → post-processing）的範式。

## Method

### Grid-based Detection（核心機制）

- 將輸入影像切為 **S × S grid**（S=7）
- 每個 grid cell 預測 **B 個 bounding boxes**（B=2）+ **confidence scores** + **C 個類別條件機率**（C=20 for VOC）
- 每個 bounding box 包含 5 個預測值：`(x, y, w, h, confidence)`
    - `(x, y)`: box 中心相對於 grid cell 的偏移（bounded 0~1）
    - `(w, h)`: 相對於整張影像的寬高（normalized 0~1）
    - `confidence = Pr(Object) × IOU_pred^truth`（無物件時為 0，有物件時等於 predicted box 與 GT 的 IOU）
- 每個 grid cell 只預測**一組**類別條件機率 `Pr(Class_i|Object)`，與 B 個 box 共用
- 最終輸出 tensor：**7 × 7 × 30**（= 7×7×(2×5 + 20)）

### Network Architecture

- **24 層 conv + 2 層 FC**，靈感來自 GoogLeNet
- 使用 1×1 reduction layers + 3×3 conv（而非 inception modules）
- 激活函數：所有隱藏層用 **Leaky ReLU**（slope=0.1），最後一層用 **linear activation**
- Fast YOLO：僅 9 層 conv + fewer filters，其餘訓練參數不變

#### 下採樣路徑（解析度變化）

```
448×448 → [7×7×64-s2 conv] → 224×224 → [maxpool] → 112×112
→ [3×3×192 conv + maxpool] → 56×56
→ [1×1×128, 3×3×256, 1×1×256, 3×3×512 + maxpool] → 28×28
→ [1×1×256, 3×3×512 ×4 + 1×1×512, 3×3×1024 + maxpool] → 14×14
→ [1×1×512, 3×3×1024 ×2 + 3×3×1024 + 3×3×1024-s2] → 7×7
→ [3×3×1024, 3×3×1024] → 7×7×1024
→ [FC 4096 → FC 7×7×30]
```

#### ImageNet 預訓練

- 前 20 層 conv + average-pooling + FC，在 ImageNet 1000 類訓練（224×224）
- 約訓練**一週**，達到 88% top-5 accuracy（與 GoogLeNet 相當）
- 使用 **Darknet 框架**
- 偵測時：加入 4 conv + 2 FC（隨機初始化），解析度提升至 448×448（因偵測需要更 fine-grained 的視覺資訊）

### Loss Function（Multi-part Sum-Squared Error）

#### 為什麼用 SSE 而非 Cross-Entropy？

YOLO 將偵測框架為**回歸問題**（非分類），SSE 簡單易優化。但這是**次優選擇**：SSE 將定位誤差和分類誤差等權處理，且無法有效處理類別不平衡。後續版本 (v4+) 改用 Focal Loss / CIoU Loss。

#### 指示函數定義

- **`𝟙_ij^obj`**：cell i 的第 j 個 predictor 是「responsible predictor」（與 GT 的 IOU 最高的那個）
- **`𝟙_i^obj`**：物件中心落在 cell i（cell 包含物件）
- **`𝟙_ij^noobj`**：cell i 的第 j 個 predictor 不負責任何物件

#### 5 個組成部分

1. **Box 中心座標 loss**（x, y）：僅 responsible predictor 計算，權重 λ_coord=5
2. **Box 尺寸 loss**（√w, √h）：僅 responsible predictor 計算，權重 λ_coord=5
    - 用平方根緩解大小 box 的不均衡（大 box 梯度 ∝ 1/(2√w)，自然變小）
3. **有物件的 confidence loss**：僅 responsible predictor，target = IOU(pred, GT)
4. **無物件的 confidence loss**：所有非 responsible predictor，權重 λ_noobj=0.5（抑制大量負樣本梯度，避免模型不穩定）
5. **類別機率 loss**：僅在包含物件的 cell 計算（`𝟙_i^obj`），與 predictor 無關

**關鍵**：座標 loss 用 `𝟙_ij^obj`（predictor 層級），分類 loss 用 `𝟙_i^obj`（cell 層級）

### Inference（推理流程）

- 每張影像預測 **98 個 bounding boxes**（7×7×2 = 98）
- Test-time class-specific score（用於排序和 NMS）：
  `Pr(Class_i|Object) × Pr(Object) × IOU_pred^truth = Pr(Class_i) × IOU_pred^truth`
- **NMS（Non-Maximum Suppression）**：
    - Grid 設計本身強制空間多樣性，大幅減少重複偵測
    - NMS 對 YOLO 不像 R-CNN/DPM 那麼關鍵，但仍提升 **2-3% mAP**
    - 主要處理：大物件或跨 cell 邊界物件被多個 cell 偵測的情況

### Training Details

- Dataset: PASCAL VOC 2007+2012，~135 epochs
- Batch size: 64, momentum: 0.9, weight decay: 0.0005
- LR schedule: 先從 10⁻³ **慢慢 warmup** 升至 10⁻²（避免初期不穩定 gradients 導致 diverge）→ 維持 75 epochs → 10⁻³ 30 epochs → 10⁻⁴ 30 epochs
- Regularization: Dropout (0.5, 僅在第一個 FC 層後) + data augmentation（random scaling/translation ±20%, HSV exposure/saturation shift ×1.5）
- **Responsible predictor 機制**：每個 grid cell 中，IOU 最高的 box predictor 負責該物件 → predictor specialization（各自學習特定 size/aspect ratio/class）

## Key Results

| Model | mAP (VOC 2007) | FPS |
|-------|---------------|-----|
| Fast YOLO | 52.7% | **155** |
| YOLO | **63.4%** | 45 |
| YOLO VGG-16 | 66.4% | 21 |
| Fast R-CNN | 70.0% | 0.5 |
| Faster R-CNN VGG-16 | 73.2% | 7 |
| Faster R-CNN ZF | 62.1% | 18 |

### Error Analysis（Hoiem et al. 方法論）

五種錯誤類型定義：

- **Correct**：正確類別 + IOU > 0.5
- **Localization**：正確類別 + 0.1 < IOU < 0.5
- **Similar**：相似類別 + IOU > 0.1
- **Other**：錯誤類別 + IOU > 0.1
- **Background**：IOU < 0.1（任何物件）

| Error Type | YOLO | Fast R-CNN |
|------------|------|------------|
| Correct | 65.5% | 71.6% |
| **Localization** | **19.0%** | 8.6% |
| Similar | 6.75% | 4.3% |
| Other | 4.0% | 1.9% |
| **Background** | **4.75%** | **13.6%** |

- YOLO 定位錯誤佔所有錯誤的比重**超過其他所有來源之和**
- Fast R-CNN 的背景誤檢是 YOLO 的 **~3 倍**
- **Fast R-CNN + YOLO 組合**：mAP 從 71.8% → **75.0%**（+3.2%），因兩者錯誤模式互補

### Generalization（藝術品偵測）

| Dataset | YOLO AP | R-CNN AP | DPM AP |
|---------|---------|----------|--------|
| VOC 2007 (person) | 59.2 | 54.2 | 43.2 |
| Picasso | **53.3** | 10.4 | 37.8 |
| People-Art | **45** | 26 | 32 |

- R-CNN 在藝術品上 AP 暴跌（54.2 → 10.4）：Selective Search 針對自然影像調優，在藝術品上產生糟糕的 proposals
- YOLO 建模 **size/shape/relationships + objects 常見位置**，這些特徵在自然影像和藝術品中一致

## Limitations

1. **空間約束**：每個 grid cell 只能預測 2 個 box、**1 個類別** → 無法處理密集小物件（如鳥群）或同 cell 多類別物件
2. **Recall 上限低**：僅 98 個 boxes，遠少於 Selective Search (~2000) → recall 上限受限
3. **小物件偵測差**：5 次 maxpool + 1 次 stride-2 conv（共 64x 下採樣），小物件特徵被消除
4. **定位精度不足**：SSE loss 對大小 box 誤差處理不完善（雖用 √w/√h 「partially address」，但仍是主要錯誤來源）
5. **泛化到不尋常 aspect ratio 困難**：直接回歸 w/h 沒有 prior，學習負擔大

## Interview Q&A

- **「為何 YOLO 比 R-CNN 快？」**：單次前向傳播 vs. 2000+ region proposals 各自分類。YOLO 丟掉整個 pipeline，用回歸取代
- **「YOLO 的 grid 設計有什麼好處？」**：強制空間多樣性、減少重複偵測、全局推理（global reasoning）→ 背景誤檢低
- **「YOLO 主要的 error type 是什麼？」**：localization error (19%)，而非 background false positive (4.75%)。Fast R-CNN 剛好相反
- **「為何用 √w/√h？」**：大 box (w=100) 梯度 ∝ 1/20，小 box (w=10) 梯度 ∝ 1/6.3，自然讓模型更關注小 box 的精度
- **「兩個物件在同一 grid cell 會怎樣？」**：只能偵測一個（只有 1 組類別機率）。v2 的 anchor boxes 解耦 class 和 spatial，改善此問題
