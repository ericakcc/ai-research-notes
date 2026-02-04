# YOLOv1 → v2 演化分析 & 面試問答

## 1. v1 → v2 因果關係對照表

| YOLOv1 的具體問題 | 數據佐證 | YOLOv2 的解決方案 | 改進效果 |
|------------------|---------|---------------------|---------|
| **Localization error 佔 19%**（主要錯誤來源）| Error analysis: localization > all other errors combined | Direct Location Prediction (σ 約束 + cell offset) | **+5% mAP**，訓練更穩定 |
| **Recall 僅 81%**（98 boxes 上限太低）| 7×7×2=98 boxes vs Selective Search ~2000 | Anchor Boxes（全卷積，13×13×5=845 boxes）| recall 81%→**88%** |
| **密集/小物件漏檢**（每 cell 1 class + 2 boxes）| 如鳥群、密集瑕疵 | Anchor Boxes 解耦 class 和 spatial + Passthrough Layer | 可同 cell 偵測多物件 |
| **小物件特徵被消除**（5 次 maxpool + stride-2 conv = 64x 下採樣）| 7×7 grid 粒度太粗 | Passthrough Layer (26×26→13×13 拼接) | **+1% mAP**，保留 fine-grained 特徵 |
| **訓練時解析度突變** (224→448) | 預訓練和偵測解析度不一致 | High Resolution Classifier（448 fine-tune 10 epochs）| **+4% mAP** |
| **過擬合需 dropout** | 第一 FC 層後 dropout 0.5 | Batch Normalization（所有 conv 層）| **+2% mAP**，可移除 dropout |
| **泛化到不尋常 aspect ratio 困難** | 直接回歸 w/h 沒有 prior | Dimension Clusters（k-means k=5，IOU 距離）| data-driven priors，5 clusters ≈ 9 hand-picked |
| **固定解析度推理** | 僅支援 448×448 | Multi-Scale Training ({320..608}) | 同一模型在不同解析度推理 |

## 2. 歷史定位（2014-2017 Detection Landscape）

```
精度 (mAP)
  80% ─┬─────────────────────────────────────────────────
       │                                    ○ YOLOv2 544 (78.6%)
       │                              ○ SSD512 (76.8%)
  75% ─┤                        ○ Faster R-CNN ResNet (76.4%)
       │                  ○ SSD300 (74.3%)
       │            ○ Faster R-CNN VGG (73.2%)
  70% ─┤      ○ Fast R-CNN (70.0%)
       │
  65% ─┤                                          ○ YOLO (63.4%)
       │
  60% ─┤
       │
       └──┬────┬────┬────┬────┬────┬────┬────┬────┬──── FPS
          0.5   5    7   19   40   45   46   67   91  155
              ← 慢                                快 →
```

### 三大範式比較

| 範式 | 代表方法 | 核心思路 | 優勢 | 劣勢 |
|------|---------|---------|------|------|
| **Two-stage** | R-CNN → Fast → Faster R-CNN | 先生成 proposals → 再分類 | 精度高（73.2%） | 慢（0.5~7 FPS） |
| **Single-stage grid** | YOLO v1/v2 | 影像切 grid → 直接回歸 | 極快（45~155 FPS）、背景誤檢低 | 定位精度差、小物件弱 |
| **Single-stage multi-scale** | SSD | 多層 feature maps 各自預測 | 速度精度均衡 | 需要精心設計 anchor |

**YOLO 的獨特定位**：
- 是 2016 年**唯一真正 real-time**（≥30 FPS）的高精度 detector
- 背景誤檢僅 R-CNN 系列的 **1/3**（global reasoning 優勢）
- 與 Fast R-CNN ensemble 可提升後者 3.2% mAP → 證明兩者錯誤模式互補
- SSD 的多尺度方案對小物件更優 → **直接促使 v3 採用 FPN**

## 3. 種子技術 → 後續版本演化追蹤

```
YOLOv2 (2016)
  │
  ├─ Passthrough Layer ──────→ YOLOv3 FPN 3 尺度 (2018)
  │   (單向, 1 個額外尺度)        → YOLOv4 PANet 雙向融合 (2020)
  │                               → YOLOv5/v8 C2f + PANet
  │
  ├─ Darknet-19 ─────────────→ YOLOv3 Darknet-53 + Residual (2018)
  │   (19 conv, 無 residual)      → YOLOv4 CSPDarknet-53 (2020)
  │                               → YOLOv5 CSPDarknet
  │                               → YOLOv6 EfficientRep (RepVGG)
  │
  ├─ Dimension Clusters ─────→ v3-v7 持續用 k-means anchor
  │   (k-means, k=5)              → YOLOv8+ Anchor-Free（拋棄 anchors）
  │                               啟示：anchor 可以是 data-driven → 不是必要的
  │
  ├─ Direct Location Pred ────→ v3-v7 持續用 σ(tx)+cx 公式
  │   (sigmoid + cell offset)     → YOLOv8 直接預測 center（不需 anchor offset）
  │
  ├─ Multi-Scale Training ───→ YOLOv3 Multi-Scale Prediction (2018)
  │   (訓練時多尺度)               (訓練+推理都在 3 尺度預測: 52², 26², 13²)
  │
  ├─ Batch Normalization ────→ v3+ 全系列 conv 層標配
  │
  ├─ 全卷積架構 ──────────────→ v3+ 全系列（支援任意尺寸輸入，32 的倍數）
  │   (移除 FC 層)
  │
  └─ WordTree ────────────────→ 概念未被後續 YOLO 繼承
      (層級分類)                    但影響了 zero-shot / open-vocabulary detection 研究
```

### 技術成熟度對照

| 技術 | v1 | v2 | v3 | v4 | v5/v8 | 狀態 |
|------|----|----|----|----|-------|------|
| BN | ✗ | ✓ | ✓ | ✓ | ✓ | 標配 |
| Anchor Boxes | ✗ | k-means | k-means | k-means | **Anchor-Free** | 已淘汰 |
| Multi-Scale Prediction | ✗ | 僅訓練 | FPN 3 尺度 | PANet | PANet+C2f | 標配 |
| Residual Connections | ✗ | ✗ | Darknet-53 | CSP | CSP+C2f | 標配 |
| NMS | 需要 | 需要 | 需要 | 需要 | v10: NMS-free | 趨勢淘汰 |

## 4. 工業應用對照表（Corning 面試導向）

| 康寧需求 | v1 適配性 | v2 適配性 | 建議方案 |
|---------|---------|---------|---------|
| **Real-time 產線** (< 50ms) | ✅ 22ms (45 FPS) | ✅ 15ms (67 FPS) | v2 已滿足，v8+TensorRT 更佳 |
| **微小瑕疵** (< 5px) | ❌ 7×7 grid 太粗 | ⚠️ 13×13 + passthrough 改善有限 | 需要 v3+ FPN (52×52 大尺度) 或 crop-then-detect |
| **密集瑕疵** (多處瑕疵) | ❌ 98 boxes, 每 cell 1 class | ✅ 845 boxes, 解耦 class/spatial | v2 已可處理中等密度 |
| **精確定位+面積測量** | ❌ loc error 19% | ⚠️ 改善至 ~10% | YOLO 粗定位 + **U-Net 精確分割** (hybrid pipeline) |
| **變動產品尺寸** | ❌ 固定 448 | ✅ Multi-scale 320~608 | v2 已支援 |
| **類別不平衡** (正常>>異常) | ⚠️ λ_noobj=0.5 緩解但不夠 | ⚠️ 同 | 改用 **Focal Loss** + **OHEM** + GAN 增強 |
| **背景複雜** (反光/紋理) | ✅ 背景誤檢低 4.75% | ✅ 同 | YOLO 的 global reasoning 是優勢 |
| **部署** (邊緣設備) | ⚠️ 需 GPU | ⚠️ 需 GPU | v8 + **TensorRT FP16/INT8** 量化 |

**綜合建議**：v1/v2 作為學習基礎，實際部署用 **v8/YOLO26 + TensorRT**，配合 **PatchCore 粗篩 + U-Net 精細分割**

## 5. 面試問答集

### Q1: 為什麼 YOLO 用 SSE 而不是 Cross-Entropy？
**A**: YOLO 將偵測框架為回歸問題（非分類），bbox 座標 (x,y,w,h) 用 L2 合理。但 class probability 也用 SSE 是次優的——SSE 將定位和分類誤差等權、且無法有效處理正負樣本不平衡（大量空 cell 的梯度壓過有物件的 cell）。v1 用 λ_noobj=0.5 緩解，但直到 v4 引入 Focal Loss 才真正解決。

### Q2: 兩個物件的中心落在同一個 grid cell，會怎樣？
**A**: v1 只能偵測**一個**（每 cell 只有 1 組 class probability）。兩個 box predictor 會各自學到與兩個 GT 最高 IOU 的 box，但 class 只能選一個。這是 v1 最大的結構性限制。v2 用 anchor boxes 解耦 class 和 spatial，每個 anchor 獨立預測類別，改善此問題。v3 的多尺度預測進一步緩解。

### Q3: 為什麼 v2 用 k-means 距離 1-IOU 而非歐氏距離？
**A**: 歐氏距離偏向大 box（大 box 的絕對座標距離天然更大），導致 priors 集中在大 box 上。IOU 是 scale-invariant 的，讓大小 box 同等對待。實驗證實：5 個 IOU-based clusters (Avg IOU=61.0) ≈ 9 個手工 anchors (60.9)，且 9 個 clusters 達到 67.2。

### Q4: Grid-based 設計對微小瑕疵偵測有什麼限制？
**A**: v1 的 7×7 grid = 每 cell 負責 64×64 px（448 輸入），v2 的 13×13 = 每 cell 32×32 px。如果瑕疵 < 5px，下採樣 32x 後特徵幾乎消失。**解決方案**：(1) v3 FPN 在 52×52 大尺度預測（每 cell 8×8 px），(2) 提高輸入解析度（608→），(3) 先裁切 ROI 再放大偵測（two-stage pipeline）。

### Q5: 如果要用 YOLO 做康寧玻璃檢測，你會選哪個版本？
**A**: 不會直接用 v1/v2。推薦 **YOLOv8/YOLO26 + TensorRT**，搭配 **hybrid pipeline**：
1. **PatchCore** 無監督粗篩（找異常區域）
2. **YOLOv8** anchor-free 偵測 + 分類（處理多類別瑕疵）
3. **U-Net** 精確分割（測量瑕疵面積）
4. **Focal Loss** 處理極度不平衡（正常 >> 異常）

但理解 v1/v2 是理解後續版本的基礎：v2 的 anchor 思想啟發了 anchor-free 演化、passthrough 是 FPN 前身、multi-scale training 成為標配。

### Q6: Passthrough Layer 和 FPN 有什麼差異？
**A**: Passthrough 是**單向、單尺度**融合（26×26 → 13×13，拼接後只在 13×13 上預測）。FPN 是**多尺度雙向金字塔**（在 52×52, 26×26, 13×13 各自預測，且 top-down + bottom-up 雙向融合）。Passthrough 只帶來 +1% mAP，FPN 帶來顯著的小物件改進。

### Q7: YOLO 的速度優勢在工業部署中意味著什麼？
**A**: (1) **Real-time 產線檢測**：v2 67 FPS (15ms) 遠快於產線速度需求（通常 30 FPS）；(2) **多攝影機同時處理**：速度餘量可用於同時處理多路視頻流；(3) **Edge deployment**：結合 TensorRT 量化，v8 可在 Jetson 等邊緣設備達 50+ FPS。但要注意 **NMS 的非確定性延遲**——box 數量不固定導致 P99 latency 波動，v10/YOLO26 的 NMS-free 解決此問題。

### Q8: 如何適應 YOLO 到極度不平衡的資料（如 1:1000 正常:異常）？
**A**: v1/v2 的 λ_noobj=0.5 是初步嘗試但不夠。建議：
1. **Focal Loss**：(1-p)^γ 讓模型更專注難樣本（v4+ 引入）
2. **OHEM**（Online Hard Example Mining）：只對最難的負樣本反傳
3. **Data augmentation**：Mosaic (v4), MixUp, Copy-Paste
4. **GAN 合成**：生成稀有瑕疵的合成影像
5. **Few-shot / anomaly detection**：PatchCore 等不需要大量正樣本的方法作為補充

### Q9: 為什麼 v2 改用 416 而非繼續用 448？
**A**: 416 / 32 = 13（奇數），確保 feature map 有唯一中心 cell。大物件常佔據影像中心，有一個中心 cell 可直接負責比 4 個相鄰 cell 共同負責更直接。448 / 32 = 14（偶數），沒有唯一中心 cell。

### Q10: √w/√h 的梯度直覺是什麼？
**A**: 對 √w 微分得 1/(2√w)。大 box (w=100) 的梯度 = 1/20，小 box (w=10) 的梯度 = 1/6.3。這讓同樣的絕對誤差在小 box 上產生**更大的梯度**，驅動模型更精確地預測小 box 的尺寸。但這只是「partially address」——v2 改用 anchor prior + exp(tw) 的 log-space 預測，更進一步解決 scale sensitivity。
