# Paper Summary: An Image is Worth 16x16 Words

**Authors**: Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby (Google Brain)
**arXiv**: [2010.11929](https://arxiv.org/abs/2010.11929)
**Year**: 2020 (ICLR 2021)

## Core Contribution

將圖像切成 patch，把每個 patch 當成一個 token，直接用**純 Transformer**做影像分類，完全不依賴 CNN 的 inductive bias（locality、translation equivariance）。核心主張：**在足夠大的資料集（14M+ 圖像）上預訓練後，ViT 效果優於當時最好的 ResNet-based 方法，且計算量更少**。

## Method

### 完整 Pipeline

```
Input: (B, C, H, W)
  ↓ 切成 N = (H/P)×(W/P) 個 patch，每個 patch 展平: N × (P²·C)
  ↓ Patch Embedding: Linear(P²·C, D)  →  (B, N, D)
  ↓ 前置 [CLS] token: (B, N+1, D)
  ↓ + Position Embedding (learnable, 1D): (B, N+1, D)
  ↓ Transformer Encoder × L 層
  ↓ 取 z₀^L（CLS token 的輸出）
  ↓ MLP Head → 分類 logits
```

### Patch Embedding

```python
# 用 Conv2d 實現：kernel_size = stride = patch_size  ← 等價於線性投影每個 patch
self.projection = nn.Conv2d(C, D, kernel_size=P, stride=P)
# (B, C, H, W) → (B, D, H/P, W/P) → rearrange → (B, N, D)
```

論文也可以用純 Linear 展平，Conv2d 是等價但更高效的實現。

### CLS Token

- 類比 BERT 的 `[CLS]`：在 patch sequence 前插入一個可學習 token
- Transformer blocks 處理後，CLS token 已 attend over 全部 patches → **全局表示**
- 不用 GAP（Global Average Pooling）的原因：讓分類頭與預訓練一致（fine-tune 時替換 MLP head 即可）

### Positional Encoding

論文測試了三種：

1. **1D learnable**（最終採用）：直接學習 N+1 個位置的 embedding
2. 2D learnable：分別學習 row 和 col embedding，concat
3. 固定 sin/cos（Transformer 原始）

**1D learnable 效果與 2D 相當**：Transformer 靠 attention 間接學到了 2D 關係，不需要顯式的 2D structure。

### Section 3.2: Fine-tuning & Higher Resolution（重點）

**大資料集預訓練 → 小資料集 fine-tune** 是 ViT 的核心用法：

1. 預訓練時：MLP head（2 層 tanh）
2. Fine-tune 時：替換為**單層 linear head**（random init），在目標資料集上訓練
3. **更高解析度 fine-tune**：patch size 不變 → sequence 變長 → 原始 position embedding 需要插值

```python
# 預訓練：224×224, patch=16 → N=196 個 tokens
# Fine-tune：384×384, patch=16 → N=576 個 tokens（序列變長 ~3×）

# Position embedding 2D 插值：
# 原始: (1, 196, D) → reshape (14, 14, D) → bilinear_interpolate → (24, 24, D) → flatten (576, D)
```

這是 Section 3.2 的核心工程細節，面試常考。

### 三種模型規模

| Model | Layers | D | MLP size | Heads | Params |
|-------|--------|---|----------|-------|--------|
| ViT-B | 12 | 768 | 3072 | 12 | 86M |
| ViT-L | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H | 32 | 1280 | 5120 | 16 | 632M |

Patch size 以 `/P` 標記（e.g., ViT-L/16 = ViT-Large with patch 16）

## Key Results

| Model | Pre-training | ImageNet top-1 |
|-------|-------------|----------------|
| ViT-L/16 | JFT-300M | **87.76%** |
| ViT-H/14 | JFT-300M | **88.55%** |
| BiT-L (ResNet) | JFT-300M | 87.54% |
| Noisy Student (EfficientNet-L2) | JFT-300M | 88.4% |

**關鍵數字**：ViT-H/14 在 JFT-300M 預訓練後，ImageNet 達到 88.55%，且訓練成本約為 BiT 的 1/8。

### 資料規模 vs 準確率（論文 Figure 3-4）

| Pre-training Data | ViT-L/16 ImageNet |
|-------------------|------------------|
| ImageNet (~1M)    | < ResNet-50 基線  |
| ImageNet-21k (14M) | ≈ BiT-M         |
| JFT-300M (300M)   | > 所有 ResNet    |

**14M 是分水嶺**：資料不足時，ViT 缺少 CNN 的 locality/spatial bias，表現更差。超過 14M 後，scale 的優勢壓過 inductive bias 的劣勢。

## Limitations

1. **資料飢渴**：小資料集（<14M）上表現劣於 CNN，沒有 CNN 的 locality inductive bias
2. **訓練成本高**：需要大規模預訓練（JFT-300M）才能發揮優勢
3. **位置插值有資訊損失**：不同解析度 fine-tune 時，bilinear 插值是近似，非完美的跨解析度表示

## 與後續工作的關係

- **DeiT (2021)**：透過 knowledge distillation 讓 ViT 在 ImageNet-only（無 JFT）也能訓練，引入 distillation token
- **MAE (2022)**：隨機 mask 75% patches，只讓 encoder 看到可見 patches，decoder 重建 pixel。**改變了 ViT 的 pre-training 範式**（discriminative → generative）
- **I-JEPA (2023)**：改進 MAE 的「predict pixels」→「predict latent representation」，避免學習低層次 pixel 細節，JEPA 系列的起點

```
ViT (backbone)
  ├── MAE: predict masked pixels  ← pixel reconstruction
  └── I-JEPA: predict masked latent  ← representation prediction (更高層次)
```

## Interview Q&A

**Q: ViT 為什麼需要大規模資料預訓練？**
A: CNN 的 locality（只看鄰近像素）和 translation equivariance（位移不變性）是 inductive bias，等於免費給模型注入了圖像先驗知識。ViT 沒有這些 bias，必須從資料中從頭學習。小資料集 (~1M) 不夠讓 ViT 學習這些先驗，14M+ 才能讓 scale 的優勢勝出。

**Q: CLS token 的作用，為何不用 GAP？**
A: CLS token 在預訓練中已學習 attend over 全部 patches，成為全局表示。Fine-tune 時只需替換 MLP head，訓練一致。GAP 需要對所有 patch embedding 平均，有時效果接近，但實作上 CLS 更自然（繼承 BERT 設計）。

**Q: Fine-tune 時解析度更高，position embedding 怎麼處理？**
A: 預訓練 position embedding 形狀不符（N_train ≠ N_finetune）。標準做法：把 1D embedding reshape 回 2D (H/P × W/P)，用 bilinear interpolation 插值到新解析度的 grid，再 flatten 回 1D。這樣保留了相對位置資訊。

**Q: ViT 的 patch size 如何影響效能？**
A: patch 越小（如 P=14 vs P=16），sequence 越長（N 更大），計算量 O(N²) 更大，但能捕捉更細緻的局部資訊。ViT-H/14 比 ViT-L/16 在高解析度任務上更強，但訓練更貴。

**Q: 為什麼 1D position encoding 效果等同於 2D？**
A: Transformer 的 self-attention 讓模型能透過觀察所有 patches 之間的 attention pattern 隱式學習 2D 空間關係，不依賴顯式的 2D structure。論文消融實驗顯示三種 PE 差距微小。

> **→ 前置知識**：Attention 機制詳解，見 [Attention Is All You Need](attention-paper-notes.md)
