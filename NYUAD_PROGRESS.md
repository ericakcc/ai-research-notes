# NYUAD Research Engineer 衝刺進度

## 目標

2026 年申請 NYUAD CAIR Research Engineer（路線圖與應徵策略見 [`topics/NYUAD.md`](topics/NYUAD.md)）

## 總覽

| 階段 | Topic | 進度 | 產出 |
|------|-------|------|------|
| 基礎 | Vision Transformers | 🟡 架構就緒 | ViT from scratch, MAE |
| 第一階段 | Self-Supervised Learning | 🔴 未開始 | VICReg, SimCLR, Barlow Twins |
| 第二階段 | JEPA | 🔴 未開始 | I-JEPA, V-JEPA, MC-JEPA |
| 第三階段 | Reinforcement Learning | 🔴 未開始 | PPO, SAC, JEPA+RL |
| 第三階段 | Embodied AI | 🔴 未開始 | Isaac Gym, JEPA Navigator |
| 持續 | World Models (VMC) | 🟡 進行中 | VAE ✅ MDN-RNN 🟡 Controller 🟡 |

## 詳細進度

### Vision Transformers

**01_vit_from_scratch 架構狀態：**
- [x] `config.py` — ViTConfig dataclass（ViT-Tiny: dim=256, depth=6, heads=8, patch_size=4）
- [x] `model.py` — PatchEmbedding, FeedForward, TransformerBlock, VisionTransformer
- [ ] `model.py` — **MultiHeadSelfAttention.forward()** ← TODO(human)，需要你親手實作
- [x] `dataset.py` — CIFAR-10 DataLoader + augmentation
- [x] `train.py` — AdamW + CosineAnnealingLR 訓練腳本
- [x] `test_vit.py` — 5 個測試（shape、residual、param count 等）
- [ ] 跑通測試 — 需先完成 MHSA.forward()
- [ ] 訓練 CIFAR-10 — 產出：accuracy > 85%

**你的第一個動手任務：**
- 檔案：`topics/vision-transformers/experiments/01_vit_from_scratch/src/model.py`
- 搜尋：`TODO(human)`
- 任務：實作 `MultiHeadSelfAttention.forward()`（約 10 行）
- 核心：QKV 投影 → multi-head reshape → scaled dot-product → softmax → 加權求和 → 輸出投影
- 驗證：`uv run pytest topics/vision-transformers/experiments/01_vit_from_scratch/tests/ -v`

**建議先讀：**
1. ViT 論文 Section 3 (Method) — 了解 patch → token 的概念
2. "Attention Is All You Need" Figure 2 — scaled dot-product attention 圖解
3. `lucidrains/vit-pytorch` 的 Attention class 作為參考

- [ ] MAE — 產出：masked autoencoder 視覺化重建結果

### Self-Supervised Learning

- [ ] 讀 LeCun EBM Tutorial — 產出：筆記 + 能量函數視覺化
- [ ] VICReg on CIFAR-10 — 產出：feature decorrelation 分析圖
- [ ] SimCLR — 產出：對比學習 t-SNE 視覺化
- [ ] Barlow Twins — 產出：cross-correlation matrix 熱力圖

### JEPA

- [ ] I-JEPA — 產出：ImageNet 子集上的 masked prediction + linear probe
- [ ] V-JEPA — 產出：Kinetics 子集上的影片特徵預測
- [ ] MC-JEPA — 產出：多模態條件注入實驗

### Reinforcement Learning

- [ ] PPO on CartPole — 產出：reward curve + policy 視覺化
- [ ] SAC on continuous — 產出：LunarLander/HalfCheetah 訓練結果
- [ ] JEPA features + RL — 產出：用 JEPA 表徵作為狀態輸入的 RL agent

### Embodied AI

- [ ] Isaac Gym basics — 產出：跑通官方 demo + 自定義環境
- [ ] JEPA-Driven Navigator — 產出：最終 Portfolio 專案

## 知識庫建置紀錄

| 日期 | 完成項目 |
|------|---------|
| 2026-02-16 | 建立 5 個新 topic 目錄結構 + CLAUDE.md |
| 2026-02-16 | ViT from scratch 架構完成（待 MHSA 實作） |
| 2026-02-16 | NYUAD_PROGRESS.md 進度追蹤建立 |

## 學習日誌

<!-- 每次學習後記錄：日期、做了什麼、學到什麼、遇到的問題 -->
