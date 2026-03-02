# Paper Summary: Attention Is All You Need

**Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
**arXiv**: [1706.03762](https://arxiv.org/abs/1706.03762)
**Year**: 2017 (NeurIPS 2017)

## Core Contribution

用純 Attention 機制取代 RNN/CNN，提出 **Transformer** 架構：完全依賴 self-attention 建模序列中所有位置之間的依賴關係，消除了 RNN 的序列計算瓶頸，實現高度平行化。奠定了後續 BERT、GPT 乃至 ViT 的基礎。

## Method

### Scaled Dot-Product Attention（核心計算）

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- **Q, K, V**：Query / Key / Value 矩陣，各由輸入線性投影而來
- **為何除以 √d_k**：內積 `QK^T` 隨 d_k 增大而方差增大，梯度會飄進 softmax 飽和區（接近 0/1）。除以 √d_k 讓方差恢復到 O(1)，梯度穩定
- **時間複雜度**：O(n² · d)，n = 序列長度，d = 維度

```
Additive Attention（Bahdanau）: O(n² · d)，但常數更大（MLP 計算 compatibility）
Dot-Product Attention:         O(n² · d)，矩陣乘法可用 BLAS 高度優化 → 實際更快
```

### Multi-Head Attention（多頭機制）

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

- **h 個頭的意義**：不同頭學習捕捉不同語義子空間（一個頭關注 syntactic，另一個頭關注 coreference…）
- **head_dim = d_model / h**：總計算量與單頭相當（d_model → h × d_k，各頭算完 concat 回 d_model）
- **輸出投影 W^O**：合併各頭後做線性投影，混合不同子空間的資訊

### 與親手實作 MHSA 的對照

```python
# 論文: Q·W_i^Q, K·W_i^K, V·W_i^V  (每頭分別投影)
# 實作: 一個 qkv = nn.Linear(dim, dim*3)，一次投影後 split  ← 等價但更高效

qkv = self.qkv(x).chunk(3, dim=-1)              # [B, n, dim] × 3
q = rearrange(q, "b n (h d) -> b h n d", h=h)   # reshape = 分頭
attn = (q @ k.transpose(-2, -1)) * self.scale    # QK^T / √d_k
out = rearrange(attn @ v, "b h n d -> b n (h d)") # concat = reshape back
return self.proj(out)                             # 輸出投影 W^O
```

### Transformer Block 結構（Encoder）

```
Input → [Multi-Head Self-Attention → Add & Norm] → [Feed-Forward → Add & Norm]
         └── residual connection ──┘                └── residual connection ──┘
```

- **Add & Norm**：先殘差加法再 LayerNorm（Post-LN，論文原始）
- ViT 實作改用 **Pre-LN**（LN 在 sublayer 前），訓練更穩定

### Position Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Attention 本身是 permutation-equivariant，必須注入位置資訊。
使用固定 sin/cos 而非可學習：允許模型外推到比訓練更長的序列。

## Key Results

| Model | EN-DE BLEU | EN-FR BLEU | Training Cost (FLOPs) |
|-------|-----------|-----------|----------------------|
| Transformer (big) | **28.4** | **41.8** | 3.3×10¹⁸ |
| ConvS2S | 25.2 | 40.5 | 1.5×10¹⁹ |
| GNMT+RL | 24.6 | 39.9 | 1.4×10²⁰ |

以 1/4 計算量超越所有基線。

## Limitations

1. **Self-Attention 是 O(n²)**：序列長度翻倍，計算量翻四倍，長序列（>8k tokens）代價昂貴
2. **無歸納偏置**：不像 CNN 有 locality/translation-invariance，需要大量資料或 pre-training
3. **Position encoding 是手動設計**：固定 sin/cos 無法學習最優位置表示（後續有 RoPE / ALiBi 改進）

## 與後續工作的關係

- **BERT** (2018)：Encoder-only + Masked LM 預訓練
- **GPT** (2018)：Decoder-only + 自回歸生成
- **ViT** (2020)：把 patch 當 token，Transformer 直接用於 CV → 見 [ViT Paper Notes](vit-paper-notes.md)
- **I-JEPA / V-JEPA**：用 ViT 作為 encoder，latent prediction 在 Transformer 特徵空間進行

## Interview Q&A

**Q: 為什麼 Attention 需要 scale（除以 √d_k）？**
A: d_k 維度高時，`QK^T` 內積的方差 = d_k（每個維度貢獻 1）。未 scale 的高值讓 softmax 梯度飽和（接近 one-hot），反向傳播梯度接近 0。除以 √d_k 讓期望方差回到 1。

**Q: Multi-head 的好處是什麼？單頭 attention 有什麼問題？**
A: 單頭只能學習一種「attend 方式」。多頭允許不同頭同時關注不同位置、不同特徵子空間（syntactic vs. semantic），concat 後保留了多種相關性。代價幾乎不增加（d_k = d_model/h）。

**Q: Transformer 為什麼比 RNN 快？**
A: RNN 必須序列計算（t 步依賴 t-1 步），無法平行。Transformer 的 self-attention 對序列中所有位置同時計算，可以高度平行化（GPU 友好的矩陣乘法）。

**Q: Self-attention 的時間複雜度是多少？為什麼仍被採用？**
A: O(n² · d)，序列長度 n 的平方。對於大多數 NLP/CV 任務 n 不大（512 tokens / 196 patches），實際可接受。且矩陣乘法在 GPU 上極其高效。長序列問題由 FlashAttention / Sparse Attention 解決。

> **→ 後續閱讀**：ViT 如何把 Attention 應用到 CV，見 [ViT Paper Notes](vit-paper-notes.md)
