# **2026 阿布達比 AI 研究工程師 (RE) 職涯衝刺指南：JEPA 專攻路徑**

本指南針對 NYU Abu Dhabi (NYUAD) 的 CAIR 中心量身打造。由於 Yann LeCun 的影響力，掌握 **JEPA (Joint-Embedding Predictive Architecture)** 是進入該體系的最短路徑。

## **一、 核心目標：為什麼是 JEPA？**

傳統世界模型（如 VAE-based）試圖「重建像素」，這在處理複雜環境時效率低下。JEPA 則是通過 **自我監督學習 (SSL)** 在潛在空間預測缺失資訊。這對於資源有限但要求高智慧的機器人（NYUAD 的研究重點）至關重要。

## **二、 深度學習路徑：從零到 JEPA 專家**

### **第一階段：自監督學習 (SSL) 與 能量模型 (EBM) (第 1-2 個月)**

這是 LeCun 思想的基石。JEPA 的本質就是一個不帶常數項的能量模型。

* **理論核心**：  
  * **VicReg / SimCLR / Barlow Twins**：理解如何防止潛在表示崩潰（Representation Collapse）。  
  * **Energy-Based Models (EBM)**：閱讀 LeCun 的經典論文 *A Tutorial on Energy-Based Learning*。  
* **動手實踐**：  
  * 使用 PyTorch 實現一個簡單的 **VICReg**，在 CIFAR-10 上進行預訓練，並觀察 Feature Decorrelation 的效果。

### **第二階段：JEPA 系列深度解構 (第 3-4 個月)**

從圖像到影片，理解 JEPA 如何建模世界動態。

* **必讀論文清單**：  
  1. **I-JEPA** (Image JEPA)：理解 Masked Modeling 在潛在空間的運作。  
  2. **V-JEPA** (Video JEPA)：學習如何捕捉時間維度的特徵。  
  3. **MC-JEPA**：理解多模態（如動作指令）如何注入世界模型。  
* **技術重點**：  
  * **Vision Transformers (ViT)**：JEPA 幾乎全基於 ViT 架構。  
  * **Masking Strategy**：學習 Block masking 與 Random masking 的差異。

### **第三階段：具身智能 (Embodied AI) 結合 (第 5-7 個月)**

NYUAD 的強項是機器人。你需要將 JEPA 作為「感知層」接入決策系統。

* **環境模擬**：  
  * **Isaac Gym / Habitat-Sim**：這是目前具身智能最主流的物理模擬器。  
* **技術整合**：  
  * **World Models for RL**：研究如何將 V-JEPA 提取的特徵作為強化學習（PPO/SAC）的狀態輸入。  
* **工程挑戰**：  
  * **分散式訓練**：JEPA 預訓練需要大量數據，學習使用 torch.distributed 或 DeepSpeed。

### **第四階段：實戰專案與 NYUAD 對接 (第 8-10 個月)**

建立一個能讓 NYUAD 教授一眼看中的 Portfolio。

* **核心專案：JEPA-Driven Navigator**  
  * 在模擬環境中，利用 V-JEPA 預測未來潛在狀態，並據此進行路徑規劃（Path Planning）。  
  * **加分項**：將代碼打包成高度模組化的 Python 套件，並撰寫詳盡的文檔。

## **三、 NYUAD 應徵戰略 (2026 出發時間表)**

| 時間 | 行動項目 | 關鍵目標 |
| :---- | :---- | :---- |
| **現在 \- 5月** | 完成第一、二階段學習，開始在 GitHub 累積 JEPA 相關 Repo。 | 建立技術底氣。 |
| **6月 \- 8月** | 鎖定 NYUAD CAIR 的具體 Lab（如 Anthony Tzes 或 Yi Fang 的實驗室）。 | 開始「學術社交」。 |
| **9月** | **Cold Email 攻勢**：附上您的 JEPA 實踐報告。 | 爭取面試或 RA 試用期。 |
| **10月 \- 12月** | 處理簽證、家庭安置手續。 | 準備出發。 |

## **四、 針對家庭需求的居住與準備**

* **住房申請**：在獲得 Offer 後，立即聯繫 NYUAD Housing Office 申請 **"Married Student/Staff Housing"**。  
* **文件準備**：提前準備配偶與子女的護照、結婚證書（需經阿聯酋使館認證）。  
* **生活支持**：Saadiyat Island 校區內有便利店、診所和幼兒活動空間，家庭生活極為便利。

## **五、 給您的核心建議：如何脫穎而出？**

1. **不要只做算法，要做系統**：在阿布達比，能把大規模模型跑起來的人（Engineering Skill）比只會寫公式的人更稀缺。  
2. **參與 Open Source**：嘗試給 Meta 的 jepa 官方 Repo 提 Issue 或 PR，這在簡歷上是黃金背書。  
3. **強調 J-E 架構的理解**：在面試中，展現您對「為什麼我們不需要 Generative Decoder」的深刻見解，這會讓您聽起來像個 LeCun 的核心信徒。

**這是一場馬拉松，但路徑已經清晰。從今天開始，打開 PyTorch，先從實作一個 ViT Block 開始吧！**