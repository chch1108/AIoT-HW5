# AI vs Human 文章分類工具

## 專案簡介  
建立一個簡單易用的 Web 應用，用於判斷一段文字是「由 AI 生成 (AI-written)」還是「人類撰寫 (Human-written)」。使用者只需在 UI 中輸入文字，即可 **即時顯示判斷結果**（例如「AI: 87% / Human: 13%」）。

目的是提供一個輕量、透明、可視化 (confidence + 統計) 的工具；對使用者、研究者、教育工作者或內容平台都可能有價值。

## 最低需求 (MVP)  

- 使用者 Interface：輸入一段文本 → 按下「判斷／偵測 (Detect)」。  
- 即時返回結果：分類為 AI 或 Human，並顯示信心分數 (e.g. AI% / Human%)。  
- 技術選擇自由：可用 `sklearn` + 傳統特徵 (e.g. TF-IDF, n-gram, readability, 統計特徵)；也可用 `transformers` (預訓練模型，或 fine-tune)；或混合特徵法。  
- 前端／介面：使用 Streamlit 作為 UI。  
- 結果可視化或簡單統計 (例如 confidence bar, 批量偵測後的統計分布等)。

## 功能／範圍 (Scope)  

| 功能 | 描述 |
|------|------|
| 單次偵測 | 使用者貼入文字 → 顯示 single-text 判斷 (AI% / Human%) |
| 批量偵測 (optional) | 可上傳多篇文章 (或多段文本)，一次輸出所有判斷結果 + 統計摘要 (分布、平均值等) |
| 可視化 | 用條形圖、圓餅圖、histogram 等簡易圖表展示 confidence / 分布 |
| 可擴充性 | 未來可換模型 (輕量 → 深度)、支援多語言 / 多風格 (正式／非正式／學術／小說) |

## 技術選型 (Tech Stack)  

- 後端 / 模型部分  
  - Option A: `scikit-learn`，使用 TF-IDF + n-gram + maybe readability / stylometric features → 用 Logistic Regression / SVM / RandomForest 等分類器。  
  - Option B: `transformers` (e.g. BERT / RoBERTa / mBERT / 中文 BERT) → 若有 dataset，可 fine-tune 做二分類 (AI vs Human)。  
  - Option C: 混合特徵 (traditional + embedding + stylometry) → 讓模型既捕捉語意，也捕捉寫作風格。  

- 前端 / UI  
  - Streamlit：簡單、即時、易用。支援文字輸入、多段上傳、即時圖表 (with `st.pyplot`, `st.bar_chart`, `st.dataframe` 等)。  

- (Optional) Dataset 與保存  
  - 如果希望批量測試與統計：可讓使用者選擇上傳 `.txt` / `.csv` / `.jsonl`。  
  - 模型 + 結果可序列化 / 保存 /下載 (CSV / JSON)。  

## 技術架構 (Architecture)  

[ Streamlit UI ] <-- web 瀏覽器
│
│── 使用者輸入 → 提交文字
▼
[ 判斷模組 (Classifier) ]
│
└── 輸出: { label: "AI" / "Human", confidence: float }
│
└── (如有批量) → aggregate → 統計 & 可視化模組
▼
[ Visualization / Result Display ]
└── 信心分數 (bar / gauge)
└── (批量) 統計圖 (分布圖 / histogram / summary table)


## UI／UX Flow  

1. 使用者開啟網頁 (Streamlit)  
2. 在文字框貼入一段 (或多段) 文字  
3. 點擊「Detect / 偵測」按鈕  
4. 若是單段：立即顯示結果 (AI% / Human%)，並以 bar / percentage 視覺化  
5. 若是上傳多段 (batch mode)：列出每篇的判斷 + confidence，並顯示整體統計 (例如 histogram, 平均 confidence)  
6. (Optional) 提供「Download results」功能 (CSV / JSON)  

## 風險與注意事項  

- 即使偵測模型表現不錯，也 **不保證** 絕對正確。正如市面上的 JustDone 等 AI-detector 都說：「沒有偵測器是完全準確的」。:contentReference[oaicite:2]{index=2}  
- 如果未來想擴展到多語言 /多風格 (學術／新聞／口語／小說)，需要對應多樣 dataset，且 classifier 容易 overfit。  
- 若使用深度模型 (transformers)，需要注意推論速度與資源 (memory / GPU / latency)；對於即時 Web App 要做權衡。  

## 未來擴充 / 進階功能 (Future Work)  

- 支援多語言 (英文、中文、其他)  
- 支援不同風格分類 (academic / casual / news / creative)  
- 加入「highlight 可疑段落」功能 — 標出最可能是 AI-generated 的句子或段落 (類似 just-in-document “heatmap”)  
- 讓使用者上傳文件 (.docx / .pdf / .txt) 自動解析並偵測  
- 加入「解釋 (explainability)」 — 顯示哪些特徵 (語句長度、詞彙重複、predictability…) 導致該判斷 (例如 stylometric / perplexity indicators)  
- 提供 API / CLI 工具，以便整合到其他系統／平台  

## 參考與靈感  

- Online 工具 JustDone AI Detector — 一個現成的 AI vs Human 偵測器，可作為功能、UI 與結果顯示的參考。:contentReference[oaicite:3]{index=3}  
- 最新研究 — 混合特徵 + 深度模型的 hybrid 方法，在檢測 AI-generated text 上有不錯效果。:contentReference[oaicite:4]{index=4}  

## Licence / Repo & Deployment  

建議使用 MIT / Apache 2.0 這類 permissive licence，方便開源／分享。  
可將專案放在 GitHub / GitLab，並用 Streamlit Cloud / Heroku / Vercel（Streamlit 適配方案）快速部署。  

---
