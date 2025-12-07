# AI vs Human 文章分類工具

## 專案概述

打造一個輕量／透明／可視化的 Streamlit Web App / 讓使用者貼上任意段落 / 即時查看「AI-written」或「Human-written」與信心分數。整體定位為研究／教育／內容平台都可使用的 Demo。

## 最低需求 (MVP)

- 單一文字輸入介面 → 按下「Detect／偵測」→ 立即顯示 AI%／Human%。  
- 回傳信心條與數值／包含基本說明。  
- 模型可採 `sklearn` 傳統特徵 (TF-IDF / n-gram / readability / stylometric) 或 `transformers` 預訓練模型／或混合手法。  
- 前端使用 Streamlit／具備即時互動。  
- 提供至少一種結果視覺化 (confidence bar / history table)。

## 功能與範圍

| 功能 | 說明 |
| --- | --- |
| 單次偵測 | 貼上文字後回傳 AI% / Human% |
| 批次偵測 (可選) | 上傳多段文本 → 顯示每篇結果 / 匯出統計 |
| 視覺化 | 條形圖／圓餅圖／histogram 呈現分布 |
| 可擴充性 | 支援更換模型／多語／不同寫作風格 |

## 技術選型

- **模型層**  
  - Option A：`scikit-learn` + TF-IDF / n-gram / stylometry → Logistic Regression / SVM / RandomForest。  
  - Option B：`transformers` (BERT / RoBERTa / mBERT / 中文 BERT) → fine-tune 做二元分類。  
  - Option C：傳統特徵與深度向量混合 → 捕捉語意與寫作風格。

- **前端層**  
  - Streamlit UI / 支援多段輸入／圖表 (`st.pyplot` / `st.bar_chart` / `st.dataframe`)。

- **資料層 (選用)**  
  - 上傳 `.txt`／`.csv`／`.jsonl` 作批量檢測。  
  - 儲存模型或檢測結果 (CSV / JSON) 供下載。

## 技術架構

```
[ Streamlit UI ]
    │ 貼上文字 / 上傳檔案
    ▼
[ 判斷模組 (Classifier) ]
    │ 產出 { label: AI / Human / confidence }
    │ 批量模式 → 聚合統計
    ▼
[ Visualization Layer ]
    ├─ 信心分數 bar / gauge
    └─ 批量統計圖 / summary 表格
```

## UI／UX Flow

1. 使用者開啟 Streamlit App。  
2. 文字框貼上單段或多段內容。  
3. 點擊「Detect／偵測」。  
4. 單段 → 顯示 AI%／Human% 與圖表。  
5. 批次 → 顯示各篇結果／平均／histogram。  
6. 選用 → 提供 CSV／JSON 下載鍵。

## 風險與注意事項

- 任何偵測器都無法保證絕對正確 → 必須提醒使用者。  
- 若要支援多語／多風格 → 需要更多 dataset／模型容易 overfit。  
- 使用深度模型需考量推論速度／記憶體／延遲 → 影響即時體驗。

## 未來擴充方向

- 多語言 (English / 中文 / 其他)。  
- 不同寫作風格 (Academic / News / Creative)。  
- 段落熱點 highlight／顯示疑似 AI 的句子。  
- 上傳 `.docx`／`.pdf`／`.txt` 自動解析。  
- Explainability：顯示造成判斷的特徵／perplexity。  
- 導出 API／CLI／可與其他平台整合。

## 參考與靈感

- JustDone AI Detector → 功能／UI／結果呈現可參考。:contentReference[oaicite:3]{index=3}  
- 混合特徵 + 深度模型的最新研究 → 在偵測 AI-generated text 表現良好。:contentReference[oaicite:4]{index=4}

## Licence／Repo／部署

- 建議使用 MIT／Apache 2.0 等寬鬆授權。  
- Repo 可放 GitHub／GitLab。  
- 部署可用 Streamlit Cloud／Heroku／Vercel／自架伺服器。

---
