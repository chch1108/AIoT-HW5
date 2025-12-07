# AI vs Human Detector

簡單的 Streamlit Web App，用於估測一段文字較可能由 AI 或人類撰寫。核心以輕量 stylometric 特徵（句長變化、字詞多樣性、重複度、停用詞比例等）為基礎，提供即時信心分數與解釋提示。

## Features

- 單次偵測：貼上文字並取得 AI/Human 百分比、特徵表與文字觀察。
- 範例文字：內建中英範例可快速測試 UI。
- 批次偵測：上傳 `text` 欄位的 CSV / JSON(或 JSONL) 批次計算，顯示統計圖並可下載結果。
- 完全本地運算，不需 API Key；若提供 `GENAI_API_KEY` 可串接 Gemini 做雲端雙重檢查。

## Getting Started

```bash
git clone https://github.com/chch1108/AIoT-HW5.git
cd AIoT-HW5
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

開啟瀏覽器後即可輸入文字或上傳檔案。若以 `python -m HW5.app` 方式啟動也支援內部匯入。

### Gemini 雙重檢查（選用）

1. 於本機環境變數或 `.streamlit/secrets.toml` 放入 `GENAI_API_KEY="你的金鑰"`（或 `GOOGLE_API_KEY`，於 Google AI Studio 取得）。
2. 重新啟動 Streamlit，即會在單次偵測結果下方自動呼叫 Gemini 分析，並以 JSON 回傳第二組信心分數。

## File Overview

| File | Description |
| --- | --- |
| `app.py` | Streamlit 介面：單次偵測、批次分析、圖表與下載按鈕。 |
| `detector.py` | `HeuristicAIHumanDetector` 核心邏輯與特徵計算。 |
| `sample_texts.py` | UI 範例段落。 |
| `project.md` | 原始作業規劃文件。 |
| `requirements.txt` | 最小相依套件（含 Streamlit、pandas、google-generativeai）。 |

## Deployment

- Streamlit Cloud：將此 repo 上傳後於 Dashboard 中新增 app，命令設定為 `streamlit run app.py`。
- 其他平台（如 Render/Vercel）：建立 Python service 並同樣執行 `streamlit run app.py`，或透過 Docker 包裝。

## Limitations & Future Ideas

- 目前使用啟發式統計特徵，主要展示概念，不代表真實鑑別能力。
- 可加入 TF-IDF/ML 模型、transformers、語言偵測與多語資料強化。
- 支援匯出 JSON、提供 API、顯示句子 hot spot 等皆可延伸。

歡迎 Issue / PR 一起改進！ 🎯
