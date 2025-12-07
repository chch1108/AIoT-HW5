from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

try:  # Allow running as `streamlit run HW5/app.py` or `python -m HW5.app`.
    from .detector import HeuristicAIHumanDetector
    from .sample_texts import SAMPLE_TEXTS
except ImportError:  # pragma: no cover
    from detector import HeuristicAIHumanDetector
    from sample_texts import SAMPLE_TEXTS


st.set_page_config(page_title="AI vs Human Detector", layout="wide")


# Caches -----------------------------------------------------------------------


@st.cache_resource
def get_detector() -> HeuristicAIHumanDetector:
    return HeuristicAIHumanDetector()


def get_genai_api_key() -> Optional[str]:
    if "GENAI_API_KEY" in st.secrets:
        return st.secrets["GENAI_API_KEY"]
    return os.environ.get("GENAI_API_KEY")


GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"


@st.cache_resource(show_spinner=False)
def get_gemini_model():
    api_key = get_genai_api_key()
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - runtime warning
        st.sidebar.warning(f"Gemini 初始化失敗：{exc}")
        return None


def main() -> None:
    st.title("AI vs Human 文章分類工具")
    st.caption("輸入文本或上傳批次檔案，使用輕量 stylometric 特徵估算 AI/Human 機率。")

    detector = get_detector()
    gemini_model = get_gemini_model()
    render_sidebar()

    single_result = render_single_detection(detector, gemini_model)
    st.divider()
    render_batch_detection(detector)

    if single_result is None:
        st.info("貼上文本並點擊「開始偵測」即可查看信心分數。")


def render_sidebar() -> None:
    st.sidebar.header("如何使用")
    st.sidebar.markdown(
        """
1. 貼上任意段落或從範例填入文字。
2. 按下 **開始偵測** 取得 AI/Human 置信度。
3. （可選）上傳 CSV/JSON 批次偵測並下載結果。

> 提醒：本偵測器為 heuristics 示範，不代表絕對真實。
"""
    )
    if not get_genai_api_key():
        st.sidebar.info("於 Streamlit secrets 設定 GENAI_API_KEY 可啟用 Gemini 雙重檢查。")

    st.sidebar.subheader("特徵說明")
    st.sidebar.write(
        "- **Complexity**: 句長與字長的綜合指標。\n"
        "- **Burstiness**: 句子的長短變化，偏高通常較人味。\n"
        "- **Repetition**: 單字重複度，越高越可能為 AI。\n"
        "- **Diversity/Entropy**: 字詞多樣性與資訊量。\n"
        "- **Stopwords/Punctuation**: 常見功能詞與標點比例。"
    )


def render_single_detection(detector: HeuristicAIHumanDetector, gemini_model=None):
    st.subheader("單次偵測")
    st.write("貼上文字後按下按鈕即可取得機率與特徵解讀。")

    st.session_state.setdefault("single_input", "")
    with st.expander("需要靈感？點擊範例文字"):
        cols = st.columns(len(SAMPLE_TEXTS))
        for idx, sample in enumerate(SAMPLE_TEXTS):
            if cols[idx].button(sample["label"]):
                st.session_state["single_input"] = sample["text"]

    with st.form("single-detect"):
        text = st.text_area("輸入欲偵測的文章段落", height=220, key="single_input")
        submitted = st.form_submit_button("開始偵測")

    if not submitted:
        return None

    if not text.strip():
        st.warning("請輸入文字後再進行偵測。")
        return None

    result = detector.predict(text)
    cols = st.columns(2)
    cols[0].metric("AI 機率", f"{result.ai_score * 100:.1f} %")
    cols[1].metric("Human 機率", f"{result.human_score * 100:.1f} %")
    st.progress(result.ai_score, text="AI 置信度")

    if result.ai_score > 0.65:
        st.error("模型推測較像 AI 撰寫，建議人工再複核。")
    elif result.ai_score < 0.35:
        st.success("模型推測較像 Human 撰寫。")
    else:
        st.info("屬於灰色區域，建議搭配其他證據判斷。")

    features_df = pd.DataFrame(
        [{"Feature": name, "Value": round(value, 3)} for name, value in result.features.items()]
    )
    st.dataframe(features_df, hide_index=True, use_container_width=True)

    explanations = build_feature_explanations(result.features)
    st.markdown("**特徵觀察**")
    for note in explanations:
        st.write(f"- {note}")

    render_gemini_section(gemini_model, text)

    return result


def build_feature_explanations(features: dict) -> List[str]:
    notes = []
    if features.get("burstiness", 0) < 0.25:
        notes.append("句子長度變化低，常見於較平鋪的 AI 敘述。")
    if features.get("repetition", 0) > 0.22:
        notes.append("關鍵詞重複度偏高。")
    if features.get("diversity", 0) < 0.5:
        notes.append("字詞多樣性較低，建議檢視是否為模板化內容。")
    if features.get("entropy", 0) > 0.65:
        notes.append("資訊熵高，呈現較自由的人類寫作風格。")
    if features.get("stopword_ratio", 0) > 0.55:
        notes.append("停用詞比例偏高，代表語句可能更規整。")
    if not notes:
        notes.append("特徵落在常見範圍，單一指標不足以判斷。")
    return notes


def render_batch_detection(detector: HeuristicAIHumanDetector) -> None:
    st.subheader("批次偵測（選用）")
    st.write("支援 CSV／JSON Lines，需包含 `text` 欄位。")

    uploaded_file = st.file_uploader("上傳檔案", type=["csv", "json"])
    if not uploaded_file:
        return

    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    df = parse_uploaded_dataframe(uploaded_file.name, file_bytes)
    if df is None or df.empty:
        st.error("無法解析檔案，請確認格式與 `text` 欄位。")
        return

    texts = df["text"].astype(str).tolist()
    results = detector.batch_predict(texts)
    result_df = pd.DataFrame(
        {
            "text": texts,
            "label": [r.label for r in results],
            "ai_score": [r.ai_score for r in results],
            "human_score": [r.human_score for r in results],
        }
    )

    counts = result_df["label"].value_counts().rename_axis("label").reset_index(name="count")
    avg_ai = result_df["ai_score"].mean()

    st.metric("平均 AI 機率", f"{avg_ai * 100:.1f} %")
    st.dataframe(result_df, use_container_width=True)
    st.bar_chart(data=counts, x="label", y="count")
    st.line_chart(result_df["ai_score"], height=200)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "下載結果 CSV",
        data=csv_buffer.getvalue(),
        file_name="batch_detection.csv",
        mime="text/csv",
    )


def parse_uploaded_dataframe(filename: str, file_bytes: bytes):
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_json(io.BytesIO(file_bytes), lines=True)
        except ValueError:
            df = pd.read_json(io.BytesIO(file_bytes))
    if "text" not in df.columns:
        return None
    return df[["text"]]


def render_gemini_section(gemini_model, text: str) -> None:
    st.markdown("### Gemini 雙重檢查")

    if genai is None:
        st.info("尚未安裝 google-generativeai 套件，無法啟用 Gemini。")
        return

    if not get_genai_api_key():
        st.info("請在 Streamlit secrets 或環境變數設定 GENAI_API_KEY。")
        return

    if gemini_model is None:
        st.warning("Gemini 初始化失敗，請稍後再試。")
        return

    with st.spinner("Gemini 分析中..."):
        try:
            result = run_gemini_check(gemini_model, text)
        except Exception as exc:
            st.error(f"Gemini 呼叫失敗：{exc}")
            return

    cols = st.columns(2)
    cols[0].metric("Gemini 判斷", result["label"])
    cols[1].metric("AI 機率 (Gemini)", f"{result['ai_probability'] * 100:.1f} %")
    st.write("Gemini 說明：", result["explanation"])


def run_gemini_check(model, text: str) -> Dict[str, float | str]:
    prompt = (
        "You are an assistant that verifies whether text was written by AI systems or humans. "
        "Return ONLY valid JSON with keys: label (\"AI-written\" or \"Human-written\"), "
        "ai_probability (0-1 float), human_probability (0-1 float), explanation (short reasoning in Traditional Chinese)."
    )
    response = model.generate_content(
        [
            prompt,
            f"請判斷以下文本：\n{text}",
        ]
    )
    raw_text = (response.text or "").strip()
    payload = _extract_json(raw_text)
    label = payload.get("label", "Unknown")
    ai_prob = float(payload.get("ai_probability", 0.5))
    human_prob = float(payload.get("human_probability", 1 - ai_prob))
    explanation = payload.get("explanation", raw_text)
    return {
        "label": label,
        "ai_probability": max(0.0, min(1.0, ai_prob)),
        "human_probability": max(0.0, min(1.0, human_prob)),
        "explanation": explanation,
    }


def _extract_json(raw_text: str) -> Dict[str, object]:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate[3:]
        candidate = candidate.strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
