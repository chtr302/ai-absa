# src/annotation/annotation_tool.py

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================
# CONFIG
# =========================

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "splits_300"

HAS_QUAD_OPTIONS = ["", "Yes", "No"]

CATEGORY_OPTIONS = [
    "",
    "None",
    "BEHAVIOR",
    "PERFORMANCE",
    "COMPARATIVE",
    "RESOURCES",
    "TOOLING",
    "CODING",
    "KNOWLEDGE",
    "QUANTIZATION",
    "REASONING",
    "FINETUNING",
    "RAG_CONTEXT",
    "Multi",
]

SENTIMENT_OPTIONS = [
    "",
    "None",
    "Positive",
    "Negative",
    "Neutral",
    "Mixed",
]


# =========================
# FILE HELPERS
# =========================

def list_annotation_files():
    # Chỉ lấy file CSV gốc cho annotator, không lấy file đã filled
    files = sorted(SPLIT_DIR.glob("annotator_*_100_samples.csv"))
    return [f for f in files if not f.name.endswith("_filled.csv")]


def output_path_of(input_path: Path):
    # annotator_1_100_samples.csv -> annotator_1_100_samples_filled.csv
    return input_path.with_name(input_path.stem + "_filled.csv")


def load_data(input_path: Path):
    output_path = output_path_of(input_path)

    # Nếu đã annotate dở thì load file filled để làm tiếp
    if output_path.exists():
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    else:
        df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_cols = [
        "human_has_quad",
        "human_aspect",
        "human_opinion",
        "human_category_label",
        "human_sentiment_label",
        "human_quads_json",
        "annotator",
        "notes",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df.fillna(""), output_path


def save_data(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# =========================
# ANNOTATION HELPERS
# =========================

def is_done(row):
    return str(row.get("human_has_quad", "")).strip() in {"Yes", "No"}


def first_unfinished_idx(df):
    for i, row in df.iterrows():
        if not is_done(row):
            return int(i)
    return 0


def safe_value(value, options):
    value = str(value).strip()
    return value if value in options else ""


def check_json(text):
    text = str(text).strip()

    if text == "":
        return True, ""

    try:
        value = json.loads(text)

        if not isinstance(value, list):
            return False, "quads_json phải là list JSON: [] hoặc [{...}]"

        for item in value:
            if not isinstance(item, dict):
                return False, "Mỗi quad phải là object JSON."

        return True, ""

    except json.JSONDecodeError as e:
        return False, f"JSON lỗi: {e}"


def build_json_from_main(aspect, opinion, category, sentiment):
    # Tạo nhanh quads_json cho trường hợp 1 quad
    if category in {"", "None"} or sentiment in {"", "None"}:
        return "[]"

    quad = [
        {
            "aspect": "" if aspect in {"", "None", "Multi"} else aspect,
            "category": category,
            "opinion": "" if opinion in {"", "None", "Multi"} else opinion,
            "sentiment": sentiment,
        }
    ]

    return json.dumps(quad, ensure_ascii=False, indent=2)


def update_row(df: pd.DataFrame, idx: int, values: dict):
    for col, val in values.items():
        df.at[idx, col] = val


# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="ABSA Annotation",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS gọn giao diện
st.markdown(
    """
<style>
:root {
    color-scheme: dark;
}

/* Nền đen đồng bộ toàn app */
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: #000000 !important;
    color: #ffffff !important;
}

/* Header Streamlit không che phần chọn file */
[data-testid="stHeader"] {
    background: #000000 !important;
    height: 2.25rem !important;
}

/* Bỏ thanh màu/decor phía trên */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Kéo nội dung xuống để selectbox File không bị che */
.block-container {
    padding-top: 4rem !important;
    padding-bottom: 0.7rem !important;
    max-width: 1780px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.28rem;
}

textarea,
.stTextInput input,
.stSelectbox div,
.stRadio div {
    font-size: 14px !important;
}

h1 {
    font-size: 1.65rem !important;
    line-height: 1.15 !important;
    margin: 0 0 0.35rem 0 !important;
    padding: 0 !important;
}

h2, h3 {
    margin-top: 0.1rem !important;
    margin-bottom: 0.35rem !important;
}

/* Khung quy ước bên trái: bỏ nền xám, cùng màu với giao diện */
.small-rule-card {
    border: 1px solid #242424;
    border-radius: 0.7rem;
    padding: 0.7rem 0.85rem;
    background: #000000 !important;
    color: #ffffff !important;
    font-size: 0.88rem;
    line-height: 1.35;
}

.small-rule-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 1rem;
    color: #ffffff !important;
}

.small-rule-card p,
.small-rule-card ul,
.small-rule-card ol {
    margin-top: 0.25rem;
    margin-bottom: 0.35rem;
}

.small-rule-card li {
    color: #ffffff !important;
}

.small-rule-card code,
.output-caption code {
    white-space: nowrap;
    background: #111111 !important;
    color: #00d084 !important;
    border: 1px solid #242424;
    border-radius: 0.35rem;
    padding: 0.05rem 0.3rem;
}

.output-caption {
    font-size: 0.78rem;
    opacity: 0.85;
    margin-top: 0.35rem;
    color: #d0d0d0 !important;
}

/* Input/textarea/select tối màu hơn để đồng bộ */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0b0b0b !important;
    color: #ffffff !important;
    border: 1px solid #262626 !important;
}

[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #9a9a9a !important;
}

/* Button đồng bộ dark */
[data-testid="stButton"] button {
    background: #0b0b0b;
    color: #ffffff;
    border: 1px solid #262626;
}

[data-testid="stButton"] button:hover {
    border-color: #555555;
    color: #ffffff;
}

/* Progress bar nền tối */
[data-testid="stProgressBar"] > div {
    background-color: #151515 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📝 ABSA Human Annotation")


# =========================
# LOAD SELECTED FILE
# =========================

files = list_annotation_files()

if not files:
    st.error(f"Không tìm thấy file annotator CSV trong: {SPLIT_DIR}")
    st.stop()

file_names = [f.name for f in files]

top_col1, top_col2, top_col3 = st.columns([1.1, 0.7, 2.2])

with top_col1:
    selected_name = st.selectbox(
        "File",
        file_names,
        label_visibility="collapsed",
    )

input_path = SPLIT_DIR / selected_name
df, output_path = load_data(input_path)

if "current_file" not in st.session_state:
    st.session_state.current_file = selected_name

if "idx" not in st.session_state:
    st.session_state.idx = first_unfinished_idx(df)

if st.session_state.current_file != selected_name:
    st.session_state.current_file = selected_name
    st.session_state.idx = first_unfinished_idx(df)

total = len(df)

if total == 0:
    st.error("File không có dòng nào.")
    st.stop()

st.session_state.idx = max(0, min(st.session_state.idx, total - 1))
idx = st.session_state.idx
row = df.iloc[idx]

done_count = int(df.apply(is_done, axis=1).sum())

with top_col2:
    st.write(f"**Tiến độ:** {done_count}/{total}")

with top_col3:
    st.progress(done_count / total if total else 0)


# =========================
# MAIN LAYOUT
# =========================

rule_col, content_col, input_col = st.columns([1.05, 1.65, 1.35], gap="medium")


# =========================
# LEFT: QUICK RULES
# =========================

with rule_col:
    st.markdown(
        """
<div class="small-rule-card">
<h4>Quy ước</h4>
<p><b>Thao tác nhanh</b></p>
<ul>
<li>Không có đánh giá: bấm <b>No Quad</b>.</li>
<li>Có 1 đánh giá: điền form rồi bấm <b>Tạo JSON</b>.</li>
<li>Có nhiều đánh giá: chọn <code>Multi</code>/<code>Mixed</code> rồi sửa JSON tay.</li>
<li>Xong một câu: bấm <b>Lưu & Tiếp</b>.</li>
</ul>
<p><b>0 quad</b></p>
<ul>
<li><code>human_has_quad = No</code></li>
<li><code>human_aspect = None</code></li>
<li><code>human_opinion = None</code></li>
<li><code>human_category_label = None</code></li>
<li><code>human_sentiment_label = None</code></li>
<li><code>human_quads_json = []</code></li>
</ul>

<p><b>1 quad</b></p>
<ul>
<li><code>human_has_quad = Yes</code></li>
<li>Điền <code>aspect</code>, <code>opinion</code>, <code>category</code>, <code>sentiment</code> chính.</li>
<li><code>human_quads_json</code> có đúng 1 object.</li>
<li>Bấm <b>Tạo JSON</b> để tạo nhanh từ form.</li>
</ul>

<p><b>Nhiều quad</b></p>
<ul>
<li><code>human_has_quad = Yes</code></li>
<li><code>human_aspect = Multi</code></li>
<li><code>human_opinion = Multi</code></li>
<li><code>human_category_label = Multi</code> nếu có nhiều category.</li>
<li><code>human_sentiment_label = Mixed</code> nếu có nhiều sentiment.</li>
<li><code>human_quads_json</code> chứa nhiều object.</li>
</ul>

<p><b>JSON mẫu</b></p>
<ul>
<li>0 quad: <code>[]</code></li>
<li>1 hoặc nhiều quad:</li>
</ul>

<pre><code>[
  {
    "aspect": "qwen",
    "category": "PERFORMANCE",
    "opinion": "fast",
    "sentiment": "Positive"
  }
]</code></pre>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="output-caption">Output: <code>{output_path.name}</code></div>',
        unsafe_allow_html=True,
    )


# =========================
# MIDDLE: CONTENT
# =========================

with content_col:
    st.subheader(f"Nội dung cần gán — {idx + 1}/{total}")

    st.markdown("**Sentence**")
    st.text_area(
        "sentence",
        value=str(row.get("sentence_text", "")),
        height=175,
        disabled=True,
        label_visibility="collapsed",
    )

    st.markdown("**Title**")
    st.text_area(
        "title",
        value=str(row.get("thread_title", "")),
        height=85,
        disabled=True,
        label_visibility="collapsed",
    )

    st.markdown("**Context**")
    st.text_area(
        "context",
        value=str(row.get("parent_context", "")),
        height=210,
        disabled=True,
        label_visibility="collapsed",
    )

    meta1, meta2, meta3 = st.columns(3)

    with meta1:
        st.caption(f"sample_id: {row.get('sample_id', '')}")

    with meta2:
        st.caption(f"sent_idx: {row.get('sentence_index', '')}")

    with meta3:
        status = "DONE" if is_done(row) else "TODO"
        st.caption(f"status: {status}")

    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        if st.button("⬅️ Trước", use_container_width=True):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with nav2:
        if st.button("Câu chưa làm", use_container_width=True):
            st.session_state.idx = first_unfinished_idx(df)
            st.rerun()

    with nav3:
        if st.button("Tiếp ➡️", use_container_width=True):
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()


# =========================
# RIGHT: INPUT FORM
# =========================

with input_col:
    st.subheader("Nhập nhãn")

    cur_has_quad = safe_value(row.get("human_has_quad", ""), HAS_QUAD_OPTIONS)
    cur_category = safe_value(row.get("human_category_label", ""), CATEGORY_OPTIONS)
    cur_sentiment = safe_value(row.get("human_sentiment_label", ""), SENTIMENT_OPTIONS)

    cur_aspect = str(row.get("human_aspect", "")).strip()
    cur_opinion = str(row.get("human_opinion", "")).strip()
    cur_quads = str(row.get("human_quads_json", "")).strip()
    cur_annotator = str(row.get("annotator", "")).strip()
    cur_notes = str(row.get("notes", "")).strip()

    human_has_quad = st.radio(
        "has_quad",
        HAS_QUAD_OPTIONS,
        index=HAS_QUAD_OPTIONS.index(cur_has_quad),
        horizontal=True,
    )

    if human_has_quad == "No":
        cur_aspect = "None"
        cur_opinion = "None"
        cur_category = "None"
        cur_sentiment = "None"
        cur_quads = "[]"

    human_aspect = st.text_input(
        "aspect",
        value=cur_aspect,
        placeholder="qwen / llama.cpp / ollama / Multi / None",
    )

    human_opinion = st.text_input(
        "opinion",
        value=cur_opinion,
        placeholder="faster / expensive / easy to use / Multi / None",
    )

    human_category = st.selectbox(
        "category",
        CATEGORY_OPTIONS,
        index=CATEGORY_OPTIONS.index(cur_category)
        if cur_category in CATEGORY_OPTIONS
        else 0,
    )

    human_sentiment = st.selectbox(
        "sentiment",
        SENTIMENT_OPTIONS,
        index=SENTIMENT_OPTIONS.index(cur_sentiment)
        if cur_sentiment in SENTIMENT_OPTIONS
        else 0,
    )

    make_json = st.button("Tạo JSON", use_container_width=True)

    if make_json:
        cur_quads = json.dumps(
            [
                {
                    "aspect": "",
                    "category": "",
                    "opinion": "",
                    "sentiment": "",
                }
            ],
            ensure_ascii=False,
        )

    human_quads_json = st.text_area(
        "quads_json",
        value=cur_quads,
        height=150,
        placeholder=(
            '[{"aspect":"","category":"",'
            '"opinion":"","sentiment":""}]'
        ),
    )

    annotator = st.text_input(
        "annotator",
        value=cur_annotator,
        placeholder="annotator_1",
        disabled=True,
    )

    notes = st.text_input(
        "notes",
        value=cur_notes,
        placeholder="ghi chú nếu cần",
    )

    json_ok, json_msg = check_json(human_quads_json)

    if not json_ok:
        st.error(json_msg)

    values = {
        "human_has_quad": human_has_quad,
        "human_aspect": human_aspect,
        "human_opinion": human_opinion,
        "human_category_label": human_category,
        "human_sentiment_label": human_sentiment,
        "human_quads_json": human_quads_json,
        "annotator": annotator,
        "notes": notes,
    }

    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("💾 Lưu", use_container_width=True):
            if json_ok:
                update_row(df, idx, values)
                save_data(df, output_path)
                st.success("Đã lưu.")
            else:
                st.error("JSON đang lỗi.")

    with btn2:
        if st.button("✅ Lưu & Tiếp", type="primary", use_container_width=True):
            if json_ok:
                update_row(df, idx, values)
                save_data(df, output_path)
                st.session_state.idx = min(total - 1, idx + 1)
                st.rerun()
            else:
                st.error("JSON đang lỗi.")

    with btn3:
        if st.button("⚡ No Quad", use_container_width=True):
            update_row(
                df,
                idx,
                {
                    "human_has_quad": "No",
                    "human_aspect": "None",
                    "human_opinion": "None",
                    "human_category_label": "None",
                    "human_sentiment_label": "None",
                    "human_quads_json": "[]",
                    "annotator": annotator,
                    "notes": notes,
                },
            )
            save_data(df, output_path)
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

    st.caption("Nếu câu không có đánh giá, bấm **No Quad**.")