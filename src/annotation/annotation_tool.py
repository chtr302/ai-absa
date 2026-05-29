# src/annotation/annotation_tool.py

import json
import html
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# CONFIG
# =========================

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "splits_300"
COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "span_selector"

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

QUAD_CATEGORY_OPTIONS = [
    o for o in CATEGORY_OPTIONS
    if o not in {"", "None", "Multi"}
]

SENTIMENT_OPTIONS = [
    "",
    "None",
    "Positive",
    "Negative",
    "Neutral",
    "Mixed",
]

QUAD_SENTIMENT_OPTIONS = [
    o for o in SENTIMENT_OPTIONS
    if o not in {"", "None", "Mixed"}
]

EMPTY_QUAD = {
    "aspect": "",
    "category": "",
    "opinion": "",
    "sentiment": "",
}

if COMPONENT_DIR.exists():
    span_selector = components.declare_component(
        "span_selector",
        path=str(COMPONENT_DIR),
    )
else:
    span_selector = None


# =========================
# FILE HELPERS
# =========================

def list_annotation_files():
    files = sorted(SPLIT_DIR.glob("annotator_*_100_samples.csv"))
    return [f for f in files if not f.name.endswith("_filled.csv")]


def output_path_of(input_path: Path):
    return input_path.with_name(input_path.stem + "_filled.csv")


def load_data(input_path: Path):
    output_path = output_path_of(input_path)

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


def clean_quad(quad):
    if not isinstance(quad, dict):
        return EMPTY_QUAD.copy()

    return {
        "aspect": str(quad.get("aspect", "")).strip(),
        "category": str(quad.get("category", "")).strip(),
        "opinion": str(quad.get("opinion", "")).strip(),
        "sentiment": str(quad.get("sentiment", "")).strip(),
    }


def parse_quads(text):
    text = str(text).strip()

    if text == "":
        return []

    value = json.loads(text)

    if not isinstance(value, list):
        raise ValueError("quads_json phải là list JSON: [] hoặc [{...}]")

    quads = []

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Mỗi quad phải là object JSON.")
        quads.append(clean_quad(item))

    return quads


def check_json(text):
    try:
        parse_quads(text)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSON lỗi: {e}"
    except ValueError as e:
        return False, str(e)


def quads_from_row(row):
    try:
        quads = parse_quads(row.get("human_quads_json", ""))
    except (json.JSONDecodeError, ValueError):
        quads = []

    if quads:
        return quads

    has_quad = str(row.get("human_has_quad", "")).strip()
    if has_quad == "No":
        return []

    aspect = str(row.get("human_aspect", "")).strip()
    opinion = str(row.get("human_opinion", "")).strip()
    category = str(row.get("human_category_label", "")).strip()
    sentiment = str(row.get("human_sentiment_label", "")).strip()

    if any(v and v not in {"None", "Multi", "Mixed"} for v in [aspect, opinion, category, sentiment]):
        return [
            {
                "aspect": "" if aspect in {"None", "Multi"} else aspect,
                "category": "" if category in {"None", "Multi"} else category,
                "opinion": "" if opinion in {"None", "Multi"} else opinion,
                "sentiment": "" if sentiment in {"None", "Mixed"} else sentiment,
            }
        ]

    return []


def non_empty_values(quads, field):
    return sorted({str(q.get(field, "")).strip() for q in quads if str(q.get(field, "")).strip()})


def summarize_quads(quads):
    clean_quads = [clean_quad(q) for q in quads]
    clean_quads = [
        q for q in clean_quads
        if q["aspect"] or q["opinion"] or q["category"] or q["sentiment"]
    ]

    if not clean_quads:
        return {
            "human_has_quad": "No",
            "human_aspect": "None",
            "human_opinion": "None",
            "human_category_label": "None",
            "human_sentiment_label": "None",
            "human_quads_json": "[]",
        }

    aspects = non_empty_values(clean_quads, "aspect")
    opinions = non_empty_values(clean_quads, "opinion")
    categories = non_empty_values(clean_quads, "category")
    sentiments = non_empty_values(clean_quads, "sentiment")

    return {
        "human_has_quad": "Yes",
        "human_aspect": primary_value(aspects, multi_label="Multi"),
        "human_opinion": primary_value(opinions, multi_label="Multi"),
        "human_category_label": primary_value(categories, multi_label="Multi"),
        "human_sentiment_label": primary_value(sentiments, multi_label="Mixed"),
        "human_quads_json": json.dumps(clean_quads, ensure_ascii=False),
    }


def primary_value(values, multi_label):
    if len(values) == 0:
        return "None"
    if len(values) == 1:
        return values[0]
    return multi_label


def update_row(df: pd.DataFrame, idx: int, values: dict):
    for col, val in values.items():
        df.at[idx, col] = val


def row_key(selected_name, idx):
    return f"{selected_name}:{idx}"


def quad_widget_key(field, selected_name, idx, active_quad):
    return f"{field}_{selected_name}_{idx}_{active_quad}"


def load_state_for_row(row, key):
    if st.session_state.get("active_row_key") == key:
        return

    st.session_state.active_row_key = key
    st.session_state.quads = quads_from_row(row)
    if not st.session_state.quads:
        st.session_state.quads = [EMPTY_QUAD.copy()]
    st.session_state.active_quad = 0
    st.session_state.last_selection_id = ""
    st.session_state.notes_value = str(row.get("notes", "")).strip()


def ensure_active_quad():
    if "quads" not in st.session_state or not st.session_state.quads:
        st.session_state.quads = [EMPTY_QUAD.copy()]

    st.session_state.active_quad = max(
        0,
        min(st.session_state.get("active_quad", 0), len(st.session_state.quads) - 1),
    )


def apply_selection(selection):
    if not selection or not isinstance(selection, dict):
        return

    selection_id = str(selection.get("id", ""))
    text = str(selection.get("text", "")).strip()
    field = str(selection.get("field", "")).strip()

    if not selection_id or not text or field not in {"aspect", "opinion"}:
        return

    if st.session_state.get("last_selection_id") == selection_id:
        return

    ensure_active_quad()
    st.session_state.quads[st.session_state.active_quad][field] = text
    st.session_state[
        quad_widget_key(
            field,
            st.session_state.get("current_file", ""),
            st.session_state.get("idx", 0),
            st.session_state.active_quad,
        )
    ] = text
    st.session_state.last_selection_id = selection_id
    st.rerun()


def save_current_row(df, idx, output_path, annotator, notes):
    clean_quads = [clean_quad(q) for q in st.session_state.quads]
    summary = summarize_quads(clean_quads)
    summary["annotator"] = annotator
    summary["notes"] = notes
    update_row(df, idx, summary)
    save_data(df, output_path)


# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="ABSA Annotation",
    page_icon="ABSA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root { color-scheme: dark; }

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: #000000 !important;
    color: #ffffff !important;
    height: 100vh !important;
    overflow: hidden !important;
}

[data-testid="stHeader"] {
    background: #000000 !important;
    height: 1.4rem !important;
}

[data-testid="stDecoration"] { display: none !important; }

.block-container {
    box-sizing: border-box !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    padding-top: 2.8rem !important;
    padding-bottom: 1.25rem !important;
    max-width: 1820px;
}

div[data-testid="stVerticalBlock"] { gap: 0.3rem; }

textarea,
.stTextInput input,
.stSelectbox div,
.stRadio div {
    font-size: 14px !important;
}

h1 {
    font-size: 1.25rem !important;
    line-height: 1.15 !important;
    margin: 0 0 0.2rem 0 !important;
    padding: 0 !important;
}

h2, h3 {
    margin-top: 0.1rem !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #0b0b0b !important;
    color: #ffffff !important;
    border: 1px solid #262626 !important;
}

[data-testid="stTextArea"] textarea:disabled,
[data-testid="stTextInput"] input:disabled {
    opacity: 1 !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-testid="stTextArea"] textarea:disabled {
    overflow-x: auto !important;
    white-space: pre !important;
}

[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #9a9a9a !important;
}

[data-testid="stButton"] button {
    background: #0b0b0b;
    color: #ffffff;
    border: 1px solid #262626;
}

[data-testid="stButton"] button:hover {
    border-color: #555555;
    color: #ffffff;
}

[data-testid="stProgressBar"] > div {
    background-color: #151515 !important;
}

.sentence-list {
    border: 1px solid #242424;
    border-radius: 8px;
    height: calc(100vh - 13.5rem);
    overflow-y: auto;
    background: #050505;
}

.sentence-item {
    display: block;
    padding: 0.52rem 0.6rem;
    border-bottom: 1px solid #171717;
    color: #e8e8e8 !important;
    text-decoration: none;
}

.sentence-item:hover {
    background: #101010;
}

.sentence-item.active {
    background: #13241c;
    border-left: 3px solid #00d084;
}

.sentence-item.done .sentence-no {
    color: #00d084;
}

.sentence-no {
    font-weight: 700;
    color: #bbbbbb;
    margin-right: 0.35rem;
}

.sentence-title {
    display: block;
    color: #bbbbbb;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.2rem;
}

.output-caption {
    font-size: 0.78rem;
    opacity: 0.85;
    margin-top: 0.35rem;
    color: #d0d0d0 !important;
}

.output-caption code {
    white-space: nowrap;
    background: #111111 !important;
    color: #00d084 !important;
    border: 1px solid #242424;
    border-radius: 0.35rem;
    padding: 0.05rem 0.3rem;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.2rem 0 0.35rem 0;
}

[data-testid="stRadio"] label {
    font-size: 0.86rem !important;
}

[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.35rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("ABSA Human Annotation")


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
    st.session_state.active_row_key = ""

total = len(df)

if total == 0:
    st.error("File không có dòng nào.")
    st.stop()

st.session_state.idx = max(0, min(st.session_state.idx, total - 1))
idx = st.session_state.idx
row = df.iloc[idx]
load_state_for_row(row, row_key(selected_name, idx))
ensure_active_quad()

done_count = int(df.apply(is_done, axis=1).sum())

with top_col2:
    st.write(f"**Tiến độ:** {done_count}/{total}")

with top_col3:
    st.progress(done_count / total if total else 0)


# =========================
# MAIN LAYOUT
# =========================

list_col, content_col, input_col = st.columns([1.05, 1.65, 1.35], gap="medium")


# =========================
# LEFT: SENTENCE LIST
# =========================

with list_col:
    st.subheader("Danh sách câu")

    search_text = st.text_input(
        "Tìm",
        value="",
        placeholder="lọc theo title hoặc sentence",
        label_visibility="collapsed",
    ).strip().lower()

    rows_for_list = []
    for i, item in df.iterrows():
        title = str(item.get("thread_title", "")).strip()
        sentence = str(item.get("sentence_text", "")).strip()
        if search_text and search_text not in f"{title} {sentence}".lower():
            continue
        rows_for_list.append((int(i), item))

    sentence_items = []
    for i, item in rows_for_list:
        title = str(item.get("thread_title", "")).strip() or "(no title)"
        sentence = str(item.get("sentence_text", "")).strip() or "(empty sentence)"
        status_class = "done" if is_done(item) else "todo"
        active_class = "active" if i == idx else ""
        marker = "✓" if is_done(item) else "·"
        sentence_items.append(
            f"""
<a class="sentence-item {status_class} {active_class}" href="?row={i}">
  <span class="sentence-no">{i + 1:03d} {marker}</span>
  <span>{html.escape(sentence[:90])}</span>
  <span class="sentence-title">{html.escape(title[:110])}</span>
</a>
""",
        )

    st.markdown(
        f'<div class="sentence-list">{"".join(sentence_items)}</div>',
        unsafe_allow_html=True,
    )

    query_row = st.query_params.get("row")
    if query_row is not None:
        try:
            query_idx = int(query_row)
            if 0 <= query_idx < total and query_idx != idx:
                st.session_state.idx = query_idx
                st.query_params.clear()
                st.rerun()
        except ValueError:
            st.query_params.clear()

    st.markdown(
        f'<div class="output-caption">Output: <code>{output_path.name}</code></div>',
        unsafe_allow_html=True,
    )


# =========================
# MIDDLE: SELECTABLE CONTENT
# =========================

with content_col:
    st.subheader(f"Nội dung cần gán - {idx + 1}/{total}")

    if span_selector is not None:
        selector_value = span_selector(
            sentence=str(row.get("sentence_text", "")),
            title=str(row.get("thread_title", "")),
            context=str(row.get("parent_context", "")),
            active_quad=st.session_state.active_quad + 1,
            key=f"selector_{selected_name}_{idx}",
            default=None,
        )
        apply_selection(selector_value)
    else:
        st.text_area(
            "Sentence",
            value=str(row.get("sentence_text", "")),
            height=175,
            disabled=True,
        )
        st.text_area(
            "Title",
            value=str(row.get("thread_title", "")),
            height=85,
            disabled=True,
        )
        st.text_area(
            "Context",
            value=str(row.get("parent_context", "")),
            height=210,
            disabled=True,
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
        if st.button("Trước", use_container_width=True):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with nav2:
        if st.button("Câu chưa làm", use_container_width=True):
            st.session_state.idx = first_unfinished_idx(df)
            st.rerun()

    with nav3:
        if st.button("Tiếp", use_container_width=True):
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()


# =========================
# RIGHT: QUAD EDITOR
# =========================

with input_col:
    st.subheader("Quads")

    quad_labels = []
    for i, quad in enumerate(st.session_state.quads, start=1):
        aspect = quad.get("aspect", "").strip() or "aspect?"
        opinion = quad.get("opinion", "").strip() or "opinion?"
        quad_labels.append(f"Quad {i}: {aspect} / {opinion}")

    active_label = st.radio(
        "Quad đang sửa",
        quad_labels,
        index=st.session_state.active_quad,
        label_visibility="collapsed",
    )
    st.session_state.active_quad = quad_labels.index(active_label)
    active_quad = st.session_state.quads[st.session_state.active_quad]
    add_col, del_col = st.columns(2)

    with add_col:
        if st.button("Thêm quad", use_container_width=True):
            st.session_state.quads.append(EMPTY_QUAD.copy())
            st.session_state.active_quad = len(st.session_state.quads) - 1
            st.rerun()

    with del_col:
        if st.button("Xóa quad", use_container_width=True):
            if len(st.session_state.quads) > 1:
                st.session_state.quads.pop(st.session_state.active_quad)
                st.session_state.active_quad = max(0, st.session_state.active_quad - 1)
            else:
                st.session_state.quads = [EMPTY_QUAD.copy()]
                st.session_state.active_quad = 0
            st.rerun()

    active_quad["aspect"] = st.text_input(
        "aspect",
        value=active_quad.get("aspect", ""),
        placeholder="bôi đen text rồi bấm Aspect",
        key=quad_widget_key("aspect", selected_name, idx, st.session_state.active_quad),
    )

    active_quad["opinion"] = st.text_input(
        "opinion",
        value=active_quad.get("opinion", ""),
        placeholder="bôi đen text rồi bấm Opinion",
        key=quad_widget_key("opinion", selected_name, idx, st.session_state.active_quad),
    )

    cur_category = safe_value(active_quad.get("category", ""), QUAD_CATEGORY_OPTIONS)
    category_index = (
        QUAD_CATEGORY_OPTIONS.index(cur_category)
        if cur_category in QUAD_CATEGORY_OPTIONS
        else None
    )
    selected_category = st.radio(
        "category",
        QUAD_CATEGORY_OPTIONS,
        index=category_index,
        horizontal=True,
        key=quad_widget_key("category", selected_name, idx, st.session_state.active_quad),
    )
    active_quad["category"] = selected_category or ""

    cur_sentiment = safe_value(active_quad.get("sentiment", ""), QUAD_SENTIMENT_OPTIONS)
    sentiment_index = (
        QUAD_SENTIMENT_OPTIONS.index(cur_sentiment)
        if cur_sentiment in QUAD_SENTIMENT_OPTIONS
        else None
    )
    selected_sentiment = st.radio(
        "sentiment",
        QUAD_SENTIMENT_OPTIONS,
        index=sentiment_index,
        horizontal=True,
        key=quad_widget_key("sentiment", selected_name, idx, st.session_state.active_quad),
    )
    active_quad["sentiment"] = selected_sentiment or ""

    summary = summarize_quads(st.session_state.quads)

    human_quads_json = st.text_area(
        "quads_json",
        value=summary["human_quads_json"],
        height=76,
        disabled=True,
    )

    annotator = str(row.get("annotator", "")).strip()

    st.session_state.notes_value = st.text_input(
        "notes",
        value=st.session_state.get("notes_value", ""),
        placeholder="ghi chú nếu cần",
    )

    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("Luu", use_container_width=True):
            save_current_row(
                df,
                idx,
                output_path,
                annotator,
                st.session_state.notes_value,
            )
            st.success("Da luu.")

    with btn2:
        if st.button("Luu & Tiep", type="primary", use_container_width=True):
            save_current_row(
                df,
                idx,
                output_path,
                annotator,
                st.session_state.notes_value,
            )
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

    with btn3:
        if st.button("No Quad", use_container_width=True):
            annotator = str(row.get("annotator", "")).strip()
            notes = st.session_state.get("notes_value", "")
            st.session_state.quads = []
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
