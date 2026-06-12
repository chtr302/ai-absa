# src/annotation/annotation_tool.py

import json
import html
import os
import logging
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# QUIET TERMINAL LOGS
# =========================
# Tool review chạy rerun liên tục, nên tắt bớt log/warning không cần thiết trên terminal.
os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

for logger_name in [
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.caching",
    "watchdog",
    "tornado",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logging.getLogger().setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "human_verification_6000"

CATEGORY_OPTIONS = [
    "",
    "PERFORMANCE",
    "INTELLIGENCE",
    "RESOURCES",
    "BEHAVIOR",
    "TECHNICAL",
    "SOFTWARE",
    "COMPARATIVE",
]

SENTIMENT_OPTIONS = ["", "Positive", "Negative", "Neutral"]
EMPTY_QUAD = {"aspect": "", "category": "", "opinion": "", "sentiment": ""}


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def list_annotation_files():
    files = sorted(SPLIT_DIR.glob("annotator*_2000.csv"))
    return [f for f in files if not f.name.endswith("_reviewed.csv") and not f.name.endswith("_filled.csv")]


def output_path_of(input_path: Path):
    return input_path.with_name(input_path.stem + "_reviewed.csv")


def read_csv_keep_schema(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df.fillna("")


def ensure_min_cols(df: pd.DataFrame):
    for col in ["id", "parent_context", "thread_title", "sentence", "quad"]:
        if col not in df.columns:
            df[col] = "[]" if col == "quad" else ""
    return df


def load_data(input_path: Path):
    output_path = output_path_of(input_path)
    original_df = ensure_min_cols(read_csv_keep_schema(input_path))
    if output_path.exists():
        df = ensure_min_cols(read_csv_keep_schema(output_path))
    else:
        df = original_df.copy()
    return original_df, df, output_path


def save_data(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def infer_done_from_changed_quads(original_df: pd.DataFrame, reviewed_df: pd.DataFrame):
    done = set()
    n = min(len(original_df), len(reviewed_df))
    if "quad" not in original_df.columns or "quad" not in reviewed_df.columns:
        return done
    for i in range(n):
        if str(original_df.at[i, "quad"]).strip() != str(reviewed_df.at[i, "quad"]).strip():
            done.add(i)
    return done


def clean_quad(quad):
    if not isinstance(quad, dict):
        return EMPTY_QUAD.copy()
    return {
        "aspect": str(quad.get("aspect", "")).strip(),
        "category": str(quad.get("category", "")).strip(),
        "opinion": str(quad.get("opinion", "")).strip(),
        "sentiment": str(quad.get("sentiment", "")).strip(),
    }


def parse_quads(value):
    if isinstance(value, list):
        raw = value
    else:
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        raw = json.loads(text)
    if not isinstance(raw, list):
        return []
    return [clean_quad(q) for q in raw if isinstance(q, dict)]


def quads_from_row(row):
    try:
        quads = parse_quads(row.get("quad", "[]"))
    except Exception:
        quads = []
    return quads if quads else [EMPTY_QUAD.copy()]


def normalize_quads_for_save(quads):
    clean = []
    for q in quads:
        cq = clean_quad(q)
        if cq["aspect"] or cq["category"] or cq["opinion"] or cq["sentiment"]:
            clean.append(cq)
    return clean


def validate_quads_before_save(quads):
    clean = normalize_quads_for_save(quads)
    if not clean:
        return True, ""
    for i, q in enumerate(clean, start=1):
        missing = [field for field in ["aspect", "category", "opinion", "sentiment"] if not q[field]]
        if missing:
            return False, f"Quad {i} thiếu: {', '.join(missing)}"
    return True, ""


def short(text, n=90):
    text = str(text or "").replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


def safe_value(value, options):
    value = str(value).strip()
    return value if value in options else ""


def row_key(selected_name, idx):
    return f"{selected_name}:{idx}"


def field_key(field, selected_name, idx, active_quad):
    return f"{field}_{selected_name}_{idx}_{active_quad}"


def load_state_for_row(row, key):
    if st.session_state.get("active_row_key") == key:
        return
    st.session_state.active_row_key = key
    st.session_state.quads = quads_from_row(row)
    st.session_state.active_quad = 0


def ensure_active_quad():
    if "quads" not in st.session_state or not st.session_state.quads:
        st.session_state.quads = [EMPTY_QUAD.copy()]
    st.session_state.active_quad = max(0, min(st.session_state.get("active_quad", 0), len(st.session_state.quads) - 1))


def first_unreviewed_idx(total, done_indexes):
    for i in range(total):
        if i not in done_indexes:
            return i
    return 0


def select_row(i: int):
    st.session_state.idx = int(i)
    st.session_state.active_row_key = ""
    st.session_state.active_quad = 0


def save_current(df, output_path, idx, done_key, done_indexes):
    ok, msg = validate_quads_before_save(st.session_state.quads)
    if not ok:
        st.error(msg)
        return False
    df.at[idx, "quad"] = compact_json(normalize_quads_for_save(st.session_state.quads))
    save_data(df, output_path)
    done_indexes.add(idx)
    st.session_state[done_key] = done_indexes
    return True


def render_selectable_content(sentence_value, title_value, context_value, idx):
    """
    Bôi đen text sẽ hiện popup ngay cạnh selection:
      A -> fill aspect
      O -> fill opinion
    Có thể bấm popup hoặc nhấn phím A/O.
    Aspect lấy từ Comment/Title/Parent context.
    Opinion chỉ lấy từ Comment.
    """
    def esc(x):
        return html.escape(str(x or ""))

    html_block = f"""
<style>
html, body {{
  margin:0;
  padding:0;
  background:#000;
  color:#fff;
  font-family:Arial,sans-serif;
  font-size:14px;
  overflow:hidden;
}}
.box-label {{
  color:#aaa;
  font-size:12px;
  margin:6px 0 3px;
}}
.box {{
  border:1px solid #333;
  background:#070707;
  padding:10px;
  border-radius:6px;
  white-space:pre-wrap;
  overflow-y:auto;
  line-height:1.4;
  user-select:text;
  -webkit-user-select:text;
}}
.comment {{ height:185px; }}
.title {{ height:65px; }}
.context {{ height:130px; }}
#msg_{idx} {{
  color:#aaa;
  font-size:12px;
  margin-bottom:6px;
  min-height:16px;
}}
#float_{idx} {{
  position:fixed;
  z-index:999999;
  display:none;
  align-items:center;
  gap:6px;
  background:#111;
  border:1px solid #444;
  border-radius:8px;
  padding:5px;
  box-shadow:0 6px 18px rgba(0,0,0,.45);
}}
#float_{idx} button {{
  background:#050505;
  color:#fff;
  border:1px solid #555;
  border-radius:6px;
  padding:4px 10px;
  cursor:pointer;
  font-weight:700;
  min-width:34px;
}}
#float_{idx} button:hover {{
  border-color:#00d084;
  color:#00d084;
}}
#float_{idx} button:disabled {{
  opacity:.35;
  cursor:not-allowed;
  color:#aaa;
}}
#float_{idx} span {{
  color:#aaa;
  font-size:12px;
}}
::selection {{ background:#0d6efd; color:#fff; }}
</style>

<div id="msg_{idx}">Bôi đen text, popup A/O sẽ hiện ngay cạnh chữ. Có thể nhấn phím A hoặc O.</div>

<div id="float_{idx}">
  <button id="float_a_{idx}" title="Fill aspect">A</button>
  <button id="float_o_{idx}" title="Fill opinion">O</button>
  <span>A=aspect, O=opinion</span>
</div>

<div class="box-label">Comment</div>
<div class="box comment" data-src="comment">{esc(sentence_value)}</div>

<div class="box-label">Thread title</div>
<div class="box title" data-src="title">{esc(title_value)}</div>

<div class="box-label">Parent context</div>
<div class="box context" data-src="context">{esc(context_value)}</div>

<script>
(function() {{
  let rootDoc = document;
  try {{
    if (window.parent && window.parent.document) rootDoc = window.parent.document;
  }} catch(e) {{
    rootDoc = document;
  }}

  const msg = document.getElementById("msg_{idx}");
  const floatBox = document.getElementById("float_{idx}");
  const btnA = document.getElementById("float_a_{idx}");
  const btnO = document.getElementById("float_o_{idx}");

  let lastText = "";
  let lastSrc = "";

  function setMsg(text, ok) {{
    msg.textContent = text || "";
    msg.style.color = ok ? "#00d084" : "#ff7777";
  }}

  function hideFloat() {{
    floatBox.style.display = "none";
  }}

  function sourceFromNode(node) {{
    let el = node && node.nodeType === 3 ? node.parentElement : node;
    while (el && el !== document.body) {{
      if (el.dataset && el.dataset.src) return el.dataset.src;
      el = el.parentElement;
    }}
    return "";
  }}

  function readSelection() {{
    const sel = document.getSelection ? document.getSelection() : window.getSelection();
    const text = sel ? String(sel.toString()).trim() : "";
    if (!text || !sel || sel.rangeCount === 0) return null;

    const srcA = sourceFromNode(sel.anchorNode);
    const srcB = sourceFromNode(sel.focusNode);

    if (!srcA || srcA !== srcB) return null;

    return {{
      text: text,
      src: srcA,
      range: sel.getRangeAt(0)
    }};
  }}

  function showFloat(info) {{
    if (!info) {{
      lastText = "";
      lastSrc = "";
      hideFloat();
      return;
    }}

    lastText = info.text;
    lastSrc = info.src;

    btnO.disabled = info.src !== "comment";

    const rect = info.range.getBoundingClientRect();
    if (!rect || (!rect.width && !rect.height)) {{
      hideFloat();
      return;
    }}

    floatBox.style.display = "flex";
    floatBox.style.left = "0px";
    floatBox.style.top = "0px";

    const boxRect = floatBox.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - boxRect.width / 2;
    let top = rect.top - boxRect.height - 8;

    if (top < 4) top = rect.bottom + 8;
    left = Math.max(4, Math.min(left, window.innerWidth - boxRect.width - 4));
    top = Math.max(4, Math.min(top, window.innerHeight - boxRect.height - 4));

    floatBox.style.left = left + "px";
    floatBox.style.top = top + "px";

    const shortText = info.text.length > 70 ? info.text.slice(0, 70) + "…" : info.text;
    const srcName = info.src === "comment" ? "Comment" : (info.src === "title" ? "Thread title" : "Parent context");
    setMsg("Đã chọn từ " + srcName + ": “" + shortText + "”", true);
  }}

  function updateSelection() {{
    showFloat(readSelection());
  }}

  function setInput(label, value) {{
    const labels = Array.from(rootDoc.querySelectorAll("label"));
    let target = null;

    for (const lab of labels) {{
      const txt = (lab.innerText || lab.textContent || "").trim().toLowerCase();
      if (txt === label.toLowerCase()) {{
        const wrap = lab.closest('[data-testid="stTextInput"]') || lab.parentElement;
        if (wrap) target = wrap.querySelector("input");
        if (target) break;
      }}
    }}

    if (!target) return false;

    const setter = Object.getOwnPropertyDescriptor(rootDoc.defaultView.HTMLInputElement.prototype, "value").set;
    setter.call(target, value);
    target.dispatchEvent(new Event("input", {{bubbles:true}}));
    target.dispatchEvent(new Event("change", {{bubbles:true}}));
    target.focus();
    return true;
  }}

  function apply(field) {{
    const info = readSelection();
    if (info) {{
      lastText = info.text;
      lastSrc = info.src;
    }}

    const text = (lastText || "").trim();
    const src = lastSrc || "";

    if (!text) {{
      setMsg("Chưa bôi đen text nào.", false);
      return;
    }}

    if (field === "opinion" && src !== "comment") {{
      setMsg("Opinion chỉ được chọn từ Comment.", false);
      return;
    }}

    const ok = setInput(field, text);
    if (ok) {{
      setMsg("Đã fill " + field + ".", true);
      hideFloat();
      const sel = document.getSelection ? document.getSelection() : window.getSelection();
      if (sel) sel.removeAllRanges();
    }} else {{
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      setMsg("Không tự fill được input, đã copy text vào clipboard.", false);
    }}
  }}

  function handleKey(e) {{
    if (!e || e.ctrlKey || e.metaKey || e.altKey) return;

    const tag = (e.target && e.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || (e.target && e.target.isContentEditable)) return;

    const key = String(e.key || "").toLowerCase();
    if (key === "a") {{
      e.preventDefault();
      e.stopPropagation();
      apply("aspect");
      return false;
    }}
    if (key === "o") {{
      e.preventDefault();
      e.stopPropagation();
      apply("opinion");
      return false;
    }}
  }}

  document.addEventListener("selectionchange", function() {{
    setTimeout(updateSelection, 0);
  }});
  document.addEventListener("mouseup", updateSelection);
  document.addEventListener("keyup", updateSelection);
  document.addEventListener("keydown", handleKey, true);

  btnA.addEventListener("mousedown", function(e) {{ e.preventDefault(); }});
  btnO.addEventListener("mousedown", function(e) {{ e.preventDefault(); }});
  btnA.addEventListener("click", function() {{ apply("aspect"); }});
  btnO.addEventListener("click", function() {{ apply("opinion"); }});

  try {{
    rootDoc.addEventListener("keydown", handleKey, true);
  }} catch(e) {{}}
}})();
</script>
"""
    components.html(html_block, height=470, scrolling=False)


st.set_page_config(page_title="ABSA Review", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
html, body, .stApp { background:#000 !important; color:#fff !important; }
[data-testid="stHeader"] { background:#000 !important; height:1.5rem !important; }
.block-container { padding-top:2.2rem !important; max-width:1850px; }
h1 { font-size:1.2rem !important; margin-bottom:0.2rem !important; }
h2, h3 { margin-top:0.1rem !important; margin-bottom:0.25rem !important; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { background:#080808 !important; color:#fff !important; border:1px solid #333 !important; }
[data-testid="stButton"] button { background:#111; color:#fff; border:1px solid #333; }
[data-testid="stButton"] button:hover { border-color:#777; color:#fff; }
.small { color:#aaa; font-size:12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("ABSA Review - chỉ sửa cột quad")

files = list_annotation_files()
if not files:
    st.error(f"Không tìm thấy annotator*_2000.csv trong: {SPLIT_DIR}")
    st.stop()

file_names = [f.name for f in files]
top1, top2, top3 = st.columns([1.2, 0.8, 2.0])
with top1:
    selected_name = st.selectbox("File", file_names, label_visibility="collapsed")

input_path = SPLIT_DIR / selected_name
original_df, df, output_path = load_data(input_path)

done_key = f"done_indexes::{selected_name}"
if done_key not in st.session_state:
    st.session_state[done_key] = set()
done_indexes = set(st.session_state[done_key]) | infer_done_from_changed_quads(original_df, df)
st.session_state[done_key] = done_indexes

total = len(df)
if total == 0:
    st.error("File không có dòng nào.")
    st.stop()

if "current_file" not in st.session_state:
    st.session_state.current_file = selected_name
if "idx" not in st.session_state:
    st.session_state.idx = first_unreviewed_idx(total, done_indexes)
if st.session_state.current_file != selected_name:
    st.session_state.current_file = selected_name
    st.session_state.idx = first_unreviewed_idx(total, done_indexes)
    st.session_state.active_row_key = ""

st.session_state.idx = max(0, min(st.session_state.idx, total - 1))
idx = st.session_state.idx
row = df.iloc[idx]
load_state_for_row(row, row_key(selected_name, idx))
ensure_active_quad()

with top2:
    st.write(f"**Tiến độ:** {len(done_indexes)}/{total}")
with top3:
    st.progress(len(done_indexes) / total if total else 0)

list_col, content_col, quad_col = st.columns([1.05, 1.65, 1.35], gap="medium")

with list_col:
    st.subheader("Danh sách câu")
    list_df = pd.DataFrame({
        "#": list(range(1, total + 1)),
        "xong": ["✓" if i in done_indexes else "" for i in range(total)],
        "câu": [short(v, 92) for v in df["sentence"].tolist()],
    })

    # Dùng dataframe native thay vì 2000 button để giảm lag.
    event = st.dataframe(
        list_df,
        height=690,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key=f"row_table_{selected_name}",
    )
    try:
        table_selection_key = f"row_table_selection::{selected_name}"
        if table_selection_key not in st.session_state:
            st.session_state[table_selection_key] = tuple()

        selected_rows = tuple(event.selection.rows)
        if selected_rows != st.session_state[table_selection_key]:
            st.session_state[table_selection_key] = selected_rows
        else:
            selected_rows = tuple()

        if selected_rows:
            chosen = int(selected_rows[0])
            if chosen != idx:
                select_row(chosen)
                st.rerun()
    except Exception:
        pass

with content_col:
    st.subheader(f"Nội dung - {idx + 1}/{total}")
    render_selectable_content(
        sentence_value=str(row.get("sentence", "")),
        title_value=str(row.get("thread_title", "")),
        context_value=str(row.get("parent_context", "")),
        idx=idx,
    )

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("Trước", use_container_width=True):
            select_row(max(0, idx - 1))
            st.rerun()
    with nav2:
        if st.button("Câu chưa review", use_container_width=True):
            select_row(first_unreviewed_idx(total, done_indexes))
            st.rerun()
    with nav3:
        if st.button("Tiếp", use_container_width=True):
            select_row(min(total - 1, idx + 1))
            st.rerun()

    st.caption(f"id: {row.get('id', '')} | status: {'REVIEWED' if idx in done_indexes else 'TODO'} | output: {output_path.name}")

with quad_col:
    st.subheader("Quads")
    ensure_active_quad()

    # Hiển thị tất cả quad, không dropdown.
    quad_indices = list(range(len(st.session_state.quads)))

    def quad_label(i):
        q = clean_quad(st.session_state.quads[i])
        return f"Q{i + 1} | A: {short(q['aspect'] or 'aspect?', 22)} | C: {q['category'] or 'category?'} | S: {q['sentiment'] or 'sentiment?'} | O: {short(q['opinion'] or 'opinion?', 32)}"

    selected_quad = st.radio(
        "Chọn quad",
        quad_indices,
        index=st.session_state.active_quad,
        format_func=quad_label,
        label_visibility="collapsed",
    )
    st.session_state.active_quad = int(selected_quad)
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

    active_quad = st.session_state.quads[st.session_state.active_quad]
    active_quad["aspect"] = st.text_input(
        "aspect",
        value=active_quad.get("aspect", ""),
        key=field_key("aspect", selected_name, idx, st.session_state.active_quad),
    )
    active_quad["opinion"] = st.text_input(
        "opinion",
        value=active_quad.get("opinion", ""),
        key=field_key("opinion", selected_name, idx, st.session_state.active_quad),
    )

    cat = safe_value(active_quad.get("category", ""), CATEGORY_OPTIONS)
    active_quad["category"] = st.radio(
        "category",
        CATEGORY_OPTIONS,
        index=CATEGORY_OPTIONS.index(cat),
        horizontal=True,
        key=field_key("category", selected_name, idx, st.session_state.active_quad),
    ) or ""

    sent = safe_value(active_quad.get("sentiment", ""), SENTIMENT_OPTIONS)
    active_quad["sentiment"] = st.radio(
        "sentiment",
        SENTIMENT_OPTIONS,
        index=SENTIMENT_OPTIONS.index(sent),
        horizontal=True,
        key=field_key("sentiment", selected_name, idx, st.session_state.active_quad),
    ) or ""

    preview = normalize_quads_for_save(st.session_state.quads)
    st.text_area("quad preview", value=compact_json(preview), height=88, disabled=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Lưu", use_container_width=True):
            if save_current(df, output_path, idx, done_key, done_indexes):
                st.rerun()
    with b2:
        if st.button("Lưu & Tiếp", type="primary", use_container_width=True):
            if save_current(df, output_path, idx, done_key, done_indexes):
                select_row(min(total - 1, idx + 1))
                st.rerun()
    with b3:
        if st.button("No Quad", use_container_width=True):
            st.session_state.quads = [EMPTY_QUAD.copy()]
            df.at[idx, "quad"] = "[]"
            save_data(df, output_path)
            done_indexes.add(idx)
            st.session_state[done_key] = done_indexes
            select_row(min(total - 1, idx + 1))
            st.rerun()
