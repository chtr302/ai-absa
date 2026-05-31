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

# Bộ Human Verification 900 mới
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "human_verification_900"


CATEGORY_OPTIONS = [
    "",
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
]

SENTIMENT_OPTIONS = [
    "",
    "Positive",
    "Negative",
    "Neutral",
]

EMPTY_QUAD = {
    "aspect": "",
    "category": "",
    "opinion": "",
    "sentiment": "",
}

REQUIRED_HUMAN_COLS = [
    "human_has_quad",
    "human_aspect",
    "human_category_label",
    "human_opinion",
    "human_sentiment_label",
    "human_quads_json",
]

OPTIONAL_COLS = [
    "notes",
]


# =========================
# FILE HELPERS
# =========================

def list_annotation_files():
    """
    Chỉ lấy 3 file annotator của bộ 900:
      annotator1_300.csv
      annotator2_300.csv
      annotator3_300.csv

    Không lấy file _filled.csv, ai_reference_900.csv, summary.
    """
    files = sorted(SPLIT_DIR.glob("annotator*_300.csv"))
    files = [f for f in files if not f.name.endswith("_filled.csv")]
    return files


def output_path_of(input_path: Path):
    return input_path.with_name(input_path.stem + "_filled.csv")


def load_data(input_path: Path):
    output_path = output_path_of(input_path)

    if output_path.exists():
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    else:
        df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Đảm bảo đúng schema tối thiểu
    base_required = [
        "sample_id",
        "id",
        "parent_context",
        "thread_title",
        "sentence",
    ]

    for col in base_required:
        if col not in df.columns:
            df[col] = ""

    for col in REQUIRED_HUMAN_COLS:
        if col not in df.columns:
            df[col] = ""

    for col in OPTIONAL_COLS:
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
        raise ValueError("human_quads_json phải là list JSON: [] hoặc [{...}]")

    quads = []

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Mỗi quad phải là object JSON.")
        quads.append(clean_quad(item))

    return quads


def quads_from_row(row):
    """
    Khi mở lại dòng đã gán:
    - Ưu tiên đọc human_quads_json.
    - Nếu JSON lỗi hoặc trống thì tạo từ các cột human_*.
    """
    try:
        quads = parse_quads(row.get("human_quads_json", ""))
    except Exception:
        quads = []

    if quads:
        return quads

    has_quad = str(row.get("human_has_quad", "")).strip()

    if has_quad == "No":
        return []

    aspect = str(row.get("human_aspect", "")).strip()
    category = str(row.get("human_category_label", "")).strip()
    opinion = str(row.get("human_opinion", "")).strip()
    sentiment = str(row.get("human_sentiment_label", "")).strip()

    # Nếu dòng cũ có 1 quad ở các cột tóm tắt thì recover lại
    if any(v and v not in {"None", "Multi", "Mixed"} for v in [aspect, category, opinion, sentiment]):
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
    values = set()

    for q in quads:
        value = str(q.get(field, "")).strip()
        if value:
            values.add(value)

    return sorted(values)


def primary_value(values, multi_label):
    if len(values) == 0:
        return "None"

    if len(values) == 1:
        return values[0]

    return multi_label


def summarize_quads(quads):
    """
    Tự động sinh các cột human_* từ danh sách quads.

    Không quad:
      No / None / [].

    1 quad:
      lấy trực tiếp aspect/category/opinion/sentiment.

    Nhiều quad:
      nếu nhiều giá trị khác nhau => Multi hoặc Mixed.
    """
    clean_quads = [clean_quad(q) for q in quads]

    # Bỏ quad rỗng hoàn toàn
    clean_quads = [
        q for q in clean_quads
        if q["aspect"] or q["category"] or q["opinion"] or q["sentiment"]
    ]

    if not clean_quads:
        return {
            "human_has_quad": "No",
            "human_aspect": "None",
            "human_category_label": "None",
            "human_opinion": "None",
            "human_sentiment_label": "None",
            "human_quads_json": "[]",
        }

    aspects = non_empty_values(clean_quads, "aspect")
    categories = non_empty_values(clean_quads, "category")
    opinions = non_empty_values(clean_quads, "opinion")
    sentiments = non_empty_values(clean_quads, "sentiment")

    return {
        "human_has_quad": "Yes",
        "human_aspect": primary_value(aspects, multi_label="Multi"),
        "human_category_label": primary_value(categories, multi_label="Multi"),
        "human_opinion": primary_value(opinions, multi_label="Multi"),
        "human_sentiment_label": primary_value(sentiments, multi_label="Mixed"),
        "human_quads_json": json.dumps(clean_quads, ensure_ascii=False, separators=(",", ":")),
    }


def validate_quads_before_save(quads):
    """
    Nếu người dùng bấm No Quad thì không cần validate.
    Nếu có quad thì mỗi quad không rỗng phải đủ 4 field.
    """
    clean_quads = [clean_quad(q) for q in quads]

    valid_quads = [
        q for q in clean_quads
        if q["aspect"] or q["category"] or q["opinion"] or q["sentiment"]
    ]

    if not valid_quads:
        return True, ""

    for i, q in enumerate(valid_quads, start=1):
        missing = []

        if not q["aspect"]:
            missing.append("aspect")
        if not q["category"]:
            missing.append("category")
        if not q["opinion"]:
            missing.append("opinion")
        if not q["sentiment"]:
            missing.append("sentiment")

        if missing:
            return False, f"Quad {i} thiếu: {', '.join(missing)}"

    return True, ""


def update_row(df: pd.DataFrame, idx: int, values: dict):
    for col, val in values.items():
        df.at[idx, col] = val


def row_key(selected_name, idx):
    return f"{selected_name}:{idx}"


def quad_widget_key(field, selected_name, idx, active_quad):
    return f"{field}_{selected_name}_{idx}_{active_quad}"


def load_state_for_row(row, key):
    """
    Khi chuyển dòng, load quads và notes của dòng đó vào session_state.
    """
    if st.session_state.get("active_row_key") == key:
        return

    st.session_state.active_row_key = key

    quads = quads_from_row(row)

    if not quads:
        quads = [EMPTY_QUAD.copy()]

    st.session_state.quads = quads
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
    """
    Nhận text được bôi đen từ component span_selector.
    Chỉ fill vào aspect hoặc opinion của quad đang active.
    """
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


def save_current_row(df, idx, output_path, notes):
    ok, msg = validate_quads_before_save(st.session_state.quads)

    if not ok:
        st.error(msg)
        return False

    clean_quads = [clean_quad(q) for q in st.session_state.quads]
    summary = summarize_quads(clean_quads)
    summary["notes"] = notes

    update_row(df, idx, summary)
    save_data(df, output_path)
    return True


def save_no_quad(df, idx, output_path, notes):
    update_row(
        df,
        idx,
        {
            "human_has_quad": "No",
            "human_aspect": "None",
            "human_category_label": "None",
            "human_opinion": "None",
            "human_sentiment_label": "None",
            "human_quads_json": "[]",
            "notes": notes,
        },
    )
    save_data(df, output_path)



def render_selectable_content(sentence_value, title_value, context_value, selected_name, idx):
    """
    Không dùng custom Streamlit component nữa nên không còn lỗi span_selector.
    Aspect lấy từ Comment / Thread title / Parent context; Opinion chỉ lấy từ Comment.
    Bôi đen text sẽ hiện popup A/O ngay cạnh vùng chọn.
    Không dùng 2 nút Aspect/Opinion cố định phía trên.
    """
    def esc(x):
        return html.escape(str(x or ""))

    html_block = f"""
<style>
:root {{ color-scheme: dark; }}
html, body {{ margin:0; padding:0; background:#000; color:#fff; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:14px; overflow:hidden; }}
* {{ box-sizing:border-box; }}
.select-hint {{ font-size:12px; color:#bdbdbd; margin:0 0 8px 0; line-height:1.25; }}
#absa_floating_toolbar_{idx} {{ position:fixed; z-index:999999; display:none; gap:6px; align-items:center; padding:5px; background:#111; border:1px solid #3a3a3a; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,.45); }}
#absa_floating_toolbar_{idx} button {{ min-width:42px; height:30px; border:1px solid #444; background:#0b0b0b; color:#fff; border-radius:7px; font-weight:800; cursor:pointer; font-size:13px; }}
#absa_floating_toolbar_{idx} button:hover {{ border-color:#00d084; color:#00d084; }}
#absa_floating_toolbar_{idx} button:disabled {{ opacity:.35; cursor:not-allowed; border-color:#333; color:#aaa; }}
#absa_floating_toolbar_{idx} .tip {{ color:#aaa; font-size:12px; padding:0 3px; }}
.selection-box-label {{ font-size:13px; color:#9a9a9a; margin:8px 0 4px 0; }}
.selectable-box {{ width:100%; background:#070707; border:1px solid #262626; border-radius:8px; color:#fff; padding:12px 14px; white-space:pre-wrap; user-select:text; -webkit-user-select:text; line-height:1.45; overflow-y:auto; outline:none; }}
.selectable-box:focus {{ border-color:#555; }}
.selectable-box.comment {{ height:176px; }}
.selectable-box.title {{ height:72px; }}
.selectable-box.context {{ height:142px; }}
::selection {{ background:#0d6efd; color:#fff; }}
.selection-status {{ min-height:18px; margin-top:7px; font-size:12px; color:#9a9a9a; }}
</style>
<div class="select-hint">
Bôi đen text xong sẽ hiện nút <b>A</b> và <b>O</b> ngay cạnh vùng chọn. <b>A</b> = Aspect. <b>O</b> = Opinion. Opinion chỉ lấy từ <b>Comment</b>.
</div>
<div id="absa_floating_toolbar_{idx}">
  <button type="button" id="absa_float_aspect_{idx}" title="Gán Aspect">A</button>
  <button type="button" id="absa_float_opinion_{idx}" title="Gán Opinion">O</button>
  <span class="tip">Aspect / Opinion</span>
</div>
<div class="selection-box-label">Comment</div>
<div class="selectable-box comment" data-absa-source="comment" tabindex="0">{esc(sentence_value)}</div>
<div class="selection-box-label">Thread title</div>
<div class="selectable-box title" data-absa-source="title" tabindex="0">{esc(title_value)}</div>
<div class="selection-box-label">Parent context</div>
<div class="selectable-box context" data-absa-source="context" tabindex="0">{esc(context_value)}</div>
<div id="absa_status_{idx}" class="selection-status"></div>
<script>
(function() {{
  let rootDoc = document;
  try {{ if (window.parent && window.parent.document) rootDoc = window.parent.document; }} catch(e) {{ rootDoc = document; }}
  const localDoc = document;
  const statusEl = localDoc.getElementById("absa_status_{idx}");
  const floatToolbar = localDoc.getElementById("absa_floating_toolbar_{idx}");
  const floatAspectBtn = localDoc.getElementById("absa_float_aspect_{idx}");
  const floatOpinionBtn = localDoc.getElementById("absa_float_opinion_{idx}");
  let lastText = "";
  let lastSource = "";

  function hideToolbar() {{
    if (floatToolbar) floatToolbar.style.display = "none";
  }}

  function showToolbarForSelection(sel, source) {{
    if (!floatToolbar || !sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    if (!rect || (!rect.width && !rect.height)) return;

    if (floatOpinionBtn) floatOpinionBtn.disabled = source !== "comment";

    floatToolbar.style.display = "flex";
    floatToolbar.style.left = "0px";
    floatToolbar.style.top = "0px";

    const toolbarRect = floatToolbar.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - toolbarRect.width / 2;
    let top = rect.top - toolbarRect.height - 8;

    if (top < 4) top = rect.bottom + 8;
    left = Math.max(4, Math.min(left, window.innerWidth - toolbarRect.width - 4));
    top = Math.max(4, Math.min(top, window.innerHeight - toolbarRect.height - 4));

    floatToolbar.style.left = left + "px";
    floatToolbar.style.top = top + "px";
  }}

  function status(msg, ok) {{
    if (!statusEl) return;
    statusEl.textContent = msg || "";
    statusEl.style.color = ok ? "#00d084" : "#ff6262";
  }}

  function sourceFromNode(node) {{
    if (!node) return "";
    let el = node.nodeType === 3 ? node.parentElement : node;
    while (el && el !== localDoc.body) {{
      if (el.dataset && el.dataset.absaSource) return el.dataset.absaSource;
      el = el.parentElement;
    }}
    return "";
  }}

  function rememberSelection() {{
    const sel = localDoc.getSelection ? localDoc.getSelection() : window.getSelection();
    const text = sel ? String(sel.toString()).trim() : "";
    if (!text) {{ hideToolbar(); return; }}
    const sourceA = sourceFromNode(sel.anchorNode);
    const sourceB = sourceFromNode(sel.focusNode);
    if (!sourceA || sourceA !== sourceB) {{
      lastText = "";
      lastSource = "";
      hideToolbar();
      status("Chỉ chọn text trong cùng 1 ô.", false);
      return;
    }}
    lastText = text;
    lastSource = sourceA;
    const name = sourceA === "comment" ? "Comment" : (sourceA === "title" ? "Thread title" : "Parent context");
    const shortText = text.length > 90 ? text.slice(0, 90) + "…" : text;
    status("Đã chọn từ " + name + ": “" + shortText + "”", true);
    showToolbarForSelection(sel, sourceA);
  }}

  function setStreamlitInput(label, value) {{
    const labels = Array.from(rootDoc.querySelectorAll('label'));
    let target = null;
    for (const lab of labels) {{
      const txt = (lab.innerText || lab.textContent || "").trim().toLowerCase();
      if (txt === label.toLowerCase()) {{
        const wrap = lab.closest('[data-testid="stTextInput"]') || lab.parentElement;
        if (wrap) target = wrap.querySelector('input');
        if (target) break;
      }}
    }}
    if (!target) {{
      const inputs = Array.from(rootDoc.querySelectorAll('input'));
      target = inputs.find(inp => (inp.getAttribute('aria-label') || '').trim().toLowerCase() === label.toLowerCase());
    }}
    if (!target) return false;
    const setter = Object.getOwnPropertyDescriptor(rootDoc.defaultView.HTMLInputElement.prototype, 'value').set;
    setter.call(target, value);
    target.dispatchEvent(new Event('input', {{ bubbles: true }}));
    target.dispatchEvent(new Event('change', {{ bubbles: true }}));
    target.focus();
    return true;
  }}

  function applyField(field) {{
    rememberSelection();
    const text = (lastText || "").trim();
    const source = lastSource || "";
    if (!text) {{ status("Chưa bôi đen text nào.", false); return; }}
    if (field === "opinion" && source !== "comment") {{
      status("Opinion chỉ được chọn từ Comment, không lấy từ Thread title / Parent context.", false);
      return;
    }}
    const ok = setStreamlitInput(field, text);
    const shortText = text.length > 90 ? text.slice(0, 90) + "…" : text;
    if (ok) {{
      status((field === "aspect" ? "Aspect" : "Opinion") + " đã fill: “" + shortText + "”", true);
      hideToolbar();
      const sel = localDoc.getSelection ? localDoc.getSelection() : window.getSelection();
      if (sel) sel.removeAllRanges();
    }} else {{
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      status("Không tự fill được input, đã copy text vào clipboard để dán tay.", false);
    }}
  }}

  function isTypingTarget(event) {{
    const tag = (event.target && event.target.tagName || "").toLowerCase();
    const editable = event.target && event.target.isContentEditable;
    return editable || ["input", "textarea", "select"].includes(tag);
  }}

  function handleShortcut(event) {{
    if (!event) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (isTypingTarget(event)) return;

    const key = String(event.key || "").toLowerCase();

    if (key === "a") {{
      event.preventDefault();
      event.stopPropagation();
      applyField("aspect");
      return false;
    }}

    if (key === "o") {{
      event.preventDefault();
      event.stopPropagation();
      applyField("opinion");
      return false;
    }}
  }}

  function installKeyHandler(doc) {{
    if (!doc || doc.__absaShortcutInstalled_{idx}) return;
    doc.__absaShortcutInstalled_{idx} = true;
    doc.addEventListener("keydown", handleShortcut, true);
    doc.addEventListener("keyup", handleShortcut, true);
  }}

  localDoc.addEventListener("selectionchange", function() {{ setTimeout(rememberSelection, 0); }});
  localDoc.addEventListener("mouseup", rememberSelection);
  localDoc.addEventListener("keyup", rememberSelection);
  floatAspectBtn.addEventListener("mousedown", function(e) {{ e.preventDefault(); }});
  floatOpinionBtn.addEventListener("mousedown", function(e) {{ e.preventDefault(); }});
  floatAspectBtn.addEventListener("click", function() {{ applyField("aspect"); }});
  floatOpinionBtn.addEventListener("click", function() {{ applyField("opinion"); }});
  localDoc.addEventListener("mousedown", function(e) {{
    if (floatToolbar && !floatToolbar.contains(e.target)) {{
      setTimeout(function() {{
        const sel = localDoc.getSelection ? localDoc.getSelection() : window.getSelection();
        if (!sel || !String(sel.toString()).trim()) hideToolbar();
      }}, 0);
    }}
  }});

  // Bắt phím ở cả iframe của components.html và document cha của Streamlit.
  // Nhiều bản deploy Streamlit để focus ở parent document nên chỉ nghe localDoc sẽ không ăn phím.
  installKeyHandler(localDoc);
  try {{ installKeyHandler(rootDoc); }} catch(e) {{}}
  try {{ if (window.parent && window.parent.document) installKeyHandler(window.parent.document); }} catch(e) {{}}
}})();
</script>
"""
    components.html(html_block, height=560, scrolling=False)

# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="ABSA Human Annotation",
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
    padding-top: 3.6rem !important;
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
    white-space: pre-wrap !important;
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

[data-testid="stSelectbox"] {
    margin-top: 0.35rem !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] {
    min-height: 42px !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 42px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    align-items: center !important;
}

.st-key-top_file_select {
    margin-top: 0.25rem !important;
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

.small-caption {
    font-size: 0.78rem;
    color: #cfcfcf;
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

st.title("ABSA Human Verification 900")


# =========================
# LOAD SELECTED FILE
# =========================

files = list_annotation_files()

if not files:
    st.error(f"Không tìm thấy file annotator CSV trong: {SPLIT_DIR}")
    st.info("Cần có file: annotator1_300.csv, annotator2_300.csv, annotator3_300.csv")
    st.stop()

file_names = [f.name for f in files]

top_col1, top_col2, top_col3 = st.columns([1.1, 0.7, 2.2])

with top_col1:
    selected_name = st.selectbox(
        "File",
        file_names,
        label_visibility="collapsed",
        key="top_file_select",
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
    st.subheader("Danh sách")

    search_text = st.text_input(
        "Tìm",
        value="",
        placeholder="lọc theo title hoặc comment",
        label_visibility="collapsed",
    ).strip().lower()

    rows_for_list = []

    for i, item in df.iterrows():
        title = str(item.get("thread_title", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        sample_id = str(item.get("sample_id", "")).strip()

        search_blob = f"{sample_id} {title} {sentence}".lower()

        if search_text and search_text not in search_blob:
            continue

        rows_for_list.append((int(i), item))

    sentence_items = []

    for i, item in rows_for_list:
        title = str(item.get("thread_title", "")).strip() or "(no title)"
        sentence = str(item.get("sentence", "")).strip() or "(empty sentence)"
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
# MIDDLE: CONTENT
# =========================

with content_col:
    st.subheader(f"Nội dung cần gán - {idx + 1}/{total}")

    sentence_value = str(row.get("sentence", ""))
    title_value = str(row.get("thread_title", ""))
    context_value = str(row.get("parent_context", ""))

    render_selectable_content(
        sentence_value=sentence_value,
        title_value=title_value,
        context_value=context_value,
        selected_name=selected_name,
        idx=idx,
    )

    meta1, meta2, meta3 = st.columns(3)

    with meta1:
        st.caption(f"sample_id: {row.get('sample_id', '')}")

    with meta2:
        st.caption(f"id: {row.get('id', '')}")

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
        category = quad.get("category", "").strip() or "category?"
        sentiment = quad.get("sentiment", "").strip() or "sentiment?"
        quad_labels.append(f"Quad {i}: {aspect} | {category} | {sentiment}")

    if not quad_labels:
        st.session_state.quads = [EMPTY_QUAD.copy()]
        quad_labels = ["Quad 1: aspect? | category? | sentiment?"]

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
        placeholder="bôi đen text rồi chọn A hoặc nhập tay",
        key=quad_widget_key("aspect", selected_name, idx, st.session_state.active_quad),
    )

    active_quad["opinion"] = st.text_input(
        "opinion",
        value=active_quad.get("opinion", ""),
        placeholder="bôi đen text ở Comment rồi chọn O hoặc nhập tay",
        key=quad_widget_key("opinion", selected_name, idx, st.session_state.active_quad),
    )

    cur_category = safe_value(active_quad.get("category", ""), CATEGORY_OPTIONS)
    category_index = CATEGORY_OPTIONS.index(cur_category) if cur_category in CATEGORY_OPTIONS else 0

    selected_category = st.radio(
        "category",
        CATEGORY_OPTIONS,
        index=category_index,
        horizontal=True,
        key=quad_widget_key("category", selected_name, idx, st.session_state.active_quad),
    )

    active_quad["category"] = selected_category or ""

    cur_sentiment = safe_value(active_quad.get("sentiment", ""), SENTIMENT_OPTIONS)
    sentiment_index = SENTIMENT_OPTIONS.index(cur_sentiment) if cur_sentiment in SENTIMENT_OPTIONS else 0

    selected_sentiment = st.radio(
        "sentiment",
        SENTIMENT_OPTIONS,
        index=sentiment_index,
        horizontal=True,
        key=quad_widget_key("sentiment", selected_name, idx, st.session_state.active_quad),
    )

    active_quad["sentiment"] = selected_sentiment or ""

    summary = summarize_quads(st.session_state.quads)

    st.text_area(
        "human_quads_json",
        value=summary["human_quads_json"],
        height=82,
        disabled=True,
    )

    st.markdown(
        f"""
<div class="small-caption">
<b>Auto summary</b><br>
has_quad: <code>{summary["human_has_quad"]}</code><br>
aspect: <code>{summary["human_aspect"]}</code><br>
category: <code>{summary["human_category_label"]}</code><br>
opinion: <code>{summary["human_opinion"]}</code><br>
sentiment: <code>{summary["human_sentiment_label"]}</code>
</div>
""",
        unsafe_allow_html=True,
    )

    st.session_state.notes_value = st.text_input(
        "notes",
        value=st.session_state.get("notes_value", ""),
        placeholder="ghi chú nếu cần",
    )

    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("Lưu", use_container_width=True):
            ok = save_current_row(
                df,
                idx,
                output_path,
                st.session_state.notes_value,
            )
            if ok:
                st.success("Đã lưu.")

    with btn2:
        if st.button("Lưu & Tiếp", type="primary", use_container_width=True):
            ok = save_current_row(
                df,
                idx,
                output_path,
                st.session_state.notes_value,
            )
            if ok:
                st.session_state.idx = min(total - 1, idx + 1)
                st.rerun()

    with btn3:
        if st.button("No Quad", use_container_width=True):
            st.session_state.quads = [EMPTY_QUAD.copy()]
            save_no_quad(
                df,
                idx,
                output_path,
                st.session_state.get("notes_value", ""),
            )
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()