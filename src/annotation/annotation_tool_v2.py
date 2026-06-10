# src/annotation/annotation_tool_6k.py
# =====================================================================
# ABSA Annotation Tool — human_verification_6000
# Kiến trúc: Pure Python HTTPServer + HTML/JS frontend
# Tích hợp công nghệ gán nhãn đám mây (Gemini Cloud Auto-Labeling)
# Cấu hình: 7 Categories (PERFORMANCE, RESOURCES, SOFTWARE, INTELLIGENCE, TECHNICAL, BEHAVIOR, COMPARATIVE)
# =====================================================================

import os
import sys
import json
import csv
import webbrowser
import socket
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import urllib.request

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# =========================
# PATHS & CONFIG
# =========================
ROOT_DIR  = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "human_verification_6000"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLS = ["id", "parent_context", "thread_title", "sentence", "quad"]
OUT_COLS = ["id", "parent_context", "thread_title", "sentence",
            "human_has_quad", "human_aspect", "human_category_label",
            "human_opinion", "human_sentiment_label", "human_quads_json"]

# Global thread status storage
BULK_THREADS = {}  # filename -> { "total": int, "processed": int, "status": str }

SYSTEM_PROMPT_ABSA = """You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA) in the domain of large language models (r/LocalLLaMA).
Your task is to analyze the input sentences and extract all aspect-sentiment quadruplets (Aspect, Category, Opinion, Sentiment).

Strictly follow these rules for each sentence:
1. Aspect: Must be a specific AI model (e.g., "Gemma 3", "Qwen 3.5", "gpt-oss 120B", "Llama-4") or a pronoun/noun referring directly to it (e.g., "it", "they", "model", "models").
   CRITICAL: Do NOT extract software tools, libraries, engines, APIs, frontends, or runners as aspects (e.g., "llama.cpp", "Ollama", "vLLM", "sglang", "LM Studio", "openclaw", "Roo Code", "Cline", "VSCode", "Hugging Face"). These are TOOLING/SOFTWARE, not AI models. If a sentence only evaluates a tool, return an empty quads array.
   CRITICAL: The aspect term must be an exact substring of the sentence text itself. Do not resolve it to an external name if that name is not in the sentence.
2. Opinion: The exact phrase in the sentence representing the evaluation or opinion of the aspect. Must be an exact substring of the sentence.
3. Category: Must be one of the following 7 attributes:
   - PERFORMANCE: Speed, latency, throughput, tokens per second, efficiency, context window size, memory footprint of the running model.
   - RESOURCES: Hardware requirements, VRAM, GPU memory footprint to run the model.
   - SOFTWARE: Deployment platforms, APIs, local UI wrappers, frontends, backends (e.g., llama.cpp, Ollama, vLLM).
   - INTELLIGENCE: Code generation, programming tasks, debugging, logic, math, reasoning, common sense, general facts, translation.
   - TECHNICAL: Model architecture, MoE vs Dense layers, file formats (GGUF, EXL2), quantization bit-widths, fine-tuning methods (LoRA, QLoRA, SFT).
   - BEHAVIOR: Chat output styles (roleplay, preachy, robotic, verbosity), alignment, censorship, safety, compliance, refusal.
   - COMPARATIVE: Comparisons between models, leaderboard rankings, cost performance.
4. Sentiment: Must be one of: "Positive", "Negative", "Neutral", "Mixed".
5. If no aspect-sentiment quadruplet can be extracted (e.g. no AI model is evaluated, or it is a pure question/fact), output an empty array: [].

Output Format:
Return a JSON array where each item corresponds to an input sentence:
[
  {
    "index": <int>,
    "quads": [
      {
        "aspect": "<aspect_text>",
        "category": "<category_label>",
        "opinion": "<opinion_text>",
        "sentiment": "<sentiment>"
      }
    ]
  }
]
"""

# =========================
# DATA HELPERS
# =========================

def list_annotation_files():
    files = sorted(SPLIT_DIR.glob("annotator*_2000.csv"))
    return [f for f in files if not f.name.endswith("_filled.csv")]


def output_path_of(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_filled.csv")


def is_row_done(row: dict) -> bool:
    val = str(row.get("human_has_quad", "")).strip().upper()
    return val in {"YES", "NO"}


def parse_existing_quad(quad_str: str) -> list:
    if not quad_str or str(quad_str).strip() in {"", "[]", "nan", "none"}:
        return []
    try:
        data = json.loads(quad_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def load_data(file_name: str):
    input_path  = SPLIT_DIR / file_name
    output_path = output_path_of(input_path)
    target = output_path if output_path.exists() else input_path

    rows = []
    if HAS_PANDAS:
        df = pd.read_csv(target, encoding="utf-8-sig", dtype=str).fillna("")
        for col in CSV_COLS + ["human_has_quad", "human_aspect", "human_category_label",
                                "human_opinion", "human_sentiment_label", "human_quads_json"]:
            if col not in df.columns:
                df[col] = ""
        for _, r in df.iterrows():
            row = r.to_dict()
            for k, v in row.items():
                if str(v).lower() in {"nan", "none", "<na>"}:
                    row[k] = ""
            rows.append(row)
    else:
        with open(target, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                row = dict(r)
                for k, v in row.items():
                    if str(v).lower() in {"nan", "none", "<na>"}:
                        row[k] = ""
                rows.append(row)

    return rows, output_path


def save_data(rows: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        for col in OUT_COLS:
            if col not in df.columns:
                df[col] = ""
        all_cols = [c for c in CSV_COLS if c in df.columns]
        for c in OUT_COLS:
            if c not in all_cols:
                all_cols.append(c)
        df[all_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        all_cols = list(dict.fromkeys(CSV_COLS + OUT_COLS))
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

# =========================
# CLOUD API HELPERS
# =========================

def clean_and_validate_quads(sentence_text: str, quads: list) -> list:
    valid_quads = []
    sentence_lower = sentence_text.lower()
    
    for q in quads:
        aspect = str(q.get("aspect", "")).strip()
        opinion = str(q.get("opinion", "")).strip()
        category = str(q.get("category", "")).strip()
        sentiment = str(q.get("sentiment", "")).strip()
        
        if not aspect or aspect.lower() == "none" or not opinion or opinion.lower() == "none":
            continue
            
        idx_a = sentence_lower.find(aspect.lower())
        idx_o = sentence_lower.find(opinion.lower())
        
        if idx_a != -1 and idx_o != -1:
            real_aspect = sentence_text[idx_a : idx_a + len(aspect)]
            real_opinion = sentence_text[idx_o : idx_o + len(opinion)]
            
            # Filter out tools
            tool_keywords = {
                "llama.cpp", "ollama", "vllm", "sglang", "lm studio", "lmstudio",
                "openclaw", "roo code", "cline", "vscode", "hugging face", 
                "huggingface", "koboldcpp", "kobold.cpp", "exllamav2", "exllama", 
                "oobabooga", "text-generation-webui", "tabbyapi", "aphrodite"
            }
            if any(tk in real_aspect.lower() for tk in tool_keywords):
                continue
                
            valid_quads.append({
                "aspect": real_aspect,
                "category": category if category in [
                    "PERFORMANCE", "RESOURCES", "SOFTWARE", "INTELLIGENCE", 
                    "TECHNICAL", "BEHAVIOR", "COMPARATIVE"
                ] else "None",
                "opinion": real_opinion,
                "sentiment": sentiment if sentiment in ["Positive", "Negative", "Neutral", "Mixed"] else "Neutral"
            })
    return valid_quads


def query_gemini_single(sentence: str, api_key: str) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT_ABSA},
                    {"text": f"Sentence: {sentence}"}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, json.dumps(payload).encode("utf-8"), timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw_quads = json.loads(content_text)
            return clean_and_validate_quads(sentence, raw_quads)
    except Exception as e:
        print(f"Gemini Single Request Error: {e}")
        return []


def query_gemini_batch(batch_data: list, api_key: str) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    prompt = f"Here is the batch of sentences to annotate:\n{json.dumps(batch_data, indent=2)}\n\nAnnotate them and return the JSON array matching the requested schema."
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT_ABSA},
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    
    max_retries = 4
    backoff = 3.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, json.dumps(payload).encode("utf-8"), timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return json.loads(content_text)
        except Exception as e:
            print(f"Gemini Batch Request Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                raise e


def save_annotations_to_df(df: pd.DataFrame, annotations: list):
    for ann in annotations:
        idx = int(ann["index"])
        raw_quads = ann.get("quads", [])
        sentence_text = df.at[idx, "sentence"]
        
        quads = clean_and_validate_quads(sentence_text, raw_quads)
        
        if len(quads) > 0:
            has_quad = "Yes"
            aspect = quads[0]["aspect"] if len(quads) == 1 else "Multi"
            opinion = quads[0]["opinion"] if len(quads) == 1 else "Multi"
            
            cats = list(set([q["category"] for q in quads]))
            category = cats[0] if len(cats) == 1 else "Multi"
            
            sents = list(set([q["sentiment"] for q in quads]))
            sentiment = sents[0] if len(sents) == 1 else "Mixed"
        else:
            has_quad = "No"
            aspect = "None"
            opinion = "None"
            category = "None"
            sentiment = "None"
            
        df.at[idx, "human_has_quad"] = has_quad
        df.at[idx, "human_aspect"] = aspect
        df.at[idx, "human_opinion"] = opinion
        df.at[idx, "human_category_label"] = category
        df.at[idx, "human_sentiment_label"] = sentiment
        df.at[idx, "human_quads_json"] = json.dumps(quads)


def start_bulk_thread(fname: str, api_key: str):
    if fname in BULK_THREADS and BULK_THREADS[fname]["status"] == "running":
        return
        
    def run_bulk():
        try:
            BULK_THREADS[fname]["status"] = "running"
            input_path = SPLIT_DIR / fname
            output_path = output_path_of(input_path)
            
            # Load
            rows, _ = load_data(fname)
            df = pd.DataFrame(rows)
            for col in OUT_COLS:
                if col not in df.columns:
                    df[col] = ""
                    
            unfinished_indices = df[df["human_has_quad"] == ""].index.tolist()
            BULK_THREADS[fname]["total"] = len(unfinished_indices)
            BULK_THREADS[fname]["processed"] = 0
            
            batch_size = 30
            for i in range(0, len(unfinished_indices), batch_size):
                if BULK_THREADS[fname]["status"] != "running":
                    break
                    
                batch_indices = unfinished_indices[i : i + batch_size]
                batch_data = []
                for idx in batch_indices:
                    batch_data.append({
                        "index": int(idx),
                        "sentence": df.at[idx, "sentence"]
                    })
                    
                annotations = query_gemini_batch(batch_data, api_key)
                if annotations:
                    save_annotations_to_df(df, annotations)
                    df.to_csv(output_path, index=False, encoding="utf-8-sig")
                    BULK_THREADS[fname]["processed"] += len(batch_indices)
                    
                time.sleep(15.0)
                
            BULK_THREADS[fname]["status"] = "completed"
        except Exception as e:
            print(f"Bulk thread error: {e}")
            BULK_THREADS[fname]["status"] = f"error: {str(e)}"
            
    t = threading.Thread(target=run_bulk, daemon=True)
    BULK_THREADS[fname] = {
        "total": 0,
        "processed": 0,
        "status": "pending"
    }
    t.start()

# =========================
# HTTP SERVER
# =========================

class AnnotationHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, content: str):
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        query  = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.send_html(HTML_TEMPLATE)
            return

        if path == "/api/files":
            files = list_annotation_files()
            self.send_json({"files": [f.name for f in files]})
            return

        if path == "/api/load":
            fname = query.get("file", [None])[0]
            if not fname:
                self.send_json({"error": "Missing file param"}, 400); return
            fname = Path(fname).name
            try:
                rows, _ = load_data(fname)
                first_unfinished = next((i for i, r in enumerate(rows) if not is_row_done(r)), 0)
                sentences = []
                for i, r in enumerate(rows):
                    sentences.append({
                        "index": i,
                        "text_snippet": str(r.get("sentence", ""))[:50] + "…",
                        "done": is_row_done(r),
                    })
                self.send_json({"total": len(rows), "first_unfinished": first_unfinished, "sentences": sentences})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/sentence":
            fname = query.get("file", [None])[0]
            idx   = query.get("index", [None])[0]
            if not fname or idx is None:
                self.send_json({"error": "Missing params"}, 400); return
            fname = Path(fname).name
            try:
                rows, _ = load_data(fname)
                i = int(idx)
                if i < 0 or i >= len(rows):
                    self.send_json({"error": "Index out of range"}, 400); return
                row = rows[i]
                existing_quads = parse_existing_quad(
                    row.get("human_quads_json") or row.get("quad", "")
                )
                self.send_json({
                    "index": i,
                    "sentence": {
                        "id":           row.get("id", ""),
                        "thread_title": row.get("thread_title", ""),
                        "parent_context": row.get("parent_context", ""),
                        "sentence_text":  row.get("sentence", ""),
                        "human_has_quad": row.get("human_has_quad", ""),
                        "existing_quads": existing_quads,
                        "notes": row.get("notes", ""),
                    }
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/bulk_status":
            fname = query.get("file", [None])[0]
            if not fname:
                self.send_json({"error": "Missing file param"}, 400); return
            fname = Path(fname).name
            status_data = BULK_THREADS.get(fname, {"status": "idle", "total": 0, "processed": 0})
            self.send_json({
                "status": status_data.get("status", "idle"),
                "total": status_data.get("total", 0),
                "processed": status_data.get("processed", 0)
            })
            return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if path == "/api/save":
            try:
                data   = json.loads(body.decode("utf-8"))
                fname  = Path(data.get("file", "")).name
                idx    = int(data.get("index", -1))
                values = data.get("values", {})
                if not fname or idx < 0 or not values:
                    self.send_json({"error": "Invalid request"}, 400); return

                rows, output_path = load_data(fname)
                if idx >= len(rows):
                    self.send_json({"error": "Index out of range"}, 400); return

                for col, val in values.items():
                    rows[idx][col] = str(val)

                save_data(rows, output_path)
                self.send_json({"status": "success", "nextIndex": min(idx + 1, len(rows) - 1)})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/auto_label_current":
            try:
                data = json.loads(body.decode("utf-8"))
                sentence = data.get("sentence", "")
                api_key = data.get("api_key", "").strip()
                if not api_key:
                    self.send_json({"error": "Thiếu Gemini API Key!"}, 400); return
                
                quads = query_gemini_single(sentence, api_key)
                self.send_json({"quads": quads})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/auto_label_bulk":
            try:
                data = json.loads(body.decode("utf-8"))
                fname = Path(data.get("file", "")).name
                api_key = data.get("api_key", "").strip()
                if not fname or not api_key:
                    self.send_json({"error": "Thiếu tham số hoặc API Key!"}, 400); return
                
                start_bulk_thread(fname, api_key)
                self.send_json({"status": "started"})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        self.send_response(404); self.end_headers()


# =========================
# HTML TEMPLATE
# =========================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ABSA Annotation — 6k Dataset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #0b0f19;
  --bg2:       #12182b;
  --bg-card:   rgba(22,28,54,0.7);
  --border:    rgba(255,255,255,0.08);
  --text:      #f1f5f9;
  --muted:     #94a3b8;
  --aspect-c:  #00f2fe;
  --aspect-bg: rgba(0,242,254,0.1);
  --aspect-br: rgba(0,242,254,0.4);
  --opinion-c: #ff9f43;
  --opinion-bg:rgba(255,159,67,0.1);
  --opinion-br:rgba(255,159,67,0.4);
  --primary:   #4f46e5;
  --primary-h: #6366f1;
  --success:   #10b981;
  --danger:    #ef4444;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column;}

/* ── Header ── */
header{background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;gap:12px;}
h1{font-family:'Outfit',sans-serif;font-size:1.25rem;font-weight:700;background:linear-gradient(135deg,#00f2fe,#4f46e5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hdr-right{display:flex;gap:10px;align-items:center;}
select{background:#1e293b;color:var(--text);border:1px solid var(--border);padding:7px 12px;border-radius:8px;font-size:.85rem;min-width:210px;outline:none;cursor:pointer;}

/* ── Buttons ── */
.btn{background:#1e293b;color:var(--text);border:1px solid var(--border);padding:7px 14px;border-radius:8px;cursor:pointer;font-weight:500;font-size:.85rem;transition:all .15s ease;display:inline-flex;align-items:center;gap:5px;}
.btn:hover{background:#334155;border-color:rgba(255,255,255,.2);}
.btn-primary{background:var(--primary);border-color:transparent;color:#fff;}
.btn-primary:hover{background:var(--primary-h);}
.btn-success{background:var(--success);border-color:transparent;color:#fff;}
.btn-success:hover{background:#059669;}
.btn-danger{background:transparent;border:1px solid rgba(239,68,68,.35);color:var(--danger);}
.btn-danger:hover{background:rgba(239,68,68,.1);}

/* ── Layout ── */
.app{display:flex;flex:1;overflow:hidden;}

/* ── Sidebar ── */
.sidebar{width:280px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
.sidebar-sec{padding:12px 14px;border-bottom:1px solid var(--border);}
.sidebar-lbl{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;margin-bottom:8px;}
.prog-wrap{height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden;margin-bottom:6px;}
.prog-fill{height:100%;background:var(--success);width:0%;transition:width .3s ease;box-shadow:0 0 6px var(--success);}
.prog-text{font-size:.8rem;font-weight:600;}
.sent-list{flex:1;overflow-y:auto;padding:6px;}
.sent-item{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:6px;cursor:pointer;font-size:.82rem;transition:background .12s;}
.sent-item:hover{background:rgba(255,255,255,.03);}
.sent-item.active{background:rgba(79,70,229,.15);border-left:3px solid var(--primary);font-weight:500;}
.dot{width:7px;height:7px;border-radius:50%;background:#475569;flex-shrink:0;}
.sent-item.done .dot{background:var(--success);box-shadow:0 0 6px var(--success);}
.sent-idx{color:var(--muted);font-weight:600;width:30px;flex-shrink:0;}
.sent-snip{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* ── Workspace ── */
.workspace{flex:1;display:flex;flex-direction:column;overflow-y:auto;padding:16px 20px;gap:14px;}
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px;backdrop-filter:blur(12px);}

/* Context */
.ctx-title{font-size:.9rem;color:#38bdf8;font-weight:600;margin-bottom:6px;}
.ctx-body{font-size:.82rem;color:var(--muted);max-height:72px;overflow-y:auto;background:rgba(0,0,0,.15);padding:8px 10px;border-radius:6px;border:1px solid rgba(255,255,255,.03);}
.ctx-meta{font-size:.72rem;color:#475569;margin-top:4px;}

/* Sentence */
.sent-display{font-size:1.35rem;line-height:1.7;font-weight:500;padding:14px;border-radius:8px;background:rgba(255,255,255,.012);border:1px dashed rgba(255,255,255,.07);user-select:text;cursor:text;min-height:80px;}
mark.a{background:var(--aspect-bg);color:var(--aspect-c);border-bottom:2px solid var(--aspect-c);padding:1px 3px;border-radius:3px;font-weight:600;}
mark.o{background:var(--opinion-bg);color:var(--opinion-c);border-bottom:2px solid var(--opinion-c);padding:1px 3px;border-radius:3px;font-weight:600;}
.hint{font-size:.75rem;color:var(--muted);margin-top:8px;display:flex;justify-content:space-between;}

/* Slots */
.slots{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px;}
.slot-lbl{font-size:.75rem;font-weight:700;letter-spacing:.05em;margin-bottom:5px;}
.slot-lbl.a{color:var(--aspect-c);}
.slot-lbl.o{color:var(--opinion-c);}
.slot-box{background:var(--bg-card);border-radius:8px;border:1px solid var(--border);padding:6px 10px;display:flex;align-items:center;gap:8px;}
.slot-box.a{background:var(--aspect-bg);border-color:var(--aspect-br);}
.slot-box.o{background:var(--opinion-bg);border-color:var(--opinion-br);}
.slot-input{flex:1;background:transparent;border:none;color:var(--text);font-size:.9rem;font-weight:500;outline:none;padding:0;}
.slot-clear{cursor:pointer;opacity:.6;font-size:1rem;line-height:1;transition:opacity .1s;}
.slot-clear:hover{opacity:1;}

/* Category grid */
.section-lbl{font-size:.78rem;color:var(--muted);font-weight:600;margin-bottom:7px;}
.cat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px;margin-bottom:12px;}
.choice{background:#1e293b;color:var(--muted);border:1px solid var(--border);padding:7px 8px;border-radius:6px;cursor:pointer;font-size:.77rem;font-weight:500;text-align:center;transition:all .12s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.choice:hover{border-color:rgba(255,255,255,.2);color:var(--text);}
.choice.active{background:var(--primary);color:#fff;border-color:transparent;box-shadow:0 0 8px rgba(79,70,229,.4);}

/* Sentiment row */
.sent-row{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:12px;}
.sent-pos.active{background:rgba(16,185,129,.2);color:#34d399;border-color:#34d39960;}
.sent-neg.active{background:rgba(239,68,68,.2);color:#f87171;border-color:#f8717160;}
.sent-neu.active{background:rgba(100,116,139,.2);color:#cbd5e1;border-color:#cbd5e160;}
.sent-mix.active{background:rgba(168,85,247,.2);color:#c084fc;border-color:#c084fc60;}

/* Add button */
.add-row{display:flex;justify-content:flex-end;margin-bottom:2px;}

/* Quad table */
.qtable{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:8px;}
.qtable th{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);color:var(--muted);font-weight:500;}
.qtable td{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04);}
.qtable tr:hover td{background:rgba(255,255,255,.01);}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.75rem;font-weight:600;}
.ba{background:var(--aspect-bg);color:var(--aspect-c);}
.bo{background:var(--opinion-bg);color:var(--opinion-c);}
.bc{background:rgba(99,102,241,.15);color:#a5b4fc;}
.bp{background:rgba(16,185,129,.15);color:#34d399;}
.bn{background:rgba(239,68,68,.15);color:#f87171;}
.bne{background:rgba(100,116,139,.15);color:#cbd5e1;}
.bm{background:rgba(168,85,247,.15);color:#c084fc;}
.no-quad-row{text-align:center;color:var(--muted);padding:16px!important;}

/* Bottom panel */
.save-panel{display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:14px;}
.nav-btns{display:flex;gap:8px;}
.right-btns{display:flex;gap:8px;align-items:center;}
.notes-inp{background:#1e293b;color:var(--text);border:1px solid var(--border);padding:7px 12px;border-radius:8px;outline:none;font-size:.83rem;width:180px;}

/* Tooltip */
.tooltip{position:fixed;background:#1e293b;border:1px solid rgba(255,255,255,.15);border-radius:8px;padding:4px;display:none;box-shadow:0 4px 20px rgba(0,0,0,.5);z-index:1000;gap:4px;}

/* Toast */
.toast{position:fixed;bottom:20px;right:20px;background:#1e293b;border-left:4px solid var(--success);padding:10px 18px;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,.3);display:none;z-index:2000;font-size:.88rem;animation:slidein .2s ease;}
@keyframes slidein{from{transform:translateX(100%);opacity:0;}to{transform:translateX(0);opacity:1;}}

/* Empty state */
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;color:var(--muted);}
.empty-state svg{opacity:.3;}
</style>
</head>
<body>

<header>
  <h1>📝 ABSA Annotation &nbsp;<span style="font-size:.75rem;font-weight:400;color:var(--muted);background:rgba(255,255,255,.06);padding:2px 8px;border-radius:10px;">6k · 7 categories</span></h1>
  <div class="hdr-right">
    <input type="password" id="api-key-input" placeholder="Nhập Gemini API Key..." onchange="saveApiKey(this.value)" style="background:#1e293b;color:var(--text);border:1px solid var(--border);padding:7px 12px;border-radius:8px;font-size:.85rem;width:200px;outline:none;">
    <select id="file-select" onchange="loadFile(this.value)">
      <option value="">— Chọn file annotator —</option>
    </select>
    <span id="hdr-prog" style="font-size:.82rem;color:var(--muted);"></span>
  </div>
</header>

<div class="app">
  <!-- SIDEBAR -->
  <div class="sidebar">
    <div class="sidebar-sec">
      <div class="sidebar-lbl">Tiến độ</div>
      <div class="prog-wrap"><div class="prog-fill" id="prog-fill"></div></div>
      <div class="prog-text" id="prog-text">0 / 0</div>
    </div>
    <div class="sidebar-sec" id="bulk-section" style="display:none; border-bottom:1px solid var(--border);">
      <button class="btn btn-success" id="btn-bulk" onclick="startBulkLabel()" style="width:100%;padding:9px 12px;justify-content:center;font-weight:600;">🤖 Gán nhãn hàng loạt (AI)</button>
      <div id="bulk-progress-text" style="font-size:.8rem;color:var(--muted);margin-top:6px;display:none;text-align:center;font-weight:500;"></div>
    </div>
    <div class="sent-list" id="sent-list">
      <div style="padding:20px;text-align:center;color:var(--muted);font-size:.85rem;">Chọn file để bắt đầu</div>
    </div>
  </div>

  <!-- WORKSPACE -->
  <div class="workspace" id="workspace" style="display:none;">

    <!-- Context -->
    <div class="card">
      <div class="ctx-title" id="thread-title">—</div>
      <div class="ctx-body"  id="parent-ctx">—</div>
      <div class="ctx-meta"  id="ctx-meta"></div>
    </div>

    <!-- Sentence -->
    <div class="card">
      <div class="sent-display" id="sent-display" onmouseup="onSelection(event)">—</div>
      <div class="hint">
        <span>💡 Bôi đen → popup A/O &nbsp;|&nbsp; Phím: <b>A</b>=aspect &nbsp;<b>O</b>=opinion &nbsp;<b>Enter</b>=thêm quad &nbsp;<b>Shift+Enter</b>=lưu&tiếp &nbsp;<b>N</b>=No Quad</span>
        <span>Sentiment: <b>1</b> Pos &nbsp;<b>2</b> Neg &nbsp;<b>3</b> Neu &nbsp;<b>4</b> Mixed</span>
      </div>
    </div>

    <!-- Builder -->
    <div class="card">
      <!-- Slots -->
      <div class="slots">
        <div>
          <div class="slot-lbl a">ASPECT</div>
          <div class="slot-box a">
            <input id="inp-aspect" class="slot-input" placeholder="bôi đen hoặc gõ…" oninput="onAspectInput()">
            <span class="slot-clear" onclick="clearAspect()">✕</span>
          </div>
        </div>
        <div>
          <div class="slot-lbl o">OPINION</div>
          <div class="slot-box o">
            <input id="inp-opinion" class="slot-input" placeholder="bôi đen hoặc gõ…" oninput="onOpinionInput()">
            <span class="slot-clear" onclick="clearOpinion()">✕</span>
          </div>
        </div>
      </div>

      <!-- Category -->
      <div class="section-lbl">CATEGORY</div>
      <div class="cat-grid" id="cat-grid"></div>

      <!-- Sentiment -->
      <div class="section-lbl">SENTIMENT &nbsp;<span style="font-weight:400;color:#475569;">1·2·3·4</span></div>
      <div class="sent-row" id="sent-row"></div>

      <!-- Add -->
      <div class="add-row">
        <button class="btn btn-primary" onclick="autoLabelCurrent()" style="padding:9px 22px;font-size:.9rem;background:#8b5cf6;border-color:transparent;margin-right:8px;color:#fff;">🤖 Gợi ý bằng AI</button>
        <button class="btn btn-primary" onclick="addQuad()" style="padding:9px 22px;font-size:.9rem;">➕ Thêm Quad (Enter)</button>
      </div>
    </div>

    <!-- Quad list -->
    <div class="card">
      <div style="font-family:'Outfit',sans-serif;font-weight:600;font-size:1rem;margin-bottom:8px;">Danh sách Quads</div>
      <table class="qtable">
        <thead><tr><th>Aspect</th><th>Opinion</th><th>Category</th><th>Sentiment</th><th style="width:70px;text-align:center;">Xóa</th></tr></thead>
        <tbody id="quad-tbody"></tbody>
      </table>
    </div>

    <!-- Save panel -->
    <div class="save-panel">
      <div class="nav-btns">
        <button class="btn" onclick="nav(-1)">◀ Trước</button>
        <button class="btn" onclick="goUnfinished()">Chưa gán</button>
        <button class="btn" onclick="nav(1)">Tiếp ▶</button>
      </div>
      <div class="right-btns">
        <input id="notes-inp" class="notes-inp" placeholder="Ghi chú…">
        <button class="btn btn-danger"  onclick="markNoQuad()">⚡ No Quad (N)</button>
        <button class="btn btn-success" onclick="saveAndNext()" style="padding:9px 22px;font-size:.9rem;">💾 Lưu &amp; Tiếp (Shift+Enter)</button>
      </div>
    </div>

  </div><!-- /workspace -->

  <!-- Empty state when no file loaded -->
  <div class="empty-state" id="empty-state">
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
    </svg>
    <span>Chọn file annotator từ dropdown phía trên</span>
  </div>
</div>

<!-- Floating tooltip -->
<div class="tooltip" id="tooltip" style="display:none;flex-direction:row;">
  <button class="btn" style="padding:3px 10px;font-size:.75rem;background:var(--aspect-c);color:#000;border:none;" onclick="setAspectSel()">Aspect (A)</button>
  <button class="btn" style="padding:3px 10px;font-size:.75rem;background:var(--opinion-c);color:#000;border:none;" onclick="setOpinionSel()">Opinion (O)</button>
</div>

<div class="toast" id="toast"></div>

<script>
const CATEGORIES = [
  "PERFORMANCE","RESOURCES","SOFTWARE","INTELLIGENCE",
  "TECHNICAL","BEHAVIOR","COMPARATIVE","None"
];
const SENTIMENTS = ["Positive","Negative","Neutral","Mixed"];
const SENT_CLS   = ["sent-pos","sent-neg","sent-neu","sent-mix"];

let curFile = "", curIdx = 0, sentData = [];
let curSentText = "";
let activeQuads = [];
let selAspect = "", selOpinion = "", selCat = "", selSent = "";
let lastSel = "";
let bulkInterval = null;

function buildGrids() {
  const catGrid = document.getElementById("cat-grid");
  catGrid.innerHTML = "";
  CATEGORIES.forEach(cat => {
    const btn = document.createElement("button");
    btn.className = "choice";
    btn.textContent = cat;
    btn.dataset.cat = cat;
    btn.onclick = () => pickCat(cat);
    catGrid.appendChild(btn);
  });

  const sentRow = document.getElementById("sent-row");
  sentRow.innerHTML = "";
  SENTIMENTS.forEach((sent, idx) => {
    const btn = document.createElement("button");
    btn.className = "choice " + SENT_CLS[idx];
    btn.textContent = `${sent} (${idx+1})`;
    btn.dataset.sent = sent;
    btn.onclick = () => pickSent(sent);
    sentRow.appendChild(btn);
  });
}

// ─── Init ───────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  const savedKey = localStorage.getItem("gemini_api_key") || "";
  document.getElementById("api-key-input").value = savedKey;

  fetchFiles();
  buildGrids();
  setupKeys();
});

function saveApiKey(val) {
  localStorage.setItem("gemini_api_key", val.trim());
  toast("Đã lưu API Key! ✓");
}

function getApiKey() {
  return localStorage.getItem("gemini_api_key") || "";
}

function fetchFiles() {
  fetch("/api/files").then(r=>r.json()).then(d=>{
    const sel = document.getElementById("file-select");
    sel.innerHTML = '<option value="">— Chọn file annotator —</option>';
    d.files.forEach(f=>{
      const o = document.createElement("option");
      o.value = o.textContent = f;
      sel.appendChild(o);
    });
    if (curFile) sel.value = curFile;
  });
}

// ─── File loading ────────────────────────────────────
function loadFile(fname) {
  if (!fname) {
    document.getElementById("workspace").style.display = "none";
    document.getElementById("empty-state").style.display = "flex";
    document.getElementById("bulk-section").style.display = "none";
    clearInterval(bulkInterval);
    return;
  }
  curFile = fname;
  fetch(`/api/load?file=${encodeURIComponent(fname)}`).then(r=>r.json()).then(d=>{
    if (d.error) { alert("Lỗi: "+d.error); return; }
    sentData = d.sentences;
    renderSidebar();
    document.getElementById("workspace").style.display = "flex";
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("bulk-section").style.display = "block";
    loadSentence(d.first_unfinished);
    checkBulkStatus();
  });
}

function renderSidebar() {
  const list = document.getElementById("sent-list");
  list.innerHTML = "";
  let done = 0;
  sentData.forEach(s=>{
    if (s.done) done++;
    const div = document.createElement("div");
    div.className = `sent-item ${s.done?"done":""} ${s.index===curIdx?"active":""}`;
    div.onclick = () => loadSentence(s.index);
    div.innerHTML = `<span class="sent-idx">#${s.index+1}</span><span class="sent-snip">${escHtml(s.text_snippet)}</span><div class="dot"></div>`;
    list.appendChild(div);
  });
  const total = sentData.length;
  const pct = total > 0 ? (done/total)*100 : 0;
  document.getElementById("prog-fill").style.width = pct+"%";
  document.getElementById("prog-text").textContent = `${done} / ${total}`;
  document.getElementById("hdr-prog").textContent = `${done}/${total} (${Math.round(pct)}%)`;
}

// ─── Sentence ────────────────────────────────────────
function loadSentence(idx) {
  curIdx = idx;

  document.querySelectorAll(".sent-item").forEach((el,i)=>{
    el.classList.toggle("active", i===idx);
    if (i===idx) el.scrollIntoView({block:"nearest"});
  });

  fetch(`/api/sentence?file=${encodeURIComponent(curFile)}&index=${idx}`)
    .then(r=>r.json()).then(res=>{
      if (res.error) { alert("Lỗi: "+res.error); return; }
      const s = res.sentence;
      curSentText = s.sentence_text;

      document.getElementById("thread-title").textContent = s.thread_title || "—";
      document.getElementById("parent-ctx").textContent   = s.parent_context || "—";
      document.getElementById("ctx-meta").textContent     = `ID: ${s.id}  |  Câu ${idx+1}/${sentData.length}  |  ${s.human_has_quad ? "Trạng thái: "+s.human_has_quad : "Chưa gán"}`;
      document.getElementById("notes-inp").value = s.notes || "";

      activeQuads = Array.isArray(s.existing_quads) ? s.existing_quads : [];

      clearAspect(); clearOpinion(); pickCat(""); pickSent("");
      renderSentDisplay();
      renderQuadTable();
    });
}

// ─── Auto-Label APIs ────────────────────────────────
function autoLabelCurrent() {
  const key = getApiKey();
  if (!key) { toast("Hãy nhập Gemini API Key trước!", true); return; }
  
  toast("Đang phân tích bằng AI...", false);
  fetch("/api/auto_label_current", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ sentence: curSentText, api_key: key })
  }).then(r=>r.json()).then(d=>{
    if (d.error) { toast("Lỗi: " + d.error, true); return; }
    activeQuads = d.quads || [];
    renderSentDisplay();
    renderQuadTable();
    toast("Đã gợi ý nhãn! Hãy kiểm tra và lưu.");
  });
}

function startBulkLabel() {
  const key = getApiKey();
  if (!key) { toast("Hãy nhập Gemini API Key trước!", true); return; }
  
  if (!confirm("Bắt đầu tự động gán nhãn cho toàn bộ các câu chưa gán trong file này theo hệ thống 7 Category mới? Tốc độ ~10 câu/giây.")) return;
  
  fetch("/api/auto_label_bulk", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ file: curFile, api_key: key })
  }).then(r=>r.json()).then(d=>{
    if (d.error) { toast("Lỗi: " + d.error, true); return; }
    toast("Đã khởi động tiến trình chạy ngầm!");
    checkBulkStatus();
    bulkInterval = setInterval(checkBulkStatus, 2000);
  });
}

function checkBulkStatus() {
  if (!curFile) return;
  fetch(`/api/bulk_status?file=${encodeURIComponent(curFile)}`)
    .then(r=>r.json()).then(d=>{
      const btn = document.getElementById("btn-bulk");
      const prog = document.getElementById("bulk-progress-text");
      
      if (d.status === "running") {
        btn.disabled = true;
        btn.textContent = "🤖 Đang gán nhãn ngầm...";
        prog.style.display = "block";
        prog.textContent = `Tiến độ: Đang chạy (${d.processed} câu mới)`;
      } else if (d.status === "completed") {
        btn.disabled = false;
        btn.textContent = "🤖 Gán nhãn hàng loạt (AI)";
        prog.style.display = "block";
        prog.textContent = "✓ Hoàn thành gán nhãn!";
        clearInterval(bulkInterval);
        loadFile(curFile);
      } else if (d.status.startsWith("error")) {
        btn.disabled = false;
        btn.textContent = "🤖 Gán nhãn hàng loạt (AI)";
        prog.style.display = "block";
        prog.textContent = `Lỗi: ${d.status}`;
        clearInterval(bulkInterval);
      } else {
        btn.disabled = false;
        btn.textContent = "🤖 Gán nhãn hàng loạt (AI)";
        prog.style.display = "none";
      }
    });
}

// ─── Sentence display & highlights ───────────────────
function renderSentDisplay() {
  const el = document.getElementById("sent-display");
  let html = escHtml(curSentText);

  const marks = [];
  activeQuads.forEach(q=>{
    if (q.aspect  && q.aspect  !== "None") marks.push({text:q.aspect,  cls:"a"});
    if (q.opinion && q.opinion !== "None") marks.push({text:q.opinion, cls:"o"});
  });
  if (selAspect)  marks.push({text:selAspect,  cls:"a"});
  if (selOpinion) marks.push({text:selOpinion, cls:"o"});

  marks.sort((a,b)=>b.text.length-a.text.length);
  const phs = [];
  marks.forEach((m,i)=>{
    const rx = new RegExp("("+escRe(m.text)+")", "gi");
    const ph = `__PH${i}__`;
    if (rx.test(html)) {
      html = html.replace(new RegExp("("+escRe(m.text)+")", "gi"), ph);
      phs.push({ph, text:m.text, cls:m.cls});
    }
  });
  phs.forEach(p=>{
    html = html.replace(new RegExp(p.ph,"g"), `<mark class="${p.cls}">${escHtml(p.text)}</mark>`);
  });
  el.innerHTML = html;
}

// ─── Selection tooltip ───────────────────────────────
function onSelection(e) {
  const sel  = window.getSelection().toString().trim();
  const tip  = document.getElementById("tooltip");
  if (!sel) { tip.style.display="none"; return; }
  lastSel = sel;
  const range = window.getSelection().getRangeAt(0);
  const rect  = range.getBoundingClientRect();
  tip.style.left    = (rect.left + rect.width/2 - 60)+"px";
  tip.style.top     = (rect.top  - 44)+"px";
  tip.style.display = "flex";
}

document.addEventListener("mousedown", e=>{
  if (!document.getElementById("tooltip").contains(e.target) &&
      !document.getElementById("sent-display").contains(e.target)) {
    document.getElementById("tooltip").style.display = "none";
  }
});

function setAspectSel()  { setAspect(lastSel);  hideTooltip(); }
function setOpinionSel() { setOpinion(lastSel); hideTooltip(); }
function hideTooltip()   { document.getElementById("tooltip").style.display="none"; window.getSelection().removeAllRanges(); }

// ─── Aspect / Opinion setters ─────────────────────────
function setAspect(t)  { selAspect  = t; document.getElementById("inp-aspect").value  = t; renderSentDisplay(); }
function setOpinion(t) { selOpinion = t; document.getElementById("inp-opinion").value = t; renderSentDisplay(); }
function clearAspect()  { setAspect(""); }
function clearOpinion() { setOpinion(""); }
function onAspectInput()  { selAspect  = document.getElementById("inp-aspect").value;  renderSentDisplay(); }
function onOpinionInput() { selOpinion = document.getElementById("inp-opinion").value; renderSentDisplay(); }

// ─── Category / Sentiment ─────────────────────────────
function pickCat(c) {
  selCat = c;
  document.querySelectorAll("#cat-grid .choice").forEach(b=>b.classList.toggle("active", b.dataset.cat===c));
}
function pickSent(s) {
  selSent = s;
  document.querySelectorAll("#sent-row .choice").forEach(b=>b.classList.toggle("active", b.dataset.sent===s));
}

// ─── Add quad ─────────────────────────────────────────
function addQuad() {
  if (!selCat)  { toast("Chọn Category trước!",  true); return; }
  if (!selSent) { toast("Chọn Sentiment trước!", true); return; }
  activeQuads.push({
    aspect:    selAspect  || "None",
    opinion:   selOpinion || "None",
    category:  selCat,
    sentiment: selSent,
  });
  clearAspect(); clearOpinion();
  renderSentDisplay();
  renderQuadTable();
}

function deleteQuad(i) {
  activeQuads.splice(i, 1);
  renderSentDisplay();
  renderQuadTable();
}

// ─── Quad table ───────────────────────────────────────
function renderQuadTable() {
  const tb = document.getElementById("quad-tbody");
  if (activeQuads.length === 0) {
    tb.innerHTML = '<tr><td colspan="5" class="no-quad-row">Chưa có quad nào. Bôi đen → A/O → chọn Category & Sentiment → Enter.</td></tr>';
    return;
  }
  tb.innerHTML = "";
  activeQuads.forEach((q,i)=>{
    const sc = (q.sentiment||"").toLowerCase();
    const sb = sc==="positive"?"bp":sc==="negative"?"bn":sc==="mixed"?"bm":"bne";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="badge ba">${escHtml(q.aspect)}</span></td>
      <td><span class="badge bo">${escHtml(q.opinion)}</span></td>
      <td><span class="badge bc">${escHtml(q.category)}</span></td>
      <td><span class="badge ${sb}">${escHtml(q.sentiment)}</span></td>
      <td style="text-align:center"><button class="btn btn-danger" style="padding:2px 8px;font-size:.73rem;" onclick="deleteQuad(${i})">Xóa</button></td>`;
    tb.appendChild(tr);
  });
}

// ─── Save ─────────────────────────────────────────────
function buildValues(hasQuad) {
  const quads = hasQuad ? activeQuads : [];
  const aspect   = quads.length===1 ? quads[0].aspect   : (quads.length>1?"Multi":"None");
  const opinion  = quads.length===1 ? quads[0].opinion  : (quads.length>1?"Multi":"None");
  const cats     = [...new Set(quads.map(q=>q.category))];
  const sents    = [...new Set(quads.map(q=>q.sentiment))];
  const category = cats.length===1  ? cats[0]  : (cats.length>1?"Multi":"None");
  const sentiment= sents.length===1 ? sents[0] : (sents.length>1?"Mixed":"None");
  return {
    human_has_quad:       hasQuad && quads.length>0 ? "Yes" : "No",
    human_aspect:         aspect,
    human_opinion:        opinion,
    human_category_label: category,
    human_sentiment_label:sentiment,
    human_quads_json:     JSON.stringify(quads),
    notes: document.getElementById("notes-inp").value,
  };
}

function doSave(hasQuad, moveNext) {
  if (hasQuad && (selAspect||selOpinion) && selCat && selSent) addQuad();

  const values = buildValues(hasQuad);
  fetch("/api/save", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({file:curFile, index:curIdx, values}),
  }).then(r=>r.json()).then(res=>{
    if (res.status !== "success") { alert("Lỗi lưu: "+res.error); return; }
    toast("Đã lưu! ✓");
    sentData[curIdx].done = true;
    renderSidebar();
    if (moveNext && curIdx < sentData.length-1) loadSentence(curIdx+1);
  });
}

function saveAndNext() { doSave(true, true); }
function markNoQuad()  { activeQuads=[]; renderQuadTable(); renderSentDisplay(); doSave(false, true); }

// ─── Navigation ───────────────────────────────────────
function nav(d) {
  const next = curIdx + d;
  if (next >= 0 && next < sentData.length) loadSentence(next);
}
function goUnfinished() {
  const u = sentData.find(s=>!s.done);
  if (u) loadSentence(u.index);
  else toast("Tất cả câu đã gán nhãn! 🎉");
}

// ─── Keyboard shortcuts ───────────────────────────────
function setupKeys() {
  document.addEventListener("keydown", e=>{
    const tag = (e.target.tagName||"").toUpperCase();
    const inInput = ["INPUT","TEXTAREA","SELECT"].includes(tag);
    const key = e.key;

    if (inInput) {
      if (key==="Enter" && e.shiftKey) { e.preventDefault(); saveAndNext(); }
      else if (key==="Enter")          { e.preventDefault(); addQuad(); }
      return;
    }

    const sel = window.getSelection().toString().trim();
    if (key==="a"||key==="A") { e.preventDefault(); if (sel){setAspect(sel);window.getSelection().removeAllRanges();}else if(lastSel)setAspect(lastSel); }
    else if (key==="o"||key==="O") { e.preventDefault(); if (sel){setOpinion(sel);window.getSelection().removeAllRanges();}else if(lastSel)setOpinion(lastSel); }
    else if (key==="Enter"&&e.shiftKey) { e.preventDefault(); saveAndNext(); }
    else if (key==="Enter")  { e.preventDefault(); addQuad(); }
    else if (key==="n"||key==="N"||key==="Escape") { e.preventDefault(); markNoQuad(); }
    else if (key==="ArrowLeft")  { e.preventDefault(); nav(-1); }
    else if (key==="ArrowRight") { e.preventDefault(); nav(1); }
    else if (["1","2","3","4"].includes(key)) { e.preventDefault(); pickSent(SENTIMENTS[parseInt(key)-1]); }
  });
}

// ─── Utilities ────────────────────────────────────────
function toast(msg, isErr=false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.borderLeftColor = isErr ? "var(--danger)" : "var(--success)";
  t.style.display = "block";
  clearTimeout(t._tid);
  t._tid = setTimeout(()=>t.style.display="none", 2000);
}
function escHtml(s) {
  if (!s) return "";
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"); }
</script>
</body>
</html>
"""
MODEL_NAME = "gemini-2.5-flash"


# =========================
# MAIN
# =========================

def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run():
    port = get_free_port()
    httpd = HTTPServer(("localhost", port), AnnotationHandler)
    url   = f"http://localhost:{port}"

    print("=" * 55)
    print(f"  ABSA Annotation Tool — human_verification_6000 (7 Categories)")
    print(f"  URL  : {url}")
    print(f"  Data : {SPLIT_DIR}")
    print("=" * 55)
    print("Ctrl+C to stop.\n")

    webbrowser.open_new_tab(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    run()
