# src/annotation/annotation_tool_v2.py

import os
import sys
import json
import csv
import webbrowser
import socket
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Import pandas if available, otherwise fallback to built-in csv module
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "final_data" / "human_verification_900"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "sample_id",
    "record_id",
    "sentence_index",
    "thread_title",
    "parent_context",
    "sentence_text",
    "human_has_quad",
    "human_aspect",
    "human_opinion",
    "human_category_label",
    "human_sentiment_label",
    "human_quads_json",
    "annotator",
    "notes",
]

# =========================
# DATA HELPERS
# =========================

def list_annotation_files():
    """Lists source csv files from the SPLIT_DIR (excluding filled outputs)."""
    files = sorted(SPLIT_DIR.glob("*.csv"))
    # Return files that don't end with _filled.csv
    return [f for f in files if not f.name.endswith("_filled.csv")]

def output_path_of(input_path: Path):
    """Maps input CSV path to its filled equivalent."""
    return input_path.with_name(input_path.stem + "_filled.csv")

def is_row_done(row):
    """Checks if a row is already annotated."""
    val = str(row.get("human_has_quad", "")).strip().upper()
    return val in {"YES", "NO"}

def sanitize_row(row):
    """Sanitizes row data to be serializable and valid."""
    sanitized = {}
    for col in REQUIRED_COLS:
        val = row.get(col, "")
        
        # 1. Handle Python None
        if val is None:
            val = ""
            
        # 2. Handle float and numpy NaN
        elif isinstance(val, float) or str(type(val)).find("float") != -1:
            if val != val:  # NaN check
                val = ""
                
        val_str = str(val).strip()
        val_str_lower = val_str.lower()
        
        # 3. Handle string representations of NaN/None that might have slipped in
        if val_str_lower in {"nan", "null", "<na>", "nat"}:
            val_str = ""
            
        # 4. Handle "none" vs "" for fields where "None" is not a valid text content
        # For sample metadata fields, "None" should be empty string
        if col in {"sample_id", "record_id", "thread_title", "parent_context", "sentence_text"}:
            if val_str_lower == "none":
                val_str = ""
                
        # For human_quads_json, if it is empty, make sure it is valid JSON "[]"
        if col == "human_quads_json":
            if val_str == "" or val_str_lower == "none":
                val_str = "[]"
                
        sanitized[col] = val_str
    return sanitized

def repair_corrupt_row(row):
    """Dynamically repairs rows that got merged/corrupted by previous tools."""
    sample_val = str(row.get("sample_id", "")).strip()
    if "," in sample_val and ("SAMPLE_" in sample_val or "S_" in sample_val):
        try:
            import io
            f_in = io.StringIO(sample_val)
            reader = csv.reader(f_in)
            parts = next(reader)
            
            repaired = {}
            for i, col in enumerate(REQUIRED_COLS):
                if i < len(parts):
                    val = parts[i].strip()
                    if val.lower() in {"nan", "null", "<na>", "nat"}:
                        val = ""
                    repaired[col] = val
                else:
                    repaired[col] = ""
            
            # Map human fields back from original row, shifting if necessary
            for col in ["human_has_quad", "human_aspect", "human_opinion", "human_category_label", "human_sentiment_label", "human_quads_json", "annotator", "notes"]:
                orig_val = str(row.get(col, "")).strip()
                if orig_val and orig_val.lower() not in {"nan", "null", "<na>", "nat"}:
                    if col == "human_quads_json" and orig_val == "[]":
                        if repaired.get(col, "") not in {"", "[]"}:
                            continue
                    repaired[col] = orig_val
            return repaired
        except Exception:
            pass
    return row

def load_data(file_name: str):
    """Loads CSV data using pandas (or fallback built-in csv reader)."""
    input_path = SPLIT_DIR / file_name
    output_path = output_path_of(input_path)

    target_path = output_path if output_path.exists() else input_path
    
    rows = []
    if HAS_PANDAS:
        df = pd.read_csv(target_path, encoding="utf-8-sig")
        # Ensure all columns are present
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = ""
        df = df.fillna("")
        for idx, r in df.iterrows():
            row_dict = r.to_dict()
            # Map legacy column names if present
            if "id" in row_dict and not row_dict.get("record_id"):
                row_dict["record_id"] = row_dict["id"]
            if "sentence" in row_dict and not row_dict.get("sentence_text"):
                row_dict["sentence_text"] = row_dict["sentence"]
            if not row_dict.get("sentence_index"):
                row_dict["sentence_index"] = str(idx)
                
            repaired = repair_corrupt_row(row_dict)
            rows.append(sanitize_row(repaired))
    else:
        with open(target_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                row_dict = dict(r)
                # Map legacy column names if present
                if "id" in row_dict and not row_dict.get("record_id"):
                    row_dict["record_id"] = row_dict["id"]
                if "sentence" in row_dict and not row_dict.get("sentence_text"):
                    row_dict["sentence_text"] = row_dict["sentence"]
                if not row_dict.get("sentence_index"):
                    row_dict["sentence_index"] = str(idx)
                for col in REQUIRED_COLS:
                    if col not in row_dict:
                        row_dict[col] = ""
                repaired = repair_corrupt_row(row_dict)
                rows.append(sanitize_row(repaired))
                
    return rows, output_path

def save_data(rows, output_path: Path):
    """Saves rows to CSV using pandas (or fallback csv writer)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    original_cols = [
        "sample_id",
        "id",
        "parent_context",
        "thread_title",
        "sentence",
        "human_has_quad",
        "human_aspect",
        "human_category_label",
        "human_opinion",
        "human_sentiment_label",
        "human_quads_json"
    ]
    
    mapped_rows = []
    for r in rows:
        row_dict = dict(r)
        if "record_id" in row_dict:
            row_dict["id"] = row_dict["record_id"]
        if "sentence_text" in row_dict:
            row_dict["sentence"] = row_dict["sentence_text"]
        
        clean_row = {}
        for col in original_cols:
            clean_row[col] = row_dict.get(col, "")
        mapped_rows.append(clean_row)
    
    if HAS_PANDAS:
        df = pd.DataFrame(mapped_rows)
        df = df[original_cols]
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=original_cols, extrasaction="ignore")
            writer.writeheader()
            for r in mapped_rows:
                writer.writerow(r)

# =========================
# WEB SERVER
# =========================

class AnnotationHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default request logging to keep console clean
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def send_html(self, html_content, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/" or path == "/index.html":
            self.send_html(HTML_TEMPLATE)
            return

        if path == "/api/files":
            files = list_annotation_files()
            self.send_json({"files": [f.name for f in files]})
            return

        if path == "/api/load":
            file_name = query.get("file", [None])[0]
            if not file_name:
                self.send_json({"error": "Missing 'file' parameter"}, 400)
                return
            
            # Security: prevent directory traversal
            file_name = Path(file_name).name
            
            try:
                rows, _ = load_data(file_name)
                # Find first unfinished index
                first_unfinished = 0
                for i, row in enumerate(rows):
                    if not is_row_done(row):
                        first_unfinished = i
                        break
                
                # Send back rows list with essential progress info
                sentence_list = []
                for i, row in enumerate(rows):
                    sentence_list.append({
                        "index": i,
                        "sample_id": row.get("sample_id", f"S_{i}"),
                        "text_snippet": row.get("sentence_text", "")[:40] + "...",
                        "done": is_row_done(row)
                    })
                
                self.send_json({
                    "total": len(rows),
                    "first_unfinished": first_unfinished,
                    "sentences": sentence_list
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/sentence":
            file_name = query.get("file", [None])[0]
            index_str = query.get("index", [None])[0]
            
            if not file_name or index_str is None:
                self.send_json({"error": "Missing 'file' or 'index' parameter"}, 400)
                return
            
            # Security: prevent directory traversal
            file_name = Path(file_name).name
            
            try:
                idx = int(index_str)
                rows, _ = load_data(file_name)
                if idx < 0 or idx >= len(rows):
                    self.send_json({"error": "Index out of range"}, 400)
                    return
                
                self.send_json({
                    "index": idx,
                    "sentence": sanitize_row(rows[idx])
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        if path == "/api/upload":
            try:
                data = json.loads(post_data.decode("utf-8"))
                filename = data.get("filename", "").strip()
                content = data.get("content", "")
                
                if not filename or not content:
                    self.send_json({"error": "Invalid file upload data"}, 400)
                    return
                
                # Security: prevent directory traversal
                filename = Path(filename).name
                
                if not filename.endswith(".csv"):
                    filename += ".csv"
                    
                target_path = SPLIT_DIR / filename
                with open(target_path, "w", encoding="utf-8-sig") as f:
                    f.write(content)
                    
                self.send_json({"status": "success", "filename": filename})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return
            
        if path == "/api/save":
            try:
                data = json.loads(post_data.decode("utf-8"))
                file_name = data.get("file")
                idx = data.get("index")
                values = data.get("values")
                
                if not file_name or idx is None or not values:
                    self.send_json({"error": "Invalid request body"}, 400)
                    return
                
                # Security: prevent directory traversal
                file_name = Path(file_name).name
                
                idx = int(idx)
                rows, output_path = load_data(file_name)
                
                if idx < 0 or idx >= len(rows):
                    self.send_json({"error": "Index out of range"}, 400)
                    return
                
                # Update row fields
                for col in REQUIRED_COLS:
                    if col in values:
                        rows[idx][col] = str(values[col])
                
                # Save CSV
                save_data(rows, output_path)
                
                # Calculate next unfinished
                next_index = idx + 1
                if next_index >= len(rows):
                    next_index = len(rows) - 1
                
                self.send_json({
                    "status": "success",
                    "nextIndex": next_index,
                    "savedIndex": idx
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        self.send_response(404)
        self.end_headers()

# =========================
# FRONTEND HTML TEMPLATE
# =========================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABSA Quadruplet Annotation Dashboard</title>
    <!-- Outfit & Inter Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #12182b;
            --bg-card: rgba(22, 28, 54, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            
            --aspect-color: #00f2fe;
            --aspect-bg: rgba(0, 242, 254, 0.12);
            --aspect-border: rgba(0, 242, 254, 0.4);
            
            --opinion-color: #ff9f43;
            --opinion-bg: rgba(255, 159, 67, 0.12);
            --opinion-border: rgba(255, 159, 67, 0.4);
            
            --primary-color: #4f46e5;
            --primary-hover: #6366f1;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        header {
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #00f2fe 0%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header-controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .btn {
            background-color: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn:hover {
            background-color: #334155;
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-primary {
            background-color: var(--primary-color);
            border-color: transparent;
        }

        .btn-primary:hover {
            background-color: var(--primary-hover);
        }

        .btn-success {
            background-color: var(--success-color);
            border-color: transparent;
        }

        .btn-success:hover {
            background-color: #059669;
        }

        .btn-danger {
            background-color: transparent;
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--danger-color);
        }

        .btn-danger:hover {
            background-color: rgba(239, 68, 68, 0.1);
        }

        select {
            background-color: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 8px;
            outline: none;
            cursor: pointer;
            font-size: 0.875rem;
            min-width: 220px;
        }

        .app-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* SIDEBAR */
        .sidebar {
            width: 320px;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-section {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
        }

        .sidebar-title {
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            font-weight: 600;
        }

        /* Progress list */
        .sentence-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }

        .sentence-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 4px;
            font-size: 0.85rem;
            transition: background 0.15s ease;
        }

        .sentence-item:hover {
            background-color: rgba(255, 255, 255, 0.03);
        }

        .sentence-item.active {
            background-color: rgba(79, 70, 229, 0.15);
            border-left: 3px solid var(--primary-color);
            color: var(--text-main);
            font-weight: 500;
        }

        .sentence-item .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #475569;
        }

        .sentence-item.done .status-dot {
            background-color: var(--success-color);
            box-shadow: 0 0 8px var(--success-color);
        }

        .sentence-index {
            color: var(--text-muted);
            margin-right: 6px;
            font-weight: 600;
        }

        .sentence-snippet {
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-right: 8px;
        }

        /* MAIN WORKSPACE */
        .workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 24px;
            gap: 20px;
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }

        /* Context header */
        .context-card {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .thread-title {
            font-size: 0.95rem;
            color: #38bdf8;
            font-weight: 600;
        }

        .parent-context {
            font-size: 0.875rem;
            color: var(--text-muted);
            max-height: 80px;
            overflow-y: auto;
            background: rgba(0,0,0,0.15);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        /* Sentence display */
        .sentence-card {
            position: relative;
        }

        .sentence-container {
            font-size: 1.5rem;
            line-height: 1.6;
            font-weight: 500;
            color: var(--text-main);
            padding: 16px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.015);
            border: 1px dashed rgba(255, 255, 255, 0.08);
            user-select: text;
        }

        /* Highlight classes */
        mark.aspect-highlight {
            background-color: var(--aspect-bg);
            color: var(--aspect-color);
            border-bottom: 2px solid var(--aspect-color);
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 600;
        }

        mark.opinion-highlight {
            background-color: var(--opinion-bg);
            color: var(--opinion-color);
            border-bottom: 2px solid var(--opinion-color);
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 600;
        }

        mark.both-highlight {
            background-color: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border-bottom: 2px solid #a855f7;
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 600;
        }

        .help-text {
            font-size: 0.775rem;
            color: var(--text-muted);
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
        }

        /* Quad Builder */
        .quad-builder {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .selection-slots {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .slot {
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }

        .slot-aspect {
            background-color: var(--aspect-bg);
            border-color: var(--aspect-border);
            color: var(--aspect-color);
        }

        .slot-opinion {
            background-color: var(--opinion-bg);
            border-color: var(--opinion-border);
            color: var(--opinion-color);
        }

        .slot .clear-slot {
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.15s ease;
            font-size: 1.1rem;
        }

        .slot .clear-slot:hover {
            opacity: 1;
        }

        .slot-empty {
            background-color: rgba(255, 255, 255, 0.02);
            border-color: var(--border-color);
            color: var(--text-muted);
            font-style: italic;
        }

        /* Selection grids */
        .option-group-title {
            font-size: 0.825rem;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 8px;
        }

        .choice-btn {
            background-color: #1e293b;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            padding: 8px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.775rem;
            font-weight: 500;
            text-align: center;
            transition: all 0.15s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .choice-btn:hover {
            border-color: rgba(255, 255, 255, 0.2);
            color: var(--text-main);
        }

        .choice-btn.active {
            background-color: var(--primary-color);
            color: white;
            border-color: transparent;
            box-shadow: 0 0 10px rgba(79, 70, 229, 0.4);
        }

        .sentiment-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }

        /* Active Quads list */
        .quad-list-container {
            margin-top: 10px;
        }

        .quad-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.875rem;
        }

        .quad-table th, .quad-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .quad-table th {
            color: var(--text-muted);
            font-weight: 500;
        }

        .quad-table tr:hover {
            background-color: rgba(255,255,255,0.01);
        }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-aspect {
            background-color: var(--aspect-bg);
            color: var(--aspect-color);
        }

        .badge-opinion {
            background-color: var(--opinion-bg);
            color: var(--opinion-color);
        }

        .badge-category {
            background-color: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
        }

        .badge-sentiment {
            background-color: rgba(255, 255, 255, 0.05);
        }

        .badge-positive { background-color: rgba(16, 185, 129, 0.15); color: #34d399; }
        .badge-negative { background-color: rgba(239, 68, 68, 0.15); color: #f87171; }
        .badge-neutral { background-color: rgba(100, 116, 139, 0.15); color: #cbd5e1; }
        .badge-mixed { background-color: rgba(168, 85, 247, 0.15); color: #c084fc; }

        /* Floating Tooltip Menu */
        .selection-tooltip {
            position: absolute;
            background-color: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            padding: 4px;
            display: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 1000;
            gap: 4px;
        }

        /* Bottom Save Panel */
        .save-panel {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
        }

        .nav-buttons {
            display: flex;
            gap: 8px;
        }

        .notes-input {
            width: 250px;
            background-color: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 8px;
            outline: none;
            font-size: 0.85rem;
        }

        /* Drag-drop overlay */
        .upload-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 2px dashed rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(255,255,255,0.01);
        }

        .upload-card:hover {
            border-color: var(--primary-color);
            background: rgba(79, 70, 229, 0.05);
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 8px;
            color: var(--text-muted);
        }

        /* Toast notifications */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background-color: #1e293b;
            border-left: 4px solid var(--success-color);
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: none;
            z-index: 2000;
            font-size: 0.9rem;
            animation: slideIn 0.2s ease;
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Circular progress styling */
        .progress-container {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .progress-bar-wrapper {
            flex: 1;
            height: 6px;
            background-color: rgba(255,255,255,0.05);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background-color: var(--success-color);
            width: 0%;
            transition: width 0.3s ease;
            box-shadow: 0 0 8px var(--success-color);
        }

        .progress-text {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
        }
    </style>
</head>
<body>

    <header>
        <h1>📝 ABSA Annotation Tool <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted); background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 12px;">v2.0</span></h1>
        <div class="header-controls">
            <select id="file-select" onchange="loadFile(this.value)">
                <option value="">-- Chọn File gán nhãn --</option>
            </select>
            <button class="btn btn-primary" onclick="triggerFileUpload()">
                📤 Upload CSV
            </button>
            <input type="file" id="file-uploader" accept=".csv" style="display: none;" onchange="handleFileSelect(event)">
        </div>
    </header>

    <div class="app-container">
        <!-- SIDEBAR -->
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">TIẾN ĐỘ</div>
                <div class="progress-container">
                    <div class="progress-bar-wrapper">
                        <div class="progress-bar-fill" id="progress-bar"></div>
                    </div>
                    <span class="progress-text" id="progress-text">0 / 0</span>
                </div>
            </div>
            
            <div class="sentence-list" id="sentence-list">
                <!-- Sentences load dynamically -->
                <div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.9rem;">
                    Vui lòng chọn hoặc upload file gán nhãn
                </div>
            </div>
        </div>

        <!-- MAIN WORKSPACE -->
        <div class="workspace" id="workspace-area" style="display: none;">
            
            <!-- Metadata and Context -->
            <div class="card context-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="thread-title" id="thread-title">No Thread Title</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted); background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px;" id="sample-meta">SAMPLE: None</span>
                </div>
                <div class="parent-context" id="parent-context">No Context Available</div>
            </div>

            <!-- Sentence Display -->
            <div class="card sentence-card">
                <div class="sentence-container" id="sentence-display" onmouseup="handleSelection(event)">
                    Sentence text here
                </div>
                <div class="help-text">
                    <span>💡 Bôi đen cụm từ trong câu: Nhấn phím <b>A</b> để chọn Aspect, nhấn phím <b>O</b> để chọn Opinion.</span>
                    <span>Phím tắt: <b>1-4</b> Chọn Sentiment</span>
                </div>
            </div>

            <!-- Quadruplet Builder -->
            <div class="card quad-builder">
                <div class="selection-slots" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 8px;">
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: var(--aspect-color); letter-spacing: 0.05em;">ASPECT TARGET</span>
                        <div class="slot slot-aspect" id="slot-aspect" style="padding: 6px 12px; display: flex; justify-content: space-between; align-items: center; border-color: var(--aspect-border); background-color: var(--aspect-bg);">
                            <input type="text" id="input-aspect" class="notes-input" style="flex: 1; border: none; background: transparent; color: var(--text-main); font-weight: 500; font-size: 0.9rem; padding: 0; outline: none; box-shadow: none;" placeholder="Nhập aspect hoặc bôi đen..." oninput="updateAspectFromInput()">
                            <span class="clear-slot" onclick="clearAspect()" style="color: var(--aspect-color); margin-left: 8px;">✕</span>
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: var(--opinion-color); letter-spacing: 0.05em;">OPINION</span>
                        <div class="slot slot-opinion" id="slot-opinion" style="padding: 6px 12px; display: flex; justify-content: space-between; align-items: center; border-color: var(--opinion-border); background-color: var(--opinion-bg);">
                            <input type="text" id="input-opinion" class="notes-input" style="flex: 1; border: none; background: transparent; color: var(--text-main); font-weight: 500; font-size: 0.9rem; padding: 0; outline: none; box-shadow: none;" placeholder="Nhập opinion hoặc bôi đen..." oninput="updateOpinionFromInput()">
                            <span class="clear-slot" onclick="clearOpinion()" style="color: var(--opinion-color); margin-left: 8px;">✕</span>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="option-group-title">
                        <span>ASPECT CATEGORY</span>
                    </div>
                    <div class="category-grid" id="category-btn-grid">
                        <!-- Dynamic Categories -->
                    </div>
                </div>

                <div>
                    <div class="option-group-title">
                        <span>SENTIMENT</span>
                        <span style="font-size: 0.75rem; font-weight: normal; color: var(--text-muted);">Phím tắt: 1 (Pos), 2 (Neg), 3 (Neu), 4 (Mixed)</span>
                    </div>
                    <div class="sentiment-row" id="sentiment-btn-row">
                        <!-- Dynamic Sentiments -->
                    </div>
                </div>

                <div style="display: flex; gap: 12px; justify-content: flex-end; align-items: center;">
                    <button class="btn btn-primary" onclick="addQuadruplet()" style="padding: 10px 24px; font-size: 0.95rem;">
                        ➕ Add Quadruplet (Enter)
                    </button>
                </div>
            </div>

            <!-- Active Quadruplets List -->
            <div class="card quad-list-container">
                <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px;">Danh sách bộ 4 đã tạo</h3>
                <table class="quad-table">
                    <thead>
                        <tr>
                            <th>Aspect</th>
                            <th>Opinion</th>
                            <th>Category</th>
                            <th>Sentiment</th>
                            <th style="width: 80px; text-align: center;">Hành động</th>
                        </tr>
                    </thead>
                    <tbody id="quad-table-body">
                        <tr>
                            <td colspan="5" style="text-align: center; color: var(--text-muted);">Chưa có bộ quadruplet nào được thêm.</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Bottom Save / Navigation panel -->
            <div class="save-panel">
                <div class="nav-buttons">
                    <button class="btn" onclick="navigateSentence(-1)">⬅️ Trước</button>
                    <button class="btn" onclick="goToUnfinished()">Câu chưa gán</button>
                    <button class="btn" onclick="navigateSentence(1)">Tiếp ➡️</button>
                </div>
                
                <div style="display: flex; gap: 12px; align-items: center;">
                    <input type="text" id="notes-field" class="notes-input" placeholder="Ghi chú câu này (nếu có)">
                    <button class="btn btn-danger" onclick="markNoQuad()">⚡ No Quad (N)</button>
                    <button class="btn btn-success" onclick="saveAndNext()" style="padding: 10px 24px; font-size: 0.95rem;">
                        💾 Lưu & Tiếp (Shift+Enter)
                    </button>
                </div>
            </div>

        </div>
    </div>

    <!-- Floating Selection Tooltip -->
    <div class="selection-tooltip" id="selection-tooltip">
        <button class="btn btn-primary" style="padding: 4px 10px; font-size: 0.75rem; background-color: var(--aspect-color); color: #000;" onclick="setAspectFromSelection()">Aspect (A)</button>
        <button class="btn btn-primary" style="padding: 4px 10px; font-size: 0.75rem; background-color: var(--opinion-color); color: #000;" onclick="setOpinionFromSelection()">Opinion (O)</button>
    </div>

    <!-- Toast Notification -->
    <div class="toast" id="toast-notif">Đã lưu nhãn thành công!</div>

    <script>
        const CATEGORIES = [
            "BEHAVIOR", "PERFORMANCE", "COMPARATIVE", "RESOURCES", 
            "TOOLING", "CODING", "KNOWLEDGE", "QUANTIZATION", 
            "REASONING", "FINETUNING", "RAG_CONTEXT", "None"
        ];
        const SENTIMENTS = ["Positive", "Negative", "Neutral", "Mixed"];

        let currentFile = "";
        let currentIndex = 0;
        let sentencesData = [];
        let currentSentenceText = "";
        let activeQuads = [];

        let selectedAspectText = "";
        let selectedOpinionText = "";
        let selectedCategory = "";
        let selectedSentiment = "";

        let lastSelectionText = "";

        // On Load
        window.addEventListener('DOMContentLoaded', () => {
            fetchFiles();
            buildOptionsGrids();
            setupKeyboardShortcuts();
        });

        // Fetch List of Labeled CSV files
        function fetchFiles() {
            fetch('/api/files')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('file-select');
                    select.innerHTML = '<option value="">-- Chọn File gán nhãn --</option>';
                    data.files.forEach(f => {
                        const opt = document.createElement('option');
                        opt.value = f;
                        opt.textContent = f;
                        select.appendChild(opt);
                    });
                    if (currentFile) {
                        select.value = currentFile;
                    }
                });
        }

        // Build HTML buttons for categories & sentiments
        function buildOptionsGrids() {
            const catGrid = document.getElementById('category-btn-grid');
            catGrid.innerHTML = '';
            CATEGORIES.forEach(cat => {
                const btn = document.createElement('button');
                btn.className = 'choice-btn';
                btn.textContent = cat;
                btn.onclick = () => selectCategory(cat);
                btn.setAttribute('data-cat', cat);
                catGrid.appendChild(btn);
            });

            const sentRow = document.getElementById('sentiment-btn-row');
            sentRow.innerHTML = '';
            SENTIMENTS.forEach((sent, idx) => {
                const btn = document.createElement('button');
                btn.className = 'choice-btn';
                btn.textContent = `${sent} (${idx + 1})`;
                btn.onclick = () => selectSentiment(sent);
                btn.setAttribute('data-sent', sent);
                sentRow.appendChild(btn);
            });
        }

        function triggerFileUpload() {
            document.getElementById('file-uploader').click();
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                const csvContent = evt.target.result;
                fetch('/api/upload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        filename: file.name,
                        content: csvContent
                    })
                })
                .then(r => r.json())
                .then(res => {
                    if (res.status === 'success') {
                        currentFile = res.filename;
                        fetchFiles();
                        setTimeout(() => {
                            loadFile(res.filename);
                        }, 200);
                        showToast(`Đã tải lên file ${res.filename}`);
                    } else {
                        alert("Lỗi upload: " + res.error);
                    }
                });
            };
            reader.readAsText(file);
        }

        function loadFile(fileName) {
            if (!fileName) {
                document.getElementById('workspace-area').style.display = 'none';
                return;
            }
            currentFile = fileName;
            fetch(`/api/load?file=${encodeURIComponent(fileName)}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        alert("Lỗi: " + data.error);
                        return;
                    }
                    sentencesData = data.sentences;
                    renderSidebarProgress();
                    document.getElementById('workspace-area').style.display = 'flex';
                    
                    // Load the first unfinished sentence
                    loadSentence(data.first_unfinished);
                });
        }

        function renderSidebarProgress() {
            const listContainer = document.getElementById('sentence-list');
            listContainer.innerHTML = '';
            
            let doneCount = 0;
            sentencesData.forEach(s => {
                if (s.done) doneCount++;
                
                const item = document.createElement('div');
                item.className = `sentence-item ${s.done ? 'done' : ''} ${s.index === currentIndex ? 'active' : ''}`;
                item.onclick = () => loadSentence(s.index);
                
                const indexSpan = document.createElement('span');
                indexSpan.className = 'sentence-index';
                indexSpan.textContent = `#${s.index + 1}`;
                
                const snippetSpan = document.createElement('span');
                snippetSpan.className = 'sentence-snippet';
                snippetSpan.textContent = s.text_snippet;
                
                const dot = document.createElement('div');
                dot.className = 'status-dot';
                
                item.appendChild(indexSpan);
                item.appendChild(snippetSpan);
                item.appendChild(dot);
                listContainer.appendChild(item);
            });

            // Update Progress bar
            const total = sentencesData.length;
            const pct = total > 0 ? (doneCount / total) * 100 : 0;
            document.getElementById('progress-bar').style.width = `${pct}%`;
            document.getElementById('progress-text').textContent = `${doneCount} / ${total}`;
        }

        function loadSentence(index) {
            currentIndex = index;
            
            // Mark active sidebar
            document.querySelectorAll('.sentence-item').forEach(item => {
                item.classList.remove('active');
            });
            const activeItem = document.querySelector(`.sentence-item:nth-child(${index + 1})`);
            if (activeItem) {
                activeItem.classList.add('active');
                activeItem.scrollIntoView({ block: 'nearest' });
            }

            fetch(`/api/sentence?file=${encodeURIComponent(currentFile)}&index=${index}`)
                .then(r => r.json())
                .then(res => {
                    if (res.error) {
                        alert("Lỗi load sentence: " + res.error);
                        return;
                    }
                    const s = res.sentence;
                    currentSentenceText = s.sentence_text;
                    
                    document.getElementById('thread-title').textContent = s.thread_title || "No Title";
                    document.getElementById('parent-context').textContent = s.parent_context || "No Context";
                    document.getElementById('sample-meta').textContent = `${s.sample_id} | Index: ${s.sentence_index}`;
                    document.getElementById('notes-field').value = s.notes || "";

                    // Reset selection slots & buttons
                    clearAspect();
                    clearOpinion();
                    selectCategory("None");
                    selectSentiment("");

                    // Parse Quads JSON if exists
                    activeQuads = [];
                    if (s.human_quads_json) {
                        try {
                            activeQuads = JSON.parse(s.human_quads_json);
                        } catch(e) {
                            activeQuads = [];
                        }
                    }
                    
                    // Render Sentence & Table
                    renderSentenceDisplay();
                    renderQuadTable();
                });
        }

        // Highlight Aspect & Opinion matches inside the sentence
        function renderSentenceDisplay() {
            const container = document.getElementById('sentence-display');
            container.innerHTML = escapeHtml(currentSentenceText);
            
            // Render marks from active quads
            let html = escapeHtml(currentSentenceText);
            
            if (activeQuads.length > 0) {
                // Collect all terms to highlight
                let marks = [];
                activeQuads.forEach(q => {
                    if (q.aspect && q.aspect !== "None" && q.aspect !== "Multi") {
                        marks.push({ text: q.aspect, type: 'aspect' });
                    }
                    if (q.opinion && q.opinion !== "None" && q.opinion !== "Multi") {
                        marks.push({ text: q.opinion, type: 'opinion' });
                    }
                });
                
                // Add current selection slots to marks
                if (selectedAspectText) {
                    marks.push({ text: selectedAspectText, type: 'aspect' });
                }
                if (selectedOpinionText) {
                    marks.push({ text: selectedOpinionText, type: 'opinion' });
                }

                // Sort marks by length descending to prevent substring issues
                marks.sort((a, b) => b.text.length - a.text.length);

                // Safe replace using placeholders
                const placeholders = [];
                marks.forEach((m, idx) => {
                    const escaped = escapeRegExp(m.text);
                    const regex = new RegExp(`(${escaped})`, 'gi');
                    const placeholder = `__PH_${m.type.toUpperCase()}_${idx}__`;
                    
                    // We check if it can match before replacing
                    if (html.match(regex)) {
                        html = html.replace(regex, placeholder);
                        placeholders.push({
                            ph: placeholder,
                            text: m.text,
                            type: m.type
                        });
                    }
                });

                // Swap placeholders back to mark tags
                placeholders.forEach(p => {
                    const className = p.type === 'aspect' ? 'aspect-highlight' : 'opinion-highlight';
                    html = html.replace(new RegExp(p.ph, 'g'), `<mark class="${className}">${escapeHtml(p.text)}</mark>`);
                });
            } else {
                // Only highlight current aspect/opinion selection slots
                if (selectedAspectText && selectedOpinionText && selectedAspectText === selectedOpinionText) {
                    const escaped = escapeRegExp(selectedAspectText);
                    html = html.replace(new RegExp(`(${escaped})`, 'gi'), `<mark class="both-highlight">$1</mark>`);
                } else {
                    if (selectedAspectText) {
                        const escaped = escapeRegExp(selectedAspectText);
                        html = html.replace(new RegExp(`(${escaped})`, 'gi'), `<mark class="aspect-highlight">$1</mark>`);
                    }
                    if (selectedOpinionText) {
                        const escaped = escapeRegExp(selectedOpinionText);
                        html = html.replace(new RegExp(`(${escaped})`, 'gi'), `<mark class="opinion-highlight">$1</mark>`);
                    }
                }
            }

            container.innerHTML = html;
        }

        // Selection Handlers
        function handleSelection(e) {
            const selection = window.getSelection().toString().trim();
            const tooltip = document.getElementById('selection-tooltip');
            
            if (selection.length > 0) {
                lastSelectionText = selection;
                
                // Show floating tooltip above selection
                const range = window.getSelection().getRangeAt(0);
                const rect = range.getBoundingClientRect();
                
                tooltip.style.left = `${rect.left + window.scrollX + (rect.width/2) - 60}px`;
                tooltip.style.top = `${rect.top + window.scrollY - 40}px`;
                tooltip.style.display = 'flex';
            } else {
                tooltip.style.display = 'none';
            }
        }

        function setAspectFromSelection() {
            if (lastSelectionText) {
                setAspect(lastSelectionText);
                document.getElementById('selection-tooltip').style.display = 'none';
                window.getSelection().removeAllRanges();
            }
        }

        function setOpinionFromSelection() {
            if (lastSelectionText) {
                setOpinion(lastSelectionText);
                document.getElementById('selection-tooltip').style.display = 'none';
                window.getSelection().removeAllRanges();
            }
        }

        function setAspect(text) {
            selectedAspectText = text;
            const input = document.getElementById('input-aspect');
            if (input) input.value = text;
            renderSentenceDisplay();
        }

        function clearAspect() {
            selectedAspectText = "";
            const input = document.getElementById('input-aspect');
            if (input) input.value = "";
            renderSentenceDisplay();
        }

        function setOpinion(text) {
            selectedOpinionText = text;
            const input = document.getElementById('input-opinion');
            if (input) input.value = text;
            renderSentenceDisplay();
        }

        function clearOpinion() {
            selectedOpinionText = "";
            const input = document.getElementById('input-opinion');
            if (input) input.value = "";
            renderSentenceDisplay();
        }

        function updateAspectFromInput() {
            selectedAspectText = document.getElementById('input-aspect').value;
            renderSentenceDisplay();
        }

        function updateOpinionFromInput() {
            selectedOpinionText = document.getElementById('input-opinion').value;
            renderSentenceDisplay();
        }

        function selectCategory(cat) {
            selectedCategory = cat;
            document.querySelectorAll('#category-btn-grid .choice-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            const activeBtn = document.querySelector(`#category-btn-grid .choice-btn[data-cat="${cat}"]`);
            if (activeBtn) activeBtn.classList.add('active');
        }

        function selectSentiment(sent) {
            selectedSentiment = sent;
            document.querySelectorAll('#sentiment-btn-row .choice-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            const activeBtn = document.querySelector(`#sentiment-btn-row .choice-btn[data-sent="${sent}"]`);
            if (activeBtn) activeBtn.classList.add('active');
        }

        // Add a quadruplet to active list
        function addQuadruplet() {
            if (!selectedCategory) {
                alert("Vui lòng chọn Category!");
                return;
            }
            if (!selectedSentiment) {
                alert("Vui lòng chọn Sentiment!");
                return;
            }

            // Create quadruplet object
            const quad = {
                aspect: selectedAspectText || "None",
                opinion: selectedOpinionText || "None",
                category: selectedCategory,
                sentiment: selectedSentiment
            };

            activeQuads.push(quad);
            renderQuadTable();
            renderSentenceDisplay();

            // Clear Aspect and Opinion selection slots
            clearAspect();
            clearOpinion();
        }

        function deleteQuad(idx) {
            activeQuads.splice(idx, 1);
            renderQuadTable();
            renderSentenceDisplay();
        }

        function renderQuadTable() {
            const tbody = document.getElementById('quad-table-body');
            tbody.innerHTML = '';

            if (activeQuads.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Chưa có bộ quadruplet nào được thêm.</td></tr>';
                return;
            }

            activeQuads.forEach((q, idx) => {
                const tr = document.createElement('tr');
                
                const tdAspect = document.createElement('td');
                tdAspect.innerHTML = `<span class="badge badge-aspect">${escapeHtml(q.aspect)}</span>`;
                
                const tdOpinion = document.createElement('td');
                tdOpinion.innerHTML = `<span class="badge badge-opinion">${escapeHtml(q.opinion)}</span>`;
                
                const tdCategory = document.createElement('td');
                tdCategory.innerHTML = `<span class="badge badge-category">${escapeHtml(q.category)}</span>`;
                
                const tdSentiment = document.createElement('td');
                const sentClass = q.sentiment ? q.sentiment.toLowerCase() : 'neutral';
                tdSentiment.innerHTML = `<span class="badge badge-sentiment badge-${sentClass}">${escapeHtml(q.sentiment)}</span>`;
                
                const tdActions = document.createElement('td');
                tdActions.style.textAlign = 'center';
                tdActions.innerHTML = `<button class="btn btn-danger" style="padding: 2px 8px; font-size: 0.75rem;" onclick="deleteQuad(${idx})">Delete</button>`;

                tr.appendChild(tdAspect);
                tr.appendChild(tdOpinion);
                tr.appendChild(tdCategory);
                tr.appendChild(tdSentiment);
                tr.appendChild(tdActions);
                tbody.appendChild(tr);
            });
        }

        // Navigation
        function navigateSentence(direction) {
            const nextIdx = currentIndex + direction;
            if (nextIdx >= 0 && nextIdx < sentencesData.length) {
                loadSentence(nextIdx);
            }
        }

        function goToUnfinished() {
            const unfinished = sentencesData.find(s => !s.done);
            if (unfinished) {
                loadSentence(unfinished.index);
            } else {
                showToast("Tất cả các câu trong file đã hoàn thành!");
            }
        }

        // Save & Next Action
        function saveAndNext() {
            // Check if there is an un-added aspect/opinion selection slot. Automatically add it if both are set!
            if ((selectedAspectText || selectedOpinionText) && selectedCategory && selectedSentiment) {
                addQuadruplet();
            }

            // Map activeQuads to schema columns
            let has_quad = "No";
            let aspect = "None";
            let opinion = "None";
            let category = "None";
            let sentiment = "None";
            
            if (activeQuads.length === 1) {
                has_quad = "Yes";
                aspect = activeQuads[0].aspect || "None";
                opinion = activeQuads[0].opinion || "None";
                category = activeQuads[0].category;
                sentiment = activeQuads[0].sentiment;
            } else if (activeQuads.length > 1) {
                has_quad = "Yes";
                aspect = "Multi";
                opinion = "Multi";
                
                // Get unique categories & sentiments
                const cats = [...new Set(activeQuads.map(q => q.category))];
                const sents = [...new Set(activeQuads.map(q => q.sentiment))];
                
                category = cats.length === 1 ? cats[0] : "Multi";
                sentiment = sents.length === 1 ? sents[0] : "Mixed";
            }

            const quadsJson = JSON.stringify(activeQuads);
            const notes = document.getElementById('notes-field').value;

            const values = {
                "human_has_quad": has_quad,
                "human_aspect": aspect,
                "human_opinion": opinion,
                "human_category_label": category,
                "human_sentiment_label": sentiment,
                "human_quads_json": quadsJson,
                "notes": notes
            };

            fetch('/api/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    file: currentFile,
                    index: currentIndex,
                    values: values
                })
            })
            .then(r => r.json())
            .then(res => {
                if (res.status === 'success') {
                    showToast("Đã lưu nhãn!");
                    // Update frontend state cache
                    sentencesData[currentIndex].done = true;
                    sentencesData[currentIndex].text_snippet = currentSentenceText.substring(0, 40) + "...";
                    renderSidebarProgress();

                    // Automatically move to next sentence
                    if (currentIndex < sentencesData.length - 1) {
                        loadSentence(currentIndex + 1);
                    } else {
                        showToast("Đã hoàn thành câu cuối cùng!");
                    }
                } else {
                    alert("Lỗi lưu nhãn: " + res.error);
                }
            });
        }

        // Quick "No Quad" Action
        function markNoQuad() {
            activeQuads = [];
            selectedAspectText = "";
            selectedOpinionText = "";
            selectedCategory = "None";
            selectedSentiment = "None";
            
            const quadsJson = "[]";
            const notes = document.getElementById('notes-field').value;

            const values = {
                "human_has_quad": "No",
                "human_aspect": "None",
                "human_opinion": "None",
                "human_category_label": "None",
                "human_sentiment_label": "None",
                "human_quads_json": quadsJson,
                "notes": notes
            };

            fetch('/api/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    file: currentFile,
                    index: currentIndex,
                    values: values
                })
            })
            .then(r => r.json())
            .then(res => {
                if (res.status === 'success') {
                    showToast("Đã ghi nhận No Quad!");
                    sentencesData[currentIndex].done = true;
                    renderSidebarProgress();
                    
                    if (currentIndex < sentencesData.length - 1) {
                        loadSentence(currentIndex + 1);
                    }
                }
            });
        }

        // Keyboard Shortcuts Setup
        function setupKeyboardShortcuts() {
            document.addEventListener('keydown', function(e) {
                const isInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
                const key = e.key.toLowerCase();

                if (isInput) {
                    if (key === 'enter') {
                        if (e.shiftKey) {
                            saveAndNext();
                        } else {
                            addQuadruplet();
                        }
                        e.preventDefault();
                    }
                    return;
                }

                const selection = window.getSelection().toString().trim();

                if (key === 'a') {
                    if (selection) {
                        setAspect(selection);
                        window.getSelection().removeAllRanges();
                    } else if (lastSelectionText) {
                        setAspect(lastSelectionText);
                    }
                    e.preventDefault();
                } 
                else if (key === 'o') {
                    if (selection) {
                        setOpinion(selection);
                        window.getSelection().removeAllRanges();
                    } else if (lastSelectionText) {
                        setOpinion(lastSelectionText);
                    }
                    e.preventDefault();
                } 
                else if (key === 'enter') {
                    if (e.shiftKey) {
                        saveAndNext();
                    } else {
                        addQuadruplet();
                    }
                    e.preventDefault();
                } 
                else if (key === 'n' || e.key === 'Escape') {
                    markNoQuad();
                    e.preventDefault();
                } 
                else if (e.key === 'ArrowLeft') {
                    navigateSentence(-1);
                    e.preventDefault();
                } 
                else if (e.key === 'ArrowRight') {
                    navigateSentence(1);
                    e.preventDefault();
                }
                else if (['1', '2', '3', '4'].includes(e.key)) {
                    selectSentiment(SENTIMENTS[parseInt(e.key) - 1]);
                    e.preventDefault();
                }
            });
        }

        function showToast(msg) {
            const toast = document.getElementById('toast-notif');
            toast.textContent = msg;
            toast.style.display = 'block';
            setTimeout(() => {
                toast.style.display = 'none';
            }, 2000);
        }

        // Helpers
        function escapeHtml(str) {
            if (!str) return "";
            return str
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function escapeRegExp(string) {
            return string.replace(/[.*+?^${}()|[\]\\+]/g, '\\$&');
        }
    </script>
</body>
</html>
"""

# =========================
# RUN MAIN SERVER
# =========================

def get_free_port():
    """Finds a free port on localhost dynamically."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run():
    # Find a free port
    port = get_free_port()
    server_address = ('localhost', port)
    
    httpd = HTTPServer(server_address, AnnotationHTTPHandler)
    url = f"http://localhost:{port}"
    
    print("=" * 60)
    print(f" ABSA Annotation Tool server successfully started!")
    print(f" URL: {url}")
    print(f" Labeled files directory: {SPLIT_DIR}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")
    
    # Auto-open browser tab
    webbrowser.open_new_tab(url)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    run()
