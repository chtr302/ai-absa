"""ONNX Runtime helpers configured for CPU-only 2-core inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


RUNTIME_NAME = "ONNX CPU 2-Core"


def create_onnx_session(model_path: str):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", "2"))
    sess_options.inter_op_num_threads = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def get_model_size_mb(model_path: str) -> float:
    return round(Path(model_path).stat().st_size / (1024 * 1024), 3)


def _parse_json_output(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    if not isinstance(value, str):
        return None

    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_quads(payload: dict[str, Any]) -> list[dict[str, str]]:
    quads = payload.get("quads") or []
    if not isinstance(quads, list):
        return []

    normalized = []
    for quad in quads:
        if not isinstance(quad, dict):
            continue
        normalized.append(
            {
                "aspect": str(quad.get("aspect") or ""),
                "opinion": str(quad.get("opinion") or ""),
                "category": str(quad.get("category") or ""),
                "sentiment": str(quad.get("sentiment") or ""),
            }
        )
    return normalized


def predict_one(session: Any, sentence: str) -> dict[str, Any]:
    inputs = session.get_inputs()

    # ── CASE 1: 4-Input KIBAC Advanced Model ──────────────────────────────────
    if len(inputs) == 4:
        from transformers import AutoTokenizer
        from collections import defaultdict
        
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        CATEGORIES = [
            'PERFORMANCE', 'INTELLIGENCE', 'RESOURCES',
            'BEHAVIOR', 'TECHNICAL', 'SOFTWARE', 'COMPARATIVE'
        ]
        tokenizer.add_special_tokens({'additional_special_tokens': [f'[{c}]' for c in CATEGORIES]})
        
        jargon_dict = {}
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        ontology_path = PROJECT_ROOT / "data" / "processed" / "ontology" / "domain_ontology.json"
        if ontology_path.exists():
            with open(ontology_path, 'r', encoding='utf-8') as f:
                jargon_dict = json.load(f).get('entries', {})
                
        class JargonOntologyBuilder:
            SENTIMENT_WEIGHT = {'Positive': 1.0, 'Negative': 1.0, 'Neutral': 0.5, None: 0.0}
            def __init__(self, jargon_dict: dict, tokenizer, max_multiword_tokens: int = 4):
                self.tokenizer = tokenizer
                self.max_span = max_multiword_tokens
                self._term_spans = {}
                for term, meta in jargon_dict.items():
                    sentiment = meta.get('sentiment')
                    weight = self.SENTIMENT_WEIGHT.get(sentiment, 0.5)
                    if weight == 0.0:
                        continue
                    token_ids = self.tokenizer.encode(term, add_special_tokens=False)
                    if token_ids:
                        self._term_spans[tuple(token_ids)] = weight

            def build(self, input_ids: np.ndarray) -> np.ndarray:
                B, L = input_ids.shape
                graph = np.zeros((B, L, L), dtype=np.float32)
                ids_list = input_ids.tolist()
                for b in range(B):
                    seq = ids_list[b]
                    matched_positions = []
                    for span_len in range(self.max_span, 0, -1):
                        for start in range(L - span_len + 1):
                            span = tuple(seq[start:start + span_len])
                            weight = self._term_spans.get(span)
                            if weight is not None:
                                end = start + span_len
                                overlaps = any(not (end <= ms or start >= me) for ms, me, _ in matched_positions)
                                if not overlaps:
                                    matched_positions.append((start, end, weight))
                    for start, end, weight in matched_positions:
                        for pos in range(start, end):
                            graph[b, pos, :] = weight
                return graph

        jargon_builder = JargonOntologyBuilder(jargon_dict, tokenizer)
        
        full_text = sentence + ' ' + ' '.join(f'[{c}]' for c in CATEGORIES)
        enc = tokenizer(
            full_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        input_ids = np.array([enc['input_ids']], dtype=np.int64)
        attention_mask = np.array([enc['attention_mask']], dtype=np.int64)
        L = input_ids.shape[1]
        
        enc_ids = enc['input_ids']
        cat_positions = {}
        for cat in CATEGORIES:
            tok_id = tokenizer.convert_tokens_to_ids(f'[{cat}]')
            if tok_id in enc_ids:
                cat_positions[cat] = enc_ids.index(tok_id)
                
        dist_matrix = np.zeros((1, L, L), dtype=np.int64)
        ontology_graph = jargon_builder.build(input_ids)
        
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dist_matrix": dist_matrix,
            "ontology_graph": ontology_graph
        }
        
        session_outputs = [o.name for o in session.get_outputs()]
        
        if "asp_boundary_logits" in session_outputs:
            ort_outs = session.run(["scores", "asp_boundary_logits", "opn_boundary_logits"], ort_inputs)
            scores = ort_outs[0][0]               # shape (L, L, 5)
            asp_boundary_logits = ort_outs[1][0]  # shape (L, 3)
            opn_boundary_logits = ort_outs[2][0]  # shape (L, 3)
            
            # argmax to get labels
            asp_preds = np.argmax(asp_boundary_logits, axis=-1).tolist()
            opn_preds = np.argmax(opn_boundary_logits, axis=-1).tolist()
            
            # Softmax to get probabilities (Tier A2)
            exp_asp = np.exp(asp_boundary_logits - np.max(asp_asp_boundary_logits if 'asp_asp_boundary_logits' in locals() else asp_boundary_logits, axis=-1, keepdims=True))
            # Wait, let's simplify the max subtraction
            exp_asp = np.exp(asp_boundary_logits - np.max(asp_boundary_logits, axis=-1, keepdims=True))
            asp_probs = exp_asp / np.sum(exp_asp, axis=-1, keepdims=True)
            
            exp_opn = np.exp(opn_boundary_logits - np.max(opn_boundary_logits, axis=-1, keepdims=True))
            opn_probs = exp_opn / np.sum(exp_opn, axis=-1, keepdims=True)
            
            first_cat_pos = min(cat_positions.values()) if cat_positions else L
            
            # 1. Decode Aspect Spans (BIO tagging)
            aspect_spans = []
            in_span = False
            start = -1
            MAX_ASP_SPAN_TOKENS = 8
            for i in range(1, first_cat_pos):
                tag = asp_preds[i]
                if tag == 1:  # B-ASP
                    if in_span:
                        aspect_spans.append((start, i - 1))
                    start = i
                    in_span = True
                elif tag == 2:  # I-ASP
                    if not in_span:
                        start = i
                        in_span = True
                    else:
                        # A1: Max Span Length Constraint
                        if i - start + 1 > MAX_ASP_SPAN_TOKENS:
                            aspect_spans.append((start, i - 1))
                            in_span = False
                            continue
                        # A2: Confidence-Based Split
                        p_continue = float(asp_probs[i, 2])
                        p_outside = float(asp_probs[i, 0])
                        if p_continue - p_outside < 0.15:
                            aspect_spans.append((start, i - 1))
                            in_span = False
                            continue
                else:  # O
                    if in_span:
                        aspect_spans.append((start, i - 1))
                        in_span = False
            if in_span:
                aspect_spans.append((start, first_cat_pos - 1))
                
            # 2. Decode Opinion Spans (BIO tagging)
            opinion_spans = []
            in_span = False
            start = -1
            MAX_OPN_SPAN_TOKENS = 12
            for i in range(1, first_cat_pos):
                tag = opn_preds[i]
                if tag == 1:  # B-OPN
                    if in_span:
                        opinion_spans.append((start, i - 1))
                    start = i
                    in_span = True
                elif tag == 2:  # I-OPN
                    if not in_span:
                        start = i
                        in_span = True
                    else:
                        # A1: Max Span Length Constraint
                        if i - start + 1 > MAX_OPN_SPAN_TOKENS:
                            opinion_spans.append((start, i - 1))
                            in_span = False
                            continue
                        # A2: Confidence-Based Split
                        p_continue = float(opn_probs[i, 2])
                        p_outside = float(opn_probs[i, 0])
                        if p_continue - p_outside < 0.15:
                            opinion_spans.append((start, i - 1))
                            in_span = False
                            continue
                else:  # O
                    if in_span:
                        opinion_spans.append((start, i - 1))
                        in_span = False
            if in_span:
                opinion_spans.append((start, first_cat_pos - 1))
                
            # C2: Mutual Exclusion Constraint - filter opinion spans that overlap with aspect spans
            filtered_opinion_spans = []
            for o_start, o_end in opinion_spans:
                overlaps = False
                for a_start, a_end in aspect_spans:
                    if not (o_end < a_start or o_start > a_end):
                        overlaps = True
                        break
                if not overlaps:
                    filtered_opinion_spans.append((o_start, o_end))
            opinion_spans = filtered_opinion_spans
                
            # Softmax relation scores
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            threshold = 0.35
            RELATION_LABELS_5 = ['ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU', 'ASP-CAT', 'NONE']

            REL_TO_SENTIMENT_5 = {
                'ASP-OPN-POS': 'Positive',
                'ASP-OPN-NEG': 'Negative',
                'ASP-OPN-NEU': 'Neutral',
            }
            pos_to_cat = {v: k for k, v in cat_positions.items()}
            
            seen, quads = set(), []
            for a_start, a_end in aspect_spans:
                for o_start, o_end in opinion_spans:
                    region_probs = probs[a_start:a_end+1, o_start:o_end+1, :3]
                    mean_prob = np.mean(region_probs, axis=(0, 1))
                    best_rel_idx = int(np.argmax(mean_prob))
                    best_p = float(mean_prob[best_rel_idx])
                    
                    if best_p >= threshold:
                        sentiment = REL_TO_SENTIMENT_5[RELATION_LABELS_5[best_rel_idx]]
                        
                        # Find Category for this aspect span using region mean pooling
                        best_cat = 'UNKNOWN'
                        best_cat_p = -1.0
                        cat_idx = RELATION_LABELS_5.index('ASP-CAT')
                        for cat_pos, cat in pos_to_cat.items():
                            if cat_pos < L:
                                cat_prob = float(np.mean(probs[a_start:a_end+1, cat_pos, cat_idx]))
                                if cat_prob > best_cat_p:
                                    best_cat_p = cat_prob
                                    best_cat = cat
                                    
                        key = (a_start, o_start, best_cat, sentiment)
                        if key not in seen:
                            seen.add(key)
                            
                            asp_chars_start = enc.token_to_chars(a_start)
                            asp_chars_end = enc.token_to_chars(a_end)
                            
                            opn_chars_start = enc.token_to_chars(o_start)
                            opn_chars_end = enc.token_to_chars(o_end)
                            
                            aspect_text = "None"
                            opinion_text = "None"
                            
                            if asp_chars_start and asp_chars_end:
                                start_char, end_char = asp_chars_start.start, asp_chars_end.end
                                if start_char < len(sentence):
                                    aspect_text = sentence[start_char:min(end_char, len(sentence))].strip()
                                    
                            if opn_chars_start and opn_chars_end:
                                start_char, end_char = opn_chars_start.start, opn_chars_end.end
                                if start_char < len(sentence):
                                    opinion_text = sentence[start_char:min(end_char, len(sentence))].strip()
                                    
                            quads.append({
                                "aspect": aspect_text if aspect_text else "None",
                                "opinion": opinion_text if opinion_text else "None",
                                "category": best_cat,
                                "sentiment": sentiment
                            })
                            
        else:
            # Fallback to old relation-only decoding logic
            ort_outs = session.run(["scores"], ort_inputs)
            scores = ort_outs[0][0]  # shape (L, L, 5)
            
            num_classes = scores.shape[-1]
            
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            preds = np.argmax(probs, axis=-1)
            conf = np.max(probs, axis=-1)
            pos_to_cat = {v: k for k, v in cat_positions.items()}
            threshold = 0.50
            
            quads = []
            
            if num_classes == 5:
                # ── 5-Class Decoding Logic (For old pretrained models) ────────────────────
                RELATION_LABELS_5 = ['ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU', 'ASP-CAT', 'NONE']
                REL_TO_SENTIMENT_5 = {
                    'ASP-OPN-POS': 'Positive',
                    'ASP-OPN-NEG': 'Negative',
                    'ASP-OPN-NEU': 'Neutral',
                }
                asp_cat_idx_5 = RELATION_LABELS_5.index('ASP-CAT')
                
                asp_opn = []
                for i in range(L):
                    for j in range(L):
                        r = preds[i, j]
                        if r in (0, 1, 2) and conf[i, j] >= threshold:
                            asp_opn.append((i, j, REL_TO_SENTIMENT_5[RELATION_LABELS_5[r]], conf[i, j]))
                            
                if asp_opn:
                    asp_cats = {}
                    for i in range(L):
                        best_cat = None
                        best_p = -1.0
                        for cat_pos, cat in pos_to_cat.items():
                            if cat_pos < L:
                                r = preds[i, cat_pos]
                                p = conf[i, cat_pos]
                                if r == asp_cat_idx_5 and p >= threshold:
                                    if p > best_p:
                                        best_p = p
                                        best_cat = cat
                        if best_cat is not None:
                            asp_cats[i] = best_cat
                            
                    seen = set()
                    for asp_tok, opn_tok, sentiment, _ in asp_opn:
                        cat = asp_cats.get(asp_tok) or 'UNKNOWN'
                        key = (asp_tok, opn_tok, cat, sentiment)
                        if key not in seen:
                            seen.add(key)
                            
                            asp_word = enc.token_to_word(asp_tok)
                            asp_chars = enc.word_to_chars(asp_word) if asp_word is not None else enc.token_to_chars(asp_tok)
                            
                            opn_word = enc.token_to_word(opn_tok)
                            opn_chars = enc.word_to_chars(opn_word) if opn_word is not None else enc.token_to_chars(opn_tok)
                            
                            aspect_text = "None"
                            opinion_text = "None"
                            
                            if asp_chars:
                                start, end = asp_chars.start, asp_chars.end
                                if start < len(sentence):
                                    aspect_text = sentence[start:min(end, len(sentence))].strip()
                                    
                            if opn_chars:
                                start, end = opn_chars.start, opn_chars.end
                                if start < len(sentence):
                                    opinion_text = sentence[start:min(end, len(sentence))].strip()
                                    
                            quads.append({
                                "aspect": aspect_text if aspect_text else "None",
                                "opinion": opinion_text if opinion_text else "None",
                                "category": cat,
                                "sentiment": sentiment
                            })
            else:
                # ── 7-Class GTS Decoding Logic (For old GTS models) ───────────────────────
                RELATION_LABELS_7 = ['A', 'O', 'ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU', 'ASP-CAT', 'NONE']
                REL_TO_SENTIMENT_7 = {
                    'ASP-OPN-POS': 'Positive',
                    'ASP-OPN-NEG': 'Negative',
                    'ASP-OPN-NEU': 'Neutral',
                }
                a_idx = RELATION_LABELS_7.index('A')
                o_idx = RELATION_LABELS_7.index('O')
                asp_cat_idx = RELATION_LABELS_7.index('ASP-CAT')
                
                rel_labels = ['ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU']
                rel_indices = [RELATION_LABELS_7.index(r) for r in rel_labels]
                rel_to_sentiment = {RELATION_LABELS_7.index(r): REL_TO_SENTIMENT_7[r] for r in rel_labels}
                
                # 1. Decode Aspect Spans from the diagonal
                aspect_spans = []
                in_aspect = False
                a_start = -1
                for i in range(L):
                    if preds[i, i] == a_idx and conf[i, i] >= threshold:
                        if not in_aspect:
                            a_start = i
                            in_aspect = True
                    else:
                        if in_aspect:
                            aspect_spans.append((a_start, i - 1))
                            in_aspect = False
                if in_aspect:
                    aspect_spans.append((a_start, L - 1))
    
                # 2. Decode Opinion Spans from the diagonal
                opinion_spans = []
                in_opinion = False
                o_start = -1
                for j in range(L):
                    if preds[j, j] == o_idx and conf[j, j] >= threshold:
                        if not in_opinion:
                            o_start = j
                            in_opinion = True
                    else:
                        if in_opinion:
                            opinion_spans.append((o_start, j - 1))
                            in_opinion = False
                if in_opinion:
                    opinion_spans.append((o_start, L - 1))
                    
                seen = set()
                for a_start, a_end in aspect_spans:
                    for o_start, o_end in opinion_spans:
                        best_rel = None
                        best_p = -1.0
                        for i in range(a_start, a_end + 1):
                            for j in range(o_start, o_end + 1):
                                r = preds[i, j]
                                p = conf[i, j]
                                if r in rel_indices and p >= threshold:
                                    if p > best_p:
                                        best_p = p
                                        best_rel = r
                                        
                        if best_rel is not None:
                            sentiment = rel_to_sentiment[best_rel]
                            
                            # 4. Find Category for this Aspect span
                            best_cat = 'UNKNOWN'
                            best_cat_p = -1.0
                            for cat_pos, cat in pos_to_cat.items():
                                if cat_pos < L:
                                    for i in range(a_start, a_end + 1):
                                        r = preds[i, cat_pos]
                                        p = conf[i, cat_pos]
                                        if r == asp_cat_idx and p >= threshold:
                                            if p > best_cat_p:
                                                best_cat_p = p
                                                best_cat = cat
                                                
                            key = (a_start, o_start, best_cat, sentiment)
                            if key not in seen:
                                seen.add(key)
                                
                                asp_chars_start = enc.token_to_chars(a_start)
                                asp_chars_end = enc.token_to_chars(a_end)
                                
                                opn_chars_start = enc.token_to_chars(o_start)
                                opn_chars_end = enc.token_to_chars(o_end)
                                
                                aspect_text = "None"
                                opinion_text = "None"
                                
                                if asp_chars_start and asp_chars_end:
                                    start, end = asp_chars_start.start, asp_chars_end.end
                                    if start < len(sentence):
                                        aspect_text = sentence[start:min(end, len(sentence))].strip()
                                        
                                if opn_chars_start and opn_chars_end:
                                    start, end = opn_chars_start.start, opn_chars_end.end
                                    if start < len(sentence):
                                        opinion_text = sentence[start:min(end, len(sentence))].strip()
                                        
                                quads.append({
                                    "aspect": aspect_text if aspect_text else "None",
                                    "opinion": opinion_text if opinion_text else "None",
                                    "category": best_cat,
                                    "sentiment": sentiment
                                })
                                
        return {
            "sentence": sentence,
            "quads": quads
        }

def predict_ensemble(session_v4: Any, session_newest: Any, sentence: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from collections import defaultdict
    import numpy as np

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    CATEGORIES = [
        'PERFORMANCE', 'INTELLIGENCE', 'RESOURCES',
        'BEHAVIOR', 'TECHNICAL', 'SOFTWARE', 'COMPARATIVE'
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[{c}]' for c in CATEGORIES]})

    STOP_WORDS = {
        'is', 'was', 'were', 'be', 'been', 'being', 'the', 'a', 'an', 'just', 'it', 'its', 'their',
        'to', 'and', 'but', 'of', 'in', 'on', 'at', 'for', 'with', 'about', 'by', 'this', 'that', 'these', 'those'
    }

    # Load Jargon Ontology
    jargon_dict = {}
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ontology_path = PROJECT_ROOT / "data" / "processed" / "ontology" / "domain_ontology.json"
    if ontology_path.exists():
        with open(ontology_path, 'r', encoding='utf-8') as f:
            jargon_dict = json.load(f).get('entries', {})

    class JargonOntologyBuilder:
        SENTIMENT_WEIGHT = {'Positive': 1.0, 'Negative': 1.0, 'Neutral': 0.5, None: 0.0}
        def __init__(self, jargon_dict: dict, tokenizer, max_multiword_tokens: int = 4):
            self.tokenizer = tokenizer
            self.max_span = max_multiword_tokens
            self._term_spans = {}
            for term, meta in jargon_dict.items():
                sentiment = meta.get('sentiment')
                weight = self.SENTIMENT_WEIGHT.get(sentiment, 0.5)
                if weight == 0.0:
                    continue
                token_ids = self.tokenizer.encode(term, add_special_tokens=False)
                if token_ids:
                    self._term_spans[tuple(token_ids)] = weight

        def build(self, input_ids: np.ndarray) -> np.ndarray:
            B, L = input_ids.shape
            graph = np.zeros((B, L, L), dtype=np.float32)
            ids_list = input_ids.tolist()
            for b in range(B):
                seq = ids_list[b]
                matched_positions = []
                for span_len in range(self.max_span, 0, -1):
                    for start in range(L - span_len + 1):
                        span = tuple(seq[start:start + span_len])
                        weight = self._term_spans.get(span)
                        if weight is not None:
                            end = start + span_len
                            overlaps = any(not (end <= ms or start >= me) for ms, me, _ in matched_positions)
                            if not overlaps:
                                matched_positions.append((start, end, weight))
                for start, end, weight in matched_positions:
                    for pos in range(start, end):
                        graph[b, pos, :] = weight
            return graph

    jargon_builder = JargonOntologyBuilder(jargon_dict, tokenizer)

    full_text = sentence + ' ' + ' '.join(f'[{c}]' for c in CATEGORIES)
    enc = tokenizer(
        full_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True
    )

    input_ids = np.array([enc['input_ids']], dtype=np.int64)
    attention_mask = np.array([enc['attention_mask']], dtype=np.int64)
    L = input_ids.shape[1]

    enc_ids = enc['input_ids']
    word_ids = enc.word_ids(0)

    cat_positions = {}
    for cat in CATEGORIES:
        tok_id = tokenizer.convert_tokens_to_ids(f'[{cat}]')
        if tok_id in enc_ids:
            cat_positions[cat] = enc_ids.index(tok_id)

    dist_matrix = np.zeros((1, L, L), dtype=np.int64)
    ontology_graph = jargon_builder.build(input_ids)

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dist_matrix": dist_matrix,
        "ontology_graph": ontology_graph
    }

    # Run v4 ONNX
    ort_outs_v4 = session_v4.run(["scores", "asp_boundary_logits", "opn_boundary_logits"], ort_inputs)
    scores_v4 = ort_outs_v4[0][0]
    asp_boundary_logits_v4 = ort_outs_v4[1][0]
    opn_boundary_logits_v4 = ort_outs_v4[2][0]

    # Run Newest ONNX
    ort_outs_new = session_newest.run(["scores", "asp_boundary_logits", "opn_boundary_logits"], ort_inputs)
    scores_new = ort_outs_new[0][0]
    asp_boundary_logits_new = ort_outs_new[1][0]
    opn_boundary_logits_new = ort_outs_new[2][0]

    first_cat_pos = min(cat_positions.values()) if cat_positions else L

    def get_word_level_spans_numpy(asp_boundary_logits, opn_boundary_logits):
        word_id_to_tokens = {}
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None and token_idx < first_cat_pos:
                if w_id not in word_id_to_tokens:
                    word_id_to_tokens[w_id] = []
                word_id_to_tokens[w_id].append(token_idx)

        sorted_word_ids = sorted(word_id_to_tokens.keys())
        num_words = len(sorted_word_ids)

        if num_words == 0:
            return [], []

        word_asp_logits = []
        word_opn_logits = []
        for w_id in sorted_word_ids:
            tok_indices = word_id_to_tokens[w_id]
            word_asp_logits.append(np.mean(asp_boundary_logits[tok_indices], axis=0))
            word_opn_logits.append(np.mean(opn_boundary_logits[tok_indices], axis=0))

        word_asp_logits = np.stack(word_asp_logits)
        word_opn_logits = np.stack(word_opn_logits)

        word_asp_preds = np.argmax(word_asp_logits, axis=-1).tolist()
        word_opn_preds = np.argmax(word_opn_logits, axis=-1).tolist()

        exp_asp = np.exp(word_asp_logits - np.max(word_asp_logits, axis=-1, keepdims=True))
        word_asp_probs = exp_asp / np.sum(exp_asp, axis=-1, keepdims=True)

        exp_opn = np.exp(word_opn_logits - np.max(word_opn_logits, axis=-1, keepdims=True))
        word_opn_probs = exp_opn / np.sum(exp_opn, axis=-1, keepdims=True)

        # Decode Aspects
        word_aspect_spans, in_span, start_w = [], False, -1
        MAX_ASP_WORDS = 6
        for i in range(num_words):
            tag = word_asp_preds[i]
            if tag == 1:
                if in_span: word_aspect_spans.append((start_w, i - 1))
                start_w, in_span = i, True
            elif tag == 2:
                if not in_span:
                    start_w, in_span = i, True
                else:
                    if i - start_w + 1 > MAX_ASP_WORDS:
                        word_aspect_spans.append((start_w, i - 1)); in_span = False; continue
                    p_cont = float(word_asp_probs[i, 2])
                    p_out  = float(word_asp_probs[i, 0])
                    if p_cont - p_out < 0.15:
                        word_aspect_spans.append((start_w, i - 1)); in_span = False; continue
            else:
                if in_span: word_aspect_spans.append((start_w, i - 1)); in_span = False
        if in_span: word_aspect_spans.append((start_w, num_words - 1))

        # Decode Opinions
        word_opinion_spans, in_span, start_w = [], False, -1
        MAX_OPN_WORDS = 8
        for i in range(num_words):
            tag = word_opn_preds[i]
            if tag == 1:
                if in_span: word_opinion_spans.append((start_w, i - 1))
                start_w, in_span = i, True
            elif tag == 2:
                if not in_span:
                    start_w, in_span = i, True
                else:
                    if i - start_w + 1 > MAX_OPN_WORDS:
                        word_opinion_spans.append((start_w, i - 1)); in_span = False; continue
                    p_cont = float(word_opn_probs[i, 2])
                    p_out  = float(word_opn_probs[i, 0])
                    if p_cont - p_out < 0.15:
                        word_opinion_spans.append((start_w, i - 1)); in_span = False; continue
            else:
                if in_span: word_opinion_spans.append((start_w, i - 1)); in_span = False
        if in_span: word_opinion_spans.append((start_w, num_words - 1))

        aspect_spans = []
        for w_s, w_e in word_aspect_spans:
            t_s = word_id_to_tokens[sorted_word_ids[w_s]][0]
            t_e = word_id_to_tokens[sorted_word_ids[w_e]][-1]
            aspect_spans.append((t_s, t_e))

        opinion_spans = []
        for w_s, w_e in word_opinion_spans:
            t_s = word_id_to_tokens[sorted_word_ids[w_s]][0]
            t_e = word_id_to_tokens[sorted_word_ids[w_e]][-1]
            word_tokens = enc_ids[t_s:t_e+1]
            word_text = tokenizer.decode(word_tokens, skip_special_tokens=True).strip().lower()
            if word_text in STOP_WORDS:
                continue
            opinion_spans.append((t_s, t_e))

        return aspect_spans, opinion_spans

    asp_spans_v4, opn_spans_v4 = get_word_level_spans_numpy(asp_boundary_logits_v4, opn_boundary_logits_v4)
    asp_spans_new, opn_spans_new = get_word_level_spans_numpy(asp_boundary_logits_new, opn_boundary_logits_new)

    # Union of aspect & opinion spans
    raw_aspect_spans = list(set(asp_spans_v4 + asp_spans_new))
    raw_opinion_spans = list(set(opn_spans_v4 + opn_spans_new))

    # Non-Maximum Suppression (NMS) to remove overlapping/duplicated spans, keeping the longer span
    def filter_overlapping_spans(spans):
        sorted_spans = sorted(spans, key=lambda x: x[1] - x[0], reverse=True)
        keep = []
        for s, e in sorted_spans:
            overlap = False
            for ks, ke in keep:
                if not (e < ks or s > ke):
                    overlap = True
                    break
            if not overlap:
                keep.append((s, e))
        return keep

    aspect_spans = filter_overlapping_spans(raw_aspect_spans)
    opinion_spans = filter_overlapping_spans(raw_opinion_spans)

    # Mutual exclusion
    filtered_opn = []
    for o_s, o_e in opinion_spans:
        overlaps = any(not (o_e < a_s or o_s > a_e) for a_s, a_e in aspect_spans)
        if not overlaps: filtered_opn.append((o_s, o_e))
    opinion_spans = filtered_opn

    # Probability Ensemble
    exp_scores_v4 = np.exp(scores_v4 - np.max(scores_v4, axis=-1, keepdims=True))
    probs_v4 = exp_scores_v4 / np.sum(exp_scores_v4, axis=-1, keepdims=True)

    exp_scores_new = np.exp(scores_new - np.max(scores_new, axis=-1, keepdims=True))
    probs_new = exp_scores_new / np.sum(exp_scores_new, axis=-1, keepdims=True)

    # Soft voting: average relations, strongly bias v4 categories
    probs_rel = 0.5 * probs_v4 + 0.5 * probs_new
    probs_cat = 0.8 * probs_v4 + 0.2 * probs_new

    threshold = 0.35
    RELATION_LABELS_5 = ['ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU', 'ASP-CAT', 'NONE']
    REL_TO_SENTIMENT_5 = {
        'ASP-OPN-POS': 'Positive',
        'ASP-OPN-NEG': 'Negative',
        'ASP-OPN-NEU': 'Neutral',
    }
    pos_to_cat = {v: k for k, v in cat_positions.items()}
    seen, quads = set(), []

    for a_start, a_end in aspect_spans:
        for o_start, o_end in opinion_spans:
            region_probs = probs_rel[a_start:a_end+1, o_start:o_end+1, :3]
            mean_prob = np.mean(region_probs, axis=(0, 1))
            best_rel_idx = int(np.argmax(mean_prob))
            best_p = float(mean_prob[best_rel_idx])

            if best_p >= threshold:
                sentiment = REL_TO_SENTIMENT_5[RELATION_LABELS_5[best_rel_idx]]

                # Find Category
                best_cat = 'UNKNOWN'
                best_cat_p = -1.0
                cat_idx = RELATION_LABELS_5.index('ASP-CAT')
                for cat_pos, cat in pos_to_cat.items():
                    if cat_pos < L:
                        cat_prob = float(np.mean(probs_cat[a_start:a_end+1, cat_pos, cat_idx]))
                        if cat_prob > best_cat_p:
                            best_cat_p = cat_prob
                            best_cat = cat

                # Heuristic filter for newest-only aspects
                is_v4_aspect = any(a_start == av_s and a_end == av_e for av_s, av_e in asp_spans_v4)
                if not is_v4_aspect and best_p < 0.45:
                    continue

                key = (a_start, o_start, best_cat, sentiment)
                if key not in seen:
                    seen.add(key)

                    asp_chars_start = enc.token_to_chars(a_start)
                    asp_chars_end = enc.token_to_chars(a_end)
                    opn_chars_start = enc.token_to_chars(o_start)
                    opn_chars_end = enc.token_to_chars(o_end)

                    aspect_text = "None"
                    opinion_text = "None"

                    if asp_chars_start and asp_chars_end:
                        start_char, end_char = asp_chars_start.start, asp_chars_end.end
                        if start_char < len(sentence):
                            aspect_text = sentence[start_char:min(end_char, len(sentence))].strip()
                            
                    if opn_chars_start and opn_chars_end:
                        start_char, end_char = opn_chars_start.start, opn_chars_end.end
                        if start_char < len(sentence):
                            opinion_text = sentence[start_char:min(end_char, len(sentence))].strip()

                    if not aspect_text or aspect_text.lower() in STOP_WORDS:
                        continue

                    quads.append({
                        "aspect": aspect_text if aspect_text else "None",
                        "opinion": opinion_text if opinion_text else "None",
                        "category": best_cat,
                        "sentiment": sentiment
                    })

    return {
        "sentence": sentence,
        "quads": quads
    }

    # ── CASE 2: Single-String Input model (baseline/fallback) ─────────────────
    if len(inputs) != 1 or "string" not in inputs[0].type.lower():
        input_signature = [
            {"name": item.name, "type": item.type, "shape": item.shape}
            for item in inputs
        ]
        raise ValueError(
            "This runtime-only Docker app expects an ONNX model with one string input "
            f"that returns JSON containing a 'quads' list. Model inputs: {input_signature}"
        )

    input_name = inputs[0].name
    outputs = session.run(None, {input_name: np.array([sentence], dtype=object)})

    for output in outputs:
        payload = _parse_json_output(output)
        if payload is not None:
            return {
                "sentence": sentence,
                "quads": _normalize_quads(payload),
            }

    output_signature = [
        {"name": item.name, "type": item.type, "shape": item.shape}
        for item in session.get_outputs()
    ]
    raise ValueError(
        "Model ran, but no JSON output with a 'quads' list was found. "
        f"Model outputs: {output_signature}"
    )
