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
        
        ort_outs = session.run(["scores"], ort_inputs)
        scores = ort_outs[0][0]  # shape (L, L, 5)
        
        RELATION_LABELS = ['ASP-OPN-POS', 'ASP-OPN-NEG', 'ASP-OPN-NEU', 'ASP-CAT', 'NONE']
        REL_TO_SENTIMENT = {
            'ASP-OPN-POS': 'Positive',
            'ASP-OPN-NEG': 'Negative',
            'ASP-OPN-NEU': 'Neutral',
        }
        ASP_CAT_IDX = RELATION_LABELS.index('ASP-CAT')
        
        threshold = 0.50
        
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        pos_to_cat = {v: k for k, v in cat_positions.items()}
        
        asp_opn = []
        for i in range(L):
            for j in range(L):
                r = preds[i, j]
                if r in (0, 1, 2) and conf[i, j] >= threshold:
                    asp_opn.append((i, j, REL_TO_SENTIMENT[RELATION_LABELS[r]], conf[i, j]))
                    
        quads = []
        if asp_opn:
            asp_cats = {}
            for i in range(L):
                best_cat = None
                best_p = -1.0
                for cat_pos, cat in pos_to_cat.items():
                    if cat_pos < L:
                        r = preds[i, cat_pos]
                        p = conf[i, cat_pos]
                        if r == ASP_CAT_IDX and p >= threshold:
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
