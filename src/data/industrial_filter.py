import json
import re
import os
import spacy
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data/processed/filtering_data.jsonl")
SUBMISSIONS_FILE = os.path.join(BASE_DIR, "data/processed/new_data.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data/processed/model_focused_data.jsonl")

# --- Regex Data ---
MODEL_FAMILIES = [
    r"llama(?:[- ]?[1234](?:\.[123])?)?(?:[- ]?(?:scout|maverick|behemoth|guard|chat|instruct|vision))?",
    r"qwen(?:[- ]?[23](?:\.[56])?)?(?:[- ]?(?:coder|vl|plus|max|thinking|math|instruct))?",
    r"deepseek(?:[- ]?(?:v[234]|r1|coder|ocr|speciale|terminus|instruct))?",
    r"mistral(?:[- ]?(?:large|small|medium|nemo|instruct|creative))?",
    r"mixtral(?:[- ]?(?:8x7b|8x22b|instruct))?",
    r"ministral(?:[- ]?[3])?",
    r"gemma(?:[- ]?[1234])?(?:[- ]?(?:it|instruct|vision))?",
    r"gemini(?:[- ]?[123](?:\.[5])?)?(?:[- ]?(?:pro|flash|ultra|nano|preview))?",
    r"phi(?:[- ]?[1234])?(?:[- ]?(?:mini|small|medium|multimodal|it|instruct))?",
    r"gpt[- ]?[345](?:\.[1245])?(?:[- ]?(?:turbo|omni|o|mini|nano|pro|instant|oss))?",
    r"claude(?:[- ]?[234](?:\.[157])?)?(?:[- ]?(?:sonnet|opus|haiku|mythos))?",
    r"grok(?:[- ]?[1234])?(?:[- ]?(?:beta|mini|fast))?",
    r"hermes(?:[- ]?[23])?", r"nous(?:[- ]?(?:hermes|llama|mixtral))?",
    r"glm(?:[- ]?[45](?:\.[1567])?)?(?:[- ]?(?:air|ocr|instruct))?",
    r"yi(?:[- ]?(?:1\.5|large|lightning|coder))?",
    r"command[- ]?r(?:[- ]?plus)?",
    r"vicuna", r"wizardlm", r"falcon(?:[- ]?(?:h1|h1r|2|3))?", r"mpt", r"starcoder",
    r"nemotron(?:[- ]?(?:3|4|ultra|super|nano))?", r"jamba", r"dbrx", r"olmo(?:[- ]?[3])?",
    r"exaone", r"kimi(?:[- ]?k[23])?", r"minimax", r"mimo(?:[- ]?v[2])?",
    r"stable[- ]?lm", r"zephyr", r"biomistral", r"codestral", r"devstral", r"magistral"
]
PARAM_SIZES = [
    r"\b[0-9]+(?:\.[0-9]+)?[bB]\b", r"\b[0-9]+x[0-9]+[bB]\b", r"\b[aA][0-9]+[bB]\b",
    r"\bgguf\b", r"\bexl2\b", r"\bawq\b", r"\bgptq\b", r"\bfp8\b", r"\bfp16\b"
]

# Compile patterns for performance
COMBINED_PATTERN = re.compile(rf"(?i)({'|'.join(MODEL_FAMILIES + PARAM_SIZES)})")

# Load Spacy for sentence splitting
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    nlp.add_pipe("sentencizer")
except Exception:
    # Fallback to a simple sentencizer if the model is not available
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

def extract_models(text):
    """Extracts model entities from text using regex."""
    if not text:
        return []
    matches = COMBINED_PATTERN.findall(text)
    # Deduplicate and normalize (lowercase)
    return sorted(list(set(m.strip().lower() for m in matches if m.strip())))

def get_thread_title_from_permalink(permalink):
    """Extracts a readable title from the Reddit permalink slug if needed."""
    try:
        parts = permalink.split('/')
        if len(parts) >= 6:
            slug = parts[5]
            title = slug.replace('_', ' ').replace('-', ' ').capitalize()
            return title
    except Exception:
        pass
    return "Unknown Thread"

def main():
    print(f"Loading submission titles from {SUBMISSIONS_FILE}...")
    thread_map = {}
    if os.path.exists(SUBMISSIONS_FILE):
        with open(SUBMISSIONS_FILE, 'r', encoding='utf-8') as f:
            try:
                submissions = json.load(f)
                for sub in submissions:
                    thread_map[sub['id']] = sub['title']
            except Exception as e:
                print(f"Warning: Could not parse submissions file: {e}")

    # Pass 1: Build comment lookup to resolve parent_context
    print(f"Pass 1: Building comment lookup from {INPUT_FILE}...")
    comment_lookup = {}
    input_line_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            input_line_count += 1
            try:
                data = json.loads(line)
                comment_lookup[data['id']] = data['body']
            except Exception:
                continue
    
    # Pass 2: Process and Filter
    print(f"Pass 2: Filtering and transforming data (95k target)...")
    retained_comments = 0
    total_sentences_kept = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=input_line_count):
            try:
                data = json.loads(line)
            except Exception:
                continue
            
            comment_id = data['id']
            parent_id = data['parent_id']
            body = data['body']
            permalink = data.get('permalink', '')
            
            # Resolve thread title
            thread_id = ""
            if '/comments/' in permalink:
                # Format: /r/Subreddit/comments/thread_id/slug/comment_id/
                parts = permalink.split('/comments/')
                if len(parts) > 1:
                    thread_id = parts[1].split('/')[0]
            
            thread_title = thread_map.get(thread_id, get_thread_title_from_permalink(permalink))
            
            # Resolve parent context
            parent_body = ""
            clean_parent_id = parent_id.split('_')[-1] if '_' in parent_id else parent_id
            if clean_parent_id in comment_lookup:
                parent_body = comment_lookup[clean_parent_id]
            
            # Identify models in context (Implicit Aspect sources)
            context_models = list(set(extract_models(thread_title) + extract_models(parent_body)))
            
            # Process sentences
            doc = nlp(body)
            comment_sentences = []
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                
                # Detect models in sentence
                sent_models = extract_models(sent_text)
                
                # Sentence is "model-focused" if it has its own models OR if context has models
                if sent_models or context_models:
                    # Union of models for detected_models list
                    all_detected = sorted(list(set(sent_models + context_models)))
                    
                    sent_obj = {
                        "text": sent_text,
                        "detected_models": all_detected,
                        "is_explicit": len(sent_models) > 0
                    }
                    comment_sentences.append(sent_obj)
                    total_sentences_kept += 1
            
            # If the comment has any focused sentences, save it
            if comment_sentences:
                output_obj = {
                    "id": comment_id,
                    "parent_context": parent_body,
                    "thread_title": thread_title,
                    "sentences": comment_sentences
                }
                f_out.write(json.dumps(output_obj) + '\n')
                retained_comments += 1

    print(f"\n--- FILTERING SUMMARY REPORT ---")
    print(f"Total Input Lines:   {input_line_count:,}")
    print(f"Retained Comments:   {retained_comments:,}")
    print(f"Total Sentences Kept: {total_sentences_kept:,}")
    print(f"Output saved to:     {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
