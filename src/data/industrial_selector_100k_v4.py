import json
import re
import os
import math
import csv
from collections import Counter

# --- Configuration ---
INPUT_FILE = "data/raw/reddit_elite_raw_candidates.jsonl"
OUTPUT_JSONL = "data/processed/reddit_100k_elite_v4.jsonl"
OUTPUT_CSV = "data/processed/reddit_100k_elite_v4.csv"
TARGET_COUNT = 100000

# Quotas for Stratified Sampling
SENTIMENT_QUOTAS = {
    "positive": 35000,
    "negative": 35000,
    "neutral_compare": 30000
}

# Entity Groups
CLOUD_ENTITIES = ["gpt", "chatgpt", "claude", "gemini", "opus", "sonnet", "haiku"]
LOCAL_ENTITIES = ["llama", "qwen", "deepseek", "mistral", "grok", "gemma", "phi", "yi"]
HARDWARE_ENTITIES = ["rtx", "gpu", "vram", "3090", "4090", "mac", "m1", "m2", "m3", "m4", "nvidia", "amd", "intel", "cuda", "rocm"]

# Keywords for Bucketing
POSITIVE_KEYWORDS = ["amazing", "good", "great", "excellent", "fast", "impressive", "recommend", "best", "love", "smooth"]
NEGATIVE_KEYWORDS = ["bad", "terrible", "slow", "buggy", "worse", "fail", "hallucinate", "expensive", "dislike", "poor"]
COMPARE_KEYWORDS = ["vs", "than", "better", "faster", "outperform", "superior", "beat", "compare"]

# Patterns
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

def get_sentiment_bucket(sent_lower):
    """Simple keyword-based bucketing for initial stratification."""
    has_pos = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in POSITIVE_KEYWORDS)
    has_neg = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in NEGATIVE_KEYWORDS)
    has_comp = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in COMPARE_KEYWORDS)
    
    if has_comp: return "neutral_compare"
    if has_neg: return "negative"
    if has_pos: return "positive"
    return "neutral_compare" # Default bucket

def get_entity_bucket(sent_lower):
    """Identify which entity group the sentence belongs to."""
    has_cloud = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in CLOUD_ENTITIES)
    has_local = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in LOCAL_ENTITIES)
    has_hw = any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in HARDWARE_ENTITIES)
    
    if has_cloud: return "cloud"
    if has_local: return "local"
    if has_hw: return "hardware"
    return "general"

def calculate_density_score(sentence_text, reddit_score, sent_lower):
    """Calculates the ABSA Density Score."""
    words = sent_lower.split()
    word_count = len(words)
    
    # 1. Base Score (Log scale of Reddit upvotes)
    # Using log to dampen the effect of extremely viral comments
    score_factor = math.log10(max(1, reddit_score + 2))
    
    # 2. Content Bonus
    content_bonus = 1.0
    
    # Entity Density Bonus
    entities_found = sum(1 for e in (CLOUD_ENTITIES + LOCAL_ENTITIES + HARDWARE_ENTITIES) 
                         if re.search(rf"\b{re.escape(e)}\b", sent_lower))
    content_bonus += min(entities_found * 0.2, 1.0) # Cap entity bonus at 1.0
    
    # Comparison Bonus
    if any(re.search(rf"\b{re.escape(k)}\b", sent_lower) for k in COMPARE_KEYWORDS):
        content_bonus += 0.5
        
    # 3. Length Strategy (15-40 words is the 'Goldilocks' zone for ABSA)
    length_multiplier = 1.0
    if 15 <= word_count <= 40:
        length_multiplier = 1.2
    elif 10 <= word_count <= 60:
        length_multiplier = 1.1
    else:
        length_multiplier = 0.8 # Penalize too short or too long
        
    return score_factor * content_bonus * length_multiplier

def main():
    print(f"🚀 Initializing Industrial Elite Selector V4...")
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    # Buckets to store candidates
    buckets = {
        "positive": [],
        "negative": [],
        "neutral_compare": []
    }
    
    seen_hashes = set()
    
    print(f"📖 Scanning {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line)
            except: continue
            
            comment_id = data.get("id", "N/A")
            reddit_score = data.get("score", 0)
            body = data.get("body", "")
            if not body or body in ['[deleted]', '[removed]']: continue
            
            # Cleaning
            body = re.sub(r'```.*?```', '', body, flags=re.DOTALL)
            body = re.sub(r'`[^`]*`', '', body)
            
            sentences = SENTENCE_SPLIT_REGEX.split(body)
            for sent in sentences:
                sent = sent.strip()
                words = sent.split()
                if not (8 <= len(words) <= 80): continue
                if URL_PATTERN.search(sent): continue
                
                sent_lower = sent.lower()
                
                # Deduplication
                sent_hash = hash(sent_lower)
                if sent_hash in seen_hashes: continue
                seen_hashes.add(sent_hash)
                
                # Check for entity presence (required for ABSA)
                ent_bucket = get_entity_bucket(sent_lower)
                if ent_bucket == "general": continue 
                
                # Bucket and Score
                sent_bucket = get_sentiment_bucket(sent_lower)
                q_score = calculate_density_score(sent, reddit_score, sent_lower)
                
                buckets[sent_bucket].append({
                    "id": comment_id,
                    "sentence": sent,
                    "sentiment_bucket": sent_bucket,
                    "entity_bucket": ent_bucket,
                    "quality_score": q_score,
                    "reddit_score": reddit_score
                })
            
            if (line_idx + 1) % 50000 == 0:
                print(f"   Processed {line_idx+1} lines... Found {sum(len(b) for b in buckets.values())} candidates.")

    # Selection with Quotas
    final_selection = []
    global_pool = []
    
    print(f"⚖️  Selecting top-tier sentences within quotas...")
    for b_name, b_data in buckets.items():
        # Sort each bucket by quality score descending
        b_data.sort(key=lambda x: x["quality_score"], reverse=True)
        
        quota = SENTIMENT_QUOTAS[b_name]
        selected = b_data[:quota]
        remainder = b_data[quota:]
        
        final_selection.extend(selected)
        global_pool.extend(remainder)
        print(f"   - {b_name}: Selected {len(selected)}/{quota}")

    # Fallback: fill remaining slots from global pool if any quota was not met
    remaining_needed = TARGET_COUNT - len(final_selection)
    if remaining_needed > 0:
        print(f"🔄 Filling {remaining_needed} remaining slots from Global Pool...")
        global_pool.sort(key=lambda x: x["quality_score"], reverse=True)
        final_selection.extend(global_pool[:remaining_needed])

    # Final Export
    print(f"💾 Saving results...")
    final_selection.sort(key=lambda x: x["quality_score"], reverse=True) # Final global sort
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_json, \
         open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f_csv:
        
        writer = csv.writer(f_csv)
        writer.writerow(["id", "sentence", "sentiment_bucket", "entity_bucket", "quality_score", "reddit_score"])
        
        for item in final_selection:
            f_json.write(json.dumps(item) + '\n')
            writer.writerow([item["id"], item["sentence"], item["sentiment_bucket"], item["entity_bucket"], f"{item['quality_score']:.4f}", item["reddit_score"]])

    print(f"\n✅ SUCCESS: Extracted {len(final_selection):,} elite sentences.")
    print(f"   Output 1: {OUTPUT_JSONL}")
    print(f"   Output 2: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
