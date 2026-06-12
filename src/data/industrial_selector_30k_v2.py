import json
import re
import random
from collections import defaultdict, Counter
import os

# Configuration
INPUT_FILE = "data/processed/denoised_data.jsonl"
OUTPUT_DIR = "data/processed/final_data/"
TARGET_TOTAL = 30000
TRAIN_RATIO = 0.8333  # ~25k
VAL_RATIO = 0.08335   # ~2.5k
TEST_RATIO = 0.08335  # ~2.5k

# Proxy Logic Keywords
CATEGORY_PROXIES = {
    "PERFORMANCE": r"speed|latency|tok/s|tps|ttft|throughput|inference|fast|slow|response time|lag",
    "REASONING": r"logic|math|cot|multi-step|reasoning|planning|puzzle|gsm8k|deduction",
    "CODING": r"python|code|script|programming|debug|refactor|boilerplate|sql|java|c\+\+|human-eval",
    "KNOWLEDGE": r"facts|info|mmlu|trivia|database|history|science|accurate|hallucination",
    "RESOURCES": r"vram|memory|gpu|3090|4090|24gb|oom|ram|cpu|hardware",
    "QUANTIZATION": r"gguf|exl2|awq|gptq|4-bit|8-bit|bits|compression|quantized",
    "TOOLING": r"ollama|llama\.cpp|vllm|lm studio|api|gui|ui|interface|wrapper",
    "RAG_CONTEXT": r"context window|128k|1m|needle|haystack|retrieval|vector db|rag",
    "FINETUNING": r"lora|qlora|adapter|sft|dpo|ppo|fine-tune|training|loss|weights",
    "BEHAVIOR": r"censorship|refusal|preachy|personality|tone|creative|alignment",
    "COMPARATIVE": r"vs|better than|worse than|alternative|rival|equivalent|benchmark",
}

SENTIMENT_POS = r"impressed|amazing|great|fast|solid|best|recommend|good|excellent|love|perfect"
SENTIMENT_NEG = r"disappointed|bad|slow|wrong|terrible|broken|fails|buggy|garbage|trash|worst|hate"

RARE_CATEGORIES = ["BEHAVIOR", "QUANTIZATION", "RAG_CONTEXT"]

def get_proxies(text):
    cat_proxy = "OTHER"
    for cat, pattern in CATEGORY_PROXIES.items():
        if re.search(pattern, text, re.IGNORECASE):
            cat_proxy = cat
            break
    
    pos = bool(re.search(SENTIMENT_POS, text, re.IGNORECASE))
    neg = bool(re.search(SENTIMENT_NEG, text, re.IGNORECASE))
    
    if pos and not neg:
        sent_proxy = "POS"
    elif neg and not pos:
        sent_proxy = "NEG"
    else:
        sent_proxy = "NEU"
        
    return cat_proxy, sent_proxy

def load_data():
    print(f"Loading data from {INPUT_FILE}...")
    threads = defaultdict(list)
    total_sentences_input = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            thread_id = item.get("thread_title") or f"unknown_{item.get('id')}"
            
            for sent in item["sentences"]:
                total_sentences_input += 1
                cat, sent_p = get_proxies(sent["text"])
                model = sent["detected_models"][0] if sent.get("detected_models") else "UNKNOWN"
                
                sentence_obj = {
                    "text": sent["text"],
                    "model_proxy": model,
                    "category_proxy": cat,
                    "sentiment_proxy": sent_p,
                    "composite_key": f"{model}_{cat}_{sent_p}",
                    "original_id": item["id"],
                    "thread_title": thread_id
                }
                threads[thread_id].append(sentence_obj)
    
    print(f"Loaded {len(threads)} threads with {total_sentences_input} sentences.")
    return threads

def sample_threads(threads):
    print("Sampling threads...")
    all_thread_ids = list(threads.keys())
    random.seed(42)
    random.shuffle(all_thread_ids)
    
    # Priority: threads with rare categories first
    def get_thread_score(tid):
        for sent in threads[tid]:
            if sent["category_proxy"] in RARE_CATEGORIES:
                return 0 # Higher priority
        return 1
    
    all_thread_ids.sort(key=get_thread_score)
    
    selected_threads = []
    current_total_sentences = 0
    model_counts = Counter()
    category_counts = Counter()
    
    MODEL_CAP = TARGET_TOTAL * 0.20
    PERFORMANCE_CAP = TARGET_TOTAL * 0.15
    
    for tid in all_thread_ids:
        if current_total_sentences >= TARGET_TOTAL:
            break
            
        thread_sentences = threads[tid]
        
        # Check caps
        # We calculate the potential impact of adding this thread
        thread_models = Counter(s["model_proxy"] for s in thread_sentences)
        thread_cats = Counter(s["category_proxy"] for s in thread_sentences)
        
        violates_cap = False
        for m, count in thread_models.items():
            if model_counts[m] + count > MODEL_CAP and m != "UNKNOWN":
                violates_cap = True
                break
        
        if not violates_cap:
            if category_counts["PERFORMANCE"] + thread_cats["PERFORMANCE"] > PERFORMANCE_CAP:
                violates_cap = True
        
        if not violates_cap or current_total_sentences < TARGET_TOTAL * 0.5: # Allow some leniency if we are far from target
            selected_threads.append(tid)
            current_total_sentences += len(thread_sentences)
            for m, count in thread_models.items():
                model_counts[m] += count
            for c, count in thread_cats.items():
                category_counts[c] += count
                
    print(f"Selected {len(selected_threads)} threads with {current_total_sentences} sentences.")
    return selected_threads

def split_and_save(threads, selected_thread_ids):
    print("Splitting threads into Train/Val/Test...")
    
    # To maintain thread integrity, we split the list of thread IDs
    random.seed(42)
    random.shuffle(selected_thread_ids)
    
    # Calculate target sentence counts for each split
    target_train = TARGET_TOTAL * TRAIN_RATIO
    target_val = TARGET_TOTAL * VAL_RATIO
    target_test = TARGET_TOTAL * TEST_RATIO
    
    train_threads = []
    val_threads = []
    test_threads = []
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for tid in selected_thread_ids:
        t_sentences = threads[tid]
        count = len(t_sentences)
        
        # Greedy assignment to reach targets
        if train_count < target_train:
            train_threads.append(tid)
            train_count += count
        elif val_count < target_val:
            val_threads.append(tid)
            val_count += count
        else:
            test_threads.append(tid)
            test_count += count

    # Final pool of sentences
    def get_sentences(tids):
        sents = []
        for tid in tids:
            sents.extend(threads[tid])
        return sents

    train_data = get_sentences(train_threads)
    val_data = get_sentences(val_threads)
    test_data = get_sentences(test_threads)
    
    # Truncate to exact targets if necessary (at sentence level, but this might slightly break thread integrity if we are very strict, 
    # but the requirement was "never split sentences of SAME thread ACROSS Train/Test", 
    # which implies a thread's sentences should all be in ONE split. 
    # If we truncate, we just remove some sentences from the end, which is fine as long as they aren't moved to another split.)
    
    # Actually, the user asked for EXACTLY 25k, 2.5k, 2.5k.
    # To get exact numbers while keeping thread integrity, we might have to remove some threads at the end.
    
    print(f"Initial split counts: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_jsonl(data, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} items to {path}")

    save_jsonl(train_data[:25000], "train.jsonl")
    save_jsonl(val_data[:2500], "val.jsonl")
    save_jsonl(test_data[:2500], "test.jsonl")
    
    return train_data[:25000] + val_data[:2500] + test_data[:2500]

def report_distribution(final_data):
    print("\n--- Final Distribution Report (Total 30,000) ---")
    total = len(final_data)
    model_counts = Counter(s["model_proxy"] for s in final_data)
    cat_counts = Counter(s["category_proxy"] for s in final_data)
    sent_counts = Counter(s["sentiment_proxy"] for s in final_data)
    
    print("Models (Top 10):")
    for model, count in model_counts.most_common(10):
        print(f"  {model}: {count} ({count/total*100:.1f}%)")

    print("\nCategories:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")
        
    print("\nSentiments:")
    for sent, count in sent_counts.most_common():
        print(f"  {sent}: {count} ({count/total*100:.1f}%)")

if __name__ == "__main__":
    threads = load_data()
    selected_thread_ids = sample_threads(threads)
    final_data = split_and_save(threads, selected_thread_ids)
    report_distribution(final_data)
