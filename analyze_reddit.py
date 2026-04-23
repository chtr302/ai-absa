import json
import re
import sys

def analyze():
    file_path = 'data/raw/reddit_elite_raw_candidates.jsonl'
    
    # Entity groups
    cloud_entities = ['gpt', 'chatgpt', 'claude', 'gemini']
    local_entities = ['llama', 'qwen', 'deepseek', 'mistral']
    hardware_entities = ['gpu', 'vram', 'rtx', 'mac', 'm1', 'm2', 'm3']
    
    # Comparison keywords
    comp_keywords = ['vs', 'than', 'better', 'faster']
    
    # Stats
    cloud_counts = {e: 0 for e in cloud_entities}
    local_counts = {e: 0 for e in local_entities}
    hardware_counts = {e: 0 for e in hardware_entities}
    
    comp_line_counts = {k: 0 for k in comp_keywords}
    
    word_lengths = []
    scores_gt_10 = 0
    scores_gt_50 = 0
    total_lines = 0
    
    # Regex for words (to handle case-insensitive and word boundaries)
    def get_pattern(words):
        return re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)

    cloud_pattern = get_pattern(cloud_entities)
    local_pattern = get_pattern(local_entities)
    hardware_pattern = get_pattern(hardware_entities)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    body = data.get('body', '')
                    score = data.get('score', 0)
                    total_lines += 1
                    
                    # Score stats
                    if score > 10:
                        scores_gt_10 += 1
                    if score > 50:
                        scores_gt_50 += 1
                    
                    # Body text processing
                    words = body.split()
                    word_count = len(words)
                    word_lengths.append(word_count)
                    
                    # Case-insensitive entity counting (count occurrences)
                    body_lower = body.lower()
                    
                    for e in cloud_entities:
                        cloud_counts[e] += len(re.findall(r'\b' + re.escape(e) + r'\b', body_lower))
                    
                    for e in local_entities:
                        local_counts[e] += len(re.findall(r'\b' + re.escape(e) + r'\b', body_lower))
                        
                    for e in hardware_entities:
                        # Special handling for m1, m2, m3 as they might be parts of words or common
                        hardware_counts[e] += len(re.findall(r'\b' + re.escape(e) + r'\b', body_lower))
                    
                    # Comparison keywords (count lines containing them)
                    for k in comp_keywords:
                        if re.search(r'\b' + re.escape(k) + r'\b', body_lower):
                            comp_line_counts[k] += 1
                            
                except json.JSONDecodeError:
                    continue
                
                if total_lines % 50000 == 0:
                    print(f"Processed {total_lines} lines...", file=sys.stderr)

        # Final calculations
        if not word_lengths:
            print("No data found.")
            return

        min_len = min(word_lengths)
        max_len = max(word_lengths)
        avg_len = sum(word_lengths) / len(word_lengths)
        
        # Report
        print(f"## BÁO CÁO PHÂN TÍCH DỮ LIỆU REDDIT ELITE")
        print(f"\n- **Tổng số dòng:** {total_lines:,}")
        
        print(f"\n### 1. Thống kê thực thể (Tổng số lần xuất hiện)")
        print(f"#### Nhóm Cloud")
        for e in cloud_entities:
            print(f"- {e.capitalize()}: {cloud_counts[e]:,}")
            
        print(f"\n#### Nhóm Local")
        for e in local_entities:
            print(f"- {e.capitalize()}: {local_counts[e]:,}")
            
        print(f"\n#### Nhóm Hardware")
        for e in hardware_entities:
            print(f"- {e.upper()}: {hardware_counts[e]:,}")
            
        print(f"\n### 2. Thống kê từ khóa so sánh (Số dòng chứa từ khóa)")
        for k in comp_keywords:
            print(f"- \"{k}\": {comp_line_counts[k]:,}")
            
        print(f"\n### 3. Thống kê độ dài văn bản (Số từ)")
        print(f"- Min: {min_len:,} từ")
        print(f"- Max: {max_len:,} từ")
        print(f"- Average: {avg_len:.2f} từ")
        
        print(f"\n### 4. Thống kê Score")
        print(f"- Score > 10: {scores_gt_10:,} ({scores_gt_10/total_lines*100:.2f}%)")
        print(f"- Score > 50: {scores_gt_50:,} ({scores_gt_50/total_lines*100:.2f}%)")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")

if __name__ == "__main__":
    analyze()
