# AI-ABSA v2: Data Intelligence & Entity-Augmented Schema Specification

Hệ thống đặc tả dataset NLP cho ABSA + Tech Entity Mining (Model / Hardware / Performance-aware NLP)

---

## 1. Data Population Statistics (Thống kê dữ liệu)

- Raw Pool: 634,942 lines (~1.2GB JSONL)
- Filtered Dataset (v2): 116,857 comment groups

### Final Split:
- Train: 20,000 samples (training model)
- Validation: 1,000 samples (tuning)
- Test: 1,000 samples (evaluation)
- Pool: 94,857 samples (unlabeled mining)

- Retention Rate: ~18.4%

---

## 2. Standard Input Schema (Cấu trúc dữ liệu đầu vào)

Mỗi sample = 1 comment + context + sentence-level analysis

```json
{
  "id": "comment_id",
  "parent_id": "t1_ or t3_ reddit id",
  "parent_type": "comment | submission",
  "parent_context": "Nội dung parent comment hoặc submission",
  "thread_title": "Tiêu đề thread đã clean",
  "sentences": [
    {
      "text": "Câu đã được tách & clean",
      "is_anchor": true,
      "tech_score": 1.0,
      "entities": [
        {
          "type": "model | hardware",
          "value": "llama | qwen | gpu | vllm",
          "mention": "canonical normalized match"
        }
      ],
      "quads": []
    }
  ]
}
```


# 3. Entity System (Hệ thống thực thể)

Mục tiêu: trích xuất thực thể MODEL + HARDWARE từ văn bản

---

## 3.1 Entity Types

| Type     | Description              | Examples |
|----------|------------------------|----------|
| model    | LLM / AI models         | GPT, Llama, Qwen, Mistral, DeepSeek, Claude |
| hardware | Compute / runtime / infra | GPU, CPU, VRAM, CUDA, vLLM, llama.cpp, Ollama |

---

## 3.2 Model Canonical Mapping

Chuẩn hóa nhiều biến thể về 1 nhóm model chính

- gpt → chatgpt, gpt3, gpt4, gpt4o, gpt5  
- qwen → qwen2, qwen3, qwen35, qwen coder, qwq  
- llama → llama2, llama3, llama31, llama32, llama.cpp  
- mistral → mistral, mixtral, ministral  
- deepseek → deepseekr1, deepseekv3, deepseek coder  
- phi → phi3, phi4  
- glm → glm4  
- gemini → gemini  
- claude → claude  
- grok → grok3, grok4  
- codex → codex  
- openai → o1, o3  

---

## 3.3 Hardware Canonical Mapping

Chuẩn hóa phần cứng + infra + runtime

- gpu → rtx, gtx, 3090, 4090, 5090, a100, h100, titan, tesla  
- cpu → intel, amd, xeon, ryzen, epyc, m2, m3  
- memory → ram, vram, ddr4, ddr5, hbm, 64gb, 128gb  
- accelerator → cuda, rocm, vulkan, tpu, onnx, metal, opencl  
- framework → vllm, llama.cpp, ollama, transformers  

---

# 4. Sentence-Level Schema

Mỗi sentence là 1 unit phân tích độc lập

| Field | Type | Description |
|------|------|-------------|
| text | string | câu đã clean |
| is_anchor | boolean | câu quan trọng nhất trong context |
| tech_score | float (0–3) | độ đậm kỹ thuật |
| entities | list | model / hardware detection |
| quads | list | ABSA output |

---

# 5. Anchor Detection Logic

Mục tiêu: chọn câu mang nhiều thông tin kỹ thuật nhất

## Scoring rule:

- +1 → có tech keyword (GPU, LLM, CUDA, inference)
- +2 → có issue keyword (fix, error, problem, why, bug)

## Selection:

```python
anchor_sentence = argmax(sentence_score)
```

# 6. Full Pipeline Architecture (Kiến trúc xử lý dữ liệu)

## Luồng xử lý tổng thể

RAW JSONL  
→ Cleaning (làm sạch text: regex + normalize)  
→ Sentence Splitting (tách câu)  
→ Tech Filtering (lọc câu có tín hiệu AI/ML)  
→ Entity Extraction (trích xuất model + hardware)  
→ Anchor Detection (chọn câu quan trọng nhất)  
→ Sentence Structuring (chuẩn hóa schema sentence-level)  
→ Dataset Splitting (train / val / test / pool)

---

## Giải thích từng bước

- Cleaning: loại URL, ký tự rác, chuẩn hóa text
- Sentence Splitting: tách đoạn văn thành từng câu
- Tech Filtering: giữ câu có liên quan AI/LLM/hardware
- Entity Extraction: nhận diện model (Llama, GPT, Qwen...) và hardware (GPU, VRAM...)
- Anchor Detection: chọn câu mang thông tin kỹ thuật quan trọng nhất
- Sentence Structuring: chuẩn hóa JSON theo ABSA schema
- Dataset Splitting: chia dữ liệu phục vụ train/eval/inference

---

# 7. Dataset Splits (Phân chia dữ liệu)

## Cấu trúc dataset

| Dataset | Size | Purpose |
|----------|------|--------|
| train_v2.jsonl | 20,000 | Huấn luyện mô hình ABSA |
| val_v2.jsonl | 1,000 | Tinh chỉnh mô hình |
| test_v2.jsonl | 1,000 | Đánh giá chính thức (gold standard) |
| pool_v2.jsonl | 94,857 | Dữ liệu chưa gán nhãn (mining / inference) |

---

## Ý nghĩa các tập dữ liệu

- train: học pattern ABSA + entity AI/ML
- val: kiểm tra overfitting trong training
- test: đánh giá cuối cùng (không dùng train)
- pool: dữ liệu mở rộng cho future labeling / active learning

---

## Ghi chú thiết kế

- Ưu tiên train data để tối ưu learning signal
- Pool giữ lại phần lớn dữ liệu để:
  - mở rộng dataset sau này
  - semi-supervised learning
  - active learning pipeline