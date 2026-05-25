# AI-ABSA: Data Intelligence & Schema Specification

Báo cáo về tài liệu

---

## 1. Data Population Statistics (Thống kê quy mô)
- **Raw Pool:** 634,942 lines (1.2 GB JSONL).
- **Elite Filtered (v5):** **175,988 Comment Groups**.
- **Retention Rate:** 27.72% (Đã khử nhiễu triệt để).

---

## 2. Standard Input Schema (Đặc tả dữ liệu đầu vào)

```json
{
  "id": "comment_id_string",
  "parent_id": "fullname_string (t1_ hoặc t3_)",
  "parent_type": "enum ['comment', 'submission']",
  "parent_context": "Text: Nội dung của câu cha hoặc tiêu đề thread",
  "thread_title": "Text: Tiêu đề bài viết đã được làm sạch",
  "sentences": [
    {
      "text": "Nội dung câu văn cụ thể",
      "is_anchor": "boolean (Tự thân có thực thể hay không)",
      "tech_score": "float (Độ đậm đặc kỹ thuật 0-10)",
      "quads": []
    }
  ]
}
```

---

## 3. Standard Output Schema (Đặc tả nhãn bộ bốn - Quadruplet)
Mục tiêu của mô hình KIBAC 3.0 là điền vào trường `quads` bộ 4 thông tin sau:

| Thành phần       | Tên biến    | Kiểu dữ liệu    | Ví dụ                               |
| :--------------- | :---------- | :-------------- | :---------------------------------- |
| **Aspect Term**  | `aspect`    | String          | "Llama 3", "VRAM", "quantization"   |
| **Category**     | `category`  | Enum (7 layers) | "PERFORMANCE", "INTELLIGENCE", etc. |
| **Opinion Term** | `opinion`   | String          | "blazing fast", "memory hog"        |
| **Sentiment**    | `sentiment` | Enum            | "Positive", "Negative", "Neutral"   |

---

## 4. The Elite Seven Categories (Hệ thống mỏ neo)
Gợi ý 7 Aspect Categories sau. Có thể nghiên cứu thêm
1. **PERFORMANCE**: Tốc độ, độ trễ, throughput.
2. **INTELLIGENCE**: Logic, reasoning, coding, accuracy.
3. **RESOURCES**: VRAM, phần cứng, mức độ tiêu thụ điện.
4. **BEHAVIOR**: Censorship, cá tính, sự tuân thủ hướng dẫn.
5. **TECHNICAL**: Định dạng tệp (GGUF), kiến trúc (MoE), kỹ thuật nén.
6. **SOFTWARE**: Công cụ hỗ trợ (Ollama, llama.cpp, UI).
7. **COMPARATIVE**: Các khía cạnh so sánh tổng quát/giá trị kinh tế.

---

## 5. Dataset Splits (Phân chia tệp tin)
- `/data/processed/balanced_30k.jsonl`: 30,000 dòng dữ liệu cho train/val/test (Sẽ có những dòng model không thể gán nhãn, nó là 1 phần của dataset).
- `/data/processed/denoised_data.jsonl`: ~115,000 dòng dữ liệu xuất hiện các ký tự liên quan đến model AI, sẽ xuất hiện rác bởi vì đây là lọc vật lý.

## 6. Explan Data (Các cột trong dữ liệu)
```json
{
  "id": "comment_id_string",
  "parent_context": "Text: Nội dung của câu cha hoặc tiêu đề thread",
  "thread_title": "Text: Tiêu đề bài viết đã được làm sạch",
  "sentences": [
    {
      "text": "Nội dung câu văn cụ thể",
      "is_anchor": "boolean (Tự thân có thực thể hay không)",
      "tech_score": "float (Độ đậm đặc kỹ thuật 0-10)",
      "quads": []
    }
  ],
  "detected_models": "Model xuất hiện (vật lý) trong câu",
  "is_explicit": "Câu ngữ cảnh ẩn hay không",
  "cleaned_models": "Kiểm tra vật lý đối với các loại data rác như Ollama bị nhận diện thành model Llama*"
}
```