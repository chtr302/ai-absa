# Rule-Based ASQP Baseline

Tài liệu này mô tả baseline được triển khai theo mục 2.2.1 trong báo cáo dự án.
Baseline không phải neural model và không cập nhật tham số bằng gradient.

## Output

```json
{
  "text": "Llama.cpp inference is fast, but memory use is not great.",
  "quads": [
    {
      "aspect": "inference",
      "category": "PERFORMANCE",
      "opinion": "fast",
      "sentiment": "Positive"
    },
    {
      "aspect": "memory use",
      "category": "RESOURCES",
      "opinion": "not great",
      "sentiment": "Negative"
    }
  ],
  "model_name": "ai-absa-rule-baseline-v1"
}
```

## Kiến trúc

```text
technical tokenization
-> domain relevance gate
-> multi-aspect extraction
-> phrase-first sentiment evidence
-> clause segmentation
-> nearest aspect-opinion linking in the same clause
-> distance-weighted sentiment aggregation
-> category mapping
-> ASQP quads
```

Baseline hỗ trợ `parent_context` và `thread_title`. Khi câu không chứa aspect
rõ ràng, model có thể lấy target từ context. Nếu toàn bộ record không có tín
hiệu AI/LLM, relevance gate trả về danh sách quad rỗng.

## Fitting với Data Final

Hai file mặc định:

```text
data/processed/final_data/train_final.jsonl
data/processed/final_data/val_final.jsonl
```

Chạy:

```bash
python -m src.models.basic.train_baseline
```

Kết quả được ghi vào:

```text
models/baseline/model.json
models/baseline/fitting_report.json
models/baseline/validation_metrics.json
models/baseline/validation_predictions.jsonl
```

Fitting thực hiện:

- loại sentence trùng giữa train và validation khỏi dữ liệu fit;
- học supervised aspect lexicon từ train;
- học category hints từ aspect và opinion đã gán nhãn;
- tính SO-PMI để đề xuất sentiment terms mới;
- serialize lexicon, threshold và metadata thành `model.json`.

Aspect, category, relevance và SO-PMI candidates không tự động được đưa vào
production lexicon. Sau khi review, tạo file JSON dạng:

```json
{
  "aspect_terms": {
    "qwen coder": "INTELLIGENCE"
  },
  "category_terms": {
    "tokens per second": "PERFORMANCE"
  },
  "relevance_terms": ["speculative decoding"],
  "sentiment_terms": {
    "abysmal": -1.2,
    "flawless": 1.2
  }
}
```

Sau đó chạy:

```bash
python -m src.models.basic.train_baseline \
  --reviewed-adaptations reviewed_adaptations.json
```

## Docker Interface

Model rule-based không phải graph tensor nên không nên ép export ONNX. Artifact
production là `model.json`; Docker image đóng gói Python source và artifact này.

```python
from src.models.basic import create_model_interface

baseline = create_model_interface("models/baseline/model.json")

result = baseline.predict({
    "text": "It is really fast.",
    "parent_context": "I installed Qwen 3 locally.",
    "thread_title": "Qwen local inference"
})
```

Contract input:

```json
{
  "text": "required string",
  "parent_context": "optional string",
  "thread_title": "optional string"
}
```

Contract output luôn gồm `text`, `quads`, và `model_name`. Flask có thể giữ một
instance `BaselineModelInterface` khi process khởi động và gọi `predict()` trong
endpoint `/api/predict`.

CLI smoke test:

```bash
python -m src.models.basic.interface \
  --text "Qwen is fast but uses too much VRAM."
```

## Đánh giá hiện tại

Baseline được đánh giá bằng exact span/match. Kết quả nằm trong
`models/baseline/validation_metrics.json`. Đây là performance floor để so với
advanced model, không phải kết quả neural training.
