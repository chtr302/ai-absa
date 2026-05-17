# Simple ABSA Baseline Model

## Mục Đích

Baseline này xử lý phiên bản đầu tiên, đơn giản nhất của bài toán AI-ABSA:

```text
raw sentence -> primary aspect -> opinion/evidence -> sentiment polarity
```

Vì dữ liệu hiện tại đến từ các comment Reddit thực tế, câu đầu vào có thể viết tắt, thiếu ngữ cảnh, có mỉa mai, hoặc không nói rõ đang đánh giá cái gì. Vì vậy baseline được thiết kế bảo thủ: mỗi câu chỉ trả về tối đa một aspect chính kèm opinion/evidence và sentiment polarity của aspect đó.

## Baseline Làm Gì

Với một câu raw tiếng Anh, model trả về:

```json
{
  "text": "Qwen is very fast.",
  "predictions": [
    {
      "aspect": "Qwen",
      "opinion": "fast",
      "sentiment": "Positive"
    }
  ]
}
```

Trong ví dụ trên:

- `Qwen` là aspect.
- `fast` là opinion/evidence.
- `Positive` là sentiment polarity được suy ra từ opinion.

Nếu không tìm thấy aspect rõ ràng trong domain GenAI/LLM, model trả về danh sách rỗng:

```json
{
  "text": "This is really impressive.",
  "predictions": []
}
```

Nếu tìm thấy aspect nhưng không có tín hiệu sentiment rõ, model trả về `Neutral`:

```json
{
  "text": "I tried Ollama yesterday.",
  "predictions": [
    {
      "aspect": "Ollama",
      "opinion": "",
      "sentiment": "Neutral"
    }
  ]
}
```

## Kiến Trúc Tổng Quan

Baseline là một rule-based NLP model:

```text
Text normalization
-> aspect detection
-> primary aspect selection
-> opinion/evidence detection
-> sentiment assignment
-> compact prediction output
```

Model không cần gold ABSA labels, không cần train supervised model, và không cần gọi external service.

## Các Thành Phần Chính

### 1. Text Normalization

File `text_utils.py` xử lý làm sạch nhẹ:

- gộp khoảng trắng thừa;
- xóa URL;
- đổi Markdown link thành text hiển thị;
- giữ lại các tên kỹ thuật như `GPT-4`, `Qwen2.5`, `VRAM`, `GGUF`, `llama.cpp`.

### 2. Aspect Detection

File `aspect_detector.py` tìm các aspect rõ ràng bằng regex và dictionary trong `baseline_lexicons.py`.

Các nhóm aspect được hỗ trợ:

- model: `Llama`, `Qwen`, `GPT`, `Claude`, `Gemma`, `Mistral`, `DeepSeek`;
- hardware/resource: `GPU`, `VRAM`, `RAM`, `RTX 4090`, `CUDA`;
- software/tool: `Ollama`, `llama.cpp`, `vLLM`, `LM Studio`;
- technical term: `quantization`, `GGUF`, `context length`, `latency`, `RAG`.

Detector cũng chuẩn hóa một số alias phổ biến, ví dụ `gpt4`, `GPT 4`, `GPT-4`.

### 3. Primary Aspect Selection

File `sentiment_rules.py` chọn một aspect chính trong câu.

Logic chấm điểm:

- aspect thuộc domain cụ thể được ưu tiên hơn cụm từ chung;
- aspect gần opinion/evidence được ưu tiên hơn;
- với comparative pattern, model ưu tiên chủ thể của phép so sánh, ví dụ `Llama 3` trong `Llama 3 is faster than GPT-4`;
- với resource usage pattern, hardware/resource aspect được ưu tiên khi phù hợp.

Mục tiêu của bước này là tránh việc trả về quá nhiều aspect trong comment ngẫu nhiên.

### 4. Opinion / Evidence Detection

Opinion/evidence là từ hoặc cụm từ khiến model gán sentiment cho aspect.

Ví dụ:

- `Qwen is very fast.` -> opinion/evidence: `fast`
- `Llama is not good at coding.` -> opinion/evidence: `good`, có negation nên sentiment là `Negative`
- `This model uses too much VRAM.` -> opinion/evidence: `too much`

Trong code hiện tại, opinion/evidence được lấy từ sentiment signal gần aspect nhất. Khi bật debug mode, trường này được trả ra dưới tên `evidence`.

### 5. Sentiment Assignment

Baseline gán một trong ba nhãn sentiment polarity:

```text
Positive
Negative
Neutral
```

Sentiment được gán bằng lexicon và rule:

- `fast`, `better`, `good`, `accurate` -> `Positive`
- `slow`, `worse`, `bad`, `too much`, `unstable` -> `Negative`
- không có opinion/evidence rõ -> `Neutral`

Negation được xử lý cục bộ:

- `not good` -> `Negative`
- `not slow` -> `Positive`

## Các File Liên Quan

```text
src/models/basic/sentiment_model.py      # Public API / orchestration
src/models/basic/aspect_detector.py      # Aspect matching và canonicalization
src/models/basic/sentiment_rules.py      # Chọn aspect chính và gán sentiment
src/models/basic/baseline_lexicons.py    # Regex/dictionary có thể cấu hình
src/models/basic/baseline_types.py       # Dataclass dùng chung
src/models/basic/text_utils.py           # Helper normalize text
```

## Public API

```python
from src.models.basic import ABSASentimentModel

model = ABSASentimentModel()
result = model.predict("Llama 3 is faster than GPT-4 but uses more VRAM.")
```

Output mặc định:

```json
{
  "text": "Llama 3 is faster than GPT-4 but uses more VRAM.",
  "predictions": [
    {
      "aspect": "Llama 3",
      "sentiment": "Positive"
    }
  ]
}
```

Có thể bật debug mode:

```python
model.predict("Qwen is very fast.", include_debug=True)
```

Debug mode sẽ thêm các trường nội bộ như normalized aspect, aspect group, confidence, evidence và negation flag.

## Hành Vi Kỳ Vọng

```text
Qwen is very fast.                               -> Qwen, fast, Positive
Llama is not good at coding.                     -> Llama, good + negation, Negative
This model uses too much VRAM.                   -> VRAM, too much, Negative
Gemma is worse than Mistral.                     -> Gemma, worse, Negative
Llama 3 is faster than GPT-4 but uses more VRAM. -> Llama 3, faster, Positive
This is really impressive.                       -> []
I tried Ollama yesterday.                        -> Ollama, no clear opinion, Neutral
```
