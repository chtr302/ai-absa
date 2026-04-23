# Data Contract: AI-ABSA Project Structure

Tài liệu này quy định cấu trúc dữ liệu hợp nhất cho toàn bộ dự án AI-ABSA. Tất cả các Module (Preprocessing, KIBAC Model, Dashboard) **BẮT BUỘC** tuân thủ định dạng này.

---

## 1. Cấu trúc Core: `ABSAQuadruplet`

| Trường | Kiểu dữ liệu | Mô tả | Ví dụ |
| :--- | :--- | :--- | :--- |
| **`aspect`** | `str` | Thực thể cụ thể (Explicit hoặc Implicit). | `"Llama 3"`, `"VRAM"` |
| **`category`** | `Enum` | Nhóm mỏ neo logic (7 tầng). | `"PERFORMANCE"`, `"RESOURCES"` |
| **`opinion`** | `str` | Từ ngữ thể hiện quan điểm. | `"blazing fast"`, `"memory hog"` |
| **`sentiment`** | `Enum` | Cảm xúc: `Positive`, `Negative`, `Neutral`. | `"Positive"` |

### 7-Layer Categories (Hệ thống mỏ neo):
`PERFORMANCE`, `INTELLIGENCE`, `RESOURCES`, `BEHAVIOR`, `TECHNICAL`, `SOFTWARE`, `COMPARATIVE`.

---

## 2. Cấu trúc Input/Output: `ABSAResult`
Dữ liệu trao đổi giữa Model và Dashboard. Cấu trúc này bảo toàn tính "Dòng họ" của Reddit.

```json
{
  "id": "comment_id_123",
  "parent_context": "Nội dung câu cha giúp giải quyết Implicit Aspect",
  "thread_title": "Tiêu đề bài viết (Global Anchor)",
  "sentences": [
    {
      "text": "Llama 3 is faster than GPT-4.",
      "quads": [
        {
          "aspect": "Llama 3",
          "category": "PERFORMANCE",
          "opinion": "faster than GPT-4",
          "sentiment": "Positive"
        }
      ]
    }
  ],
  "model_info": {
    "name": "KIBAC-3.0-Biaffine",
    "version": "v1.0",
    "inference_ms": 45.0
  }
}
```

---

## 3. Team Workflows (Phân công phối hợp)

1.  **Thức (Model Basemline):**
    - Xây dựng `Baseline` cho ứng dụng.

2.  **Hậu (Model Architecture):**
    - Nhận Input JSON, thực hiện Biaffine Scoring để trích xuất `quads`.
    - Đảm bảo Output đúng Schema `ABSAQuadruplet`.
    - Tích hợp **AI Jargon Ontology** để xử lý các thực thể ẩn.

3.  **Quân (Data Engineering - Frontend/Dashboard):**
    - Đảm bảo Pipeline cung cấp đúng JSON lồng nhau có `parent_context`.
    - Chịu trách nhiệm về `tech_score` và làm sạch dữ liệu (Denoising).
    - Sử dụng `category` làm mỏ neo chính để vẽ biểu đồ so sánh.
    - Tận dụng `thread_title` để nhóm các phân tích theo chủ đề.
    - Hiển thị mối quan hệ giữa `parent_context` và câu trả lời để người dùng hiểu ngữ cảnh.
