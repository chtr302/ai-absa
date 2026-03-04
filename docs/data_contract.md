# Data Contract: AI-ABSA Project Structure

Tài liệu này quy định cấu trúc dữ liệu chung cho toàn bộ dự án AI-ABSA. Tất cả các module (Data Preprocessing, Model Training, Dashboard) **BẮT BUỘC** phải tuân thủ định dạng này để đảm bảo hệ thống có thể tích hợp (Integration) thành công.

---

## 1. Cấu trúc Core: `ABSATriplet`
Đây là đơn vị dữ liệu nhỏ nhất, đại diện cho một bộ ba cảm xúc.

| Trường | Kiểu dữ liệu | Mô tả | Ví dụ |
| :--- | :--- | :--- | :--- |
| `aspect` | `str` | Thực thể được đánh giá (Bắt buộc). | `"ChatGPT"`, `"UI"` |
| `opinion` | `str | None` | Từ ngữ chỉ quan điểm (Tùy chọn ở Tier 1). | `"fast"`, `"confusing"` |
| `sentiment` | `Enum` | Cảm xúc (Bắt buộc). | `"positive"`, `"negative"`, `"neutral"` |

---

## 2. Cấu trúc Output: `ABSADocument`
Kết quả phân tích cho một câu văn hoàn chỉnh. Đây là dữ liệu Dashboard(Quân) lấy để hiển thị.

### Ví dụ JSON Output:
```json
{
  "raw_text": "ChatGPT is very fast but sometimes it provides hallucinating answers.",
  "triplets": [
    {
      "aspect": "ChatGPT",
      "opinion": "fast",
      "sentiment": "positive"
    },
    {
      "aspect": "answers",
      "opinion": "hallucinating",
      "sentiment": "negative"
    }
  ],
  "model_name": "ai-absa-v1-gen",
  "inference_time_ms": 12.5
}
```

---

## 3. Workflows

1.  **Thức:** Làm BIO Tagging trên Mendeley, đảm bảo đầu ra cuối cùng có thể map được vào cấu trúc `ABSATriplet`.
2.  **Hậu (Model):** Model Generative ABSA sẽ sinh ra chuỗi văn bản, sau đó sẽ được hàm Parser chuyển đổi thành Object `ABSADocument`.
3.  **Quân:** Dùng JSON Output ở trên để mock dữ liệu để tạo dashboard chứ không cần chờ model hoàn chỉnh.

