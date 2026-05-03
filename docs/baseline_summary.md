# Baseline Model — Tóm tắt luồng hoạt động

## Mục tiêu

Phân loại cảm xúc của bình luận trên r/LocalLlama thành **positive / negative / neutral**.

---

## Luồng xử lý

```
raw JSONL
   │
   ▼
data_loader.load_labeled()
   • Đọc từng dòng JSONL
   • Bỏ qua [removed], [deleted], body rỗng
   • Gán nhãn tạm thời bằng bộ từ khoá (regex):
       - positive : good, great, best, amazing, recommend, …
       - negative : bad, slow, bug, issue, error, hallucinate, …
       - neutral  : compare, vs, gpu, vram, token, benchmark, …
   • Chọn nhãn có tần suất khớp cao nhất; bỏ nếu không khớp gì
   │
   ▼
data_loader.balance()
   • Undersample về N_PER_CLASS mẫu mỗi lớp (mặc định 33 000)
   • Đảm bảo 3 lớp cân bằng nhau
   │
   ▼
data_loader.split()
   • 70 % → dev pool  → chia tiếp 70 / 15 / 15 → train / val / test
   • 30 % → leaderboard holdout (không dùng khi huấn luyện)
   │
   ▼
preprocess.preprocess()  — áp dụng cho mỗi tập
   • Lowercase, xoá URL, code block, ký tự phi alpha
   • Xử lý phủ định: gắn tiền tố NOT_ cho tối đa 3 token sau
     các từ "not / no / never / …"  (ví dụ: "not good" → "NOT_good")
   │
   ▼
model.build() + model.fit()
   • TF-IDF vectorizer  (unigram + bigram, sublinear TF, max 60 000 đặc trưng)
   • Classifier:
       - "nb" → Multinomial Naive Bayes  (alpha = 0.1)
       - "lr" → Logistic Regression      (C = 5, class_weight = balanced)
   • fit() trên tập train
   │
   ▼
model.predict()  — trên test + leaderboard
   │
   ▼
evaluate.metrics()
   • Accuracy
   • Classification Report (precision / recall / F1 mỗi lớp + macro avg)
   │
   ▼
evaluate.to_batch()
   • Bọc kết quả vào BatchABSA (chuẩn Pydantic)
   • Mỗi bản ghi là ABSADocument với một ABSATriplet
     aspect = "__DOC_LEVEL__", sentiment = nhãn dự đoán
   │
   ▼
results/
   ├── test_predictions.json      — BatchABSA cho tập test
   ├── leaderboard_predictions.json — BatchABSA cho leaderboard
   └── summary.json               — accuracy + kích thước tập
```

---

## Cấu trúc file

| File | Vai trò |
| :--- | :--- |
| `src/data/preprocess.py` | Làm sạch văn bản + xử lý phủ định |
| `src/data/data_loader.py` | Đọc JSONL, gán nhãn, cân bằng, chia tập |
| `src/models/model.py` | Pipeline TF-IDF + classifier |
| `src/evaluation/evaluate.py` | Tính metric, xuất Pydantic |
| `src/data/schemas.py` | Định nghĩa ABSATriplet / ABSADocument / BatchABSA |
| `main.py` | Điều phối toàn bộ luồng, in kết quả ra terminal |

---

## Cách chạy

```bash
python main.py
```

Terminal sẽ in:

```
=== Test Set  (n=...) ===
Accuracy : 0.XXXX

              precision    recall  f1-score   support
    negative     ...
     neutral     ...
    positive     ...
    ...

=== Leaderboard  (n=...) ===
Accuracy : 0.XXXX
...
```

Kết quả dự đoán và summary được lưu vào thư mục `results/`.

---

## Tham số có thể điều chỉnh trong `main.py`

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `DATA_PATH` | `data/r_LocalLlama_comments.jsonl` | Đường dẫn file dữ liệu |
| `MODEL_KIND` | `"nb"` | `"nb"` hoặc `"lr"` |
| `N_PER_CLASS` | `33 000` | Số mẫu tối đa mỗi lớp trước khi chia |
