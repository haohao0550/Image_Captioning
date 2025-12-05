# Hướng dẫn chạy Backend Image Captioning

## Bước 1: Chuẩn bị file cần thiết

Sau khi train xong model trong notebook, bạn cần 2 file:

### 1.1. Model weights: `best_e2e_model.weights.h5`
- Đây là file đã có từ quá trình training
- Copy vào: `backend/weight/best_e2e_model.weights.h5`

### 1.2. Vocabulary file: `vocabulary.json`
Chạy code sau trong **Jupyter Notebook** (sau khi đã build vocabulary ở cell thứ 8):

```python
# Chạy trong notebook sau khi có word_to_idx, idx_to_word, vocab_size, max_length
vocabulary_data = {
    "word_to_idx": word_to_idx,
    "idx_to_word": {str(k): v for k, v in idx_to_word.items()},
    "vocab_size": vocab_size,
    "max_length": max_length
}

import json
with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary_data, f, ensure_ascii=False, indent=2)

print("✅ Vocabulary saved to vocabulary.json")
```

Sau đó copy file `vocabulary.json` vào: `backend/weight/vocabulary.json`

---

## Bước 2: Cấu trúc thư mục backend/weight

```
backend/
  weight/
    ├── best_e2e_model.weights.h5  ← Model weights
    └── vocabulary.json             ← Vocabulary file
```

---

## Bước 3: Chạy backend

```cmd
cd backend

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

## Bước 4: Test API

### Test với curl:
```cmd
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\image.jpg"
```

### Hoặc mở Swagger UI:
http://127.0.0.1:8000/docs

---

## API Response Format

```json
{
  "filename": "example.jpg",
  "caption": "a dog is running on the grass",
  "method": "beam_search"
}
```

---

## Tham số API

- `beam_search` (bool, default=True): Dùng beam search hoặc greedy search
- `beam_width` (int, default=5): Số lượng beams (chỉ dùng với beam_search=True)

### Ví dụ với tham số:
```cmd
curl.exe -X POST "http://127.0.0.1:8000/predict?beam_search=true&beam_width=10" -F "file=@image.jpg"
```
