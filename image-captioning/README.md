# ImageCaption AI — Setup & Run

## Mô tả nhanh

- **Backend**: FastAPI (Python) với endpoint `/predict` cho image captioning
- **Frontend**: Next.js (React) với API route proxy `/api/predict`
- **Model**: ResNet50 Encoder + Transformer Decoder

---

## Yêu cầu

- Python 3.8+ và `pip`
- Node.js 18+ và `npm`
- TensorFlow 2.x

---

## 1. Chuẩn bị Model & Vocabulary

### Sau khi train xong model trong notebook:

**1.1. Export vocabulary** (Chạy trong notebook sau cell training):

```python
# Export vocabulary
vocabulary_data = {
    "word_to_idx": word_to_idx,
    "idx_to_word": {str(k): v for k, v in idx_to_word.items()},
    "vocab_size": vocab_size,
    "max_length": max_length
}

import json
with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary_data, f, ensure_ascii=False, indent=2)
```

**1.2. Copy files vào backend:**
- `best_e2e_model.weights.h5` → `backend/weight/`
- `vocabulary.json` → `backend/weight/`

---

## 2. Backend — cài đặt và chạy

```powershell
# vào thư mục backend
cd backend

# (tùy chọn) tạo và kích hoạt virtualenv
python -m venv .venv
.\.venv\Scripts\Activate

# cài đặt dependencies
pip install -r requirements.txt

# chạy server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Kiểm tra**: Mở http://127.0.0.1:8000/docs

---

## 3. Frontend — cài đặt và chạy

```powershell
# vào thư mục frontend
cd frontend

# cài đặt dependencies
npm install

# chạy frontend (Next.js dev)
npm run dev
```

**Kiểm tra**: Mở http://localhost:3000

---

## 4. Test nhanh

### Test backend trực tiếp:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@D:\path\to\test.jpg"
```

### Test với tham số:
```powershell
# Dùng beam search với beam_width=10
curl.exe -X POST "http://127.0.0.1:8000/predict?beam_search=true&beam_width=10" -F "file=@image.jpg"

# Dùng greedy search (nhanh hơn)
curl.exe -X POST "http://127.0.0.1:8000/predict?beam_search=false" -F "file=@image.jpg"
```

### Test qua frontend:
1. Mở http://localhost:3000
2. Upload ảnh
3. Nhấn "Tạo mô tả ngay"
4. Xem caption được tạo

---

## 5. API Response Format

```json
{
  "filename": "example.jpg",
  "caption": "a dog is running on the grass",
  "method": "beam_search"
}
```

---

## 6. Cấu trúc thư mục quan trọng

```
backend/
  ├── main.py                           # FastAPI server
  ├── inference/
  │   └── image_captioning.py          # Model inference
  └── weight/
      ├── best_e2e_model.weights.h5   # Model weights ⚠️
      └── vocabulary.json               # Vocabulary ⚠️

frontend/
  ├── app/
  │   ├── page.tsx                     # Main UI
  │   └── api/predict/route.ts         # API proxy
  └── lib/
      └── utils.ts                      # Utilities
```

---

## 7. Troubleshooting

**Lỗi "Model weights not found":**
→ Đảm bảo file `best_e2e_model.weights.h5` ở `backend/weight/`

**Lỗi "Vocabulary not found":**
→ Export vocabulary từ notebook (xem phần 1) và copy vào `backend/weight/`

**Lỗi TensorFlow:**
→ `pip install tensorflow==2.15.0` (hoặc version tương thích)

**Port đã dùng:**
→ Đổi port: `uvicorn main:app --reload --port 8001`

**Caption không đúng:**
→ Kiểm tra lại model weights và vocabulary có khớp với model đã train

---

## 8. Upload code lên GitHub

### Lần đầu (khởi tạo repo)

```powershell
# 1. Khởi tạo git (nếu chưa có)
git init

# 2. Thêm tất cả file (trừ file trong .gitignore)
git add .

# 3. Commit lần đầu
git commit -m "Initial commit: ImageCaption AI - Image captioning app"

# 4. Tạo repo trên GitHub (https://github.com/new), copy URL, rồi:
git remote add origin https://github.com/your-username/your-repo.git

# 5. Push lên GitHub
git branch -M main
git push -u origin main
```

### Các lần sau (cập nhật code)

```powershell
# 1. Kiểm tra file đã thay đổi
git status

# 2. Thêm file muốn commit
git add .

# 3. Commit với message mô tả thay đổi
git commit -m "Mô tả thay đổi của bạn"

# 4. Push lên GitHub
git push
```

### Lưu ý quan trọng

- **Model weights (file .weights.h5, .pth, .h5) đã bị chặn bởi .gitignore** vì quá nặng cho GitHub.
- Nếu cần share model, dùng:
  - **Git LFS** (Large File Storage): `git lfs install` → `git lfs track "*.h5"` → commit
  - **Google Drive / Dropbox**: upload model, share link trong README
  - **GitHub Releases**: attach file nặng vào release thay vì commit

- Để pull model từ teammate:
  ```powershell
  # Tải model từ link (VD: Google Drive) và đặt vào:
  # backend/weight/best_e2e_model.weights.h5
  # backend/weight/vocabulary.json
  ```