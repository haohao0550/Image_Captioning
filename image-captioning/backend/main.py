from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path

# Import image captioning model
# from inference.image_captioning import ImageCaptioningModel
from inference.blip_captioning import BlipCaptioningModel

# Khởi tạo app
app = FastAPI(title="Image Captioning API - BLIP")

# Cho phép frontend từ domain khác truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:3000'] nếu muốn giới hạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (paths resolved relative to this file)
# BASE_DIR = Path(__file__).resolve().parent
# captioning_model = ImageCaptioningModel(
#     weights_path=str(BASE_DIR / "weight" / "best_e2e_model.weights.h5"),
#     vocab_path=str(BASE_DIR / "weight" / "vocabulary_5.json"),
# )

# Load BLIP model (paths resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent
captioning_model = BlipCaptioningModel(
    model_path=str(BASE_DIR / "best_model"),
    max_length=50,
)


# Health check
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


# Predict từ ảnh upload
@app.post("/predict")
async def predict(file: UploadFile = File(...), beam_search: bool = True, beam_width: int = 5):
    """
    Nhận ảnh từ frontend và trả về caption mô tả ảnh
    
    Args:
        file: Ảnh upload
        beam_search: Sử dụng beam search (True) hoặc greedy search (False)
        beam_width: Độ rộng beam (chỉ dùng khi beam_search=True)
    """
    try:
        # Đọc ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Tạo caption
        if beam_search:
            result = captioning_model.generate_caption_beam_search(
                image, beam_width=beam_width, alpha=0.7
            )
        else:
            result = captioning_model.generate_caption_greedy(image)

        return {
            "filename": file.filename,
            "caption": result["caption"],
            "method": "beam_search" if beam_search else "greedy",
        }
    except Exception as e:
        return {"error": str(e)}
