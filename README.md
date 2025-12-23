# 🖼️ Image Captioning Project

## 📌 Giới thiệu

Đề tài **Image Captioning** tập trung vào bài toán **tự động sinh mô tả
(caption) cho hình ảnh**, kết hợp giữa **thị giác máy tính (Computer
Vision)** và **xử lý ngôn ngữ tự nhiên (NLP)**.\
Dự án nhằm so sánh hiệu quả giữa hai chiến lược tiếp cận: mô hình truyền
thống CNN--Transformer và mô hình hiện đại đa phương thức BLIP.

------------------------------------------------------------------------

## 📊 Dữ liệu (Dataset)

### Flickr8k Dataset

-   Nguồn: Flickr8k
-   Số lượng:
    -   8,000 hình ảnh
    -   Mỗi ảnh có **5 caption** do con người gán nhãn
-   Chia tập:
    -   Train
    -   Validation
    -   Test
-   Đặc điểm:
    -   Đa dạng ngữ cảnh đời sống
    -   Phù hợp cho bài toán image captioning cơ bản và nâng cao

------------------------------------------------------------------------

## 🧠 Chiến lược mô hình

### 1️⃣ ResNet50 + Transformer (Baseline)

-   **Encoder**:
    -   ResNet50 trích xuất đặc trưng ảnh\
    -   Sử dụng feature map từ tầng convolution cuối
-   **Decoder**:
    -   Transformer Decoder sinh caption
-   Huấn luyện:
    -   Huấn luyện từ đầu hoặc fine-tune nhẹ
    -   Cross-Entropy Loss
-   Ưu điểm:
    -   Kiến trúc rõ ràng, dễ triển khai
    -   Phù hợp làm baseline để so sánh

------------------------------------------------------------------------

### 2️⃣ BLIP Video Captioning Base (Fine-tune)

-   Mô hình: **BLIP (Bootstrapping Language-Image Pretraining)**\
-   Phiên bản: **BLIP Video Captioning Base**
-   Chiến lược:
    -   Fine-tune mô hình pretrained trên dataset Flickr8k
    -   Áp dụng cho bài toán image captioning (video được xem như 1
        frame)
-   Ưu điểm:
    -   Khả năng hiểu ngữ cảnh mạnh
    -   Sinh caption tự nhiên và chính xác hơn
    -   Tận dụng sức mạnh của pretraining đa phương thức

------------------------------------------------------------------------

## ⚙️ Quy trình huấn luyện

1.  Tiền xử lý ảnh:
    -   Resize, normalize
2.  Tiền xử lý văn bản:
    -   Tokenization
    -   Padding
3.  Huấn luyện mô hình theo từng chiến lược
4.  Đánh giá và so sánh kết quả

------------------------------------------------------------------------

## 📈 Đánh giá

-   Các chỉ số đánh giá:
    -   BLEU-1, BLEU-4
    -   METEOR
    -   CIDEr
-   So sánh:
    -   Caption sinh ra
    -   Chất lượng ngữ nghĩa và độ tự nhiên của câu

------------------------------------------------------------------------

## 🚀 Kết luận

-   ResNet50 + Transformer là baseline hiệu quả và dễ triển khai
-   BLIP fine-tune cho kết quả vượt trội về chất lượng caption
-   Pretrained đa phương thức giúp mô hình hiểu ngữ cảnh hình ảnh tốt
    hơn
-   Flickr8k phù hợp cho cả nghiên cứu cơ bản và thử nghiệm mô hình hiện
    đại

------------------------------------------------------------------------

## 🛠 Công nghệ sử dụng

-   Python\
-   PyTorch / TensorFlow\
-   HuggingFace Transformers\
-   NumPy, OpenCV

------------------------------------------------------------------------

## 📄 Giấy phép

Dự án phục vụ cho mục đích **học tập và nghiên cứu**.
