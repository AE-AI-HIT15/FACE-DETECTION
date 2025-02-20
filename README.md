# 📌 FACE-DETECTION

## 1️⃣ Giới thiệu
Dự án **FACE DETECTION** giúp phát hiện khuôn mặt trong **thời gian thực** từ webcam hoặc từ ảnh, video được tải lên.

## 2️⃣ Chức năng
Hệ thống hỗ trợ ba chế độ phát hiện khuôn mặt chính:
- 📷 **Phát hiện qua Webcam**
- 🖼 **Phát hiện qua Ảnh**
- 🎥 **Phát hiện qua Video**

## 3️⃣ Tổng quan Hệ thống
- 🏗 **Mô hình**: YOLOv8n Face
- ⚙ **Backend**: FastAPI
- 💻 **Frontend**: Streamlit

## 4️⃣ Hướng dẫn Cài đặt
### 🔹 Yêu cầu hệ thống
- Python 3.8+
- pip

### 🔹 Sao chép kho lưu trữ
```bash
git clone https://github.com/AE-AI-HIT15/FACE-DETECTION.git
cd FACE-DETECTION
```

### 🔹 Cấu trúc thư mục
```
📂 FACE-DETECTION
├── 📂 app               # Thư mục chính
│   ├── __init__.py
│   ├── app.py          # Ứng dụng Streamlit
│   ├── main.py         # Máy chủ API FastAPI
│   ├── requirements.txt # Chứa các thư viện cần thiết
│   ├── 📂 routers      # Router API
│   │   ├── __init__.py
│   │   ├── face_detect.py
│   │   ├── upload.py   # Xử lý tải lên
│   ├── 📂 domain       # Xử lý logic ứng dụng
│   │   ├── __init__.py
│   │   ├── face_detector.py
│   ├── 📂 schemas      # Định nghĩa mô hình dữ liệu
│   │   ├── __init__.py
│   │   ├── base_model.py
│   ├── 📂 utils        # Tiện ích & cấu hình
│   │   ├── __init__.py
│   │   ├── configs.py
│   ├── 📂 api          # Xử lý middleware & logging
│   │   ├── __init__.py
│   │   ├── exception_handler.py
│   │   ├── logger.py
│   │   ├── middleware.py
```

## 5️⃣ Hướng dẫn Chạy Hệ thống
### 🔹 Cài đặt môi trường ảo và các thư viện cần thiết
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows
venv\Scripts\activate
# Trên macOS/Linux
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 🔹 Tải mô hình
- Truy cập đường link:(https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt) để tải mô hình

### 🔹 Chạy Backend FastAPI
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 🔹 Chạy Frontend Streamlit
```bash
cd app
streamlit run app_streamlit.py
```

Sau khi khởi động, hệ thống sẽ tự động mở trình duyệt với giao diện phát hiện khuôn mặt. Người dùng có thể sử dụng các tính năng phát hiện trực tiếp qua webcam, ảnh hoặc video.