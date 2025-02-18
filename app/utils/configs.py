import torch
from ultralytics import YOLO

# Đường dẫn đến mô hình YOLOv8n-face (cần tải trước)
MODEL_PATH = "yolov8n-face.pt"

# Tham số tùy chỉnh
CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng tin cậy để giữ lại khuôn mặt
IOU_THRESHOLD = 0.5         # Ngưỡng NMS để loại bỏ bbox trùng lặp
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Dùng GPU nếu có

def model():
    """Load mô hình YOLOv8n face"""
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)  # Chuyển mô hình sang GPU (nếu có)
    return model
