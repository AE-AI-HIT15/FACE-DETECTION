import torch
from ultralytics import YOLO

# Path to the YOLOv8n-face model (must be downloaded beforehand)
MODEL_PATH = "yolov8n-face.pt"

# ========================== CONFIGURATION PARAMETERS ==========================
FRAME_SKIP = 2  # Number of frames to skip before processing (reduces computational load)

CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold to retain detected faces
IOU_THRESHOLD = 0.5         # Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS)

# Select device: Use GPU if available, otherwise default to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STREAM = True  # Enable streaming mode for processing video frames
IMAGE_SIZE = (480, 640)  # Target image size (height, width) for model input


def create_model():
    """
    Load the YOLOv8n-face model and move it to the selected device (GPU or CPU).

    :return: YOLO model instance ready for inference.
    """
    model = YOLO(MODEL_PATH)  # Load the model from the specified path
    model.to(DEVICE)  # Move the model to GPU if available, otherwise use CPU
    return model
