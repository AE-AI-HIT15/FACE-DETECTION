import time
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.configs import model

# Kiểm tra đầu vào là video hay ảnh
def is_webcam(source):
    """Kiểm tra xem đầu vào có phải là webcam realtime hay không"""
    return source == 0

def is_video(source):
    """Kiểm tra xem đầu vào có phải là video hay không"""
    return isinstance(source, str) and source.lower().endswith(('.mp4', '.avi', '.mov'))

def is_image(source):
    """Kiểm tra xem đầu vào có phải là ảnh hay không"""
    return isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg'))

def process_video(yolo_model, source) -> list[list[int]]:
    """Xử lý video để nhận diện khuôn mặt"""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Không thể mở video!")
        exit()

    while True:
        start_time = time.time()  # Bắt đầu tính thời gian
        
        ret, frame = cap.read()
        if not ret:
            break  # Dừng nếu hết video

        # Phát hiện khuôn mặt bằng YOLO
        results = yolo_model.predict(frame)

    #     # Vẽ bounding box
    #     for result in results:
    #         for box in result.boxes.xyxy:
    #             x1, y1, x2, y2 = map(int, box[:4])
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #     # Tính FPS
    #     fps = 1 / (time.time() - start_time)
    #     fps_text = f"FPS: {fps:.2f}"

    #     flip_frame = cv2.flip(frame, 1)
        
    #     # Hiển thị FPS trên video
    #     cv2.putText(flip_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    #     # Hiển thị video
    #     cv2.imshow("YOLO Face Detection", flip_frame)

    #     # Nhấn 'q' để thoát
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()

"""def main(source):
    # Tải mô hình YOLO
    yolo_model = model()

    if is_video(source):
        # Nếu là video, gọi hàm xử lý video
        process_video(yolo_model, source)
    elif is_webcam(source):
        process_video(yolo_model, source)
    elif is_image(source):
        # Nếu là ảnh, gọi hàm xử lý ảnh
        process_image(yolo_model, source)
    else:
        print("Đầu vào không hợp lệ. Vui lòng cung cấp một file ảnh hoặc video!")

    # Giải phóng tài nguyên
    cv2.destroyAllWindows()"""

"""if __name__ == "__main__":
    # Đặt đường dẫn đến video hoặc ảnh ở đây
    source = '' 
    # Thay đổi thành đường dẫn file video hoặc ảnh hoặc webcam (file path hoặc 0 - dành cho webcam) của bạn
    main(source)
"""
