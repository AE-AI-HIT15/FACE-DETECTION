import time
import cv2
import sys
import os
from typing import List, Union
# import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.configs import model, CONFIDENCE_THRESHOLD

def frame_process(input: Union[str, int], confident: float = CONFIDENCE_THRESHOLD, source: str = "webcam") -> List[List[int]]:
    """
        Hàm xử lí nhận diện khuôn mặt trong từng khung hình

        Tham số:
        - input: Đường dẫn ảnh, video, chỉ số webcam hoặc mảng NumPy chứa frame.
        - confident: Ngưỡng độ tin cậy để nhận diện khuôn mặt.
        - source: Loại dữ liệu đầu vào ('image', 'video', 'webcam').

        Trả về vị trí của khung khuôn mặt, confident score
    """
    yolo_model = model()
    res = []
    
    if source == "image":
        if isinstance(input, str):
            frame = cv2.imread(input)
        else:
            frame = input
        predict_frame = yolo_model.predict(frame, conf=confident, stream=True)
        for result in predict_frame:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                res.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

         # Hiển thị ảnh sau khi nhận diện khuôn mặt
        cv2.imshow("Detected Faces", frame)
        cv2.waitKey(0)  # Chờ nhấn phím bất kỳ để đóng cửa sổ
        cv2.destroyAllWindows()        
        return res, confident
    
    elif source in ["video", "webcam"]:
        cap = cv2.VideoCapture(input)  # Webcam: 0, Video: đường dẫn file
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            predict_frame = yolo_model.predict(frame, conf=confident, stream=True)  # Fix lỗi CONF
            frame_res = []
            for result in predict_frame:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    frame_res.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Webcam Face Detection", frame)  # Hiển thị video
            print(f"Faces detected: {frame_res}")

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
                break

        cap.release()
        cv2.destroyAllWindows()
    
    return res, confident


if __name__ == '__main__':
    """
    Đưa đường đẫn của ảnh, video, webcam vào file_path.
    1. Nếu sử dụng ảnh thì ở tham số source truyền vào "image"
    2. Nếu sử dụng video hoặc webcam thì ở tham số source truyền vào "video", "webcam" tương ứng
    3. Nếu sử dụng webcam thì file_path = 0
    4. Tham số confident dùng để tùy chỉnh ngưỡng tin cậy khi nhân diện khuân măt của mô hình
    """
    file_path = "/"
    # frame = cv2.imread(file_path)
    print(frame_process(file_path, confident= 0.1 , source= "image"))
    # frame_process(0, 0.5, source="webcam")
