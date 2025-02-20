import json
import os
import tempfile
import requests
import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Streamlit UI
st.set_page_config(page_title="Webcam App", layout="wide")
st.title("Nhận diện Khuôn mặt")

# Layout với 3 cột
col1, col2, col3 = st.columns([1, 3, 1])

# Biến trạng thái
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Cột 1: Upload ảnh / video
with col1:
    st.header("Upload Ảnh / Video")
    path = ""
    image_file = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])
    if image_file is not None:

        if image_file is not None:
            # Tạo file tạm thời
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, image_file.name)

            # Lưu file vào đường dẫn tạm thời
            with open(temp_path, "wb") as f:
                f.write(image_file.read())

            path = temp_path
        img_cv = cv2.imread(path)
        img_list = img_cv.tolist()

        url = "http://127.0.0.1:8000/image_process"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"frame": img_list})

        response = requests.post(url, headers=headers, data=payload)
        bboxes = response.json()["bbox"]
        confs = response.json()["confident"]
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        for (bbox, conf) in zip(bboxes, confs):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        col2.image(img_cv, caption="Ảnh với khuôn mặt đã nhận diện", use_container_width=True)
        col2.write(f"**Số mặt phát hiện:** {len(bboxes)}")


    video_file = st.file_uploader("Tải lên video", type=["mp4", "avi", "mov"])
    path = ""
    if video_file is not None:
        # Tạo file tạm thời
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, video_file.name)

        # Lưu file vào đường dẫn tạm thời
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        path = temp_path

        VIDEO_URL = f"http://localhost:8000/video_process?path={path}"
        with col2:
            st.title("Live Video Stream")

            # Tạo khung hiển thị
            frame_placeholder = st.empty()

            # Đọc video từ stream
            cap = cv2.VideoCapture(VIDEO_URL)

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.write("Không thể nhận video.")
                    break

                # Chuyển frame sang RGB để hiển thị trên Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Cập nhật hình ảnh trong Streamlit
                frame_placeholder.image(frame, channels="RGB")
            cap.release()

# Cột 3: Nút START & STOP
with col3:
    if st.button("START"):
        st.session_state.webcam_active = True
        VIDEO_URL = f"http://localhost:8000/video_process?path={0}"
        # Cột 2: Hiển thị camera
        with col2:
            st.title("Live Video Stream")

            # Tạo khung hiển thị
            frame_placeholder = st.empty()

            # Đọc video từ stream
            cap = cv2.VideoCapture(VIDEO_URL)

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.write("Không thể nhận video.")
                    break

                # Chuyển frame sang RGB để hiển thị trên Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Cập nhật hình ảnh trong Streamlit
                frame_placeholder.image(frame, channels="RGB")
            cap.release()
        cv2.destroyAllWindows()
    if st.button("STOP"):
        st.session_state.webcam_active = False