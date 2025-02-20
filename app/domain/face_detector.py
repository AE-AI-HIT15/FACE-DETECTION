import time
import cv2
import sys
import os
from typing import List, Tuple, Union, Generator
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.configs import create_model, CONFIDENCE_THRESHOLD, FRAME_SKIP, DEVICE, STREAM, IMAGE_SIZE

try:
    model = create_model()
except Exception as e:
    logging.error(e)

def image_process(input: np.ndarray, confident: float = CONFIDENCE_THRESHOLD, stream=STREAM, device=DEVICE, imgsz = IMAGE_SIZE) -> Tuple[List[List[int]], List[float]]:

    """
    Process an image and return bounding boxes with confidence scores.

    :param input: Input image (NumPy array)
    :param confident: Confidence threshold for detection
    :return: List of bounding boxes and their confidence scores
    """
    # predict result from yolov8-face
    predict = model.predict(input, conf=confident, stream=STREAM, device=DEVICE, imgsz=IMAGE_SIZE)
    #  List store face after detect
    res = []    
    conf_scores = []  # List store confidence score of face

    for result in predict:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):  # Take bbox and confidence scores
            x1, y1, x2, y2 = map(int, box[:4])
            res.append([x1, y1, x2, y2])
            conf_scores.append(float(conf)) 

    return res, conf_scores  # Return list of face and confidence scores

def video_process(input: str) -> Generator[bytes, None, None]:
    """
    Processes a video stream from a file path or webcam and generates a sequence of JPEG-encoded frames.

    :param input: Video source, either a file path (str) or "0" (str) for the webcam.
    :return: A generator yielding JPEG-encoded video frames as bytes.
    """

    frame_count = 0
    cap = cv2.VideoCapture(int(input)) if input == "0" else cv2.VideoCapture(input)
    fps_time = time.time()

    while cap.isOpened():
        start_time = time.time() # Store the time before processing

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        # Process the frame (bounding boxes and confidence scores)
        bboxes, confs = image_process(frame, confident=CONFIDENCE_THRESHOLD, stream=STREAM, device=DEVICE, imgsz=IMAGE_SIZE )

        for (bbox, conf) in zip(bboxes, confs):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps = 1 / (time.time() - fps_time) # Computing FPS based on the actual frame processing time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.waitKey(33)

        # Encode frame as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = jpeg.tobytes()

        # Yield the frame as an HTTP MJPEG response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

