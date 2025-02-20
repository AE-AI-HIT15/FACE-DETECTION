from fastapi import APIRouter, FastAPI, Query
from starlette.responses import StreamingResponse
from app.domain.face_detector import image_process, video_process
from app.schemas.base_model import APIResponse, APIInput

try:
    router = APIRouter()
except Exception as e:
    print(e)
    print("Error")


@router.post("/image_process")
async def process(input: APIInput) -> APIResponse:
    """
    Process a single frame image for face detection.

    Args:
        input (APIInput): The input object containing image data.

    Returns:
        APIResponse: An object containing a list of bounding boxes and confidence scores.
    """
    image = input.to_numpy()  # Convert frame to numpy
    bboxes, conf_scores = image_process(image)
    result = APIResponse(bbox=bboxes, confident=conf_scores)  # Process image
    return result


@router.get("/video_process")
async def video_feed(
        path: str = Query(default="0", description="Video path or '0' to open the camera")
) -> StreamingResponse:
    """
    Process video stream and detect faces in real-time.

    Args:
        path (str, optional): The video path or '0' to use the webcam. Default is "0".

    Returns:
        StreamingResponse: A streaming response with multipart/x-mixed-replace format.
    """
    return StreamingResponse(video_process(path), media_type="multipart/x-mixed-replace; boundary=frame")
