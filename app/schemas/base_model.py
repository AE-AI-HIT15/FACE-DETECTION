from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class APIInput(BaseModel):
    frame: Optional[List[List[List[int]]]] = None  # Image data (if a frame is provided)

    def to_numpy(self) -> Optional[np.ndarray]:
        """
        Convert the frame from a list to a numpy.ndarray (if available).

        Returns:
            Optional[np.ndarray]: The frame as a NumPy array, or None if no frame is provided.
        """
        if self.frame is not None:
            return np.array(self.frame, dtype=np.uint8)
        return None


class APIResponse(BaseModel):
    """
    Response model for face detection results.

    Attributes:
        bbox (List[List[int]]): A list of bounding boxes for detected faces.
        confident (List[float]): A list of confidence scores corresponding to each detected face.
    """
    bbox: List[List[int]]
    confident: List[float]

