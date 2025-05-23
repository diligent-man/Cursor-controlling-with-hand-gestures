from typing import Tuple, List, Iterator, Iterable

import cv2 as cv
import numpy as np

# legacy api
from mediapipe.framework.formats import landmark_pb2


__all__ = [
    "get_bbox_from_landmarks"
]


def get_bbox_from_landmarks(
        landmarks: landmark_pb2.NormalizedLandmarkList |
                   Iterator[landmark_pb2.NormalizedLandmark] |
                   Iterable[landmark_pb2.NormalizedLandmark]
) -> List[Tuple[int | float, int | float]]:
    """
    :param landmarks:
    :return: (top_left(int, int), bottom_right(int, int)) with shape (H,W)
    """
    ops: Tuple = (min, max)

    if isinstance(landmarks, landmark_pb2.NormalizedLandmarkList):
        landmarks: List[landmark_pb2.NormalizedLandmark] = list(landmarks.landmark)
    elif isinstance(landmarks, Iterator):
        landmarks: List[landmark_pb2.NormalizedLandmark] = list(landmarks)

    x_coordinates: List[float] = [landmark.x for landmark in landmarks]  # img width
    y_coordinates: List[float] = [landmark.y for landmark in landmarks]  # img height

    bbox_coord: List[Tuple[int | float, int | float]] = []
    for op in ops:
        bbox_coord.append((op(y_coordinates), op(x_coordinates)))
    return bbox_coord
