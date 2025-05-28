from typing import Tuple, List, Iterator, Iterable

import cv2 as cv
import numpy as np

# legacy api
from mediapipe.framework.formats import landmark_pb2


from .utils import get_screen_center_origin


__all__ = [
    "display",
    "denormalize_coord",
    "get_bbox_from_landmarks"
]


def denormalize_coord(img: np.ndarray,
                      landmark_lst: landmark_pb2.NormalizedLandmarkList
                      ) -> List[Tuple[int, int, float]]:
    """
    :param img: image for referring size
    :param landmark_lst: list of landmark(s) of detected object(s)
    :return: List of denormalized coord tuples
    """
    h, w = img.shape[:2]
    denormalized_landmark_lst: List[Tuple[int, int, float]] = []
    for i in range(len(landmark_lst.landmark)):
        denormalized_landmark: Tuple[int, int, float] = (
            int(round(w * landmark_lst.landmark[i].x)),
            int(round(h * landmark_lst.landmark[i].y)),
            landmark_lst.landmark[i].z
        )

        denormalized_landmark_lst.append(denormalized_landmark)
    return denormalized_landmark_lst


def get_bbox_from_landmarks(
        landmarks:
        landmark_pb2.NormalizedLandmarkList |
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

    cx_lst: List[float] = [landmark.x for landmark in landmarks]  # img width
    cy_lst: List[float] = [landmark.y for landmark in landmarks]  # img height

    bbox_coord: List[Tuple[int | float, int | float]] = []
    for op in ops:
        bbox_coord.append((op(cy_lst), op(cx_lst)))
    return bbox_coord


def display(frame: np.ndarray, winname: str, move_to_center: bool = True) -> None:
    cv.imshow(winname, frame)

    if move_to_center:
        cv.moveWindow(winname, *get_screen_center_origin((frame.shape[1], frame.shape[0])))
