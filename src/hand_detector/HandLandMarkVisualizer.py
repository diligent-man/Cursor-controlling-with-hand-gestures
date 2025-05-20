import cv2 as cv
import numpy as np
from typing import List, Tuple, Mapping


# legacy api
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import (
    DrawingSpec,
    get_default_hand_landmarks_style,
    get_default_hand_connections_style
)

# new api
from mediapipe.tasks.python.components.containers.category import Category


from ..utils.FPSCalculator import FPSCalculator
from .HandLandmarksConnections import HandLandmarksConnections


__all__ = ["HandLandMarkVisualizer"]


class HandLandMarkVisualizer(object):
    __FONT: int = cv.FONT_HERSHEY_PLAIN
    __FONT_SIZE: int = 1
    __FONT_THICKNESS: int = 1
    __LINE_TYPE: int = cv.LINE_AA

    __FPS_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 0)  # cyan, channel order: BGR

    __HANDEDNESS_MARGIN: int = 10
    __HANDEDNESS_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 255)  # red, channel order: BGR

    # Adapt new api for old draw_landmarks api
    __HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS = [
        (con.start, con.end) for con in HandLandmarksConnections.HAND_CONNECTIONS
    ]

    def __init__(self, legacy_mediapipe_api: bool = True) -> None:
        super(HandLandMarkVisualizer, self).__init__()
        self.__fps_calculator: FPSCalculator = FPSCalculator()
        self.__legacy_mediapipe_api: bool = legacy_mediapipe_api

    def __draw_handedness(self,
                          img: np.ndarray,
                          handedness: List[Category],
                          hand_landmarks: landmark_pb2.NormalizedLandmarkList
                          ):
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = img.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]

        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - self.__HANDEDNESS_MARGIN

        # Draw handedness (left or right hand) on the image.
        text: str = f"{handedness[0].category_name}"

        cv.putText(img,
                   text,
                   (text_x, text_y),
                   self.__FONT,
                   self.__FONT_SIZE,
                   self.__HANDEDNESS_TEXT_COLOR,
                   self.__FONT_THICKNESS,
                   self.__LINE_TYPE
                   )

    def __call__(self,
                 img: np.ndarray,
                 handedness_lst: List[List[Category]],
                 hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList],
                 include_fps: bool = True,
                 include_handedness: bool = True,
                 include_landmarks: bool = True,
                 *,
                 hand_landmarks_style: Mapping[int, DrawingSpec] = get_default_hand_landmarks_style(),
                 hand_connections_style: Mapping[Tuple[int, int], DrawingSpec] = get_default_hand_connections_style()
                 ) -> np.ndarray:
        if len(hand_landmarks_lst) > 0:
            # Loop through list of detected hands and landmarks.
            for hand_landmarks, handedness in zip(hand_landmarks_lst, handedness_lst):
                if include_landmarks:
                    draw_landmarks(img,
                                   hand_landmarks,
                                   self.__HAND_CONNECTIONS,
                                   hand_landmarks_style,
                                   hand_connections_style
                                   )

                if include_handedness:
                    self.__draw_handedness(img, handedness, hand_landmarks)

        # Draw fps
        if include_fps:
            cv.putText(img,
                       f"FPS: {int(self.__fps_calculator())}",
                       (int(img.shape[0] * .05), int(img.shape[1] * .05)),
                       self.__FONT,
                       self.__FONT_SIZE,
                       self.__FPS_TEXT_COLOR,
                       self.__FONT_THICKNESS,
                       self.__LINE_TYPE
                       )
        return img
