import cv2 as cv
import numpy as np

from typing import List

# legacy api
from mediapipe.python import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import draw_landmarks

# new api
from mediapipe.tasks.python.components.containers.category import Category

from ..utils.FPSCalculator import FPSCalculator


__all__ = ["HandLandMarkVisualizer"]


class HandLandMarkVisualizer(object):
    def __init__(self, legacy_mediapipe_api: bool = True):
        self.__fps_calculator = FPSCalculator()
        self.__legacy_mediapipe_api = legacy_mediapipe_api

    def __call__(self,
                 img: np.ndarray,
                 handedness_lst: List[List[Category]],
                 hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList]
                 ):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        if len(hand_landmarks_lst) > 0:
            # Loop through list of detected hands and landmarks.
            for hand_landmarks, handedness in zip(hand_landmarks_lst, handedness_lst):
                draw_landmarks(
                    img,
                    hand_landmarks,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = img.shape

                x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
                y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]

                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                img = cv.putText(img, f"{handedness[0].category_name}",
                                 (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA
                                 )

        # Draw fps
        cv.putText(img, f"FPS: {int(self.__fps_calculator())}", (20, 45),
                   cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)
        return img
