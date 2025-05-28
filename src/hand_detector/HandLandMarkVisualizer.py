import cv2 as cv
import numpy as np

from typing import List, Tuple, Mapping, Iterator

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


from ..utils import get_bbox_from_landmarks

from ..utils.GlobalVar import gl
from ..utils.FPSCalculator import FPSCalculator

from .HandLandmarksConnections import HandLandmarksConnections


__all__ = ["HandLandMarkVisualizer"]


class HandLandMarkVisualizer(object):
    __HANDEDNESS_FONT: int = gl.HANDEDNESS_FONT
    __HANDEDNESS_FONT_SIZE: int = gl.HANDEDNESS_FONT_SIZE
    __HANDEDNESS_TEXT_COLOR: Tuple[int, int, int] = gl.HANDEDNESS_TEXT_COLOR
    __HANDEDNESS_FONT_THICKNESS: int = gl.HANDEDNESS_FONT_THICKNESS
    __HANDEDNESS_LINE_TYPE: int = gl.HANDEDNESS_LINE_TYPE

    __HAND_BBOX_COLOR: Tuple[int, int, int] = gl.HAND_BBOX_COLOR
    __HAND_BBOX_THICKNESS = gl.HAND_BBOX_THICKNESS
    __HAND_BBOX_LINE_TYPE = gl.HAND_BBOX_LINE_TYPE

    __PALM_BBOX_COLOR: Tuple[int, int, int] = gl.PALM_BBOX_COLOR
    __PALM_BBOX_THICKNESS = gl.PALM_BBOX_THICKNESS
    __PALM_BBOX_LINE_TYPE = gl.PALM_BBOX_LINE_TYPE

    __FINGER_UP_ORIG: Tuple[float, float] = gl.FINGER_UP_ORIG
    __FINGER_UP_FONT: int = gl.FINGER_UP_FONT
    __FINGER_UP_FONT_SIZE: int = gl.FINGER_UP_FONT_SIZE
    __FINGER_UP_TEXT_COLOR: Tuple[int, int, int] = gl.FINGER_UP_TEXT_COLOR
    __FINGER_UP_FONT_THICKNESS: int = gl.FINGER_UP_FONT_THICKNESS
    __FINGER_UP_LINE_TYPE: int = gl.FINGER_UP_LINE_TYPE

    __FPS_ORIG: Tuple[float, float] = gl.FPS_ORIG
    __FPS_FONT: int = gl.FPS_FONT
    __FPS_FONT_SIZE: int = gl.FPS_FONT_SIZE
    __FPS_TEXT_COLOR: Tuple[int, int, int] = gl.FPS_TEXT_COLOR
    __FPS_FONT_THICKNESS: int = gl.FPS_FONT_THICKNESS
    __FPS_LINE_TYPE: int = gl.FPS_LINE_TYPE

    __BBOX_MARGIN: int = gl.BBOX_MARGIN
    __PALM_MARGIN: int = gl.PALM_MARGIN

    __fps_calculator: FPSCalculator = FPSCalculator()

    # Adapt new api for old draw_landmarks api
    __HAND_CONNECTIONS: List[Tuple[int, int]] = [
        (con.start, con.end) for con in HandLandmarksConnections.HAND_CONNECTIONS
    ]

    __PALM_IDX: List[int] = [0, 1, 5, 9, 13, 17]

    def __init__(self,
                 legacy_mediapipe_api: bool = True,
                 include_fps: bool = True,
                 include_landmarks: bool = True,
                 include_hand_bbox: bool = True,
                 include_palm_bbox: bool = True,
                 include_finger_up: bool = True,
                 include_handedness: bool = True,
                 ) -> None:
        super(HandLandMarkVisualizer, self).__init__()
        
        self.__legacy_mediapipe_api: bool = legacy_mediapipe_api
        self.__include_fps: bool = include_fps
        self.__include_landmarks: bool = include_landmarks
        self.__include_hand_bbox: bool = include_hand_bbox
        self.__include_palm_bbox: bool = include_palm_bbox
        self.__include_finger_up: bool = include_finger_up
        self.__include_handedness: bool = include_handedness

    def __call__(self,
                 img: np.ndarray,
                 handedness_lst: List[List[Category]],
                 hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList],
                 finger_up_lst: List[List[bool]] = None,
                 *,
                 hand_landmarks_style: Mapping[int, DrawingSpec] = get_default_hand_landmarks_style(),
                 hand_connections_style: Mapping[Tuple[int, int], DrawingSpec] = get_default_hand_connections_style()
                 ) -> np.ndarray:
        h, w, _ = img.shape
        NormalizedLandmark = landmark_pb2.NormalizedLandmark

        if len(hand_landmarks_lst) > 0:
            # Loop through list of detected hands and landmarks.
            for hand_landmarks, handedness in zip(hand_landmarks_lst, handedness_lst):
                if self.__include_landmarks:
                    draw_landmarks(img,
                                   hand_landmarks,
                                   self.__HAND_CONNECTIONS,
                                   hand_landmarks_style,
                                   hand_connections_style
                                   )

                if self.__include_handedness or self.__include_hand_bbox:
                    hand_bbox: List[Tuple, Tuple] = [
                        (int(x * w), int(y * h)) for y, x in get_bbox_from_landmarks(hand_landmarks)
                    ]

                    if self.__include_handedness:
                        cv.putText(img,
                                   f"{handedness[0].category_name}",
                                   (hand_bbox[0][0] - self.__BBOX_MARGIN, hand_bbox[0][1] - self.__BBOX_MARGIN),
                                   self.__HANDEDNESS_FONT, self.__HANDEDNESS_FONT_SIZE,
                                   self.__HANDEDNESS_TEXT_COLOR, self.__HANDEDNESS_FONT_THICKNESS,
                                   self.__HANDEDNESS_LINE_TYPE
                                   )

                    if self.__include_hand_bbox:
                        cv.rectangle(img,
                                     (hand_bbox[0][0] - self.__BBOX_MARGIN, hand_bbox[0][1] - self.__BBOX_MARGIN),
                                     (hand_bbox[1][0] + self.__BBOX_MARGIN, hand_bbox[1][1] + self.__BBOX_MARGIN),
                                     self.__HAND_BBOX_COLOR,
                                     self.__HAND_BBOX_THICKNESS,
                                     self.__HAND_BBOX_LINE_TYPE
                                     )

                if self.__include_palm_bbox:
                    palm_landmarks: Iterator[Tuple[int, NormalizedLandmark]] = filter(
                        lambda x: x if x[0] in self.__PALM_IDX else None, enumerate(hand_landmarks.landmark))
                    palm_landmarks: Iterator[NormalizedLandmark] = map(lambda x: x[1], palm_landmarks)

                    palm_bbox: List[Tuple, Tuple] = get_bbox_from_landmarks(palm_landmarks)
                    palm_bbox = [(int(x * w), int(y * h)) for y, x in palm_bbox]

                    cv.rectangle(img,
                                 (palm_bbox[0][0] + self.__BBOX_MARGIN, palm_bbox[0][1] + self.__BBOX_MARGIN),
                                 (palm_bbox[1][0] - self.__BBOX_MARGIN, palm_bbox[1][1] - self.__BBOX_MARGIN),
                                 self.__PALM_BBOX_COLOR,
                                 self.__PALM_BBOX_THICKNESS,
                                 self.__PALM_BBOX_LINE_TYPE
                                 )

        if self.__include_fps:
            cv.putText(img,
                       f"FPS: {int(self.__fps_calculator())}",
                       (int(w * self.__FPS_ORIG[0]), int(h * self.__FPS_ORIG[1])),
                       self.__FPS_FONT,
                       self.__FPS_FONT_SIZE,
                       self.__FPS_TEXT_COLOR,
                       self.__FPS_FONT_THICKNESS,
                       self.__FPS_LINE_TYPE
                       )

        if self.__include_finger_up:
            if finger_up_lst is None:
                txt: str = "Fingers: None"
            else:
                total_fingers = sum([sum(is_up_fingers) for is_up_fingers in finger_up_lst])
                txt: str = f"Fingers: {total_fingers}"

            cv.putText(img,
                       txt,
                       (int(w * self.__FINGER_UP_ORIG[0]), int(h * self.__FINGER_UP_ORIG[1])),
                       self.__FINGER_UP_FONT,
                       self.__FINGER_UP_FONT_SIZE,
                       self.__FINGER_UP_TEXT_COLOR,
                       self.__FINGER_UP_FONT_THICKNESS,
                       self.__FINGER_UP_LINE_TYPE
                       )
        return img
