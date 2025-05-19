from typing import List
from dataclasses import dataclass

import numpy as np

# legacy api
from mediapipe.framework.formats import landmark_pb2

# new api
from mediapipe import Image
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


__all__ = ["HandDetectorResult"]


@dataclass
class HandDetectorResult:
    """
    Wrap detected result from HandDetector class.
    """
    hand_landmarker_result: HandLandmarkerResult = None
    img: Image | np.ndarray = None
    ts: int = None

    def __init__(self,
                 result: HandLandmarkerResult = None,
                 img: np.ndarray = None,
                 ts: int = None,
                 to_normalized_landmark_lst: bool = True
                 ) -> None:
        self.hand_landmarker_result = result
        self.img = img
        self.ts = ts
        self.__to_normalized_landmark_lst: bool = to_normalized_landmark_lst

    def __post_init__(self) -> None:
        if self.__to_normalized_landmark_lst and self.hand_landmarker_result is not None:
            self._to_normalized_landmark_lst()

    def _to_normalized_landmark_lst(self) -> None:
        """
        convert List[List[landmark_module.NormalizedLandmark]] into List[NormalizedLandmarkList]
        for using legacy drawing_utils. Check src/utils/drawing_utils.py

        src of NormalizedLandmarkList(): https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/formats/landmark.proto
        Note: visibility, presence fields are also ignored cuz it belongs to legacy api
        """
        pb2_NormalizedLandmark: landmark_pb2.NormalizedLandmark = landmark_pb2.NormalizedLandmark
        pb2_NormalizedLandmarkList: landmark_pb2.NormalizedLandmarkList = landmark_pb2.NormalizedLandmarkList

        hand_landmarks_lst: List[List[NormalizedLandmark]] = self.hand_landmarker_result.hand_landmarks
        converted_hand_landmarks_lst: List[pb2_NormalizedLandmarkList] = []

        for hand_landmarks in hand_landmarks_lst:
            converted_hand_landmarks: pb2_NormalizedLandmarkList = pb2_NormalizedLandmarkList()
            converted_hand_landmarks.landmark.extend(
                [
                    pb2_NormalizedLandmark(
                        x=landmarks.x,
                        y=landmarks.y,
                        z=landmarks.z,
                    ) for landmarks in hand_landmarks
                ]
            )
            converted_hand_landmarks_lst.append(converted_hand_landmarks)
        self.hand_landmarker_result.hand_landmarks = converted_hand_landmarks_lst
