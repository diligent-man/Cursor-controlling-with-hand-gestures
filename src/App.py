import os
import sys
import time
import asyncio

from typing import Dict, List, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from screeninfo import Monitor

from pynput.mouse import Controller as mController
from pynput.keyboard import Controller as kbController

from mediapipe.tasks.python.vision.hand_landmarker import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode

from .utils.GlobalVar import gl

from .hand_detector import (
    HandDetector,
    HandDetectorResult,
    HandLandMarkVisualizer
)

from .utils import (
    display,
    draw_control_region,
    get_primary_monitor_info,
    denormalize_coord
)

__all__ = ["App"]


class App(object):
    __THUMB_X_OFFSET: float = gl.THUMB_X_OFFSET
    __THUMB_Y_OFFSET: float = gl.THUMB_Y_OFFSET

    __m_controller: mController = mController()
    __kb_controller: kbController = kbController()
    __monitor: Monitor = get_primary_monitor_info()

    __SUPPORT_INP_TYPES: Dict[str, VMode] = {
        "image": VMode.IMAGE,
        "video": VMode.VIDEO,
        "live_stream": VMode.LIVE_STREAM
    }

    __FINGER_TIP_IDX: np.ndarray[np.uint8] = np.array([
        v.value for k, v in HandLandmark.__members__.items() if k.endswith("TIP")
    ], dtype=np.uint8)

    __hand_detector: HandDetector
    __visualizer: HandLandMarkVisualizer

    def __init__(self,
                 hand_detector: HandDetector = None,
                 hand_visualizer: HandLandMarkVisualizer = None
                 ) -> None:
        super(App, self).__init__()

        if hand_detector is None:
            hand_detector = HandDetector(None, VMode.IMAGE, 2, is_mirrored=gl.IS_MIRRORED)
        else:
            assert isinstance(hand_detector, HandDetector), ValueError

        if hand_visualizer is None:
            hand_visualizer = HandLandMarkVisualizer()
        else:
            assert isinstance(hand_visualizer, HandLandMarkVisualizer)

        self.__hand_detector = hand_detector
        self.__hand_visualizer = hand_visualizer

    def get_detected_result(self) -> HandDetectorResult:
        assert self.__hand_detector.get_running_mode() == VMode.IMAGE, ValueError(
            "Currently support get detected result for image input")
        return self.__hand_detector.get_result()

    def _check_fingers_up(self,
                          handedness_lst: List[List[Category]],
                          hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList]
                          ) -> List[List[bool]]:
        """
        :param handedness_lst: contains hand type pred
        :param hand_landmarks_lst: contains each hand 21 detected landmarks. Each landmark is NormalizedLandmark obj
        :return: list of bool for each finger in corresponding detected hand.

        Assumption: Anterior (ventral) side of hand is facing with camera and posterior (dorsal) side is opposite.
        Landmark note: x (width), y (height), z

        Landmark ref: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png
        Check: /misc/finger_up_check.png
        """
        NormalizedLandmark = landmark_pb2.NormalizedLandmark

        result: List[List[bool]] = []
        for hand_landmarks, handedness in zip(hand_landmarks_lst, handedness_lst):
            hand_type: str = handedness[0].category_name

            tip_landmarks: Iterator[Tuple[int, NormalizedLandmark]] = filter(
                lambda x: x if x[0] in self.__FINGER_TIP_IDX else None, enumerate(hand_landmarks.landmark))
            tip_landmarks: Iterator[NormalizedLandmark] = map(lambda x: x[1], tip_landmarks)

            # except thumb, which is called MCP
            pip_landmarks: Iterator[Tuple[int, NormalizedLandmark]] = filter(
                lambda x: x if x[0] in self.__FINGER_TIP_IDX - 2 else None, enumerate(hand_landmarks.landmark))
            pip_landmarks: Iterator[NormalizedLandmark] = map(lambda x: x[1], pip_landmarks)

            is_up = [False] * len(self.__FINGER_TIP_IDX)
            for i, (tip_landmark, pip_landmark) in enumerate(zip(tip_landmarks, pip_landmarks)):
                if i == 0:
                    if (
                            hand_type == "Left" and
                            tip_landmark.x - self.__THUMB_X_OFFSET > pip_landmark.x and
                            tip_landmark.y + self.__THUMB_Y_OFFSET > pip_landmark.y
                    ) or (
                            hand_type == "Right" and
                            tip_landmark.x + self.__THUMB_X_OFFSET < pip_landmark.x and
                            tip_landmark.y + self.__THUMB_Y_OFFSET < pip_landmark.y
                    ):
                        is_up[i] = True
                else:
                    if tip_landmark.y < pip_landmark.y:
                        is_up[i] = True
            result.append(is_up)
        return result

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        if sys.version_info < (3, 10):
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        else:
            try:
                loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            except RuntimeError:
                loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        return loop

    async def _run(self, inp: str | int) -> None:
        cap: cv.VideoCapture = cv.VideoCapture(inp)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        executor: ThreadPoolExecutor = ThreadPoolExecutor()
        loop: asyncio.AbstractEventLoop = self._get_event_loop()
        while True:
            try:
                # bgr frame with 480x640x3 (H,W,C) shape
                success, frame = await loop.run_in_executor(executor, cap.read)

                if (not success) or cv.waitKey(1) == ord("q"):
                    break

                self.__hand_detector.detect(frame, int(time.time() * 1000))
                detected_result: HandDetectorResult = self.__hand_detector.get_result()

                finger_up_lst: List[List[bool]] = self._check_fingers_up(
                    detected_result.hand_landmarker_result.handedness,
                    detected_result.hand_landmarker_result.hand_landmarks
                )

                draw_control_region(detected_result.img, gl.CONTROL_REGION)

                #     gl.PREVIOUS_X, gl.PREVIOUS_Y = cursor_control(img, fingers, landmarks_lst,
                #                                                   gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                #                                                   new_d, new_h,
                #                                                   monitor.width, monitor.height,
                #                                                   gl.PREVIOUS_X, gl.PREVIOUS_Y, gl.SMOOTH_FACTOR,
                #                                                   kb_controller, m_controller
                #                                                   )

                frame = self.__hand_visualizer(detected_result.img,
                                               detected_result.hand_landmarker_result.handedness,
                                               detected_result.hand_landmarker_result.hand_landmarks,
                                               finger_up_lst
                                               )

                frame = cv.resize(frame,
                                  (int(self.__monitor.width * gl.SCALE_FACTOR),
                                   int(self.__monitor.height * gl.SCALE_FACTOR)),
                                  interpolation=cv.INTER_CUBIC
                                  )

                display(frame, gl.WINDOW_NAME)
            except Exception as e:
                print(e)

    async def run(self, inp_type: str, inp: str | int | np.ndarray = None) -> None:
        assert (inp_type in list(self.__SUPPORT_INP_TYPES.keys())) \
               and (self.__SUPPORT_INP_TYPES[inp_type] == self.__hand_detector.get_running_mode()), ValueError

        if inp_type == "image":
            self.__hand_detector.detect(inp)
        else:
            if inp_type == "video":
                assert inp is not None and os.path.isfile(inp), ValueError
            else:
                inp = 0

            await self._run(inp)
