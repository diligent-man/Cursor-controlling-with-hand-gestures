import asyncio
import os
from concurrent.futures import ThreadPoolExecutor


import cv2 as cv
import mediapipe
import numpy as np

from queue import Queue
from typing import Tuple, List, Dict


from dotenv import load_dotenv
from pynput.mouse import Controller as mController
from pynput.keyboard import Controller as kbController
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from .utils import GlobalVar, get_primary_monitor_info, get_screen_center_origin
from .hand_detector import HandDetector, HandLandMarkVisualizer, HandDetectorResult


__all__ = ["App"]


class App(object):
    __m_controller: mController = mController()
    __kb_controller: kbController = kbController()
    __read_frame_queue: Queue[np.ndarray] = Queue()
    __detected_queue: Queue[HandDetectorResult] = Queue()

    __support_inp_types: Dict[str, VMode] = {
        "image": VMode.IMAGE,
        "video": VMode.VIDEO,
        "live_stream": VMode.LIVE_STREAM
    }

    __gl: GlobalVar
    __hand_detector: HandDetector
    __visualizer: HandLandMarkVisualizer

    def __init__(self,
                 hand_detector: HandDetector = None,
                 hand_visualizer: HandLandMarkVisualizer = None
                 ) -> None:
        super(App, self).__init__()

        if hand_detector is None:
            hand_detector = HandDetector(None, VMode.IMAGE, 2, is_mirrored=self.__gl.IS_MIRRORED)
        else:
            assert isinstance(hand_detector, HandDetector), ValueError

        if hand_visualizer is None:
            hand_visualizer = HandLandMarkVisualizer()
        else:
            assert isinstance(hand_visualizer, HandLandMarkVisualizer)

        self.__gl = globals().get("gl", GlobalVar())
        self.__hand_detector = hand_detector
        self.__hand_visualizer = hand_visualizer

    async def __read_frame(self, inp: str | int) -> None:
        cap: cv.VideoCapture = cv.VideoCapture(inp)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        executor = ThreadPoolExecutor()
        loop = asyncio.get_event_loop()

        while True:
            try:
                # bgr frame with 480x640x3 (W,H,C) shape
                success, frame = await loop.run_in_executor(executor, cap.read)

                if (not success) or cv.waitKey(1) == ord("q"):
                    break

                # detector.detect(frame, int(time.time() * 1000))
                # detected_result: HandDetectorResult = detector.get_result()

                # Check which fingers are up
                # fingers, total_fingers = detector.check_fingers_up(
                #     detected_result.img,
                #     detected_result.hand_landmarker_result.hand_landmarks
                # )

                # frame: np.ndarray = visualizer(
                #     detected_result.img,
                #     detected_result.hand_landmarker_result.handedness,
                #     detected_result.hand_landmarker_result.hand_landmarks,
                # )

                # new_d, new_h = int(gl.SCALE_FACTOR * monitor.width), int(gl.SCALE_FACTOR * monitor.height)
                # frame = cv.resize(frame, (new_d, new_h), interpolation=cv.INTER_CUBIC)

                # landmarks_lst, bounding_box = detector.findPosition(img)  # Getting position of hand
                # print(landmarks_lst, bounding_box)

                #
                #     # Draw control region cursor
                #     control_region(img,
                #                    gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                #                    monitor.width, monitor.height
                #                    )
                #
                #     gl.PREVIOUS_X, gl.PREVIOUS_Y = cursor_control(img, fingers, landmarks_lst,
                #                                                   gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                #                                                   new_d, new_h,
                #                                                   monitor.width, monitor.height,
                #                                                   gl.PREVIOUS_X, gl.PREVIOUS_Y, gl.SMOOTHEN_FACTOR,
                #                                                   kb_controller, m_controller
                #                                                   )

                self.__display(frame)
            except Exception as e:
                print(e)

    def __display(self, frame: np.ndarray):
        cv.imshow(self.__gl.WINDOW_NAME, frame)
        cv.moveWindow(self.__gl.WINDOW_NAME, *get_screen_center_origin((frame.shape[1], frame.shape[0])))

    async def run(self, inp_type: str, inp: str | int | np.ndarray = None) -> None:
        assert (inp_type in list(self.__support_inp_types.keys()))\
            and (self.__support_inp_types[inp_type] == self.__hand_detector.get_running_mode()), ValueError

        if inp_type == "image":
            self.__hand_detector.detect(inp)
            self.__detected_queue.put(self.__hand_detector.get_result())
        else:
            if inp_type == "video":
                assert inp is not None and os.path.isfile(inp), ValueError
            else:
                inp = 0

            await self.__read_frame(inp)
