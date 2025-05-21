import os
import pathlib
from queue import Queue


import cv2 as cv
import numpy as np

# new api
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)

from .HandDetectorResult import HandDetectorResult


__all__ = ["HandDetector"]

__package__ = pathlib.Path(__file__).parent.resolve()


class HandDetector(object):
    """
    Ref: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult
    Note: hand_world_landmarks from det result are ignored

                        Hand screen "keypoints" or "landmarks" interpretation
        Each hand per se contains 21 landmarks with (x, y, z) tuple. (x, y) is a normalized coordinate w.r.t an image/ frame.
    While normalized z is relative to wrist z-origin, if landmark has z > 0 then it is out of the page w.r.t wrist
    (0th landmark), else z < 0 when a landmark is into the page in regard to wrist.
    (see more at https://github.com/google-ai-edge/mediapipe/issues/742).

    More info about calculating z coord: https://github.com/google-ai-edge/mediapipe/issues/742#issuecomment-639104199

    To reconvert normalized coordinate into pixel-based location, we just respectively multiply x and y with image's width and height
    """
    __default_ckpt: str = os.path.join(__package__, "..", "models", "hand_landmarker.task")
    __finger_tip_idx = [0, 4, 8, 12, 16, 20]
    __is_mirrored: bool = True

    __result_queue: Queue[HandDetectorResult] = Queue()

    __opts: HandLandmarkerOptions
    __hand_detector: HandLandmarker

    def __init__(self,
                 base_options: HandLandmarkerOptions = None,
                 running_mode: VMode = VMode.IMAGE,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = .5,
                 min_hand_presence_confidence: float = .5,
                 min_tracking_confidence: float = 0.5,
                 return_fm: str = "bgr",
                 is_mirrored: bool = True
                 ):
        assert return_fm in ("rgb", "bgr"), ValueError
        super(HandDetector, self).__init__()
        self.__opts: HandLandmarkerOptions = HandLandmarkerOptions(
            base_options=base_options if base_options is not None else BaseOptions(self.__default_ckpt, None, 0),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self.__async_callback if running_mode in (VMode.LIVE_STREAM, ) else None
        )

        self.__hand_detector = HandLandmarker.create_from_options(self.__opts)
        self.__return_fm = return_fm
        self.__is_mirrored = is_mirrored

    def get_running_mode(self) -> VMode:
        return self.__opts.running_mode

    def get_result(self) -> HandDetectorResult:
        return self.__result_queue.get_nowait()

    def __async_callback(self, detected_result: HandLandmarkerResult, rgb_img: Image, timestamp_ms: int) -> None:
        rgb_img = np.copy(rgb_img.numpy_view())

        detected_result: HandDetectorResult = HandDetectorResult(
            detected_result,
            rgb_img if self.__return_fm == "rgb" else cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB),
            timestamp_ms,
            self.__is_mirrored
        )

        detected_result.__post_init__()
        self.__result_queue.put_nowait(detected_result)

    def detect(self, img: np.ndarray, timestamp_ms: int = None) -> None:
        """
        :param img: input image to perform detection upon
        :param timestamp_ms: frame timestamp
        :return: "bgr" | "rgb" image and detected corresponding landmarks in result_queue field,
                 which is retrieved via get_result() method
        """
        if self.__is_mirrored:
            img = cv.flip(img, 1)

        rgb_img: Image = Image(image_format=ImageFormat.SRGB, data=cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if self.__opts.running_mode in (VMode.IMAGE, VMode.VIDEO):
            if self.__opts.running_mode == VMode.IMAGE:
                detected_result: HandLandmarkerResult = self.__hand_detector.detect(rgb_img)
            else:
                assert timestamp_ms is not None, ValueError
                detected_result: HandLandmarkerResult = self.__hand_detector.detect_for_video(rgb_img, timestamp_ms)

            detected_result: HandDetectorResult = HandDetectorResult(
                detected_result,
                np.copy(rgb_img.numpy_view()) if self.__return_fm == "rgb" else img,
                timestamp_ms,
                self.__is_mirrored
            )
            detected_result.__post_init__()
            self.__result_queue.put_nowait(detected_result)
        else:
            assert timestamp_ms is not None, ValueError
            self.__hand_detector.detect_async(rgb_img, timestamp_ms)
