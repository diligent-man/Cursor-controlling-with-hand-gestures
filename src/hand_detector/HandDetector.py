import os
import pathlib
from queue import Queue
from typing import List

import cv2 as cv
import numpy as np

# legacy api
from mediapipe.framework.formats import landmark_pb2

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
        return self.__result_queue.get()

    def __async_callback(self, detected_result: HandLandmarkerResult, rgb_img: Image, timestamp_ms: int) -> None:
        rgb_img = np.copy(rgb_img.numpy_view())

        detected_result: HandDetectorResult = HandDetectorResult(
            detected_result,
            rgb_img if self.__return_fm == "rgb" else cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB),
            timestamp_ms,
            self.__is_mirrored
        )

        detected_result.__post_init__()
        self.__result_queue.put(detected_result)

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
            self.__result_queue.put(detected_result)
        else:
            assert timestamp_ms is not None, ValueError
            self.__hand_detector.detect_async(rgb_img, timestamp_ms)

    # Checks which fingers are up
    def check_fingers_up(self, img: np.ndarray, hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList]):
        """
        :param img: input bgr image (W, H, C)
        :param hand_landmarks_lst:
        :return:

        Landmark ref: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png
        Landmark note: x (width), y (height), z
        """

        # cv.putText(frame, 'Fingers: ' + str(int(total_fingers)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2,
        #            cv.LINE_AA)
        for hand_landmarks in hand_landmarks_lst:
            # NormalizedLandmark object contains 21 landmarks for each hand
            tips = list(filter(lambda x: x if x[0] in self.__finger_tip_idx else None, enumerate(hand_landmarks.landmark)))
            tips = list(map(lambda x: (round(x[1].y, 3), round(x[1].z, 3)), tips))
            print(tips)

        # up_fingers: List = []

        # Thumb check
        # thumb_tip = self.__finger_tip_idx[0]
        # thumb_tip_y = self.landmarks_lst[thumb_tip][1]
        #
        # thumb_ip = self.__finger_tip_idx[0] - 1
        # thumb_ip_y = self.landmarks_lst[thumb_ip][1]
        #
        # if thumb_tip_y > thumb_ip_y:
        #     up_fingers.append(1)
        # else:
        #     up_fingers.append(0)
        #
        # # Rest of fingers check
        # for id in range(1, 5):
        #     # Lấy cách 2 đốt cho dễ nhận diện
        #     finger_tip = self.__finger_tip_idx[id]
        #     finger_tip_y_coord = self.landmarks_lst[finger_tip][2]
        #
        #     finger_pip = self.__finger_tip_idx[id] - 2
        #     finger_pip_y_coord = self.landmarks_lst[finger_pip][2]
        #
        #     if finger_tip_y_coord < finger_pip_y_coord:
        #         up_fingers.append(1)
        #     else:
        #         up_fingers.append(0)
        #
        # total_fingers = up_fingers.count(1)
        # return up_fingers, total_fingers
        return None, None

    # Find position of hand from input frame
    # def findPosition(self, img, handNo=0, draw=True):
    #     xList = []
    #     yList = []
    #     bounding_box = []
    #     self.landmarks_lst = []
    #
    #     if self.__det_result.multi_hand_landmarks:
    #         my_hand = self.__det_result.multi_hand_landmarks[handNo]
    #         for id, lm in enumerate(my_hand.landmark):
    #             # id: index
    #             # lm: normalized landmark coordinate (x.y) with x,y in [0,1]
    #             height, width, channels = img.shape
    #             # find centroid of an img
    #             cx, cy = int(lm.x * width), int(lm.y * height) # rescale proportional to the sesized resolution
    #             xList.append(cx)
    #             yList.append(cy)
    #             self.landmarks_lst.append([id, cx, cy])
    #             if draw:
    #                 pink = (255, 0, 255)
    #                 cv.circle(img, (cx, cy), 3, pink, cv.FILLED)
    #
    #         xmin, xmax = min(xList), max(xList)
    #         ymin, ymax = min(yList), max(yList)
    #         bounding_box = xmin, ymin, xmax, ymax
    #         if draw:
    #             cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
    #                           (0, 255, 0), 2)
    #     return self.landmarks_lst, bounding_box
