import os
from queue import Queue

import cv2 as cv
import numpy as np


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


class HandDetector(object):
    """
    mp_drawing.draw_landmarks(
        image
        landmark_list: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: mediapipe.python.solutions.drawing_utils.DrawingSpec = DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        connection_drawing_spec: mediapipe.python.solutions.drawing_utils.DrawingSpec = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    )
    """
    __default_ckpt: str = os.path.join(os.getcwd(), "src", "models", "hand_landmarker.task")
    __finger_tip_idx = [4, 8, 12, 16, 20]

    __result_queue: Queue[HandDetectorResult] = Queue()

    __opts: HandLandmarkerOptions
    __hand_detector: HandLandmarker

    # Ref: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult
    # hand_world_landmarks from det result are ignored

    def __init__(self,
                 base_options: HandLandmarkerOptions = None,
                 running_mode: VMode = VMode.IMAGE,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = .5,
                 min_hand_presence_confidence: float = .5,
                 min_tracking_confidence: float = 0.5,
                 return_fm: str = "bgr"
                 ):
        assert return_fm in ("rgb", "bgr"), ValueError

        self.__opts: HandLandmarkerOptions = HandLandmarkerOptions(
            base_options=base_options if base_options is not None else BaseOptions(self.__default_ckpt, None, 0),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self.__async_callback if running_mode in (VMode.VIDEO, VMode.LIVE_STREAM) else None
        )

        self.__hand_detector = HandLandmarker.create_from_options(self.__opts)
        self.__return_fm = return_fm

    def get_result(self) -> HandDetectorResult:
        return self.__result_queue.get()

    def __async_callback(self, detected_result: HandLandmarkerResult, rgb_img: Image, timestamp_ms: int) -> None:
        rgb_img = np.copy(rgb_img.numpy_view())

        detected_result: HandDetectorResult = HandDetectorResult(
            detected_result,
            rgb_img if self.__return_fm == "rgb" else cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB),
            timestamp_ms
        )

        detected_result.__post_init__()
        self.__result_queue.put(detected_result)

    def detect(self, img: np.ndarray, timestamp_ms: int = None) -> None:
        rgb_img: Image = Image(image_format=ImageFormat.SRGB, data=cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if self.__opts.running_mode in (VMode.IMAGE, VMode.VIDEO):
            if VMode == VMode.IMAGE:
                detected_result: HandLandmarkerResult = self.__hand_detector.detect(rgb_img)
            else:
                assert timestamp_ms is not None, ValueError
                detected_result: HandLandmarkerResult = self.__hand_detector.detect_for_video(rgb_img, timestamp_ms)

            detected_result: HandDetectorResult = HandDetectorResult(
                detected_result,
                np.copy(rgb_img.numpy_view()) if self.__return_fm == "rgb" else img,
                timestamp_ms
            )
            detected_result.__post_init__()
            self.__result_queue.put(detected_result)
        else:
            assert timestamp_ms is not None, ValueError
            self.__hand_detector.detect_async(rgb_img, timestamp_ms)

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

    # Checks which fingers are up
    # def fingersUp(self):
    #     # Landmark ref: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png
    #     fingers = []
    #     # Thumb check
    #     thumb_tip = self.__finger_tip_idx[0]
    #     thumb_tip_y = self.landmarks_lst[thumb_tip][1]
    #
    #     thumb_ip = self.__finger_tip_idx[0] - 1
    #     thumb_ip_y = self.landmarks_lst[thumb_ip][1]
    #
    #     if thumb_tip_y > thumb_ip_y:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    #
    #     # Rest of fingers check
    #     for id in range(1, 5):
    #         # Lấy cách 2 đốt cho dễ nhận diện
    #         finger_tip = self.__finger_tip_idx[id]
    #         finger_tip_y_coord = self.landmarks_lst[finger_tip][2]
    #
    #         finger_pip = self.__finger_tip_idx[id] - 2
    #         finger_pip_y_coord = self.landmarks_lst[finger_pip][2]
    #
    #         if finger_tip_y_coord < finger_pip_y_coord:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #
    #     total_fingers = fingers.count(1)
    #     return fingers, total_fingers
