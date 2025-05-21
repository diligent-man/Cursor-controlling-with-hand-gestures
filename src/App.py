import os
import sys
import time
import asyncio

from queue import Queue
from typing import Dict
from concurrent.futures import ThreadPoolExecutor


import cv2 as cv
import numpy as np

from pynput.mouse import Controller as mController
from pynput.keyboard import Controller as kbController
from mediapipe.tasks.python.vision.hand_landmarker import landmark_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from .utils import GlobalVar, get_screen_center_origin
from .hand_detector import HandDetector, HandLandMarkVisualizer, HandDetectorResult


__all__ = ["App"]


class App(object):
    __m_controller: mController = mController()
    __kb_controller: kbController = kbController()
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

    def get_detected_result(self) -> HandDetectorResult:
        assert self.__hand_detector.get_running_mode() == VMode.IMAGE, ValueError("Currently support get detected result for image input")
        return self.__hand_detector.get_result()

    # Checks which fingers are up
    # def check_fingers_up(self, img: np.ndarray, hand_landmarks_lst: List[landmark_pb2.NormalizedLandmarkList]):
    #     """
    #     :param img: input bgr image (W, H, C)
    #     :param hand_landmarks_lst:
    #     :return:
    #
    #     Landmark ref: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png
    #     Landmark note: x (width), y (height), z
    #     """
    #
    #     # cv.putText(frame, 'Fingers: ' + str(int(total_fingers)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2,
    #     #            cv.LINE_AA)
    #     for hand_landmarks in hand_landmarks_lst:
    #         # NormalizedLandmark object contains 21 landmarks for each hand
    #         tips = list(
    #             filter(lambda x: x if x[0] in self.__finger_tip_idx else None, enumerate(hand_landmarks.landmark)))
    #         tips = list(map(lambda x: (round(x[1].y, 3), round(x[1].z, 3)), tips))
    #         print(tips)
    #
    #     # up_fingers: List = []
    #
    #     # Thumb check
    #     # thumb_tip = self.__finger_tip_idx[0]
    #     # thumb_tip_y = self.landmarks_lst[thumb_tip][1]
    #     #
    #     # thumb_ip = self.__finger_tip_idx[0] - 1
    #     # thumb_ip_y = self.landmarks_lst[thumb_ip][1]
    #     #
    #     # if thumb_tip_y > thumb_ip_y:
    #     #     up_fingers.append(1)
    #     # else:
    #     #     up_fingers.append(0)
    #     #
    #     # # Rest of fingers check
    #     # for id in range(1, 5):
    #     #     # Lấy cách 2 đốt cho dễ nhận diện
    #     #     finger_tip = self.__finger_tip_idx[id]
    #     #     finger_tip_y_coord = self.landmarks_lst[finger_tip][2]
    #     #
    #     #     finger_pip = self.__finger_tip_idx[id] - 2
    #     #     finger_pip_y_coord = self.landmarks_lst[finger_pip][2]
    #     #
    #     #     if finger_tip_y_coord < finger_pip_y_coord:
    #     #         up_fingers.append(1)
    #     #     else:
    #     #         up_fingers.append(0)
    #     #
    #     # total_fingers = up_fingers.count(1)
    #     # return up_fingers, total_fingers
    #     return None, None

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

    @staticmethod
    def __get_event_loop() -> asyncio.AbstractEventLoop:
        if sys.version_info < (3, 10):
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        else:
            try:
                loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            except RuntimeError:
                loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        return loop

    def __display(self, frame: np.ndarray) -> None:
        cv.imshow(self.__gl.WINDOW_NAME, frame)
        cv.moveWindow(self.__gl.WINDOW_NAME, *get_screen_center_origin((frame.shape[1], frame.shape[0])))

    async def __read_frame(self, inp: str | int) -> None:
        cap: cv.VideoCapture = cv.VideoCapture(inp)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
        loop: asyncio.AbstractEventLoop = self.__get_event_loop()

        while True:
            try:
                # bgr frame with 480x640x3 (W,H,C) shape
                success, frame = await loop.run_in_executor(executor, cap.read)

                if (not success) or cv.waitKey(1) == ord("q"):
                    break

                self.__hand_detector.detect(frame, int(time.time() * 1000))
                detected_result: HandDetectorResult = self.__hand_detector.get_result()

                # Check which fingers are up
                # fingers, total_fingers = detector.check_fingers_up(
                #     detected_result.img,
                #     detected_result.hand_landmarker_result.hand_landmarks
                # )

                frame: np.ndarray = self.__hand_visualizer(
                    detected_result.img,
                    detected_result.hand_landmarker_result.handedness,
                    detected_result.hand_landmarker_result.hand_landmarks,
                )

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

    async def run(self, inp_type: str, inp: str | int | np.ndarray = None) -> None:
        assert (inp_type in list(self.__support_inp_types.keys()))\
            and (self.__support_inp_types[inp_type] == self.__hand_detector.get_running_mode()), ValueError

        if inp_type == "image":
            self.__hand_detector.detect(inp)
        else:
            if inp_type == "video":
                assert inp is not None and os.path.isfile(inp), ValueError
            else:
                inp = 0

            await self.__read_frame(inp)
