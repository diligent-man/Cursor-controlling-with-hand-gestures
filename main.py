import time


import cv2 as cv
import numpy as np

from screeninfo import Monitor
from dotenv import load_dotenv
from pynput.keyboard import Controller as kbController
from pynput.mouse import Controller as mController
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from src.utils import (
    GlobalVar,
    get_primary_monitor_info,
    get_screen_center_origin
)

from src.hand_detector import (
    HandDetector,
    HandDetectorResult,
    HandLandMarkVisualizer
)

load_dotenv("./.env")

from mediapipe.python.solutions.hands import Hands


def main() -> None:
    gl: GlobalVar = GlobalVar()
    monitor: Monitor = get_primary_monitor_info()

    m_controller: mController = mController()
    kb_controller: kbController = kbController()

    visualizer: HandLandMarkVisualizer = HandLandMarkVisualizer()
    detector: HandDetector = HandDetector(None, VMode.LIVE_STREAM, 1, is_mirrored=gl.IS_MIRRORED)

    cap: cv.VideoCapture = cv.VideoCapture(0)  # 640 x 480
    while cap.isOpened():
        _, frame = cap.read()

        detector.detect(frame, int(time.time() * 1000))
        detected_result: HandDetectorResult = detector.get_result()

        frame: np.ndarray = visualizer(
            detected_result.img,
            detected_result.hand_landmarker_result.handedness,
            detected_result.hand_landmarker_result.hand_landmarks,
        )

        new_d, new_h = int(gl.SCALE_FACTOR * monitor.width), int(gl.SCALE_FACTOR * monitor.height)
        frame = cv.resize(frame, (new_d, new_h), interpolation=cv.INTER_CUBIC)

        # landmarks_lst, bounding_box = detector.findPosition(img)  # Getting position of hand
        # print(landmarks_lst, bounding_box)

        # Checking if fingers are upwards
        # fingers, total_fingers = detector.fingersUp()

        # print(fingers)
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

        # Display
        cv.imshow(gl.WINDOW_NAME, frame)
        cv.moveWindow(gl.WINDOW_NAME, *get_screen_center_origin((frame.shape[1], frame.shape[0])))

        k = cv.waitKey(1)
        if k == ord("q"):
            exit()
    return None


if __name__ == "__main__":
    main()
