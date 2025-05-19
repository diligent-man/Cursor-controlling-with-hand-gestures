import cv2 as cv

from screeninfo import Monitor
from dotenv import load_dotenv

from pynput.keyboard import Controller as kbController
from pynput.mouse import Controller as mController


from src.utils import *
from src.utils.HandDetector import HandDetector
from src.utils.FPSCalculator import FPSCalculator


load_dotenv("./.env")


def main() -> None:
    gl: GlobalVar = GlobalVar()
    monitor: Monitor = get_primary_monitor_info()

    m_controller: mController = mController()
    kb_controller: kbController = kbController()
    fps_calculator: FPSCalculator = FPSCalculator()

    cap: cv.VideoCapture = cv.VideoCapture(0)  # default resolution: 640 x 480 -> resize after
    detector: HandDetector = HandDetector(max_num_hands=1)

    while cap.isOpened():
        _, img = cap.read()

        new_d, new_h = int(gl.SCALE_FACTOR * monitor.width), int(gl.SCALE_FACTOR * monitor.height)

        img = cv.flip(img, 1)
        img = cv.resize(img, (new_d, new_h), interpolation=cv.INTER_CUBIC)
        img = detector.detect(img)

        landmarks_lst, bounding_box = detector.findPosition(img)  # Getting position of hand
        # print(landmarks_lst, bounding_box)

        if len(landmarks_lst) != 0:
            # Checking if fingers are upwards
            fingers, total_fingers = detector.fingersUp()
            cv.putText(img, 'Fingers: ' + str(int(total_fingers)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2,
                       cv.LINE_AA)
            # print(fingers)

            # Draw control region cursor
            control_region(img,
                           gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                           monitor.width, monitor.height
                           )

            gl.PREVIOUS_X, gl.PREVIOUS_Y = cursor_control(img, fingers, landmarks_lst,
                                                          gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                                                          new_d, new_h,
                                                          monitor.width, monitor.height,
                                                          gl.PREVIOUS_X, gl.PREVIOUS_Y, gl.SMOOTHEN_FACTOR,
                                                          kb_controller, m_controller
                                                          )

        # Compute fps
        cv.putText(img, f"FPS: {int(fps_calculator())}", (20, 45), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)

        # Display
        cv.imshow("Image", img)

        k = cv.waitKey(1)
        if k == ord("q"):
            exit()
    return None


if __name__ == '__main__':
    main()
