import cv2 as cv
from dotenv import load_dotenv
from screeninfo import get_monitors
from pynput.keyboard import Controller as kbController
from pynput.mouse import Controller as mController

from src.utils import *
from src.utils import HandTracking as ht


load_dotenv("./.env")


def main() -> None:
    gl: GlobalVar = GlobalVar()
    kb_controller: kbController = kbController()
    m_controller: mController = mController()

    cap = cv.VideoCapture(0)  # 640 x 480 -> resize after
    detector = ht.handDetector(max_num_hands=1)
    screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

    while True:
        success, img = cap.read()
        img, frame_width, frame_height = resize_img(img, screen_width, screen_height)
        img = detector.findHands(img)

        landmarks_lst, bounding_box = detector.findPosition(img)  # Getting position of hand
        # print(landmarks_lst, bounding_box)
        if len(landmarks_lst) != 0:
            # Checking if fingers are upwards
            fingers, total_fingers = detector.fingersUp()
            cv.putText(img, 'Fingers: ' + str(int(total_fingers)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2,
                       cv.LINE_AA)
            # print(fingers)
            control_region(img, gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y, frame_width,
                           frame_height)  # Draw control region cursor

            gl.PREVIOUS_X, gl.PREVIOUS_Y = cursor_control(img, fingers, landmarks_lst,
                                                    gl.FRAME_REDUCTION_X, gl.FRAME_REDUCTION_Y,
                                                    frame_width, frame_height,
                                                    screen_width, screen_height,
                                                    gl.PREVIOUS_X, gl.PREVIOUS_Y, gl.SMOOTHEN_FACTOR,
                                                    kb_controller, m_controller
                                                    )

        # Compute fps
        previous_time, fps = calculate_FPS(gl.PREVIOUS_TIME)
        cv.putText(img, 'FPS: ' + str(int(fps)), (20, 45), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)
        # Display
        cv.imshow("Image", img)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    return None


if __name__ == '__main__':
    main()
