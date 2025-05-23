import time
import math
from typing import Tuple


import autopy
import cv2 as cv
import numpy as np

from pynput import keyboard, mouse
from screeninfo import get_monitors, Monitor


__all__ = [
    "find_distance",
    "cursor_control",
    "get_primary_monitor_info",
    "get_screen_center_origin",
]


def find_distance(finger_1, finger_2, img, draw=True, r=15, t=3):
    finger_1_x, finger_1_y = finger_1
    finger_2_x, finger_2_y = finger_2
    # middle of p1 and p2
    cx, cy = (finger_1_x + finger_2_x) // 2, (finger_1_y + finger_2_y) // 2

    if draw:
        cv.line(img, (finger_1_x, finger_1_y), (finger_2_x, finger_2_y), (255, 0, 255), t)
        cv.circle(img, (finger_1_x, finger_1_y), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (finger_2_x, finger_2_y), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
    length = np.hypot(finger_2_x - finger_1_x, finger_2_y - finger_1_y)
    return length, img, [finger_1_x, finger_1_y, finger_2_x, finger_2_y, cx, cy]


def cursor_control(img, fingers, landmarks_lst,
                   frame_reduction_x, frame_reduction_y,
                   frame_width, frame_height,
                   scr_width, scr_height,
                   previous_x, previous_y, smooth_factor,
                   keyboard_controller, mouse_controller):
    thumb = fingers[0]
    fore_finger = fingers[1]
    middle_finger = fingers[2]
    ring_finger = fingers[3]
    pinky_finger = fingers[4]

    # Move cursor
    if fore_finger == 1 and thumb == 0 and middle_finger == 0 and ring_finger == 0 and pinky_finger == 0:
        fore_finger_x, fore_finger_y = landmarks_lst[8][1:]
        # Convert coord from control region to host resolution
        x3 = np.interp(fore_finger_x, (frame_reduction_x, frame_width - frame_reduction_x), (0, scr_width))
        y3 = np.interp(fore_finger_y, (frame_reduction_y, frame_height - frame_reduction_y), (0, scr_height))

        # smoothen movement
        current_x = int(previous_x + (x3 - previous_x) / smooth_factor)
        current_y = int(previous_y + (y3 - previous_y) / smooth_factor)
        previous_x, previous_y = current_x, current_y

        # Always keep cursor in host resolution
        if current_x <= 0:
            current_x = 1
        if current_y <= 0:
            current_y = 1

        if current_x >= scr_width:
            current_x = scr_width-1
        if current_y >= scr_height:
            current_y = scr_height-1

        autopy.mouse.move(scr_width - current_x, current_y)
        pink = (255, 0, 255)
        cv.circle(img, (fore_finger_x, fore_finger_y), 5, pink, 15, cv.FILLED)

    # Right-click
    if fore_finger == 1 and thumb == 0 and middle_finger == 1 and ring_finger == 0 and pinky_finger == 0:
        fore_finger_x, fore_finger_y = landmarks_lst[8][1:]
        finger_1 = (fore_finger_x, fore_finger_y)

        middle_finger_x, middle_finger_y = landmarks_lst[12][1:]
        finger_2 = (middle_finger_x, middle_finger_y)

        length, img, lineInfo = find_distance(finger_1, finger_2, img)
        if length < 60:
            cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), 5, cv.FILLED)
            mouse_controller.click(mouse.Button.right)
            time.sleep(.3)

    # Left-click
    if fore_finger == 1 and thumb == 1 and middle_finger == 1 and ring_finger == 1 and pinky_finger == 1:
        mouse_controller.click(mouse.Button.left)
        time.sleep(.3)

    # Volume up
    if fore_finger == 1 and thumb == 1 and middle_finger == 0 and ring_finger == 0 and pinky_finger == 0:
        fore_finger_x, fore_finger_y = landmarks_lst[8][1:]
        finger_1 = (fore_finger_x, fore_finger_y)

        thumb_x, thumb_y = landmarks_lst[4][1:]
        finger_2 = (thumb_x, thumb_y)

        length, img, lineInfo = find_distance(finger_1, finger_2, img)
        if length < 130:
            cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), 5, cv.FILLED)
            keyboard_controller.tap(keyboard.Key.media_volume_up)
            time.sleep(.1)

    # Volume down
    if fore_finger == 1 and thumb == 1 and middle_finger == 1 and ring_finger == 0 and pinky_finger == 0:
        thumb_x, thumb_y = landmarks_lst[4][1:]
        finger_1 = (thumb_x, thumb_y)

        fore_finger_x, fore_finger_y = landmarks_lst[8][1:]
        finger_2 = (fore_finger_x, fore_finger_y)

        middle_finger_x, middle_finger_y = landmarks_lst[12][1:]
        finger_3 = (middle_finger_x, middle_finger_y)

        length_1, img, lineInfo_1 = find_distance(finger_1, finger_2, img)
        length_2, img, lineInfo_2 = find_distance(finger_2, finger_3, img)
        if length_1 < 130 and length_2 < 80:
            cv.circle(img, (lineInfo_1[4], lineInfo_1[5]), 15, (0, 255, 0), 5, cv.FILLED)
            cv.circle(img, (lineInfo_2[4], lineInfo_2[5]), 15, (0, 255, 0), 5, cv.FILLED)
            keyboard_controller.tap(keyboard.Key.media_volume_down)
            time.sleep(.1)

    # Tab switching
    if fore_finger == 1 and thumb == 0 and middle_finger == 1 and ring_finger == 1 and pinky_finger == 1:
        fore_finger_x, fore_finger_y = landmarks_lst[8][1:]
        finger_1 = (fore_finger_x, fore_finger_y)

        middle_finger_x, middle_finger_y = landmarks_lst[12][1:]
        finger_2 = (middle_finger_x, middle_finger_y)

        ring_finger_x, ring_finger_y = landmarks_lst[16][1:]
        finger_3 = (ring_finger_x, ring_finger_y)

        pinky_finger_x, pinky_finger_y = landmarks_lst[20][1:]
        finger_4 = (pinky_finger_x, pinky_finger_y)

        length_1, img, lineInfo_1 = find_distance(finger_1, finger_2, img)
        length_2, img, lineInfo_2 = find_distance(finger_2, finger_3, img)
        length_3, img, lineInfo_3 = find_distance(finger_3, finger_4, img)
        if length_1 > 60:
            if length_2 < 70:
                cv.circle(img, (lineInfo_2[4], lineInfo_2[5]), 15, (0, 255, 0), 5, cv.FILLED)
                keyboard_controller.press(keyboard.Key.alt)
                if length_3 < 110:
                    print(length_3)
                    cv.circle(img, (lineInfo_3[4], lineInfo_3[5]), 15, (0, 255, 0), 5, cv.FILLED)
                    keyboard_controller.tap(keyboard.Key.tab)
                    time.sleep(.2)

        if length_1 <= 60:
            cv.circle(img, (lineInfo_1[4], lineInfo_1[5]), 15, (0, 255, 0), 5, cv.FILLED)
            keyboard_controller.release(keyboard.Key.alt)

    # Window closing
    if fore_finger == 1 and thumb == 1 and middle_finger == 0 and ring_finger == 0 and pinky_finger == 1:
        with keyboard_controller.pressed(keyboard.Key.alt_l):
            keyboard_controller.tap(keyboard.Key.f4)
        time.sleep(.2)

    # Cease programme
    if fore_finger == 0 and thumb == 0 and middle_finger == 0 and ring_finger == 0 and pinky_finger == 0:
        keyboard_controller.tap('q')
        time.sleep(.5)
    return previous_x, previous_y


def get_primary_monitor_info() -> Monitor:
    return_monitor: None | Monitor = None
    for monitor in get_monitors():
        if monitor.is_primary:
            return_monitor = monitor
    return return_monitor


def get_screen_center_origin(img_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    :param img_size: (width, height) of image
    """
    window_width, window_height = img_size[0], img_size[1]

    screen = get_primary_monitor_info()
    x = int((screen.width / 2) - (window_width / 2))
    y = int((screen.height / 2) - (window_height / 2))
    return x, y
