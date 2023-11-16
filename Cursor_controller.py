import time
import autopy
import cv2 as cv
import numpy as np
import HandTracking as ht
from screeninfo import get_monitors
from pynput import keyboard, mouse


def calculate_FPS(previous_time=0, current_time=0):
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    return (previous_time, fps)

def resize_img(img, screen_width, screen_height, scale_factor=3/3):
    b, g, r = cv.split(img)
    resized_height = int(screen_height*scale_factor)
    resized_width = int(screen_width*scale_factor)
    b = cv.resize(b, (resized_width, resized_height), interpolation=cv.INTER_NEAREST)
    g = cv.resize(g, (resized_width, resized_height), interpolation=cv.INTER_NEAREST)
    r = cv.resize(r, (resized_width, resized_height), interpolation=cv.INTER_NEAREST)
    img = cv.merge([b, g, r])
    return img, resized_width, resized_height 


def control_region(img, frame_reduction_x, frame_reduction_y, resized_width, resized_height):
    red = (0, 0, 255)
    upper_left_x, upper_left_y = frame_reduction_x, frame_reduction_y
    bottom_right_x = resized_width - frame_reduction_x
    bottom_right_y = resized_height - frame_reduction_y
    cv.rectangle(img, (upper_left_x, upper_left_y), (bottom_right_x, bottom_right_y), red, 4)


def find_distance(finger_1, finger_2, img, draw=True,r=15, t=3):
        finger_1_x, finger_1_y = finger_1
        finger_2_x, finger_2_y = finger_2
        # middle of p1 and p2
        cx, cy = (finger_1_x+finger_2_x)//2, (finger_1_y+finger_2_y)//2

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
                   previous_x, previous_y, smoothen_factor,
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
        x3 = np.interp(fore_finger_x, (frame_reduction_x, frame_width - frame_reduction_x), (0,scr_width))
        y3 = np.interp(fore_finger_y, (frame_reduction_y, frame_height - frame_reduction_y), (0, scr_height))

        # smoothen movement
        current_x = int(previous_x + (x3 - previous_x) / smoothen_factor)
        current_y = int(previous_y + (y3 - previous_y) / smoothen_factor)
        previous_x, previous_y = current_x, current_y

        # Always keep cursor in host resolution
        if current_x <= 0: current_x = 1
        if current_y <= 0: current_y = 1

        if current_x >= scr_width: current_x = scr_width-1
        if current_y >= scr_height: current_y = scr_height-1

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
        if  length_1 > 60:
            if length_2 < 60:
                cv.circle(img, (lineInfo_2[4], lineInfo_2[5]), 15, (0, 255, 0), 5, cv.FILLED)
                keyboard_controller.press(keyboard.Key.alt)
                if length_3 < 80:
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



def main():
    ### Variables Declaration
    previous_time = 0                                             # Used to calculate frame rate
    smoothen_factor = 10
    previous_x = 0
    previous_y = 0                                                # Used for cursor movement smoothening 
    frame_reduction_x, frame_reduction_y = 550, 300               # The coord for rectangular box of control region
    # from pynput.mouse import Controller as mouseController, Button
    # from pynput.keyboard import Controller as keyboardController, KeyCode, HotKey, Key
    keyboard_controller = keyboard.Controller()
    mouse_controller = mouse.Controller()

    cap = cv.VideoCapture(0)                                      # 640 x 480 -> resize after
    detector = ht.handDetector(max_num_hands=1)
    screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

    while True:
        success, img = cap.read()
        img, frame_width, frame_height = resize_img(img, screen_width, screen_height)
        img = detector.findHands(img)       

        landmarks_lst, bounding_box = detector.findPosition(img) # Getting position of hand
        #print(landmarks_lst, bounding_box)
        if len(landmarks_lst)!=0:
            # Checking if fingers are upwards
            fingers, total_fingers = detector.fingersUp()
            cv.putText(img, 'Fingers: ' + str(int(total_fingers)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)    
            #print(fingers)
            control_region(img, frame_reduction_x, frame_reduction_y, frame_width, frame_height)  # Draw control region cursor
            previous_x, previous_y = cursor_control(img, fingers, landmarks_lst,
                                                    frame_reduction_x, frame_reduction_y,
                                                    frame_width, frame_height, 
                                                    screen_width, screen_height, 
                                                    previous_x, previous_y, smoothen_factor,
                                                    keyboard_controller, mouse_controller)

        # Compute fps
        previous_time, fps = calculate_FPS(previous_time)
        cv.putText(img, 'FPS: ' + str(int(fps)), (20, 45), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)
        # Display
        cv.imshow("Image", img)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    return 0

if __name__ == '__main__':
    main()