import cv2 as cv # Can be installed using "pip install opencv-python"
import mediapipe as mp  # Can be installed using "pip install mediapipe"
import time
import math
import numpy as np


'''
mp_drawing.draw_landmarks(
    image
    landmark_list: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: mediapipe.python.solutions.drawing_utils.DrawingSpec = DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    connection_drawing_spec: mediapipe.python.solutions.drawing_utils.DrawingSpec = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
)
'''
class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1,
                 min_detection_confidence=.5, min_tracking_confidence=0.7):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                                        static_image_mode = self.static_image_mode,
                                        max_num_hands = self.max_num_hands,
                                        model_complexity = self.model_complexity,
                                        min_detection_confidence = self.min_detection_confidence,
                                        min_tracking_confidence = self.min_tracking_confidence)
        
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_index = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    black = (0, 0, 0); cyan = (255, 255, 0)
                    landmark_drawing_spec = self.mp_draw.DrawingSpec(color=black, thickness=5)
                    connection_drawing_spec = self.mp_draw.DrawingSpec(color=cyan, thickness=3, circle_radius=1)
                    self.mp_draw.draw_landmarks(image = img,
                                                landmark_list = hand_landmarks,
                                                connections = self.mp_hands.HAND_CONNECTIONS,
                                                landmark_drawing_spec = landmark_drawing_spec,
                                                connection_drawing_spec = connection_drawing_spec)
        return img

    # Find position of hand from input frame
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bounding_box = []
        self.landmarks_lst = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):
                # id: index
                # lm: normalized landmark coordinate (x.y) with x,y in [0,1]
                height, width, channels = img.shape
                # find centroid of an img
                cx, cy = int(lm.x * width), int(lm.y * height) # rescale proportional to the sesized resolution
                xList.append(cx)
                yList.append(cy)
                self.landmarks_lst.append([id, cx, cy])
                if draw:
                    pink = (255, 0, 255)
                    cv.circle(img, (cx, cy), 3, pink, cv.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bounding_box = xmin, ymin, xmax, ymax
            if draw:
                cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return self.landmarks_lst, bounding_box


    # Checks which fingers are up
    def fingersUp(self):    
        # Landmark ref: https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png
        fingers = []
        # Thumb check
        thumb_tip = self.tip_index[0]
        thumb_tip_y = self.landmarks_lst[thumb_tip][1]

        thumb_ip = self.tip_index[0] - 1
        thumb_ip_y = self.landmarks_lst[thumb_ip][1]
        
        if thumb_tip_y > thumb_ip_y:
            fingers.append(1)
        else:
            fingers.append(0)

        # Rest of fingers check
        for id in range(1, 5):
            # Lấy cách 2 đốt cho dễ nhận diện
            finger_tip = self.tip_index[id]
            finger_tip_y_coord = self.landmarks_lst[finger_tip][2]

            finger_pip = self.tip_index[id] - 2
            finger_pip_y_coord = self.landmarks_lst[finger_pip][2]

            if finger_tip_y_coord < finger_pip_y_coord:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        return fingers, total_fingers


def calculate_FPS(previous_time=0, current_time=0):
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    return (previous_time, fps)

# For testing
def main():
    previous_time = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while cap.isOpened():
        success, img = cap.read()
        img = detector.findHands(img)
        landmarks_lst, bbox = detector.findPosition(img)
        # Check thumb tip
        if len(landmarks_lst) != 0:
            print(landmarks_lst[4])

        
        previous_time, fps = calculate_FPS(previous_time)
        cv.putText(img, 'FPS: ' + str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv.LINE_AA)
        cv.imshow("Image", img)
        k = cv.waitKey(1)
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()