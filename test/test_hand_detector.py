import cv2 as cv

from src.utils.FPSCalculator import FPSCalculator
from src.utils.HandDetector import HandDetector


def main():
    cap: cv.VideoCapture = cv.VideoCapture(0)
    detector: HandDetector = HandDetector()
    fps_calculator = FPSCalculator()

    while cap.isOpened():
        success, img = cap.read()
        img = detector.detect(img)
        landmarks_lst, bbox = detector.findPosition(img)

        # Check thumb tip
        # if len(landmarks_lst) != 0:
        #     print(landmarks_lst[4])

        cv.putText(img, f"FPS {int(fps_calculator())}", (10, 35), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
        cv.imshow("Image", img)

        k = cv.waitKey(1)
        if k == ord("q"):
            exit()


if __name__ == "__main__":
    main()
