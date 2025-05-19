import time

import cv2 as cv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode

from src.hand_detector import HandDetector, HandDetectorResult, HandLandMarkVisualizer


def main() -> None:
    detector: HandDetector = HandDetector(num_hands=2, running_mode=VMode.LIVE_STREAM)
    visualizer: HandLandMarkVisualizer = HandLandMarkVisualizer()

    cap: cv.VideoCapture = cv.VideoCapture(0)  # default resolution: 640 x 480 -> resize after

    while cap.isOpened():
        _, frame = cap.read()

        detector.detect(frame, int(time.time() * 1000))
        detected_result: HandDetectorResult = detector.get_result()

        frame = visualizer(
            detected_result.img,
            detected_result.hand_landmarker_result.handedness,
            detected_result.hand_landmarker_result.hand_landmarks,
        )

        cv.imshow("Image", frame)

        k = cv.waitKey(1)
        if k == ord("q"):
            exit()
    return None


if __name__ == "__main__":
    main()
