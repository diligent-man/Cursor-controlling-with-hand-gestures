import os
import time

from pathlib import Path
from pprint import pformat
from dataclasses import asdict


import cv2 as cv
import numpy as np

from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from src.utils.GlobalVar import GlobalVar
from src.utils import get_screen_center_origin
from src.hand_detector import HandDetector, HandDetectorResult, HandLandMarkVisualizer


def run_with_image(detector: HandDetector,
                   visualizer: HandLandMarkVisualizer,
                   img_path: str,
                   spath: str
                   ) -> None:
    os.makedirs(Path(spath).parent, exist_ok=True)

    img: np.ndarray = cv.imread(img_path, cv.IMREAD_COLOR_BGR)
    img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

    detector.detect(img)
    detected_result: HandDetectorResult = detector.get_result()

    img: np.ndarray = visualizer(
        detected_result.img,
        detected_result.hand_landmarker_result.handedness,
        detected_result.hand_landmarker_result.hand_landmarks,
    )

    cv.imwrite(spath, img)

    spath: Path = Path(spath)
    with open(os.path.join(spath.parent, f"{spath.stem}.txt"), "w", encoding="utf-8") as f:
        hand_landmarker_result = asdict(detected_result)["hand_landmarker_result"]
        f.write(pformat(hand_landmarker_result, indent=0))


def run_with_video(detector: HandDetector,
                   visualizer: HandLandMarkVisualizer,
                   vid_path: str,
                   spath: str
                   ) -> None:
    cap: cv.VideoCapture = cv.VideoCapture(vid_path)

    writer: cv.VideoWriter = cv.VideoWriter(
        spath, cv.VideoWriter_fourcc(*"X264"), cap.get(cv.CAP_PROP_FPS),
        (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    )

    success, frame = cap.read()
    while success:
        detector.detect(frame, int(time.time() * 1000))
        detected_result: HandDetectorResult = detector.get_result()

        frame = visualizer(
            detected_result.img,
            detected_result.hand_landmarker_result.handedness,
            detected_result.hand_landmarker_result.hand_landmarks,
        )

        writer.write(frame)
        success, frame = cap.read()

    cap.release()
    writer.release()


def run_with_live_video(gl: GlobalVar,
                        detector: HandDetector,
                        visualizer: HandLandMarkVisualizer,
                        ) -> None:
    cap: cv.VideoCapture = cv.VideoCapture(0)  # default resolution: 640 x 480 -> resize after

    cv.namedWindow(gl.WINDOW_NAME, cv.WINDOW_AUTOSIZE)
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
        cv.moveWindow(gl.WINDOW_NAME, *get_screen_center_origin((frame.shape[1], frame.shape[0])))

        k = cv.waitKey(1)
        if k == ord("q"):
            exit()
    return None


def main() -> None:
    gl = GlobalVar("./data/cfg.yaml")

    visualizer: HandLandMarkVisualizer = HandLandMarkVisualizer(
        include_fps=False,
        include_handedness=False,
        include_hand_bbox=False
    )

    run_with_image(
        HandDetector(num_hands=1, running_mode=VMode.IMAGE),
        visualizer,
        "./data/hand_detector/image/two_hands.jpg",
        "./result/hand_detector/image/two_hands.jpg"
    )

    run_with_video(
        HandDetector(num_hands=2, running_mode=VMode.VIDEO),
        visualizer,
        "./data/hand_detector/video/one_hand.mp4",
        "./result/hand_detector/video/one_hand.mp4"
    )

    run_with_live_video(
        gl,
        HandDetector(num_hands=2, running_mode=VMode.LIVE_STREAM),
        visualizer,
    )


if __name__ == "__main__":
    main()
