import os
import asyncio

from pathlib import Path
from pprint import pprint as pp


import cv2 as cv

from dotenv import load_dotenv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from src.App import App
from src.hand_detector import HandDetector, HandLandMarkVisualizer


load_dotenv("./.env")


async def main():
    app = App(
        HandDetector(None, VMode.VIDEO, 1),
        HandLandMarkVisualizer(
            include_fps=False,
            include_handedness=False,
            include_landmarks=True,
            include_hand_bbox=False
        )
    )

    await app.run("video", "./test/data/hand_detector/one_hand.mp4")
    # await app.run("live_stream", 0)


if __name__ == "__main__":
    asyncio.run(main())
    # TODO: Algo for landmark pos for gesture

    # TODO: Install model-explorer
    #     https://github.com/google-ai-edge/model-explorer/wiki/1.-Installation

    # TODO: Check Body Pre Focusing mechanism for distant detection (2m < D < 10m)
    #     https: // github.com / geaxgx / depthai_hand_tracker / tree / main
    # add env var read for GlobalVar
