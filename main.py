import asyncio

import cv2 as cv
from dotenv import load_dotenv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from src.utils import GlobalVar
from src.hand_detector import HandDetector


load_dotenv("./.env")
gl = GlobalVar()


async def main():
    from src.App import App
    global gl

    app = App(
        HandDetector(None, VMode.LIVE_STREAM, 2, is_mirrored=gl.IS_MIRRORED)
    )

    # await app.run("video", "./test/data/hand_detector/one_hand.mp4")
    await app.run("live_stream", 0)


if __name__ == "__main__":
    asyncio.run(main())
    # TODO: Adapt FPSHandler from luxion AI src
    # TODO: Algo for landmark pos for gesture
    # TODO: Install model-explorer
    #     https://github.com/google-ai-edge/model-explorer/wiki/1.-Installation
    # TODO: Check Body Pre Focusing mechanism for distant detection (2m < D < 10m)
    #     https: // github.com / geaxgx / depthai_hand_tracker / tree / main
    # TODO: Reconsider input queue for async running
