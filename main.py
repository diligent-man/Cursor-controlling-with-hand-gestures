import asyncio


from dotenv import load_dotenv
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VMode


from src.App import App
from src.hand_detector import HandDetector, HandLandMarkVisualizer


load_dotenv("./.env")


async def main():
    app = App(
        HandDetector(None, VMode.VIDEO, 2),
        HandLandMarkVisualizer(
            include_fps=True,
            include_handedness=True,
            include_landmarks=True,
            include_hand_bbox=True
        )
    )

    await app.run("video", "./test/data/hand_detector/two_hands.mp4")
    # await app.run("live_stream", 0)


if __name__ == "__main__":
    asyncio.run(main())
    # TODO: Algo for landmark pos for gesture

    # TODO: Install model-explorer
    #     https://github.com/google-ai-edge/model-explorer/wiki/1.-Installation

    # TODO: Check Body Pre Focusing mechanism for distant detection (2m < D < 10m)
    #     https: // github.com / geaxgx / depthai_hand_tracker / tree / main
