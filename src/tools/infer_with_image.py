import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../"))

from pathlib import Path
from pprint import pformat
from functools import partial
from dataclasses import asdict
from multiprocessing import Pool
from warnings import filterwarnings
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Generator, Callable


import cv2 as cv
import numpy as np

from tqdm import tqdm


from src.utils.GlobalVar import  GlobalVar
from src.hand_detector import (
    HandDetector,
    HandDetectorResult,
    HandLandMarkVisualizer
)

filterwarnings("ignore", category=UserWarning)


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


def main(args: Namespace) -> None:
    cfg: Path = Path(args.config)
    assert cfg.exists(), ValueError

    global gl
    gl = GlobalVar(cfg)
    # globals().update(E=None, gl=gl)

    src: Path = Path(args.src_dir)
    assert src.exists(), ValueError

    dst: Path = Path(args.dst_dir)
    if not dst.exists():
        os.makedirs(dst)

    detector = HandDetector()

    visualizer: HandLandMarkVisualizer = HandLandMarkVisualizer(
        include_fps=False,
        include_handedness=False,
        include_hand_bbox=False
    )

    fn: Callable = partial(run_with_image, detector, visualizer)
    obj_lst: Generator[Path] = src.glob(f"*.{args.extension}")

    batch: List[Tuple[str, str]] = []
    for f in tqdm(obj_lst, total=len(list(obj_lst)), colour="cyan"):
        batch.append((str(f), os.path.join(dst, f.name)))

        if len(batch) % args.batch_size - 1:
            with Pool(processes=args.processes) as pool:
                pool.starmap_async(fn, batch)
            batch = []
    return None


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--config",
                                 type=str,
                                 help="Path to config file",
                                 default=os.path.join(Path(__file__).parent.parent.parent, "config", "inference.yaml")
                                 )

    argument_parser.add_argument("--batch_size",
                                 default=32,
                                 type=int
                                 )

    argument_parser.add_argument("--processes",
                                 default=os.cpu_count()//2 if os.cpu_count() is not None else 1,
                                 type=int
                                 )

    argument_parser.add_argument("--src_dir",
                                 type=str,
                                 required=True
                                 )

    argument_parser.add_argument("--dst_dir",
                                 type=str,
                                 required=True
                                 )

    argument_parser.add_argument("--extension",
                                 type=str,
                                 required=True
                                 )

    parsed_args = argument_parser.parse_args()
    main(parsed_args)
