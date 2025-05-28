from typing import Tuple


import cv2 as cv
import numpy as np


__all__ = ["draw_control_region"]


def draw_control_region(img: np.ndarray,
                        region_size: Tuple[int, int],
                        color=(0, 0, 255)
                        ):
    """
    :param img: bgr image with (H,W,C) shape
    :param region_size: (W,H)
    :param color: bgr color code
    :return:
    """
    cy, cx = img.shape[0] // 2, img.shape[1] // 2

    upper_left_y, upper_left_x = cy - region_size[1] // 2, cx - region_size[0] // 2
    bottom_right_y, bottom_right_x = cy + region_size[1] // 2, cx + region_size[0] // 2

    cv.rectangle(img, (upper_left_x, upper_left_y), (bottom_right_x, bottom_right_y), color, 1)
