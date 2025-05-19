import dataclasses
from typing import List

__all__ = ["HandLandmarksConnections"]


class HandLandmarksConnections:
    """
    The connections between hand landmarks.
    Temporary fix for https://github.com/google-ai-edge/mediapipe/issues/5972
    Mediapipe version: 0.10.21
    """

    @dataclasses.dataclass
    class Connection:
        """The connection class for hand landmarks."""

        start: int
        end: int

    HAND_PALM_CONNECTIONS: List[Connection] = [
        Connection(0, 1),
        Connection(0, 5),
        Connection(9, 13),
        Connection(13, 17),
        Connection(5, 9),
        Connection(0, 17),
    ]

    HAND_THUMB_CONNECTIONS: List[Connection] = [
        Connection(1, 2),
        Connection(2, 3),
        Connection(3, 4),
    ]

    HAND_INDEX_FINGER_CONNECTIONS: List[Connection] = [
        Connection(5, 6),
        Connection(6, 7),
        Connection(7, 8),
    ]

    HAND_MIDDLE_FINGER_CONNECTIONS: List[Connection] = [
        Connection(9, 10),
        Connection(10, 11),
        Connection(11, 12),
    ]

    HAND_RING_FINGER_CONNECTIONS: List[Connection] = [
        Connection(13, 14),
        Connection(14, 15),
        Connection(15, 16),
    ]

    HAND_PINKY_FINGER_CONNECTIONS: List[Connection] = [
        Connection(17, 18),
        Connection(18, 19),
        Connection(19, 20),
    ]

    HAND_CONNECTIONS: List[Connection] = (
            HAND_PALM_CONNECTIONS +
            HAND_THUMB_CONNECTIONS +
            HAND_INDEX_FINGER_CONNECTIONS +
            HAND_MIDDLE_FINGER_CONNECTIONS +
            HAND_RING_FINGER_CONNECTIONS +
            HAND_PINKY_FINGER_CONNECTIONS
    )
