import os

from pathlib import Path
from warnings import warn
from functools import partial
from typing import Dict, Tuple, Any, Callable


import yaml

__package__ = Path(__file__).parent.resolve()

__all__ = ["GlobalVar"]


class GlobalVar(object):
    IS_MIRRORED: bool
    WINDOW_NAME: str

    SCALE_FACTOR: float
    CONTROL_REGION: tuple[int, int]

    # Used for cursor movement smoothening
    PREVIOUS_X: int
    PREVIOUS_Y: int
    SMOOTH_FACTOR: float

    # HandLandMarkVisualizer
    HANDEDNESS_FONT: int
    HANDEDNESS_FONT_SIZE: int
    HANDEDNESS_TEXT_COLOR: tuple[int, int, int]
    HANDEDNESS_FONT_THICKNESS: int
    HANDEDNESS_LINE_TYPE: int

    HAND_BBOX_COLOR: tuple[int, int, int]
    HAND_BBOX_THICKNESS: int
    HAND_BBOX_LINE_TYPE: int

    PALM_BBOX_COLOR: tuple[int, int, int]
    PALM_BBOX_THICKNESS: int
    PALM_BBOX_LINE_TYPE: int

    FPS_ORIG: tuple[int, int]
    FPS_FONT: int
    FPS_FONT_SIZE: int
    FPS_TEXT_COLOR: tuple[int, int, int]
    FPS_FONT_THICKNESS: int
    FPS_LINE_TYPE: int

    BBOX_MARGIN: int
    PALM_MARGIN: int

    def __init__(self, cfg: str | Path = None) -> None:
        def _getter(inst: GlobalVar, attr_name: str) -> Any | Exception:
            """
            :param inst: class's instance
            :param attr_name: name of attr to get
            :return: callable for get attr
            """
            return getattr(
                inst, f"_{attr_name}",
                AttributeError(f"'{attr_name}' attr is missing")
            )

        def _setter(*_, attr_name: str) -> Exception:
            """
            :param _: tuple of (inst, new_val) which is ignored
            :param attr_name: name of attr to set
            :return: AttributeError
            """
            raise AttributeError(f"'{attr_name}' attr is immutable")

        def _del(*_, attr_name: str) -> Exception:
            """
            :param _: tuple of (inst, new_val) which is ignored
            :param attr_name: name of attr to set
            :return: AttributeError
            """
            raise AttributeError(f"'{attr_name}' attr is indelible")

        _defaults: Dict[str, Tuple[Callable, Any]] = {
            "IS_MIRRORED": (bool, True),
            "WINDOW_NAME": (str, "App"),

            "SCALE_FACTOR": (float, 0.5),
            "CONTROL_REGION": (tuple, (300, 150)),

            "PREVIOUS_X": (int, 0),
            "PREVIOUS_Y": (int, 0),
            "SMOOTH_FACTOR": (float, 10),

            "HANDEDNESS_FONT": (int, 1),
            "HANDEDNESS_FONT_SIZE": (int, 2),
            "HANDEDNESS_TEXT_COLOR": (tuple, (0, 0, 255)),
            "HANDEDNESS_FONT_THICKNESS": (int, 2),
            "HANDEDNESS_LINE_TYPE": (int, 16),

            "HAND_BBOX_COLOR": (tuple, (255, 0, 0)),
            "HAND_BBOX_THICKNESS": (int, 1),
            "HAND_BBOX_LINE_TYPE": (int, 16),

            "PALM_BBOX_COLOR": (tuple, (0, 0, 255)),
            "PALM_BBOX_THICKNESS": (int, 1),
            "PALM_BBOX_LINE_TYPE": (int, 16),

            "FPS_ORIG": (tuple, (.015, .05)),
            "FPS_FONT": (int, 1),
            "FPS_FONT_SIZE": (int, 2),
            "FPS_TEXT_COLOR": (tuple, (0, 0, 255)),
            "FPS_FONT_THICKNESS": (int, 2),
            "FPS_LINE_TYPE": (int, 16),

            "BBOX_MARGIN": (int, 10),
            "PALM_MARGIN": (int, 35),
        }

        cfg: Dict[str, Any] = self._read_cfg(cfg)

        # Dtype of collection's ele (e.g. list, tuple, ...) are not verified
        # Value range of all attrs is not appropriately checked
        for name, val in _defaults.items():
            read_val: Any = cfg.get(name, None)

            if type(read_val) != val[0]:
                warn(f"{name} gets wrong config dtype. Run with default: {val[1]}")
                read_val = val[1]

            setattr(self, f"_{name}", read_val)  # why attr name must have "_" ?
            setattr(self.__class__,
                    name,
                    property(
                        partial(_getter, attr_name=f"{name}"),
                        partial(_setter, attr_name=f"{name}"),
                        partial(_del, attr_name=f"{name}")
                    ))

    @staticmethod
    def _read_cfg(cfg: str | Path) -> Dict[str, Any]:
        if cfg is not None:
            cfg: Path = Path(cfg)
            assert cfg.exists(), FileNotFoundError
        else:
            cfg: str = os.path.join(__package__.parent, "cfg.yaml")

        with open(cfg) as stream:
            try:
                # allows Python-based obj, but it's still insecure for code written in Python at the end of the day.
                cfg: Dict[str, Any] = yaml.full_load(stream)
            except yaml.YAMLError as e:
                raise e
        return cfg

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}("

        obj_dct: Dict[str, Any] = self.__dict__
        for i, (k, v) in enumerate(obj_dct.items()):
            msg += f"{k[1:]}={v}"

            if i < len(obj_dct) - 1:
                msg += ", "
            else:
                msg += ")"
        return msg
