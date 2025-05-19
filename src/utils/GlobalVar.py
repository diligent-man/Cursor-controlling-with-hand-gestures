import os
from typing import Callable, Any, List
from dataclasses import dataclass


__all__ = [
    "GlobalVar"
]


@dataclass
class GlobalVar(object):
    SCALE_FACTOR: float

    # Used for cursor movement smoothening
    PREVIOUS_X: int
    PREVIOUS_Y: int
    SMOOTHEN_FACTOR: float

    # The coord for rectangular box of control region
    FRAME_REDUCTION_X: int
    FRAME_REDUCTION_Y: int

    def __init__(self):
        super(GlobalVar, self).__init__()
        _vars: List[str] = ["SCALE_FACTOR", "PREVIOUS_X", "PREVIOUS_Y",
                            "SMOOTHEN_FACTOR", "FRAME_REDUCTION_X", "FRAME_REDUCTION_Y"
                            ]
        _dtypes: List[Callable] = [float, int, int, float, int, int]
        _defaults: List[Any] = [1., 0, 0, 10, 550, 300]

        for i in range(len(_vars)):
            read_var: None | str = os.environ.get(_vars[i], None)

            if read_var is not None and _defaults[i] is not None:
                try:
                    read_var: _dtypes[i] = _dtypes[i](read_var)
                except ValueError as e:
                    print(f"Try to cast '{read_var}' into '{_defaults[i]}' but get {e}")
            else:
                read_var = _defaults[i]

            setattr(self, _vars[i], read_var)
