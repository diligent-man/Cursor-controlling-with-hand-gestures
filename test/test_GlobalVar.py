import pytest
from typing import List, Any, Dict

from dotenv import load_dotenv
from src.utils import GlobalVar


@pytest.mark.parametrize("env_path", ["./data/env"])
def test_GlobalVar(env_path: str):
    load_dotenv(env_path)

    gl: GlobalVar = GlobalVar()

    var_map: Dict[str, List[Any]] = {
        "IS_MIRRORED": [bool, True],
        "WINDOW_NAME": [str, "Image"],
        "SCALE_FACTOR": [float, 0.5],
        "SMOOTH_FACTOR": [float, 10.]
    }

    for k, v in gl.__dict__.items():
        if k in var_map.keys():
            assert type(v) == var_map[k][0], ValueError(f"Wrong data type, expected '{var_map[k][0]}'")
            assert v == var_map[k][1], ValueError(f"Wrong value, expected '{var_map[k][1]}'")
