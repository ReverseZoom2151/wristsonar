from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "blender" / "wristsonar_blender.py"
    spec = importlib.util.spec_from_file_location("wristsonar_blender", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blender_sink_accepts_the_runtime_pose_wire_format() -> None:
    module = _module()
    payload = {
        "version": 1,
        "type": "wristsonar.pose",
        "frame": "wrist-relative",
        "joints": [[0, 0, 0]] * 21,
    }
    assert len(module.parse_pose_line(json.dumps(payload))) == 21


def test_blender_sink_refuses_a_world_space_or_short_pose() -> None:
    module = _module()
    with pytest.raises(ValueError, match="wrist-relative"):
        module.parse_pose_line(
            json.dumps(
                {
                    "version": 1,
                    "type": "wristsonar.pose",
                    "frame": "world",
                    "joints": [],
                }
            )
        )
