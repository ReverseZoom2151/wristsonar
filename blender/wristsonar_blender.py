"""Blender 4.x sink for Wristsonar JSON Lines pose frames.

Run this from Blender's Text Editor.  It listens on a local TCP connection so
the inference process can stream one JSON object per line without embedding a
Python interpreter inside Blender.
"""

from __future__ import annotations

import json
import socket
import threading
from queue import Empty, Queue
from typing import Any

JOINT_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


def parse_pose_line(line: str) -> list[tuple[float, float, float]]:
    """Validate an untrusted stream record without importing Blender."""
    payload: dict[str, Any] = json.loads(line)
    if payload.get("version") != 1 or payload.get("type") != "wristsonar.pose":
        raise ValueError("not a Wristsonar v1 pose frame")
    if payload.get("frame") != "wrist-relative":
        raise ValueError("Wristsonar only emits wrist-relative coordinates")
    joints = payload.get("joints")
    if not isinstance(joints, list) or len(joints) != len(JOINT_NAMES):
        raise ValueError("expected 21 joints")
    return [tuple(map(float, point)) for point in joints]


def install(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Create joint empties and start a background JSONL receiver."""
    import bpy

    queue: Queue[list[tuple[float, float, float]]] = Queue()
    collection = bpy.data.collections.get("Wristsonar")
    if collection is None:
        collection = bpy.data.collections.new("Wristsonar")
        bpy.context.scene.collection.children.link(collection)
    for name in JOINT_NAMES:
        if name not in collection.objects:
            bpy.ops.object.empty_add(type="SPHERE", radius=0.008)
            obj = bpy.context.object
            obj.name = name
            for parent in list(obj.users_collection):
                parent.objects.unlink(obj)
            collection.objects.link(obj)

    def receive() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            while True:
                connection, _ = server.accept()
                with connection.makefile("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            queue.put(parse_pose_line(line))
                        except (ValueError, json.JSONDecodeError):
                            continue

    def apply() -> float:
        try:
            joints = queue.get_nowait()
        except Empty:
            return 1.0 / 60.0
        for name, (x, y, z) in zip(JOINT_NAMES, joints, strict=True):
            # Wristsonar is right-handed; Blender displays Z up.
            collection.objects[name].location = (x, -z, y)
        return 1.0 / 60.0

    threading.Thread(target=receive, daemon=True).start()
    bpy.app.timers.register(apply)
    print(f"Wristsonar listening on {host}:{port}")
