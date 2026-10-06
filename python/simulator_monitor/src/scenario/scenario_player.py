"""
Scenario player and frame generator using OpenCV.
Parses .scenario files and renders visual stimuli frame-by-frame.
"""

import os
import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class ScenarioItem:
    def __init__(self, duration: float, item_type: str, params: Dict[str, Any]):
        self.duration = duration  # seconds
        self.item_type = item_type.lower()
        self.params = params

    def __repr__(self):
        return f"ScenarioItem(type={self.item_type}, duration={self.duration}s, params={self.params})"


class ScenarioPlayer:
    """
    Renders visual stimuli in real-time according to a .scenario file using OpenCV.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.items: List[ScenarioItem] = []
        self.current_index = 0
        self.start_time: Optional[float] = None
        self.item_start_time: Optional[float] = None
        self.is_running = False
        self.is_finished = False
        self.loop = False
        self._cached_image: Optional[np.ndarray] = None
        self._cached_image_path: Optional[str] = None

    def load_scenario(self, file_path: str) -> bool:
        """
        Loads and parses a .scenario file.
        Lines format:
        duration, type, key=value, key=value...
        Example:
        3.0, cross, size=50, color=255,255,255
        5.0, text, message=Focus on Target, color=0,255,0, scale=2.0
        """
        self.items.clear()
        self.current_index = 0

        if not os.path.exists(file_path):
            print(f"[Scenario] Warning: File not found: {file_path}. Creating default scenario.")
            self._create_default_items()
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        duration = float(parts[0])
                    except ValueError:
                        continue
                    item_type = parts[1].lower()
                    params = {}

                    # Parse remaining key=value pairs
                    i = 2
                    while i < len(parts):
                        kv = parts[i]
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            # Color special handling: color=R,G,B or color=B,G,R
                            if k in ("color", "bg_color") and i + 2 < len(parts) and "=" not in parts[i+1]:
                                v = f"{v},{parts[i+1]},{parts[i+2]}"
                                i += 2
                            params[k] = v
                        else:
                            # Positional fallback (e.g. for text or path)
                            if "message" not in params:
                                params["message"] = kv
                        i += 1

                    self.items.append(ScenarioItem(duration, item_type, params))

            if not self.items:
                self._create_default_items()
            return True
        except Exception as e:
            print(f"[Scenario] Error loading scenario: {e}")
            self._create_default_items()
            return False

    def _create_default_items(self):
        self.items = [
            ScenarioItem(5.0, "blank", {}),
            ScenarioItem(3.0, "cross", {"size": "60", "thickness": "3", "color": "0,255,0"}),
            ScenarioItem(5.0, "text", {"message": "Focus on the screen", "scale": "1.8", "color": "255,255,255"}),
            ScenarioItem(3.0, "cross", {"size": "40", "thickness": "2", "color": "255,255,255"}),
            ScenarioItem(5.0, "circle", {"radius": "80", "color": "0,0,255"}),
            ScenarioItem(4.0, "blank", {}),
        ]

    def start(self):
        self.current_index = 0
        now = time.time()
        self.start_time = now
        self.item_start_time = now
        self.is_running = True
        self.is_finished = False

    def stop(self):
        self.is_running = False
        self.is_finished = False

    def reset(self):
        self.current_index = 0
        now = time.time()
        self.start_time = now
        self.item_start_time = now
        self.is_finished = False

    def update_resolution(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cached_image = None
        self._cached_image_path = None

    def get_frame(self) -> np.ndarray:
        """
        Renders and returns current frame as BGR numpy array.
        When not running, returns pure black canvas.
        """
        # Base black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not self.is_running or not self.items:
            return canvas

        if self.item_start_time is None:
            self.start()

        now = time.time()
        elapsed = now - self.item_start_time
        current_item = self.items[self.current_index]

        # Check if current item duration has elapsed
        if elapsed >= current_item.duration:
            self.current_index += 1
            if self.current_index >= len(self.items):
                if self.loop:
                    self.current_index = 0
                    self.item_start_time = now
                    current_item = self.items[self.current_index]
                else:
                    self.is_running = False
                    self.is_finished = True
                    return canvas
            else:
                self.item_start_time = now
                current_item = self.items[self.current_index]

        # Render item onto canvas
        self._render_item(canvas, current_item)
        return canvas

    def _parse_color(self, color_str: Optional[str], default: tuple = (255, 255, 255)) -> tuple:
        if not color_str:
            return default
        try:
            parts = [int(p.strip()) for p in color_str.split(",")]
            if len(parts) == 3:
                # BGR
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
        return default

    def _render_item(self, canvas: np.ndarray, item: ScenarioItem):
        cx = self.width // 2
        cy = self.height // 2

        if item.item_type == "blank":
            bg = self._parse_color(item.params.get("bg_color"), (0, 0, 0))
            if bg != (0, 0, 0):
                canvas[:] = bg

        elif item.item_type in ("cross", "fixation_cross"):
            size = int(item.params.get("size", 50))
            thickness = int(item.params.get("thickness", 3))
            color = self._parse_color(item.params.get("color"), (255, 255, 255))

            cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, thickness)
            cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, thickness)

        elif item.item_type == "text":
            text = item.params.get("message", "Stimulus")
            scale = float(item.params.get("scale", 2.0))
            thickness = int(item.params.get("thickness", 2))
            color = self._parse_color(item.params.get("color"), (255, 255, 255))

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(canvas, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

        elif item.item_type == "circle":
            radius = int(item.params.get("radius", 60))
            color = self._parse_color(item.params.get("color"), (0, 0, 255))
            thickness = int(item.params.get("thickness", -1))  # -1 is filled
            cv2.circle(canvas, (cx, cy), radius, color, thickness)

        elif item.item_type == "rectangle":
            rw = int(item.params.get("width", 200))
            rh = int(item.params.get("height", 150))
            color = self._parse_color(item.params.get("color"), (0, 255, 0))
            thickness = int(item.params.get("thickness", -1))
            cv2.rectangle(canvas, (cx - rw // 2, cy - rh // 2), (cx + rw // 2, cy + rh // 2), color, thickness)

        elif item.item_type == "image":
            path = item.params.get("path") or item.params.get("file")
            if path and os.path.exists(path):
                if self._cached_image_path != path:
                    img = cv2.imread(path)
                    if img is not None:
                        self._cached_image = img
                        self._cached_image_path = path

                if self._cached_image is not None:
                    # Place image at center
                    ih, iw = self._cached_image.shape[:2]
                    # Resize if larger than screen
                    scale = min(self.width / iw, self.height / ih, 1.0)
                    if scale < 1.0:
                        nw, nh = int(iw * scale), int(ih * scale)
                        resized = cv2.resize(self._cached_image, (nw, nh))
                    else:
                        resized = self._cached_image
                        nw, nh = iw, ih

                    x1 = cx - nw // 2
                    y1 = cy - nh // 2
                    canvas[y1 : y1 + nh, x1 : x1 + nw] = resized
