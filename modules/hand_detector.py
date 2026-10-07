import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore[import]
    _MP_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions")
except ImportError:
    mp = None  # type: ignore[assignment]
    _MP_SOLUTIONS_AVAILABLE = False

if not _MP_SOLUTIONS_AVAILABLE:
    try:
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            HandLandmarksConnections,
            drawing_styles,
            drawing_utils,
        )
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ImportError(
            "MediaPipe installation does not support either mp.solutions or the "
            "tasks API required by this application."
        ) from exc


class HandDetector:
    """Detects hands using MediaPipe and draws landmarks on frames."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.use_solutions = _MP_SOLUTIONS_AVAILABLE

        if self.use_solutions:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            print("INFO: mp.solutions unavailable - falling back to MediaPipe tasks API.")
            self.mp_hands = HandLandmarksConnections
            self.mp_drawing = drawing_utils
            self.mp_drawing_styles = drawing_styles
            self.model_path = model_path or self._find_default_model()
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"MediaPipe hand landmarker model file not found: {self.model_path}.\n"
                    "Either place a valid hand_landmarker.task file in the project root, "
                    "set HAND_LANDMARKER_MODEL_PATH to the model path, or install a "
                    "MediaPipe wheel that provides mp.solutions.\n"
                    "Example: pip install mediapipe==0.9.1.0 if your platform supports it."
                )
            base_options = BaseOptions(model_asset_path=self.model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.hands = HandLandmarker.create_from_options(options)

    def _find_default_model(self) -> str:
        print("INFO: searching for hand_landmarker.task model file...")
        env_path = os.getenv("HAND_LANDMARKER_MODEL_PATH")
        candidate_paths = []
        if env_path:
            candidate_paths.append(Path(env_path))
        candidate_paths.extend([
            Path.cwd() / "hand_landmarker.task",
            Path(__file__).resolve().parent / "hand_landmarker.task",
            Path(__file__).resolve().parent.parent / "hand_landmarker.task",
        ])

        for path in candidate_paths:
            if path.exists():
                print(f"INFO: found hand_landmarker.task at {path}")
                return str(path)

        # Search installed site-packages for a .task file if present.
        for search_path in sys.path:
            try:
                search_root = Path(search_path)
            except TypeError:
                continue
            if not search_root.exists():
                continue
            for candidate in search_root.rglob("hand_landmarker.task"):
                return str(candidate)

        # Try to download the model into the project root if the network is available.
        download_path = Path.cwd() / "hand_landmarker.task"
        print(f"INFO: attempting to download hand_landmarker.task to {download_path}")
        if self._download_default_model(download_path):
            print(f"INFO: successfully downloaded hand_landmarker.task to {download_path}")
            return str(download_path)

        print("WARN: unable to locate or download hand_landmarker.task model file.")
        return str(candidate_paths[0])

    def _download_default_model(self, download_path: Path) -> bool:
        model_url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
        try:
            download_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(model_url, timeout=30) as response:
                if response.status != 200:
                    print(f"WARN: model download failed with status {response.status}")
                    return False
                with open(download_path, "wb") as f:
                    f.write(response.read())
            return True
        except Exception as exc:
            print(f"WARN: error downloading hand_landmarker.task: {exc}")
            if download_path.exists():
                download_path.unlink(missing_ok=True)
            return False

    def detect(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[str]]:
        """Detect hands and return the frame with visualized results."""
        if frame is None:
            return frame, [], []

        if self.use_solutions:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            if not results.multi_hand_landmarks:
                return frame, [], []
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(ImageFormat.SRGB, rgb_frame)
            results = self.hands.detect(mp_image)
            if not results.hand_landmarks:
                return frame, [], []

        return self._draw_hands(frame, results)

    def _draw_hands(
        self, frame: np.ndarray, results: Any
    ) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[str]]:
        hand_landmarks_list: list[list[tuple[int, int]]] = []
        handedness_list: list[str] = []

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks if self.use_solutions else results.hand_landmarks,
            results.multi_handedness if self.use_solutions else results.handedness,
        ):
            label = self._get_handedness_label(handedness)
            handedness_list.append(label)
            landmark_points = self._get_landmark_points(frame, hand_landmarks)
            hand_landmarks_list.append(landmark_points)
            self._draw_landmarks(frame, hand_landmarks)
            self._draw_landmark_ids(frame, landmark_points)
            self._draw_bounding_box(frame, label, landmark_points)

        return frame, hand_landmarks_list, handedness_list

    def _get_handedness_label(self, handedness: Any) -> str:
        if hasattr(handedness, "classification"):
            return handedness.classification[0].label
        if isinstance(handedness, list) and handedness:
            category = handedness[0]
            return getattr(category, "category_name", None) or getattr(
                category, "display_name", "Unknown"
            )
        return "Unknown"

    def _get_landmark_points(
        self, frame: np.ndarray, hand_landmarks: Any
    ) -> list[tuple[int, int]]:
        height, width = frame.shape[:2]
        points: list[tuple[int, int]] = []

        for landmark in (
            hand_landmarks.landmark
            if self.use_solutions
            else hand_landmarks
        ):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        return points

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks: Any,
    ) -> None:
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def _draw_landmark_ids(
        self, frame: np.ndarray, landmark_points: list[tuple[int, int]]
    ) -> None:
        for index, (x, y) in enumerate(landmark_points):
            cv2.putText(
                frame,
                str(index),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        label: str,
        landmark_points: list[tuple[int, int]],
    ) -> None:
        x_coordinates = [point[0] for point in landmark_points]
        y_coordinates = [point[1] for point in landmark_points]
        x_min = max(min(x_coordinates) - 16, 0)
        y_min = max(min(y_coordinates) - 16, 0)
        x_max = min(max(x_coordinates) + 16, frame.shape[1])
        y_max = min(max(y_coordinates) + 16, frame.shape[0])

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{label} Hand",
            (x_min, max(y_min - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
