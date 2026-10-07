import math
from typing import Dict, List, Optional, Tuple


class HandTracker:
    """Track hand IDs independently using MediaPipe handedness."""

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self.hand_id_map: Dict[str, str] = {}
        self.hand_order: List[str] = []
        self.missed_frames: Dict[str, int] = {}

    def update(self, hand_positions: List[Tuple[str, int]]) -> Dict[str, str]:
        """Assign stable hand IDs based on detected handedness and position."""
        sorted_hands = sorted(hand_positions, key=lambda item: item[1])
        current_hands = {handedness for handedness, _ in sorted_hands}

        new_map: Dict[str, str] = {}
        used_ids = set()

        for index, (handedness, _) in enumerate(sorted_hands, start=1):
            if handedness in self.hand_id_map:
                hand_id = self.hand_id_map[handedness]
            else:
                hand_id = f"hand_{index}"
                while hand_id in used_ids:
                    index += 1
                    hand_id = f"hand_{index}"
            new_map[handedness] = hand_id
            used_ids.add(hand_id)
            self.missed_frames[handedness] = 0

        for handedness in list(self.hand_id_map):
            if handedness not in current_hands:
                self.missed_frames[handedness] = self.missed_frames.get(handedness, 0) + 1
                if self.missed_frames[handedness] >= self.timeout:
                    self.hand_id_map.pop(handedness, None)
                    self.missed_frames.pop(handedness, None)

        self.hand_id_map = new_map
        return new_map


class GestureDetector:
    """Determine finger open/closed states from hand landmarks and motion history."""

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    SWIPE_THRESHOLD = 100
    WAVE_DIRECTION_CHANGES = 4
    WAVE_AMPLITUDE = 100
    ORIENTATION_SMOOTHING = 5
    SIZE_SMALL_THRESHOLD = 160
    SIZE_MEDIUM_THRESHOLD = 240

    def __init__(self, max_history: int = 20, history_timeout: int = 15) -> None:
        self.max_history = max_history
        self.history_timeout = history_timeout
        self.hand_position_history: Dict[str, List[Tuple[int, int]]] = {}
        self.hand_size_history: Dict[str, List[float]] = {}
        self.orientation_history: Dict[str, List[float]] = {}
        self.missed_frames: Dict[str, int] = {}

    @staticmethod
    def get_finger_states(
        landmark_points: List[Tuple[int, int]], handedness: str
    ) -> List[int]:
        """Return finger states [Thumb, Index, Middle, Ring, Pinky]."""
        if len(landmark_points) != 21:
            return [0, 0, 0, 0, 0]

        thumb_state = GestureDetector._get_thumb_state(landmark_points, handedness)
        index_state = GestureDetector._is_finger_open(landmark_points, 8, 6)
        middle_state = GestureDetector._is_finger_open(landmark_points, 12, 10)
        ring_state = GestureDetector._is_finger_open(landmark_points, 16, 14)
        pinky_state = GestureDetector._is_finger_open(landmark_points, 20, 18)

        return [thumb_state, index_state, middle_state, ring_state, pinky_state]

    @staticmethod
    def _get_thumb_state(
        landmark_points: List[Tuple[int, int]], handedness: str
    ) -> int:
        thumb_tip_x = landmark_points[4][0]
        thumb_ip_x = landmark_points[3][0]

        if handedness == "Right":
            return 1 if thumb_tip_x > thumb_ip_x else 0

        return 1 if thumb_tip_x < thumb_ip_x else 0

    @staticmethod
    def _is_finger_open(
        landmark_points: List[Tuple[int, int]], tip_index: int, pip_index: int
    ) -> int:
        tip_y = landmark_points[tip_index][1]
        pip_y = landmark_points[pip_index][1]
        return 1 if tip_y < pip_y else 0

    @staticmethod
    def detect_gesture(finger_states: List[int]) -> str:
        """Return a gesture name for the given finger state pattern."""
        gesture_map = {
            (0, 0, 0, 0, 0): "Fist",
            (1, 1, 1, 1, 1): "Open Palm",
            (1, 0, 0, 0, 0): "Thumbs Up",
            (0, 1, 0, 0, 0): "Point",
            (0, 1, 1, 0, 0): "Peace",
            (1, 1, 0, 0, 1): "Rock",
            (1, 0, 0, 0, 1): "Call Me",
            (0, 1, 1, 1, 1): "Four Fingers",
        }
        gesture = gesture_map.get(tuple(finger_states))
        if gesture:
            return gesture

        count = sum(finger_states)
        if count == 0:
            return "Fist"
        if count == 1:
            return "One Finger"
        if count == 2:
            return "Two Fingers"
        if count == 3:
            return "Three Fingers"
        if count == 4:
            return "Four Fingers"

        return "Unknown"

    @staticmethod
    def format_states(finger_states: List[int]) -> str:
        """Format finger state list for display."""
        return f"[{','.join(str(state) for state in finger_states)}]"

    @staticmethod
    def get_hand_center(landmark_points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Compute the center of the hand from landmark positions."""
        x_coordinates = [pt[0] for pt in landmark_points]
        y_coordinates = [pt[1] for pt in landmark_points]
        x_min = min(x_coordinates)
        x_max = max(x_coordinates)
        y_min = min(y_coordinates)
        y_max = max(y_coordinates)
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)

    @staticmethod
    def calculate_distance(
        point_a: Tuple[int, int], point_b: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return math.hypot(dx, dy)

    def calculate_measurements(
        self, landmark_points: List[Tuple[int, int]]
    ) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
        """Calculate thumb-index, index-middle, palm width, and hand size."""
        if len(landmark_points) != 21:
            return None, None, None, "Unknown"

        thumb_index = self.calculate_distance(landmark_points[4], landmark_points[8])
        index_middle = self.calculate_distance(landmark_points[8], landmark_points[12])
        palm_width = self.calculate_distance(landmark_points[5], landmark_points[17])
        hand_height = self.calculate_distance(landmark_points[0], landmark_points[9])
        hand_size = self.classify_hand_size(palm_width, hand_height)

        return (
            int(round(thumb_index)),
            int(round(index_middle)),
            int(round(palm_width)),
            hand_size,
        )

    def calculate_hand_size_measure(
        self, landmark_points: List[Tuple[int, int]]
    ) -> float:
        """Return a numeric hand size measure used for relative distance detection."""
        if len(landmark_points) != 21:
            return 0.0

        palm_width = self.calculate_distance(landmark_points[5], landmark_points[17])
        hand_height = self.calculate_distance(landmark_points[0], landmark_points[9])
        return (palm_width + hand_height) / 2.0

    def classify_hand_size(
        self, palm_width: float, hand_height: float
    ) -> str:
        """Classify approximate hand size from palm width and hand height."""
        if palm_width == 0 or hand_height == 0:
            return "Unknown"

        size_measure = (palm_width + hand_height) / 2
        if size_measure < self.SIZE_SMALL_THRESHOLD:
            return "Small"
        if size_measure < self.SIZE_MEDIUM_THRESHOLD:
            return "Medium"
        return "Large"

    def estimate_orientation(
        self,
        hand_id: str,
        landmark_points: List[Tuple[int, int]],
        handedness: str,
    ) -> Tuple[str, str, str]:
        """Estimate palm direction, rotation, and tilt angle for a hand."""
        if len(landmark_points) != 21:
            return "Unknown", "Unknown", "--"

        wrist = landmark_points[0]
        index_mcp = landmark_points[5]
        pinky_mcp = landmark_points[17]
        middle_mcp = landmark_points[9]

        palm_direction = self._estimate_palm_direction(
            wrist, index_mcp, pinky_mcp, handedness
        )
        rotation = self._estimate_rotation(index_mcp, pinky_mcp)
        tilt_angle = self._estimate_tilt_angle(wrist, middle_mcp, handedness)
        smooth_angle = self._smooth_tilt_angle(tilt_angle, hand_id)

        return palm_direction, rotation, f"{int(round(smooth_angle))}°"

    def _estimate_palm_direction(
        self,
        wrist: Tuple[int, int],
        index_mcp: Tuple[int, int],
        pinky_mcp: Tuple[int, int],
        handedness: str,
    ) -> str:
        x_diff = abs(index_mcp[0] - pinky_mcp[0])
        y_diff = abs(index_mcp[1] - pinky_mcp[1])
        if x_diff > y_diff * 1.4:
            return "Side View"
        if y_diff > x_diff * 1.3:
            return "Facing Camera"
        return "Facing Away"

    def _estimate_rotation(
        self, index_mcp: Tuple[int, int], pinky_mcp: Tuple[int, int]
    ) -> str:
        angle = math.degrees(math.atan2(pinky_mcp[1] - index_mcp[1], pinky_mcp[0] - index_mcp[0]))
        if angle > 45:
            return "Tilted Left"
        if angle < -45:
            return "Tilted Right"
        return "Upright"

    def _estimate_tilt_angle(
        self,
        wrist: Tuple[int, int],
        middle_mcp: Tuple[int, int],
        handedness: str,
    ) -> float:
        dx = middle_mcp[0] - wrist[0]
        dy = wrist[1] - middle_mcp[1]
        if dx == 0:
            return 0.0
        angle = math.degrees(math.atan2(dy, dx))
        return angle if handedness == "Right" else -angle

    def _smooth_tilt_angle(self, angle: float, hand_id: str) -> float:
        history = self.orientation_history.get(hand_id, [])
        history.append(angle)
        if len(history) > self.ORIENTATION_SMOOTHING:
            history.pop(0)
        self.orientation_history[hand_id] = history
        return sum(history) / len(history)

    def update_dynamic_gestures(
        self,
        hand_centers: Dict[str, Tuple[int, int]],
        hand_size_measures: Dict[str, float],
    ) -> Dict[str, tuple[str, str]]:
        """Update motion history and return current dynamic and distance gestures."""
        if not hand_centers:
            for hand_id in list(self.missed_frames.keys()):
                self.missed_frames[hand_id] += 1
                if self.missed_frames[hand_id] >= self.history_timeout:
                    self.hand_position_history.pop(hand_id, None)
                    self.hand_size_history.pop(hand_id, None)
                    self.orientation_history.pop(hand_id, None)
                    self.missed_frames.pop(hand_id, None)
            return {}

        dynamic_gestures: Dict[str, tuple[str, str]] = {}
        active_hand_ids = set(hand_centers.keys())

        for hand_id in list(self.hand_position_history.keys()):
            if hand_id not in active_hand_ids:
                self.missed_frames[hand_id] = self.missed_frames.get(hand_id, 0) + 1
                if self.missed_frames[hand_id] >= self.history_timeout:
                    self.hand_position_history.pop(hand_id, None)
                    self.hand_size_history.pop(hand_id, None)
                    self.orientation_history.pop(hand_id, None)
                    self.missed_frames.pop(hand_id, None)

        for hand_id, center in hand_centers.items():
            self.missed_frames[hand_id] = 0
            history = self.hand_position_history.get(hand_id, [])
            history.append(center)
            if len(history) > self.max_history:
                history.pop(0)
            self.hand_position_history[hand_id] = history

            size_history = self.hand_size_history.get(hand_id, [])
            size_history.append(hand_size_measures.get(hand_id, 0.0))
            if len(size_history) > self.max_history:
                size_history.pop(0)
            self.hand_size_history[hand_id] = size_history

            movement_gesture = self._detect_dynamic_gesture(history)
            distance_gesture = self._detect_distance_gesture(size_history)
            dynamic_gestures[hand_id] = (movement_gesture, distance_gesture)

        return dynamic_gestures

    def _detect_dynamic_gesture(self, history: List[Tuple[int, int]]) -> str:
        if len(history) < 5:
            return "None"

        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx > self.SWIPE_THRESHOLD and abs_dy < abs_dx * 0.4:
            return "Swipe Right" if dx > 0 else "Swipe Left"

        if abs_dy > self.SWIPE_THRESHOLD and abs_dx < abs_dy * 0.4:
            return "Swipe Down" if dy > 0 else "Swipe Up"

        return self._detect_wave(history)

    def _detect_wave(self, history: List[Tuple[int, int]]) -> str:
        directions: List[str] = []
        for index in range(1, len(history)):
            delta_x = history[index][0] - history[index - 1][0]
            if abs(delta_x) < 15:
                continue
            directions.append("R" if delta_x > 0 else "L")

        cleaned_directions = [
            directions[i]
            for i in range(len(directions))
            if i == 0 or directions[i] != directions[i - 1]
        ]

        if len(cleaned_directions) >= self.WAVE_DIRECTION_CHANGES:
            x_values = [pos[0] for pos in history]
            if max(x_values) - min(x_values) > self.WAVE_AMPLITUDE:
                return "Wave"

        return "None"

    def _detect_distance_gesture(self, size_history: List[float]) -> str:
        if len(size_history) < 4:
            return "Stable"

        start_average = sum(size_history[:2]) / 2
        end_average = sum(size_history[-2:]) / 2
        threshold = max(12.0, start_average * 0.08)
        difference = end_average - start_average

        if difference > threshold:
            return "Moving Closer"
        if difference < -threshold:
            return "Moving Away"
        return "Stable"
