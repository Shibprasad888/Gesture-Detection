import time
from typing import Any

import cv2

from modules.camera import Camera
from modules.gesture_detector import GestureDetector, HandTracker
from modules.hand_detector import HandDetector


class FpsCounter:
    """Lightweight FPS counter updated every second."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def update(self) -> None:
        """Update frame count and refresh FPS every second."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = int(self.frame_count / elapsed)
            self.frame_count = 0
            self.start_time = time.time()

    def get_fps(self) -> int:
        """Return the current FPS value."""
        return self.fps


def draw_overlay(frame: Any, fps: int, width: int, height: int, hand_count: int) -> None:
    """Draw FPS, resolution, and hand count overlay on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Resolution: {width} x {height}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_finger_states(
    frame: Any,
    hand_id: str,
    handedness: str,
    finger_states: list[int],
    static_gesture: str,
    movement_gesture: str,
    distance_gesture: str,
    palm_direction: str,
    rotation: str,
    tilt: str,
    thumb_index_distance: str,
    index_middle_distance: str,
    palm_width: str,
    hand_size: str,
    top_left_y: int,
) -> int:
    """Draw finger state, gesture, orientation, and measurement text."""
    cv2.putText(
        frame,
        f"{hand_id} ({handedness})",
        (10, top_left_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Static Gesture: {static_gesture}",
        (10, top_left_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Movement: {movement_gesture}",
        (10, top_left_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Distance: {distance_gesture}",
        (10, top_left_y + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Palm: {palm_direction}",
        (10, top_left_y + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Rotation: {rotation}",
        (10, top_left_y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Tilt: {tilt}",
        (10, top_left_y + 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Measurements",
        (10, top_left_y + 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Thumb-Index: {thumb_index_distance}",
        (10, top_left_y + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Index-Middle: {index_middle_distance}",
        (10, top_left_y + 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Palm Width: {palm_width}",
        (10, top_left_y + 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Hand Size: {hand_size}",
        (10, top_left_y + 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for index, finger_name in enumerate(GestureDetector.FINGER_NAMES):
        state_text = "Open" if finger_states[index] else "Closed"
        cv2.putText(
            frame,
            f"{finger_name}: {state_text}",
            (10, top_left_y + 230 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Finger States: {GestureDetector.format_states(finger_states)}",
        (10, top_left_y + 230 + len(GestureDetector.FINGER_NAMES) * 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return top_left_y + 260 + len(GestureDetector.FINGER_NAMES) * 20


def draw_motion_trail(frame: Any, history: list[tuple[int, int]]) -> None:
    """Draw a small trail for the recent hand motion path."""
    for index in range(1, len(history)):
        start = history[index - 1]
        end = history[index]
        cv2.line(frame, start, end, (0, 255, 255), 2)
        cv2.circle(frame, end, 3, (0, 255, 255), -1)


def draw_measurement_lines(
    frame: Any,
    points: list[tuple[int, int]],
    thumb_index_text: str,
    index_middle_text: str,
    palm_width_text: str,
) -> None:
    """Draw measurement lines and labels for hand geometry."""
    if len(points) != 21:
        return

    segments = [
        (points[4], points[8], "Thumb-Index", thumb_index_text),
        (points[8], points[12], "Index-Middle", index_middle_text),
        (points[5], points[17], "Palm Width", palm_width_text),
    ]

    for start, end, label, value_text in segments:
        cv2.line(frame, start, end, (255, 0, 0), 2)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            frame,
            label,
            (midpoint[0] - 40, midpoint[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value_text,
            (midpoint[0] - 40, midpoint[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    """Run the hand detection application."""
    camera = Camera()
    if not camera.is_opened():
        print("ERROR: Unable to access camera.")
        return

    hand_detector = HandDetector()
    gesture_detector = GestureDetector()
    hand_tracker = HandTracker()
    fps_counter = FpsCounter()
    camera_width = int(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                cv2.imshow(
                    "Hand Gesture Recognition",
                    frame if frame is not None else cv2.UMat(1, 1, cv2.CV_8UC3),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            processed_frame, landmark_points_list, handedness_list = hand_detector.detect(frame)
            fps_counter.update()
            draw_overlay(
                processed_frame,
                fps_counter.get_fps(),
                camera_width,
                camera_height,
                len(handedness_list),
            )

            hand_positions = []
            hand_items = []
            hand_centers = {}
            hand_size_measures = {}
            for points, handedness in zip(landmark_points_list, handedness_list):
                center = GestureDetector.get_hand_center(points)
                hand_positions.append((handedness, center[0]))
                hand_items.append((handedness, points, center))

            hand_map = hand_tracker.update(hand_positions)

            hand_items_with_id = []
            for handedness, points, center in hand_items:
                hand_id = hand_map.get(handedness, "hand_?")
                hand_items_with_id.append((hand_id, handedness, points))
                hand_centers[hand_id] = center
                hand_size_measures[hand_id] = gesture_detector.calculate_hand_size_measure(points)

            dynamic_gestures = gesture_detector.update_dynamic_gestures(
                hand_centers, hand_size_measures
            )

            offset_y = 120
            for hand_id, handedness, points in hand_items_with_id:
                finger_states = gesture_detector.get_finger_states(points, handedness)
                static_gesture_name = gesture_detector.detect_gesture(finger_states)
                movement_gesture, distance_gesture = dynamic_gestures.get(
                    hand_id, ("None", "Stable")
                )
                palm_direction, rotation, tilt_angle = gesture_detector.estimate_orientation(
                    hand_id, points, handedness
                )
                thumb_index_distance, index_middle_distance, palm_width, hand_size = (
                    gesture_detector.calculate_measurements(points)
                )
                thumb_index_text = (
                    f"{thumb_index_distance} px" if thumb_index_distance is not None else "--"
                )
                index_middle_text = (
                    f"{index_middle_distance} px" if index_middle_distance is not None else "--"
                )
                palm_width_text = (
                    f"{palm_width} px" if palm_width is not None else "--"
                )
                draw_measurement_lines(processed_frame, points, thumb_index_text, index_middle_text, palm_width_text)
                draw_motion_trail(
                    processed_frame,
                    gesture_detector.hand_position_history.get(hand_id, []),
                )
                offset_y = draw_finger_states(
                    processed_frame,
                    hand_id,
                    handedness,
                    finger_states,
                    static_gesture_name,
                    movement_gesture,
                    distance_gesture,
                    palm_direction,
                    rotation,
                    tilt_angle,
                    thumb_index_text,
                    index_middle_text,
                    palm_width_text,
                    hand_size,
                    offset_y,
                )

            cv2.imshow("Hand Gesture Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        hand_detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
