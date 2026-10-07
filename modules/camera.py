import cv2


class Camera:
    """Simple camera wrapper for webcam access."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(self.camera_index)
        self._configure_camera()

    def _configure_camera(self) -> None:
        """Configure camera resolution settings."""
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def is_opened(self) -> bool:
        """Return whether the camera is opened."""
        return self.capture.isOpened()

    def read(self) -> tuple[bool, any]:
        """Read a frame from the camera safely."""
        if not self.is_opened():
            return False, None
        return self.capture.read()

    def release(self) -> None:
        """Release the camera resource."""
        if self.capture.isOpened():
            self.capture.release()
