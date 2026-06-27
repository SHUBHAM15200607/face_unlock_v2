import cv2
from app.config import CAMERA_ID


class Camera:

    def __init__(self):
        self.cap = None

    def open(self):

        if self.cap is not None and self.cap.isOpened():
            return

        # Use the V4L2 backend on Linux
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Reduce startup latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):

        if self.cap is None:
            raise RuntimeError("Camera is not open")

        return self.cap.read()

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None
