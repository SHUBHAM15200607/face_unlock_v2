import cv2
from app.config import *

class Detector:

    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            INPUT_SIZE,
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            5000
        )

    def detect(self, frame):

        h, w = frame.shape[:2]

        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        return faces
