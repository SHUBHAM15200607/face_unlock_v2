import cv2
import math
import mediapipe as mp


class LivenessDetector:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.blinked = False

    def distance(self, p1, p2):

        return math.hypot(
            p1.x - p2.x,
            p1.y - p2.y,
        )

    def eye_ratio(self, landmarks, eye):

        p1 = landmarks[eye[0]]
        p2 = landmarks[eye[1]]
        p3 = landmarks[eye[2]]
        p4 = landmarks[eye[3]]
        p5 = landmarks[eye[4]]
        p6 = landmarks[eye[5]]

        vertical = self.distance(p2, p6) + self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        return vertical / (2.0 * horizontal)

    def verify(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        left = self.eye_ratio(landmarks, self.LEFT_EYE)
        right = self.eye_ratio(landmarks, self.RIGHT_EYE)

        ear = (left + right) / 2

        if ear < 0.20:
            self.blinked = True

        if self.blinked and ear > 0.25:
            self.blinked = False
            return True

        return False
