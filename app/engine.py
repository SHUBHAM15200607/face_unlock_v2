import cv2
import time
from app.head_pose import get_head_direction
from app.challenge import ChallengeManager
from app.camera import Camera
from app.detector import Detector
from app.recognizer import Recognizer
from app.liveness import LivenessDetector
from app.antispoof import AntiSpoofDetector
from app.logger import logger
from app.config import (
    DEBUG,
    MATCH_THRESHOLD,
    SIMILARITY_THRESHOLD,
    SCAN_TIMEOUT,
)


class FaceUnlockEngine:

    def __init__(self):

        logger.info("=" * 40)
        logger.info(" Loading Face Unlock Engine")
        logger.info("=" * 40)

        self.cam = Camera()
        self.detector = Detector()
        self.recognizer = Recognizer()
        self.liveness = LivenessDetector()
        self.antispoof = AntiSpoofDetector()

    def scan_and_unlock(self):
        try:
            self.cam.open()
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            return False
        start_time = time.time()
        match_count = 0

        challenge = ChallengeManager()

        challenge_name = challenge.current()
        challenge_completed = False
        print("Challenge:", challenge_name)
        try:

            while True:

                if time.time() - start_time >= SCAN_TIMEOUT:

                    print("[INFO] Scan timed out.")

                    return False

                ret, frame = self.cam.read()

                if not ret:
                    return False

                faces = self.detector.detect(frame)

                if faces is None:

                    match_count = 0

                else:

                    face = max(faces, key=lambda f: f[2] * f[3])
                    direction = get_head_direction(face)
                    if not challenge_completed:

                        if challenge.verify(direction):

                            challenge_completed = True

                            logger.info(f"Challenge completed: {challenge_name}")
                    similarity, confidence = self.recognizer.recognize(
                        frame,
                        face
                    )

                    x, y, w, h = face[:4].astype(int)
                    print("Direction:", direction)
                    if DEBUG:

                        cv2.putText(
                            frame,
                            f"Direction: {direction}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2,
                         )
                    if similarity >= SIMILARITY_THRESHOLD:

                        match_count += 1

                        if DEBUG:

                            cv2.rectangle(
                                frame,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 0),
                                2,
                            )

                            cv2.putText(
                                frame,
                                f"Shubham {confidence:.2f}% ({match_count}/{MATCH_THRESHOLD})",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                            cv2.putText(
                                frame,
                                f"Direction: {direction}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 0),
                                2,
                            )
                            cv2.putText(
                                frame,
                                f"Turn: {challenge_name}",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                            )

                        if match_count >= MATCH_THRESHOLD:

                                logger.info("Face recognized")

                        if DEBUG:

                            cv2.putText(
                                frame,
                                "Turn " + challenge_name if not challenge_completed else "Blink to Unlock",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 0),
                                2,
                            )

                            if challenge_completed:

                                if challenge_completed:

                                    if self.liveness.verify(frame):

                                        logger.info("Blink detected")

                                        if self.antispoof.verify(frame):

                                            logger.info("Anti-Spoof Passed")
                                            logger.info("Identity Confirmed")

                                            return True

                                        else:

                                            logger.warning("Anti-Spoof Failed")

                    else:

                        match_count = 0

                        if DEBUG:

                            cv2.rectangle(
                                frame,
                                (x, y),
                                (x + w, y + h),
                                (0, 0, 255),
                                2,
                            )

                            cv2.putText(
                                frame,
                                f"Unknown {confidence:.2f}%",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                if DEBUG:

                    cv2.imshow("Face Unlock", frame)

                    if cv2.waitKey(1) == 27:
                        return False

        finally:

            self.cam.release()

            if DEBUG:
                cv2.destroyAllWindows()

    def shutdown(self):

        self.cam.release()

        if DEBUG:
            cv2.destroyAllWindows()
