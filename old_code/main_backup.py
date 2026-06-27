import cv2

from app.camera import Camera
from app.detector import Detector
from app.recognizer import Recognizer
from app.unlock import unlock

# Number of consecutive successful matches required
MATCH_THRESHOLD = 3

cam = Camera()
detector = Detector()
recognizer = Recognizer()

print("===================================")
print(" Face Unlock V2 Started")
print("===================================")

match_count = 0

while True:

    ret, frame = cam.read()

    if not ret:
        break

    faces = detector.detect(frame)

    if faces is not None:

        # Take the largest detected face
        face = max(faces, key=lambda f: f[2] * f[3])

        similarity, confidence = recognizer.recognize(frame, face)

        x, y, w, h = face[:4].astype(int)

        if similarity >= 0.50:

            match_count += 1

            color = (0, 255, 0)

            text = f"Shubham {confidence:.2f}% ({match_count}/{MATCH_THRESHOLD})"

            # Unlock only after 3 consecutive successful frames
            if match_count >= MATCH_THRESHOLD:

                print("\n===================================")
                print(" Identity Confirmed")
                print(" Unlocking Session...")
                print("===================================\n")

                unlock()

                break

        else:

            match_count = 0

            color = (0, 0, 255)

            text = f"Unknown {confidence:.2f}%"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    else:
        match_count = 0

    cv2.imshow("Face Unlock V2", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
