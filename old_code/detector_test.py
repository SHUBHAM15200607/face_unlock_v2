import cv2
import time

model_path = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

start = time.time()
frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, width, height = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

    frames += 1

    elapsed = time.time() - start
    fps = frames / elapsed

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YuNet Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
