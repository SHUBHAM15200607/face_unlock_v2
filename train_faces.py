import cv2
import os
import numpy as np

FACES_DIR = "faces"
OUTPUT_FILE = "data/embeddings.npz"

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (640, 640),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

embeddings = []

print("Training started...\n")

for filename in sorted(os.listdir(FACES_DIR)):

    path = os.path.join(FACES_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    h, w = image.shape[:2]

    scale = 640 / max(h, w)

    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale))
    )

    detector.setInputSize(
        (resized.shape[1], resized.shape[0])
    )

    _, faces = detector.detect(resized)

    if faces is None:
        print(f"Skipped {filename}")
        continue

    face = max(faces, key=lambda f: f[2] * f[3])

    face = face.copy()
    face[:14] /= scale

    aligned = recognizer.alignCrop(image, face)

    feature = recognizer.feature(aligned)

    feature = feature / np.linalg.norm(feature)

    embeddings.append(feature.flatten())

    print(f"Processed {filename}")

embeddings = np.array(embeddings)

np.savez(
    OUTPUT_FILE,
    embeddings=embeddings
)

print("\nTraining Complete")
print("Images:", len(embeddings))
print("Saved:", OUTPUT_FILE)
