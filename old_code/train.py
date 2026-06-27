import cv2
import os
import numpy as np

FACES_DIR = "faces"
OUTPUT_FILE = "data/embeddings.npz"

# Load models
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

    # Resize large images while preserving aspect ratio
    h, w = image.shape[:2]

    max_side = 640
    scale = max_side / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    detector.setInputSize((new_w, new_h))

    _, faces = detector.detect(resized)

    if faces is None:
        print(f"Skipped {filename} (No face)")
        continue

    # Use the largest detected face
    face = max(faces, key=lambda f: f[2] * f[3])

    # Convert coordinates back to original image
    face = face.copy()
    face[:14] /= scale

    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)

    embeddings.append(feature)

    print(f"Processed: {filename}")


embeddings = np.vstack(embeddings)

# Normalize each embedding
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Average embedding
profile = np.mean(embeddings, axis=0)

# Normalize again
profile = profile / np.linalg.norm(profile)

np.savez(
    OUTPUT_FILE,
    profile=profile
)

print("\nTraining Complete!")
print(f"Faces processed: {len(embeddings)}")
print("Saved profile:", OUTPUT_FILE)
