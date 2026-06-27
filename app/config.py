# ===========================
# Face Unlock V2 Configuration
# ===========================

# ---------------------------
# Debug Mode
# ---------------------------
# True  -> Show camera window, face box, confidence
# False -> Silent production mode
DEBUG = True

# ---------------------------
# Camera
# ---------------------------
CAMERA_ID = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------------------------
# Face Detection
# ---------------------------
INPUT_SIZE = (640, 640)

SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

# ---------------------------
# Face Recognition
# ---------------------------
SIMILARITY_THRESHOLD = 0.75

# Number of consecutive matches
MATCH_THRESHOLD = 3
SCAN_TIMEOUT = 10
# ---------------------------
# Display
# ---------------------------
SHOW_CAMERA = DEBUG

# ---------------------------
# Models
# ---------------------------
DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

# ---------------------------
# Database
# ---------------------------
EMBEDDINGS_FILE = "data/embeddings.npz"
