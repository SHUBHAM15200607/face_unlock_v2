import cv2
import numpy as np
import onnxruntime as ort

from pathlib import Path
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "antispoof"
    / "face-antispoof-onnx"
    / "models"
    / "best_model_quantized.onnx"
)
class AntiSpoofDL:

    def __init__(self):

        print("=" * 40)
        print("Loading Deep Anti-Spoof Model")
        print("=" * 40)

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        print("Deep Anti-Spoof Model Loaded")
