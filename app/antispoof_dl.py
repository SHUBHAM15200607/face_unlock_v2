from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


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

        print("=" * 50)
        print("Loading Deep Anti-Spoof Model")
        print("=" * 50)

        self.model_size = 128

        self.threshold = 0.5

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("Model Loaded Successfully")

    # --------------------------------------------------

    def preprocess(self, image):

        h, w = image.shape[:2]

        ratio = self.model_size / max(h, w)

        nh = int(h * ratio)

        nw = int(w * ratio)

        if ratio > 1:
            interp = cv2.INTER_LANCZOS4
        else:
            interp = cv2.INTER_AREA

        image = cv2.resize(
            image,
            (nw, nh),
            interpolation=interp,
        )

        canvas = np.zeros(
            (
                self.model_size,
                self.model_size,
                3,
            ),
            dtype=np.uint8,
        )

        y = (self.model_size - nh) // 2

        x = (self.model_size - nw) // 2

        canvas[
            y:y + nh,
            x:x + nw,
        ] = image

        canvas = canvas.astype(np.float32)

        canvas /= 255.0

        canvas = canvas.transpose(2, 0, 1)

        return np.expand_dims(canvas, axis=0)
    # --------------------------------------------------

    def infer(self, face):

        input_tensor = self.preprocess(face)

        output = self.session.run(
            None,
            {
                self.input_name: input_tensor
            },
        )[0]

        return output[0]

    # --------------------------------------------------

    def process_logits(self, logits):

        real_logit = float(logits[0])

        spoof_logit = float(logits[1])

        score = real_logit - spoof_logit

        confidence = abs(score)

        is_real = score >= self.threshold

        return {

            "is_real": is_real,

            "score": score,

            "confidence": confidence,

            "real_logit": real_logit,

            "spoof_logit": spoof_logit,

        }

    # --------------------------------------------------

    def predict(self, face):

        logits = self.infer(face)

        return self.process_logits(logits)
    # --------------------------------------------------

    def verify(self, face):

        result = self.predict(face)

        print("\n========== Deep Anti-Spoof ==========")
        print(f"Real Logit : {result['real_logit']:.3f}")
        print(f"Spoof Logit: {result['spoof_logit']:.3f}")
        print(f"Score      : {result['score']:.3f}")
        print(f"Confidence : {result['confidence']:.3f}")

        if result["is_real"]:
            print("Prediction : REAL")
        else:
            print("Prediction : SPOOF")

        print("=====================================\n")

        return result["is_real"], result["confidence"]
