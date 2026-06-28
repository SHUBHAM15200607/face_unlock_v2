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

        print("=" * 50)
        print("Loading Deep Anti-Spoof Model")
        print("=" * 50)

        self.model_size = 128
        self.expand_factor = 1.5
        self.threshold = 0.5

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("Model Loaded Successfully")

    # ----------------------------------------------------------

    def expand_bbox(self, frame, face):

        h, w = frame.shape[:2]

        x, y, fw, fh = face[:4].astype(int)

        cx = x + fw / 2
        cy = y + fh / 2

        size = int(max(fw, fh) * self.expand_factor)

        x1 = int(cx - size / 2)
        y1 = int(cy - size / 2)

        x2 = x1 + size
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2

    # ----------------------------------------------------------

    def crop_face(self, frame, face):

        x1, y1, x2, y2 = self.expand_bbox(frame, face)

        crop = frame[y1:y2, x1:x2]

        return crop

    # ----------------------------------------------------------

    def preprocess(self, face):

        h, w = face.shape[:2]

        ratio = self.model_size / max(h, w)

        nh = int(h * ratio)
        nw = int(w * ratio)

        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LANCZOS4

        face = cv2.resize(face, (nw, nh), interpolation=interp)

        top = (self.model_size - nh) // 2
        bottom = self.model_size - nh - top

        left = (self.model_size - nw) // 2
        right = self.model_size - nw - left

        face = cv2.copyMakeBorder(
            face,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )

        face = face.transpose(2, 0, 1).astype(np.float32)

        face /= 255.0

        face = np.expand_dims(face, axis=0)

        return face

    # ----------------------------------------------------------

    def infer(self, crop):

        input_tensor = self.preprocess(crop)

        output = self.session.run(
            None,
            {
                self.input_name: input_tensor
            },
        )[0]

        return output[0]

    # ----------------------------------------------------------

    def process_logits(self, logits):

        real = float(logits[0])
        spoof = float(logits[1])

        score = real - spoof

        confidence = abs(score)

        is_real = score >= self.threshold

        return {
            "is_real": is_real,
            "score": score,
            "confidence": confidence,
            "real_logit": real,
            "spoof_logit": spoof,
        }

    # ----------------------------------------------------------

    def verify(self, frame, face):

        crop = self.crop_face(frame, face)

        logits = self.infer(crop)

        result = self.process_logits(logits)

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
