import cv2
import numpy as np

from app.config import *


class Recognizer:

    def __init__(self):

        self.recognizer = cv2.FaceRecognizerSF.create(
            RECOGNIZER_MODEL,
            ""
        )

        data = np.load(EMBEDDINGS_FILE)

        self.embeddings = data["embeddings"]

    def recognize(self, frame, face):

        aligned = self.recognizer.alignCrop(frame, face)

        feature = self.recognizer.feature(aligned)

        feature = feature / np.linalg.norm(feature)

        similarities = np.dot(
            self.embeddings,
            feature.flatten()
        )

        best_similarity = float(np.max(similarities))

        confidence = best_similarity * 100

        return best_similarity, confidence
