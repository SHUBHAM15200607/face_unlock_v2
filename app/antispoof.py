import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class AntiSpoofDetector:

    def __init__(self):

        self.sharpness_threshold = 80

        self.lbp_threshold = 0.35

    def sharpness_score(self, gray):

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def texture_score(self, gray):

        radius = 2
        points = radius * 8

        lbp = local_binary_pattern(
            gray,
            points,
            radius,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, points + 3),
            range=(0, points + 2)
        )

        hist = hist.astype("float")

        hist /= (hist.sum() + 1e-7)

        uniformity = hist.max()

        return uniformity

    def verify(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharpness = self.sharpness_score(gray)

        texture = self.texture_score(gray)

        print(f"[Sharpness] {sharpness:.2f}")
        print(f"[Texture]   {texture:.3f}")

        sharp_ok = sharpness > self.sharpness_threshold

        texture_ok = texture < self.lbp_threshold

        print("Sharpness :", sharp_ok)
        print("Texture   :", texture_ok)

        return sharp_ok and texture_ok

