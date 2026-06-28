import numpy as np


LEFT = "LEFT"
RIGHT = "RIGHT"
CENTER = "CENTER"
UP = "UP"


def get_head_direction(face):

    left_eye = np.array(face[4:6])
    right_eye = np.array(face[6:8])
    nose = np.array(face[8:10])

    eye_center = (left_eye + right_eye) / 2

    dx = nose[0] - eye_center[0]
    dy = nose[1] - eye_center[1]

    eye_distance = np.linalg.norm(right_eye - left_eye)

    x_ratio = dx / eye_distance
    y_ratio = dy / eye_distance
    print(f"x_ratio = {x_ratio:.3f}")
    if x_ratio < -0.10:
        return LEFT

    if x_ratio > 0.10:
        return RIGHT

    if y_ratio < -0.05:
        return UP

    return CENTER

