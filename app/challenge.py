import random

LEFT = "LEFT"
RIGHT = "RIGHT"
CENTER = "CENTER"

CHALLENGES = [
    LEFT,
    RIGHT,
    CENTER,
]


class ChallengeManager:

    def __init__(self):
        self.challenge = random.choice(CHALLENGES)

    def current(self):
        return self.challenge

    def verify(self, direction):
        return direction == self.challenge
