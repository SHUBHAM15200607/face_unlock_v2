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
        if direction == LEFT: 
            direction = RIGHT
        elif direction == RIGHT:
            direction = LEFT

        return direction == self.challenge
