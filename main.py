from app.engine import FaceUnlockEngine

engine = FaceUnlockEngine()

try:
    engine.scan_and_unlock()

finally:
    engine.shutdown()
