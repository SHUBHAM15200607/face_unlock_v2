import os
import socket

from app.engine import FaceUnlockEngine
from app.unlock import unlock

SOCKET_PATH = os.path.join(
    os.environ["XDG_RUNTIME_DIR"],
    "face_unlock.sock"
)

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(5)

print("=" * 50)
print("Face Unlock Daemon Running")
print("Socket:", SOCKET_PATH)
print("=" * 50)

engine = FaceUnlockEngine()

try:

    while True:

        conn, _ = server.accept()

        try:

            command = conn.recv(1024).decode().strip()

            print("Received:", command)

            if command == "scan":

                print("[INFO] Starting face scan...")

                verified = engine.scan_and_unlock()

                if verified:

                    print("[INFO] Verification successful.")

                    conn.sendall(b"SUCCESS")

                    unlock()

                else:

                    print("[INFO] Verification failed.")

                    conn.sendall(b"FAIL")

        except Exception as e:

            print("[ERROR]", e)

            try:
                conn.sendall(b"FAIL")
            except Exception:
                pass

        finally:

            conn.close()

finally:

    engine.shutdown()

    server.close()

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
