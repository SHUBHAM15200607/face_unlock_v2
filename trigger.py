import os
import socket

SOCKET_PATH = os.path.join(
    os.environ["XDG_RUNTIME_DIR"],
    "face_unlock.sock"
)

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

client.connect(SOCKET_PATH)

client.sendall(b"scan")

response = client.recv(1024).decode().strip()

client.close()

print("Daemon Response:", response)

if response == "SUCCESS":
    exit(0)
else:
    exit(1)
