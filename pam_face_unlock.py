#!/usr/bin/env python3

import os
import socket
import sys

SOCKET_PATH = os.path.join(
    os.environ["XDG_RUNTIME_DIR"],
    "face_unlock.sock"
)


def authenticate():

    try:

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        client.connect(SOCKET_PATH)

        client.sendall(b"scan")

        response = client.recv(1024).decode().strip()

        client.close()

        print("Daemon Response:", response)

        return response == "SUCCESS"

    except Exception as e:

        print("Authentication Error:", e)

        return False


def main():

    if authenticate():
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
