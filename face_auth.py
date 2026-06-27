#!/usr/bin/env python3

import sys

from app.engine import FaceUnlockEngine


def main():

    engine = FaceUnlockEngine()

    try:

        verified = engine.scan_and_unlock()

        if verified:
            print("Authentication SUCCESS")
            sys.exit(0)

        print("Authentication FAILED")
        sys.exit(1)

    except KeyboardInterrupt:

        print("Authentication Cancelled")
        sys.exit(1)

    except Exception as e:

        print(f"Authentication Error: {e}")
        sys.exit(1)

    finally:

        engine.shutdown()


if __name__ == "__main__":
    main()
