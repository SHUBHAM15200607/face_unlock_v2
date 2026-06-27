import os
import socket

from gi.repository import GLib, Gio

SOCKET_PATH = os.path.join(
    os.environ["XDG_RUNTIME_DIR"],
    "face_unlock.sock"
)


def signal_handler(
    connection,
    sender_name,
    object_path,
    interface_name,
    signal_name,
    parameters,
):

    if signal_name != "ActiveChanged":
        return

    active = parameters.unpack()[0]

    print("Screen Active:", active)

    if active:

        return

    try:

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        client.connect(SOCKET_PATH)

        client.sendall(b"scan")

        client.close()

        print("Scan Trigger Sent")

    except Exception as e:

        print("Socket Error:", e)


bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

bus.signal_subscribe(
    "org.gnome.ScreenSaver",
    "org.gnome.ScreenSaver",
    "ActiveChanged",
    "/org/gnome/ScreenSaver",
    None,
    Gio.DBusSignalFlags.NONE,
    signal_handler,
)

print("=" * 40)
print("GNOME Listener Running")
print("Socket:", SOCKET_PATH)
print("=" * 40)

GLib.MainLoop().run()
