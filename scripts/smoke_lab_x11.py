#!/usr/bin/env python3
"""Exercise real Lab input and graceful close on an existing X11 display.

Run under xvfb-run in CI. Requires xdotool and a Vulkan compute adapter.
Only the child process's own window receives input or WM_DELETE_WINDOW.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def wait_for_log(process, path, text, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        contents = path.read_text(errors="replace")
        if text in contents:
            return
        if process.poll() is not None:
            raise RuntimeError(f"Lab exited before {text!r}\n{contents}")
        time.sleep(0.05)
    raise TimeoutError(f"No acknowledgement {text!r}\n{path.read_text()}")


def key(*keys):
    subprocess.run(["xdotool", "key", "--clearmodifiers", *keys], check=True)


def close_window(window):
    class Data(ctypes.Union):
        _fields_ = [("bytes", ctypes.c_char * 20),
                    ("shorts", ctypes.c_short * 10),
                    ("longs", ctypes.c_long * 5)]

    class Message(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong),
                    ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
                    ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
                    ("format", ctypes.c_int), ("data", Data)]

    class Event(ctypes.Union):
        _fields_ = [("message", Message), ("padding", ctypes.c_long * 24)]

    x11 = ctypes.CDLL("libX11.so.6")
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                              ctypes.c_long, ctypes.POINTER(Event)]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(os.environ["DISPLAY"].encode())
    if not display:
        raise RuntimeError("Cannot connect to the test X11 display")
    try:
        event = Event()
        event.message.type = 33  # ClientMessage
        event.message.display = display
        event.message.window = window
        event.message.message_type = x11.XInternAtom(display, b"WM_PROTOCOLS", 0)
        event.message.format = 32
        event.message.data.longs[0] = x11.XInternAtom(display, b"WM_DELETE_WINDOW", 0)
        if not x11.XSendEvent(display, window, 0, 0, ctypes.byref(event)):
            raise RuntimeError("WM_DELETE_WINDOW was not sent")
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


def exercise(binary, experiment):
    with tempfile.TemporaryDirectory(prefix="manonica-x11-") as directory:
        path = Path(directory) / "lab.log"
        with path.open("w") as output:
            environment = os.environ.copy()
            environment.pop("WAYLAND_DISPLAY", None)
            process = subprocess.Popen([binary, experiment, "--backend", "gpu"],
                                       stdout=output, stderr=subprocess.STDOUT,
                                       env=environment)
        try:
            wait_for_log(process, path, "Manonica: FIFO VSync")
            result = subprocess.check_output([
                "xdotool", "search", "--all", "--pid", str(process.pid),
                "--name", "Manonica Lab"], text=True)
            window = result.strip().splitlines()[0]
            owner = int(subprocess.check_output([
                "xdotool", "getwindowpid", window], text=True))
            if owner != process.pid:
                raise RuntimeError("Refusing to control a window owned by another process")
            subprocess.run(["xdotool", "windowfocus", "--sync", window], check=True)
            key("e", "ctrl+1", "Return")
            wait_for_log(process, path, "applied experiment revision 1")
            key("e", "ctrl+a")
            time.sleep(0.1)  # Idle event must preserve the active selection.
            subprocess.run(["xdotool", "type", "--clearmodifiers", "cos("], check=True)
            key("Return")
            wait_for_log(process, path, "rejected experiment revision 2")
            key("ctrl+1", "KP_Enter")
            wait_for_log(process, path, "preparing experiment revision 3")
            started = time.monotonic()
            close_window(int(window))
            code = process.wait(timeout=5)
            if code != 0:
                raise RuntimeError(f"Lab exited with {code}\n{path.read_text()}")
            print("PASS: Enter applies, invalid edits retain live state, keypad Enter works")
            print(f"PASS: close during pending apply finished in {time.monotonic()-started:.3f}s")
        except Exception:
            print(path.read_text(errors="replace"), file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: smoke_lab_x11.py <lab-binary> <experiment.toml>")
    exercise(str(Path(sys.argv[1]).resolve()), str(Path(sys.argv[2]).resolve()))
