# Pico Image Auto Clicker - firmware (CircuitPython)
#
# Listens on the usb_cdc "data" serial channel for a small text protocol
# and drives a USB HID relative mouse in response. See pc/pico_serial.py
# for the PC-side counterpart.
#
# Protocol (newline-terminated ASCII):
#   PING            -> replies PONG
#   MOVE:<dx>:<dy>  -> relative mouse move by (dx, dy) pixels, chunked to
#                      the HID report's +-127 range -> OK:MOVE / ERR:MOVE
#   CLICK[:<ms>]    -> left button down, wait <ms> (default 20), up
#                      -> OK:CLICK / ERR:CLICK
#   PRESS           -> left button down, held (for drag) -> OK:PRESS
#   RELEASE         -> left button up -> OK:RELEASE
#   STOP            -> no-op acknowledgement, used for emergency stop
#                      -> OK:STOP
#   anything else   -> ERR:UNKNOWN

import time

import usb_cdc
import usb_hid
from adafruit_hid.mouse import Mouse

DEFAULT_CLICK_PULSE_MS = 20
MAX_HID_STEP = 127  # boot-protocol relative mouse reports are signed 8-bit

serial = usb_cdc.data
if serial is None:
    # boot.py did not run (no hard reset yet) -- nothing we can do but stop.
    raise RuntimeError("usb_cdc.data is not enabled; power-cycle the Pico after copying boot.py")

mouse = Mouse(usb_hid.devices)

_rx_buffer = b""


def send(message):
    serial.write((message + "\n").encode("utf-8"))


def do_move(dx, dy):
    dx, dy = int(dx), int(dy)
    while dx != 0 or dy != 0:
        step_x = max(-MAX_HID_STEP, min(MAX_HID_STEP, dx))
        step_y = max(-MAX_HID_STEP, min(MAX_HID_STEP, dy))
        mouse.move(step_x, step_y, 0)
        dx -= step_x
        dy -= step_y


def do_click(pulse_ms):
    mouse.press(Mouse.LEFT_BUTTON)
    time.sleep(pulse_ms / 1000.0)
    mouse.release(Mouse.LEFT_BUTTON)


def handle_command(cmd):
    if cmd == "PING":
        send("PONG")
    elif cmd == "STOP":
        send("OK:STOP")
    elif cmd == "PRESS":
        try:
            mouse.press(Mouse.LEFT_BUTTON)
            send("OK:PRESS")
        except Exception:
            send("ERR:PRESS")
    elif cmd == "RELEASE":
        try:
            mouse.release(Mouse.LEFT_BUTTON)
            send("OK:RELEASE")
        except Exception:
            send("ERR:RELEASE")
    elif cmd == "CLICK" or cmd.startswith("CLICK:"):
        parts = cmd.split(":")
        pulse_ms = DEFAULT_CLICK_PULSE_MS
        if len(parts) == 2:
            try:
                pulse_ms = int(parts[1])
            except ValueError:
                send("ERR:CLICK")
                return
        try:
            do_click(pulse_ms)
            send("OK:CLICK")
        except Exception:
            send("ERR:CLICK")
    elif cmd.startswith("MOVE:"):
        parts = cmd.split(":")
        if len(parts) == 3:
            try:
                do_move(parts[1], parts[2])
                send("OK:MOVE")
            except ValueError:
                send("ERR:MOVE")
        else:
            send("ERR:MOVE")
    else:
        send("ERR:UNKNOWN")


while True:
    if serial.in_waiting > 0:
        _rx_buffer += serial.read(serial.in_waiting)
        while b"\n" in _rx_buffer:
            line, _rx_buffer = _rx_buffer.split(b"\n", 1)
            text = line.decode("utf-8", "ignore").strip()
            if text:
                handle_command(text)
    time.sleep(0.005)
