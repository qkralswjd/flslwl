# Runs once at power-on/reset, before code.py.
#
# CircuitPython exposes one USB CDC serial port by default (the "console",
# used for the REPL). We enable a SECOND CDC channel ("data") dedicated to
# our PING/MOVE/CLICK/KEYDOWN/KEYUP protocol so it never collides with the REPL.
#
# usb_hid.enable() is called explicitly to include BOTH mouse AND keyboard HID
# devices. Without this, only the default boot-protocol devices are present and
# Keyboard(usb_hid.devices) may fail to find a keyboard descriptor.
#
# NOTE: boot.py only takes effect after a hard reset (unplug/replug or
# press the RESET button) -- saving this file alone is not enough.

import usb_cdc
import usb_hid
from adafruit_hid.keyboard import Keyboard  # noqa: F401 – ensures keyboard descriptor exists

usb_cdc.enable(console=True, data=True)

# Explicitly enable standard HID devices: boot mouse + boot keyboard.
# This guarantees Keyboard(usb_hid.devices) finds a keyboard descriptor in code.py.
usb_hid.enable((usb_hid.Device.MOUSE, usb_hid.Device.KEYBOARD))
