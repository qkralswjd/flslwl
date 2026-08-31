# Runs once at power-on/reset, before code.py.
#
# CircuitPython exposes one USB CDC serial port by default (the "console",
# used for the REPL). We enable a SECOND CDC channel ("data") dedicated to
# our PING/MOVE/CLICK/STOP protocol so it never collides with the REPL.
# The default USB HID devices (including the boot mouse) are already
# enabled, so no usb_hid changes are needed here.
#
# NOTE: boot.py only takes effect after a hard reset (unplug/replug or
# press the RESET button) -- saving this file alone is not enough.

import usb_cdc

usb_cdc.enable(console=True, data=True)
