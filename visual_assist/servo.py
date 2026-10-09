"""Pan-tilt head carrying the camera and LiDAR.

Two ways to drive the servos:
  esp8266 - the Pi sends text commands over USB serial to the ESP8266
            running firmware/esp8266_pan_tilt (the setup in the report).
  gpio    - the servos' signal wires go straight to the Pi's GPIO pins.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


def _clamp(angle: float) -> int:
    return int(max(0, min(180, round(angle))))


class NullPanTilt:
    """No servos: the camera only ever looks straight ahead."""

    can_move = False

    def look(self, bearing: float, tilt: float = 0.0, settle: float = None) -> None:
        pass

    def center(self) -> None:
        pass

    def close(self) -> None:
        pass


class _PanTilt:
    can_move = True

    def __init__(self, cfg):
        self._cfg = cfg

    def look(self, bearing: float, tilt: float = 0.0, settle: float = None) -> None:
        """Point `bearing` degrees right of straight ahead and `tilt` degrees up.

        `settle` is how long to wait for the head to stop; small steps of a
        LiDAR sweep need less than the default.
        """
        # A standard servo turns anticlockwise (left) as its angle increases.
        sign = 1 if self._cfg.pan_invert else -1
        self._write(
            _clamp(self._cfg.pan_center + sign * bearing),
            _clamp(self._cfg.tilt_center + tilt),
        )
        time.sleep(self._cfg.settle_s if settle is None else settle)

    def center(self) -> None:
        self.look(0.0, 0.0)

    def _write(self, pan: int, tilt: int) -> None:
        raise NotImplementedError


class Esp8266PanTilt(_PanTilt):
    def __init__(self, cfg):
        super().__init__(cfg)
        import serial

        self._serial = serial.Serial(cfg.port, cfg.baudrate, timeout=1.0)
        # Opening the port resets a NodeMCU; let it boot, then drop its boot log.
        time.sleep(2.0)
        self._serial.reset_input_buffer()

    def _send(self, line: str) -> None:
        self._serial.write((line + "\n").encode("ascii"))
        reply = self._serial.readline().decode("ascii", "replace").strip()
        if reply != "OK":
            log.warning("ESP8266 replied %r to %r", reply, line)

    def _write(self, pan: int, tilt: int) -> None:
        self._send(f"PAN {pan}")
        self._send(f"TILT {tilt}")

    def close(self) -> None:
        self._serial.close()


class GpioPanTilt(_PanTilt):
    def __init__(self, cfg):
        super().__init__(cfg)
        from gpiozero import AngularServo

        def servo(pin):
            return AngularServo(
                pin, min_angle=0, max_angle=180,
                min_pulse_width=0.0005, max_pulse_width=0.0025,
            )

        self._pan = servo(cfg.pan_pin)
        self._tilt = servo(cfg.tilt_pin)

    def _write(self, pan: int, tilt: int) -> None:
        self._pan.angle = pan
        self._tilt.angle = tilt

    def close(self) -> None:
        self._pan.close()
        self._tilt.close()


def open_pantilt(cfg):
    if cfg.backend == "none":
        return NullPanTilt()
    if cfg.backend == "gpio":
        return GpioPanTilt(cfg)
    try:
        return Esp8266PanTilt(cfg)
    except Exception as exc:
        if cfg.backend != "auto":
            raise
        log.warning("Pan-tilt not available (%s); the head cannot turn, so there is "
                    "no room sweep or route planning.", exc)
        return NullPanTilt()
