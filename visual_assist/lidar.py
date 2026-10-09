"""TF-Luna LiDAR over UART.

The sensor streams 9-byte frames at 115200 baud:

    0x59 0x59 Dist_L Dist_H Amp_L Amp_H Temp_L Temp_H Checksum

Distance is in centimetres; the checksum is the low byte of the sum of the
first eight bytes.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

HEADER = 0x59
FRAME_LEN = 9
MIN_STRENGTH = 100  # below this the datasheet calls the reading unreliable
MIN_RANGE_M = 0.2
MAX_RANGE_M = 8.0


def find_frame(buffer: bytes) -> Optional[bytes]:
    """Return the most recent complete, checksum-valid frame in `buffer`."""
    for start in range(len(buffer) - FRAME_LEN, -1, -1):
        frame = buffer[start : start + FRAME_LEN]
        if (
            frame[0] == HEADER
            and frame[1] == HEADER
            and sum(frame[:8]) & 0xFF == frame[8]
        ):
            return frame
    return None


def parse_frame(frame: bytes) -> Optional[float]:
    """Distance in metres from a valid frame, or None if the reading is unusable."""
    distance_m = (frame[2] + frame[3] * 256) / 100.0
    strength = frame[4] + frame[5] * 256
    if strength < MIN_STRENGTH or strength == 0xFFFF:
        return None
    if not MIN_RANGE_M <= distance_m <= MAX_RANGE_M:
        return None
    return distance_m


class NullLidar:
    """Stands in when no LiDAR is connected: every reading is 'unknown'."""

    available = False

    def read(self) -> Optional[float]:
        return None

    def close(self) -> None:
        pass


class TFLuna:
    available = True

    def __init__(self, port: str, baudrate: int = 115200):
        import serial

        self._serial = serial.Serial(port, baudrate, timeout=0.3)

    def read(self) -> Optional[float]:
        """Latest distance straight ahead in metres, or None."""
        # Drop buffered frames so the reading reflects where we point *now*.
        self._serial.reset_input_buffer()
        # One frame is usually enough; a second covers starting mid-frame.
        data = self._serial.read(FRAME_LEN)
        frame = find_frame(data)
        if frame is None:
            frame = find_frame(data + self._serial.read(FRAME_LEN))
        return parse_frame(frame) if frame else None

    def close(self) -> None:
        self._serial.close()


def open_lidar(cfg):
    if cfg.backend == "none":
        return NullLidar()
    try:
        return TFLuna(cfg.port, cfg.baudrate)
    except Exception as exc:
        if cfg.backend != "auto":
            raise
        log.warning("LiDAR not available (%s); distances will be camera estimates "
                    "and there is no route planning.", exc)
        return NullLidar()
