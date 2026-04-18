from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterable

from ptz_pano.models import CameraConfig, CameraPose

VISCA_TERMINATOR = 0xFF


def _encode_nibbles(value: int, width: int = 4) -> bytes:
    if value < 0:
        value = (1 << (width * 4)) + value
    if not 0 <= value < (1 << (width * 4)):
        raise ValueError(f"value {value} does not fit in {width} VISCA nibbles")
    return bytes((value >> shift) & 0x0F for shift in range((width - 1) * 4, -1, -4))


def _decode_nibbles(data: bytes, signed: bool = False) -> int:
    value = 0
    for item in data:
        if item > 0x0F:
            raise ValueError(f"VISCA nibble out of range: 0x{item:02X}")
        value = (value << 4) | item
    if signed and value >= 1 << (len(data) * 4 - 1):
        value -= 1 << (len(data) * 4)
    return value


def _ensure_command(command: Iterable[int]) -> bytes:
    data = bytes(command)
    if not data or data[-1] != VISCA_TERMINATOR:
        data += bytes([VISCA_TERMINATOR])
    return data


@dataclass
class PtzOpticsViscaTcpController:
    config: CameraConfig
    pan_speed: int = 0x18
    tilt_speed: int = 0x14

    def __post_init__(self) -> None:
        if self.config.transport != "tcp":
            raise ValueError("PtzOpticsViscaTcpController requires tcp transport")
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.timeout_sec,
        )
        sock.settimeout(self.config.timeout_sec)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def send_raw(self, command: bytes) -> list[bytes]:
        self.connect()
        assert self._socket is not None
        self._socket.sendall(_ensure_command(command))
        return self._read_responses()

    def home(self) -> None:
        self.send_raw(bytes.fromhex("81 01 06 04 FF"))

    def stop(self) -> None:
        self.send_raw(bytes([0x81, 0x01, 0x06, 0x01, self.pan_speed, self.tilt_speed, 0x03, 0x03, 0xFF]))

    def move_absolute(self, pose: CameraPose) -> None:
        command = (
            bytes([0x81, 0x01, 0x06, 0x02, self.pan_speed, self.tilt_speed])
            + _encode_nibbles(pose.pan)
            + _encode_nibbles(pose.tilt)
            + bytes([0xFF])
        )
        self.send_raw(command)
        self.set_zoom(pose.zoom)

    def get_position(self) -> CameraPose:
        responses = self._send_inquiry(bytes.fromhex("81 09 06 12 FF"))
        response = _find_information_response(responses)
        payload = response[2:-1]
        if len(payload) < 8:
            raise RuntimeError(f"unexpected pan/tilt inquiry response: {response.hex(' ').upper()}")
        pan = _decode_nibbles(payload[:4], signed=True)
        tilt = _decode_nibbles(payload[4:8], signed=True)
        return CameraPose(pan=pan, tilt=tilt, zoom=self.get_zoom())

    def get_zoom(self) -> int:
        responses = self._send_inquiry(bytes.fromhex("81 09 04 47 FF"))
        response = _find_information_response(responses, allow_missing=True)
        if response is not None:
            payload = response[2:-1]
            if len(payload) >= 4:
                return _decode_nibbles(payload[:4])

        responses = self._send_inquiry(bytes.fromhex("81 09 7E 7E 00 FF"))
        response = _find_information_response(responses)
        payload = response[2:-1]
        if len(payload) < 4:
            raise RuntimeError(f"unexpected lens block inquiry response: {response.hex(' ').upper()}")
        return _decode_nibbles(payload[:4])

    def set_zoom(self, zoom: int) -> None:
        command = bytes([0x81, 0x01, 0x04, 0x47]) + _encode_nibbles(zoom) + bytes([0xFF])
        self.send_raw(command)

    def _send_inquiry(self, command: bytes) -> list[bytes]:
        self.connect()
        assert self._socket is not None
        self._socket.sendall(_ensure_command(command))
        return self._read_responses(expected_responses=1)

    def _read_responses(self, expected_responses: int = 2) -> list[bytes]:
        assert self._socket is not None
        responses: list[bytes] = []
        current = bytearray()

        while True:
            try:
                chunk = self._socket.recv(64)
            except socket.timeout:
                break
            if not chunk:
                break
            for item in chunk:
                current.append(item)
                if item == VISCA_TERMINATOR:
                    responses.append(bytes(current))
                    current.clear()
            if len(responses) >= expected_responses:
                break
        return responses


def _find_information_response(
    responses: list[bytes],
    allow_missing: bool = False,
) -> bytes | None:
    for response in responses:
        if len(response) >= 3 and response[1] == 0x50:
            return response
        if len(response) >= 3 and response[1] == 0x60 and not allow_missing:
            raise RuntimeError(f"VISCA inquiry error: {response.hex(' ').upper()}")
    if allow_missing:
        return None
    raise RuntimeError("VISCA inquiry did not return an information response")
