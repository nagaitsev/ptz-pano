from __future__ import annotations

import socket

from ptz_pano.camera.ptzoptics_visca_tcp import (
    PtzOpticsViscaTcpController,
    _find_information_response,
)
from ptz_pano.models import CameraConfig


class FakeSocket:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        raise socket.timeout

    def close(self) -> None:
        pass


def test_send_inquiry_reads_until_information_response_after_stale_completion() -> None:
    expected_responses = [
        bytes.fromhex("90 51 FF"),
        bytes.fromhex("90 50 00 00 00 00 00 00 00 00 FF"),
    ]
    fake_socket = FakeSocket(expected_responses)
    controller = PtzOpticsViscaTcpController(
        CameraConfig(vendor="ptzoptics", host="127.0.0.1", port=5678)
    )
    controller._socket = fake_socket

    responses = controller._send_inquiry(bytes.fromhex("81 09 06 12 FF"))

    assert fake_socket.sent == [bytes.fromhex("81 09 06 12 FF")]
    assert responses == expected_responses


def test_missing_information_response_error_includes_received_packets() -> None:
    response = bytes.fromhex("90 51 FF")

    try:
        _find_information_response([response])
    except RuntimeError as exc:
        assert "90 51 FF" in str(exc)
    else:
        raise AssertionError("expected missing information response error")
