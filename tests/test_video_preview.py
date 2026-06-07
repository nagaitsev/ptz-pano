from __future__ import annotations

import numpy as np

from ptz_pano.api.main import _jpeg_preview_frame, _latest_frame_mjpeg_stream, _mjpeg_stream


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_mjpeg_stream_yields_jpeg_part_and_releases_capture() -> None:
    capture = FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])

    def capture_factory(source: str) -> FakeCapture:
        assert source == "rtsp://camera/1"
        return capture

    def jpeg_encoder(frame: np.ndarray) -> tuple[bool, np.ndarray]:
        assert frame.shape == (2, 2, 3)
        return True, np.frombuffer(b"jpeg-bytes", dtype=np.uint8)

    stream = _mjpeg_stream(
        "rtsp://camera/1",
        capture_factory=capture_factory,
        jpeg_encoder=jpeg_encoder,
        frame_delay_sec=0,
        max_frames=1,
    )

    part = next(stream)

    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"Cache-Control: no-store" in part
    assert b"jpeg-bytes" in part
    assert part.endswith(b"\r\n")

    try:
        next(stream)
    except StopIteration:
        pass
    else:
        raise AssertionError("stream should stop after max_frames")
    assert capture.released is True


def test_latest_frame_stream_uses_most_recent_frame_without_capture_reads() -> None:
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 255, dtype=np.uint8)
    frames = [frame_a, frame_b]

    def latest_frame() -> np.ndarray:
        return frames.pop(0)

    encoded = []

    def jpeg_encoder(frame: np.ndarray) -> tuple[bool, np.ndarray]:
        encoded.append(int(frame[0, 0, 0]))
        return True, np.frombuffer(f"jpeg-{encoded[-1]}".encode("ascii"), dtype=np.uint8)

    stream = _latest_frame_mjpeg_stream(
        latest_frame,
        jpeg_encoder=jpeg_encoder,
        frame_delay_sec=0,
        max_frames=2,
    )

    first = next(stream)
    second = next(stream)

    assert b"jpeg-0" in first
    assert b"jpeg-255" in second
    assert encoded == [0, 255]


def test_jpeg_preview_frame_reads_one_frame_and_releases_capture() -> None:
    capture = FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])

    def capture_factory(source: str) -> FakeCapture:
        assert source == "rtsp://camera/1"
        return capture

    def jpeg_encoder(frame: np.ndarray) -> tuple[bool, np.ndarray]:
        return True, np.frombuffer(b"single-jpeg", dtype=np.uint8)

    frame = _jpeg_preview_frame(
        "rtsp://camera/1",
        capture_factory=capture_factory,
        jpeg_encoder=jpeg_encoder,
    )

    assert frame == b"single-jpeg"
    assert capture.released is True
