from __future__ import annotations

import json
import socket
from pathlib import Path

from ptz_pano.tools.portable_server import (
    _default_whep_url,
    _ensure_portable_capture_defaults,
    _ensure_runtime_layout,
    _mediamtx_config_text,
    _wait_for_local_port,
)


def test_default_whep_url_is_localhost() -> None:
    assert _default_whep_url() == "http://127.0.0.1:8889/camera/whep"


def test_ensure_portable_capture_defaults_sets_rtsp_host_and_whep_url(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.local.json"
    config_path.write_text(
        json.dumps(
            {
                "camera": {"host": "192.168.100.6", "port": 5678},
                "capture": {
                    "kind": "rtsp",
                    "source": "rtsp://192.168.1.50:554/1",
                    "resolution": [1920, 1080],
                },
            }
        ),
        encoding="utf-8",
    )

    updated = _ensure_portable_capture_defaults(config_path, _default_whep_url())

    assert updated["capture"]["source"] == "rtsp://192.168.100.6:554/1"
    assert updated["capture"]["whep_url"] == "http://127.0.0.1:8889/camera/whep"

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["capture"]["source"] == "rtsp://192.168.100.6:554/1"
    assert persisted["capture"]["whep_url"] == "http://127.0.0.1:8889/camera/whep"


def test_mediamtx_config_text_uses_camera_rtsp_source() -> None:
    config_text = _mediamtx_config_text("rtsp://192.168.100.6:554/1")

    assert "source: rtsp://192.168.100.6:554/1" in config_text
    assert "webrtcAdditionalHosts:" in config_text
    assert "127.0.0.1" in config_text


def test_ensure_runtime_layout_creates_logs_dir(tmp_path: Path) -> None:
    _ensure_runtime_layout(tmp_path)

    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "data" / "scans").is_dir()


def test_wait_for_local_port_detects_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()

        assert host == "127.0.0.1"
        assert _wait_for_local_port(port, timeout_sec=0.5)
