from __future__ import annotations

from ptz_pano.models import capture_config_from_dict


def test_capture_config_accepts_optional_whep_url() -> None:
    config = capture_config_from_dict(
        {
            "kind": "rtsp",
            "source": "rtsp://10.1.1.13:554/1",
            "resolution": [1920, 1080],
            "whep_url": "http://127.0.0.1:8889/camera/whep",
        }
    )

    assert config.whep_url == "http://127.0.0.1:8889/camera/whep"
    assert config.resolution == (1920, 1080)
