from __future__ import annotations

from ptz_pano.api.main import _next_zoom_value


def test_next_zoom_value_steps_and_clamps() -> None:
    assert _next_zoom_value(1000, 1, 400) == 1400
    assert _next_zoom_value(1000, -1, 400) == 600
    assert _next_zoom_value(50, -1, 400) == 0
    assert _next_zoom_value(0x3FFE, 1, 400) == 0x4000
