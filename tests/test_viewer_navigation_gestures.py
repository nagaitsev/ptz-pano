from __future__ import annotations

from pathlib import Path


def test_viewer_supports_double_click_and_double_tap_navigation() -> None:
    html = Path("backend/app/ptz_pano/api/viewer.html").read_text(encoding="utf-8")

    assert 'viewport.addEventListener("dblclick"' in html
    assert "handleTapNavigation" in html
    assert "moveCameraAtScreenPoint" in html


def test_viewer_contains_draggable_joystick_window() -> None:
    html = Path("backend/app/ptz_pano/api/viewer.html").read_text(encoding="utf-8")

    assert 'id="cameraJoystick"' in html
    assert 'id="cameraJoystickHandle"' in html
    assert "bindJoystickWindowDrag" in html
    assert "bindJoystickPad" in html
    assert 'id="cameraZoomInZone"' in html
    assert 'id="cameraZoomOutZone"' in html
    assert "cameraZoomLabel cameraZoomLabel--left" in html
    assert "cameraZoomLabel cameraZoomLabel--right" in html
    assert "bindJoystickZoomZones" in html


def test_viewer_supports_webrtc_preview_with_image_fallback() -> None:
    html = Path("backend/app/ptz_pano/api/viewer.html").read_text(encoding="utf-8")

    assert "startWebRtcPreview" in html
    assert "fallbackToImagePreview" in html
    assert "videoElement" in html


def test_viewer_deduplicates_joystick_commands_and_uses_friendly_error_status() -> None:
    html = Path("backend/app/ptz_pano/api/viewer.html").read_text(encoding="utf-8")

    assert "lastCommandKey" in html
    assert "scheduleJoystickCommand" in html
    assert "Ошибка связи с камерой" in html


def test_viewer_pauses_telemetry_while_camera_controls_are_active() -> None:
    html = Path("backend/app/ptz_pano/api/viewer.html").read_text(encoding="utf-8")

    assert "telemetryRequestInFlight" in html
    assert "isCameraControlActive" in html
