from __future__ import annotations

from ptz_pano.api.main import (
    _apply_local_target_corrections,
    _remember_local_correction,
)


def test_local_target_correction_applies_reverse_observed_shift() -> None:
    corrections = [
        {
            "desired_yaw_deg": 10.0,
            "desired_pitch_deg": 5.0,
            "observed_yaw_deg": 14.0,
            "observed_pitch_deg": 2.0,
            "delta_yaw_deg": -4.0,
            "delta_pitch_deg": 3.0,
        }
    ]

    corrected = _apply_local_target_corrections(10.0, 5.0, corrections)

    assert corrected["yaw_deg"] == 6.0
    assert corrected["pitch_deg"] == 8.0
    assert corrected["applied"] is True


def test_local_target_correction_ignores_far_points() -> None:
    corrections = [
        {
            "desired_yaw_deg": 10.0,
            "desired_pitch_deg": 5.0,
            "observed_yaw_deg": 14.0,
            "observed_pitch_deg": 2.0,
            "delta_yaw_deg": -4.0,
            "delta_pitch_deg": 3.0,
        }
    ]

    corrected = _apply_local_target_corrections(120.0, 5.0, corrections)

    assert corrected["yaw_deg"] == 120.0
    assert corrected["pitch_deg"] == 5.0
    assert corrected["applied"] is False


def test_local_target_correction_uses_only_same_scan_when_requested() -> None:
    corrections = [
        {
            "scan_id": "scan_a",
            "desired_yaw_deg": 10.0,
            "desired_pitch_deg": 5.0,
            "observed_yaw_deg": 14.0,
            "observed_pitch_deg": 2.0,
            "delta_yaw_deg": -4.0,
            "delta_pitch_deg": 3.0,
        },
        {
            "scan_id": "scan_b",
            "desired_yaw_deg": 10.0,
            "desired_pitch_deg": 5.0,
            "observed_yaw_deg": 8.0,
            "observed_pitch_deg": 7.0,
            "delta_yaw_deg": 2.0,
            "delta_pitch_deg": -2.0,
        },
    ]

    corrected = _apply_local_target_corrections(10.0, 5.0, corrections, scan_id="scan_b")

    assert corrected["yaw_deg"] == 12.0
    assert corrected["pitch_deg"] == 3.0
    assert corrected["applied"] is True
    assert corrected["correction_count"] == 1


def test_local_correction_replaces_previous_sample_in_same_small_area() -> None:
    existing = [
        {
            "scan_id": "scan_a",
            "desired_yaw_deg": 10.0,
            "desired_pitch_deg": 5.0,
            "delta_yaw_deg": -4.0,
            "delta_pitch_deg": 1.0,
            "created_at": "2026-06-06T12:00:00",
        },
        {
            "scan_id": "scan_a",
            "desired_yaw_deg": 10.8,
            "desired_pitch_deg": 5.2,
            "delta_yaw_deg": -3.0,
            "delta_pitch_deg": 0.5,
            "created_at": "2026-06-06T12:00:01",
        },
    ]
    new_correction = {
        "scan_id": "scan_a",
        "desired_yaw_deg": 10.6,
        "desired_pitch_deg": 5.1,
        "delta_yaw_deg": -2.0,
        "delta_pitch_deg": 0.25,
        "created_at": "2026-06-06T12:00:10",
    }

    remembered = _remember_local_correction(existing, new_correction)

    assert len(remembered) == 1
    assert remembered[0]["delta_yaw_deg"] == -2.0


def test_local_corrections_keep_only_latest_five_in_same_area() -> None:
    existing = [
        {
            "desired_yaw_deg": 10.0 + index * 4.0,
            "desired_pitch_deg": 5.0,
            "delta_yaw_deg": float(index),
            "delta_pitch_deg": 0.0,
            "created_at": f"2026-06-06T12:00:0{index}",
        }
        for index in range(5)
    ]
    new_correction = {
        "desired_yaw_deg": 18.0,
        "desired_pitch_deg": 5.0,
        "delta_yaw_deg": 99.0,
        "delta_pitch_deg": 0.0,
        "created_at": "2026-06-06T12:00:10",
    }

    remembered = _remember_local_correction(existing, new_correction)

    assert len(remembered) == 5
    assert [item["delta_yaw_deg"] for item in remembered] == [0.0, 1.0, 3.0, 4.0, 99.0]


def test_local_corrections_do_not_evict_other_areas() -> None:
    existing = [
        {
            "desired_yaw_deg": 10.0 + index * 0.5,
            "desired_pitch_deg": 5.0,
            "delta_yaw_deg": float(index),
            "delta_pitch_deg": 0.0,
            "created_at": f"2026-06-06T12:00:0{index}",
        }
        for index in range(5)
    ]
    far_area = {
        "desired_yaw_deg": 120.0,
        "desired_pitch_deg": -10.0,
        "delta_yaw_deg": 50.0,
        "delta_pitch_deg": 1.0,
        "created_at": "2026-06-06T12:01:00",
    }

    remembered = _remember_local_correction(existing, far_area)

    assert len(remembered) == 6
    assert remembered[-1]["delta_yaw_deg"] == 50.0
