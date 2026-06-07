from __future__ import annotations

import logging

from ptz_pano.logging_utils import setup_logging


def test_setup_logging_writes_messages_to_file(tmp_path) -> None:
    log_path = tmp_path / "logs" / "ptz-pano.log"

    setup_logging(log_path)
    logger = logging.getLogger("ptz_pano.test")
    logger.info("camera preview failed")

    for handler in logging.getLogger("ptz_pano").handlers:
        handler.flush()

    assert log_path.exists()
    assert "camera preview failed" in log_path.read_text(encoding="utf-8")
