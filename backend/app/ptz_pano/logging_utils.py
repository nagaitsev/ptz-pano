from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MANAGED_HANDLER_ATTR = "_ptz_pano_managed"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ptz_pano")
    logger.setLevel(logging.DEBUG)

    resolved_path = Path(log_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTR, False):
            current_path = getattr(handler, "baseFilename", None)
            if current_path != str(resolved_path):
                logger.removeHandler(handler)
                handler.close()

    if not any(
        getattr(handler, _MANAGED_HANDLER_ATTR, False)
        and getattr(handler, "baseFilename", None) == str(resolved_path)
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            resolved_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        setattr(handler, _MANAGED_HANDLER_ATTR, True)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger
