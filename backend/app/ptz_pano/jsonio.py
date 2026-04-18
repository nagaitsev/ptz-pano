from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ptz_pano.models import to_jsonable


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(value), file, ensure_ascii=False, indent=2)
        file.write("\n")

