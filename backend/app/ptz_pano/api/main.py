from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from ptz_pano.storage.scan_repository import ScanRepository

app = FastAPI(title="PTZ Pano")
repository = ScanRepository(Path("data/scans"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str) -> dict:
    document = repository.load_document(scan_id)
    return {
        "id": document.id,
        "camera": document.camera,
        "capture": document.capture,
        "frames": document.frames,
    }

