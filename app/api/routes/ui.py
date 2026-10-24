"""Serve the built frontend and its static assets."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["web UI"])
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
INDEX_FILE = FRONTEND_DIST / "index.html"


@router.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend build is missing. Run npm install and npm run build in frontend/.",
        )
    return FileResponse(INDEX_FILE, media_type="text/html")


@router.get("/{asset_path:path}", include_in_schema=False)
def frontend_asset(asset_path: str) -> FileResponse:
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend build is missing. Run npm install and npm run build in frontend/.",
        )

    root = FRONTEND_DIST.resolve()
    requested_file = (root / asset_path).resolve()
    try:
        requested_file.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc

    if requested_file.is_file():
        return FileResponse(requested_file)
    if Path(asset_path).suffix:
        raise HTTPException(status_code=404, detail="Asset not found")
    # Allow client-side routes to fall back to the SPA entry point.
    return FileResponse(INDEX_FILE, media_type="text/html")
