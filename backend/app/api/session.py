"""Anonymous session endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..models import Workspace
from ..schemas import SessionOut
from .deps import get_workspace

router = APIRouter()


@router.get("/session", response_model=SessionOut)
def read_session(
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
) -> SessionOut:
    return SessionOut(
        created_at=workspace.created_at,
        retention_days=settings.retention_days,
        max_csv_bytes=settings.max_csv_bytes,
        max_zip_bytes=settings.max_zip_bytes,
    )
