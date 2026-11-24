"""Signed anonymous workspace cookies."""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import Settings


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.resolved_secret_key(), salt="idop-workspace")


def sign_workspace(settings: Settings, workspace_id: str) -> str:
    return _serializer(settings).dumps(workspace_id)


def unsign_workspace(settings: Settings, value: str | None) -> str | None:
    if not value:
        return None
    try:
        workspace_id = _serializer(settings).loads(
            value, max_age=settings.session_max_age_days * 24 * 60 * 60
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(workspace_id, str):
        return None
    return workspace_id
