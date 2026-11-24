"""FastAPI application entry point."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import artifacts, datasets, runs, session, tasks
from .bootstrap import setup_runtime
from .config import get_settings
from .db import init_db
from .security import sign_workspace, unsign_workspace


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    setup_runtime()
    yield


app = FastAPI(
    title="idopNetwork Web API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def workspace_cookie_middleware(request, call_next):
    settings = get_settings()
    workspace_id = unsign_workspace(settings, request.cookies.get(settings.session_cookie))
    is_new = workspace_id is None
    if is_new:
        workspace_id = str(uuid.uuid4())
    request.state.workspace_id = workspace_id
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            key=settings.session_cookie,
            value=sign_workspace(settings, workspace_id),
            max_age=settings.session_max_age_days * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=settings.cookie_secure,
            path="/",
        )
    return response


app.include_router(session.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(artifacts.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "idopNetwork Web"}
