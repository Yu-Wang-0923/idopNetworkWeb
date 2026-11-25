"""Derived NetRecon charts regenerated from a stored export ZIP."""
from __future__ import annotations

import io
import json
import re
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .models import Artifact, Run, Workspace
from .pure import netrecon as netrecon_pure
from .pure.charts import netrecon_html
from .storage import Storage, sha256_file, zip_members

DERIVED_SEMANTIC = {
    "network": "netrecon_derived_network_html",
    "effect": "netrecon_derived_effect_html",
    "heatmap": "netrecon_derived_heatmap_html",
}
EXPORT_SEMANTICS = (
    "single_layer_idopnetwork_export",
    "multi_layer_idopnetwork_export",
)


class DerivedChartError(Exception):
    """A derived chart request that cannot be fulfilled."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value.replace("/", "-")).strip("-")
    return cleaned[:80] or "network"


def _read_csv(archive: zipfile.ZipFile, name: str, *, index_col=0) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle, index_col=index_col)


def _export_artifact(db: Session, run: Run) -> Artifact:
    artifact = db.scalars(
        select(Artifact)
        .where(
            Artifact.run_id == run.id,
            Artifact.semantic_type.in_(EXPORT_SEMANTICS),
            Artifact.is_primary.is_(True),
        )
        .order_by(Artifact.created_at.desc())
    ).first()
    if artifact is None:
        artifact = db.scalars(
            select(Artifact)
            .where(
                Artifact.run_id == run.id,
                Artifact.semantic_type.in_(EXPORT_SEMANTICS),
            )
            .order_by(Artifact.created_at.desc())
        ).first()
    if artifact is None:
        raise DerivedChartError(
            "derived.export_missing", "The run has no NetRecon export archive"
        )
    return artifact


def _resolve_member(
    run: Run, artifact: Artifact, names: set[str], member: str | None
) -> tuple[str, str]:
    """Return (base prefix, label) for the requested subnet."""
    if artifact.semantic_type == "multi_layer_idopnetwork_export":
        members = zip_members(sorted(names))
        if member:
            normalized = member.replace("\\", "/")
            if normalized not in members:
                raise DerivedChartError(
                    "derived.member_not_found", f"Unknown member: {member}"
                )
            selected = normalized
        elif members:
            selected = members[0]
        else:
            raise DerivedChartError(
                "derived.no_member", "The archive contains no from_to network"
            )
        base = "" if selected == "from_to.csv" else selected.rsplit("/", 1)[0]
        return base, base or "from_to.csv"
    if member:
        raise DerivedChartError(
            "derived.member_not_supported",
            "A single-layer export only contains one network",
        )
    try:
        params = json.loads(run.params_json)
    except ValueError:
        params = {}
    condition = str(params.get("condition") or "")
    return "", condition or "single"


def _intercepts(
    archive: zipfile.ZipFile,
    names: set[str],
    base: str,
    targets: list[str],
    effect_map: dict[str, pd.DataFrame],
    predictions: pd.DataFrame | None,
    *,
    recover: bool,
) -> tuple[pd.Series, bool]:
    name = f"{base}/intercepts.csv" if base else "intercepts.csv"
    if name in names:
        frame = _read_csv(archive, name, index_col=None)
        mapping = dict(zip(frame["target"].astype(str), frame["intercept"].astype(float)))
        return pd.Series({target: mapping.get(target, 0.0) for target in targets}), True
    if recover and predictions is not None:
        values: dict[str, float] = {}
        for target, effect in effect_map.items():
            if target not in predictions.columns:
                continue
            delta = predictions[target].reindex(effect.index) - effect.sum(axis=1)
            valid = delta.dropna()
            if not valid.empty:
                values[target] = float(valid.mean())
        if values:
            return pd.Series({target: values.get(target, 0.0) for target in targets}), True
    return pd.Series(0.0, index=targets), False


def _predictions_for(db: Session, settings: Settings, run: Run) -> pd.DataFrame | None:
    artifact = db.scalars(
        select(Artifact).where(
            Artifact.run_id == run.id,
            Artifact.logical_name == "tables/predictions.csv",
        )
    ).first()
    if artifact is None:
        return None
    path = Storage(settings).artifact_path(run.id, artifact.relative_path)
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0)


def create_derived_chart(
    db: Session,
    settings: Settings,
    run: Run,
    *,
    kind: str,
    member: str | None,
    top_edges: int,
) -> Artifact:
    if kind not in DERIVED_SEMANTIC:
        raise DerivedChartError("derived.invalid_kind", f"Unknown chart kind: {kind}")
    artifact = _export_artifact(db, run)
    storage = Storage(settings)
    export_path = storage.artifact_path(run.id, artifact.relative_path)
    if not export_path.is_file():
        raise DerivedChartError(
            "derived.export_missing", "The export file is no longer on disk"
        )
    data = export_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        base, label = _resolve_member(run, artifact, names, member)
        adjacency_name = f"{base}/adjacency_matrix.csv" if base else "adjacency_matrix.csv"
        if adjacency_name not in names:
            raise DerivedChartError(
                "derived.export_invalid", "The export archive has no adjacency matrix"
            )
        adjacency = _read_csv(archive, adjacency_name)
        targets = [str(column) for column in adjacency.index]

        if kind == "network":
            if not targets:
                raise DerivedChartError(
                    "derived.export_invalid", "The adjacency matrix is empty"
                )
            with tempfile.TemporaryDirectory(
                dir=settings.tmp_dir, prefix="derived-"
            ) as temp:
                path = netrecon_pure.build_network_html(
                    adjacency, Path(temp), top_edges=int(top_edges)
                )
                html = path.read_bytes()
        elif kind == "heatmap":
            html = netrecon_html.render_heatmap(
                adjacency,
                title="Adjusted Matrix Heatmap",
                subtitle=f"{label} · derived",
            ).encode("utf-8")
        else:
            effect_map: dict[str, pd.DataFrame] = {}
            for target in targets:
                name = f"{base}/effect/{target}.csv" if base else f"effect/{target}.csv"
                if name in names:
                    effect_map[target] = _read_csv(archive, name)
            if not effect_map:
                raise DerivedChartError(
                    "derived.export_invalid", "The export archive has no effect tables"
                )
            predictions = (
                _predictions_for(db, settings, run)
                if artifact.semantic_type == "single_layer_idopnetwork_export"
                else None
            )
            intercept, complete = _intercepts(
                archive,
                names,
                base,
                list(effect_map),
                effect_map,
                predictions,
                recover=artifact.semantic_type == "single_layer_idopnetwork_export",
            )
            first_effect = next(iter(effect_map.values()))
            predicted = pd.DataFrame(
                index=first_effect.index,
                columns=list(effect_map),
                dtype=float,
            )
            effects = [effect_map[target].reindex(first_effect.index) for target in effect_map]
            subtitle = f"{label} · derived"
            if not complete:
                subtitle += " · intercept not stored in this export; baseline shown as 0"
            html = netrecon_html.render_effects(
                predicted,
                effects,
                intercept,
                title="Effect decomposition",
                subtitle=subtitle,
            ).encode("utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    relative = f"derived/{kind}_{_slug(label)}_{stamp}-{token}.html"
    destination = storage.run_artifacts_dir(run.id) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(html)
    artifact_row = Artifact(
        run_id=run.id,
        workspace_id=run.workspace_id,
        logical_name=relative,
        relative_path=relative,
        semantic_type=DERIVED_SEMANTIC[kind],
        kind="html",
        mime="text/html",
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        is_primary=False,
    )
    db.add(artifact_row)
    workspace = db.get(Workspace, run.workspace_id)
    if workspace is not None:
        workspace.storage_bytes = (workspace.storage_bytes or 0) + artifact_row.size_bytes
    db.commit()
    db.refresh(artifact_row)
    return artifact_row
