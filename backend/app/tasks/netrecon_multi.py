"""NetRecon multi-layer reconstruction from a FunClu-K export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register
from app.pure.charts import netrecon_html
from app.pure.netrecon import (
    build_multilayer_export_zip,
    build_network_html,
    build_network_summary,
    collect_network_export_artifacts,
    fit_single_network,
    load_funclu_export,
    sort_cluster_names,
)
from app.tasks.common import write_text
from app.tasks.netrecon import NetReconSingleParams


class NetReconMultiParams(NetReconSingleParams):
    scope: Literal["all", "inter", "intra"] = "all"
    clusters: list[str] = Field(default_factory=list)
    member_cap: int = Field(default=0, ge=0, le=50000)


MI_STAGES = (
    StageSpec("netrecon.stage.load", 5),
    StageSpec("netrecon.stage.prepare", 5),
    StageSpec("netrecon.stage.multilayer", 75),
    StageSpec("netrecon.stage.overview", 10),
    StageSpec("netrecon.stage.export", 5),
)


def _funclu_zip_input(ctx: TaskContext) -> Path:
    inputs = [item for item in ctx.inputs if item.role == "funclu_zip"]
    if len(inputs) != 1:
        raise ValueError("Exactly one FunClu-K export ZIP is required")
    path = Path(inputs[0].path)
    if path.suffix.lower() != ".zip":
        raise ValueError("The NetRecon multilayer input must be a FunClu-K ZIP")
    return path


def _network_label(layer: str, condition: str, cluster: str) -> str:
    if layer == "inter_cluster":
        return f"{condition} · inter"
    return f"{condition} · {cluster}"


def _run_netrecon_multi(ctx: TaskContext) -> None:
    params = NetReconMultiParams.model_validate(ctx.params)
    path = _funclu_zip_input(ctx)
    ctx.check_cancelled()
    ctx.progress(
        "netrecon.stage.load", 0.02, "netrecon.message.loading", name=path.name
    )
    (
        labels_df,
        cluster_sizes_df,
        centers,
        center_responses,
        members,
        member_responses,
    ) = load_funclu_export(path.read_bytes())
    condition_names = list(centers)
    available_clusters = sort_cluster_names(
        {name for cluster_map in members.values() for name in cluster_map}
    )
    selected_clusters = (
        [name for name in params.clusters if name in set(available_clusters)]
        if params.clusters
        else available_clusters
    )
    build_inter = params.scope in {"all", "inter"}
    build_intra = params.scope in {"all", "intra"}
    if not selected_clusters and build_intra and not build_inter:
        raise ValueError("No selected cluster is present in the FunClu-K export")
    aggregation = (
        "energy_share"
        if params.algorithm == "dynamic_idop"
        else params.adjacency_aggregation
    )
    ctx.progress(
        "netrecon.stage.prepare",
        0.08,
        "netrecon.message.multi_prepare",
        conditions=len(condition_names),
        clusters=len(selected_clusters),
    )

    skipped: list[dict] = []
    work: list[dict] = []
    if build_inter:
        for condition in condition_names:
            if condition not in center_responses:
                skipped.append(
                    {
                        "layer": "condition",
                        "condition": condition,
                        "cluster": "",
                        "n_nodes": int(centers[condition].shape[1]),
                        "reason": "missing cluster_center_quasi_dynamic in FunClu-K export",
                    }
                )
                continue
            work.append(
                {
                    "layer": "inter_cluster",
                    "condition": condition,
                    "cluster": "",
                    "curve": centers[condition],
                    "response": center_responses[condition],
                }
            )
    if build_intra:
        for condition in condition_names:
            for cluster in selected_clusters:
                member = members.get(condition, {}).get(cluster)
                if member is None:
                    continue
                response = member_responses.get(condition, {}).get(cluster)
                if member.shape[1] < 2:
                    skipped.append(
                        {
                            "layer": "intra_cluster",
                            "condition": condition,
                            "cluster": cluster,
                            "n_nodes": int(member.shape[1]),
                            "reason": "fewer than 2 members",
                        }
                    )
                    continue
                if params.member_cap > 0 and member.shape[1] > params.member_cap:
                    skipped.append(
                        {
                            "layer": "intra_cluster",
                            "condition": condition,
                            "cluster": cluster,
                            "n_nodes": int(member.shape[1]),
                            "reason": f"n_nodes exceeds member cap ({params.member_cap})",
                        }
                    )
                    continue
                if response is None:
                    skipped.append(
                        {
                            "layer": "intra_cluster",
                            "condition": condition,
                            "cluster": cluster,
                            "n_nodes": int(member.shape[1]),
                            "reason": "missing member quasi_dynamic response",
                        }
                    )
                    continue
                work.append(
                    {
                        "layer": "intra_cluster",
                        "condition": condition,
                        "cluster": cluster,
                        "curve": member,
                        "response": response,
                    }
                )

    inter_cluster: dict[str, dict] = {}
    intra_cluster: dict[str, dict[str, dict]] = {}
    total_items = max(1, len(work) + len(skipped))
    completed = len(skipped)
    if skipped:
        ctx.log(
            "netrecon.message.multi_skipped",
            level="warning",
            count=len(skipped),
            reason="; ".join(sorted({str(item["reason"]) for item in skipped})),
        )
    for index, job in enumerate(work):
        ctx.check_cancelled()
        layer = str(job["layer"])
        condition = str(job["condition"])
        cluster = str(job["cluster"])
        label = _network_label(layer, condition, cluster)
        ctx.progress(
            "netrecon.stage.multilayer",
            0.10 + 0.72 * completed / total_items,
            "netrecon.message.multi_job",
            index=index + 1,
            total=len(work),
            name=label,
        )
        try:
            network = fit_single_network(
                job["curve"],
                job["response"],
                algorithm=params.algorithm,
                max_order=params.max_order,
                nonneg_self=params.nonneg_self,
                max_interactions=params.max_interactions,
                adjacency_aggregation=aggregation,
                lasso_alpha=params.lasso_alpha,
                lasso_windows=params.lasso_windows,
                lasso_threshold=params.lasso_threshold,
                n_fourier=params.n_fourier,
                dynamic_window=params.dynamic_window,
                dynamic_fs=params.dynamic_fs,
            )
        except Exception as exc:  # noqa: BLE001 - per-network failures are skipped
            skipped.append(
                {
                    "layer": layer,
                    "condition": condition,
                    "cluster": cluster,
                    "n_nodes": int(job["curve"].shape[1]),
                    "reason": str(exc),
                }
            )
            ctx.log(
                "netrecon.message.multi_network_failed",
                level="warning",
                name=label,
                raw_text=f"{label}: {exc}",
            )
        else:
            network["condition"] = condition
            network["cluster"] = cluster
            network["algorithm"] = params.algorithm
            if layer == "inter_cluster":
                inter_cluster[condition] = network
            else:
                intra_cluster.setdefault(condition, {})[cluster] = network
        completed += 1

    if not inter_cluster and not intra_cluster:
        raise ValueError(
            "No multilayer network could be fitted; see the skipped networks in the log"
        )
    result = {
        "params": params.model_dump(),
        "scope": {
            "layers": params.scope,
            "clusters": selected_clusters,
            "member_cap": int(params.member_cap),
        },
        "inter_cluster": inter_cluster,
        "intra_cluster": intra_cluster,
        "skipped": skipped,
    }

    ctx.check_cancelled()
    charts_dir = ctx.output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    ctx.progress(
        "netrecon.stage.overview", 0.84, "netrecon.message.multi_overview"
    )
    inter_adj = {
        condition: network["adj_df"]
        for condition, network in inter_cluster.items()
    }
    intra_adj = {
        condition: {
            cluster: network["adj_df"]
            for cluster, network in cluster_map.items()
        }
        for condition, cluster_map in intra_cluster.items()
        if cluster_map
    }
    overview_html = netrecon_html.render_multilayer_overview(
        inter_adj,
        intra_adj,
        conditions=condition_names,
        modules=sort_cluster_names(
            {cluster for cluster_map in intra_adj.values() for cluster in cluster_map}
        ),
        title="Multilayer IdopNetwork overview",
        subtitle=(
            f"{params.algorithm} · scope={params.scope} · "
            f"{len(condition_names)} conditions · {len(selected_clusters)} clusters"
        ),
    )
    overview_path = write_text(
        charts_dir / "multilayer_overview.html", overview_html
    )
    ctx.add_artifact(
        overview_path,
        "charts/multilayer_overview.html",
        "netrecon_multilayer_overview_html",
        kind="html",
        mime="text/html",
    )
    for condition, network in inter_cluster.items():
        directory = charts_dir / "networks" / "inter" / condition
        directory.mkdir(parents=True, exist_ok=True)
        network_path = build_network_html(
            network["adj_df"], directory, top_edges=params.top_edges
        )
        ctx.add_artifact(
            network_path,
            f"charts/networks/inter/{condition}.html",
            "netrecon_network_html",
            kind="html",
            mime="text/html",
        )
    for condition, cluster_map in intra_cluster.items():
        for cluster, network in cluster_map.items():
            directory = charts_dir / "networks" / "intra" / condition / cluster
            directory.mkdir(parents=True, exist_ok=True)
            network_path = build_network_html(
                network["adj_df"], directory, top_edges=params.top_edges
            )
            ctx.add_artifact(
                network_path,
                f"charts/networks/intra/{condition}/{cluster}.html",
                "netrecon_network_html",
                kind="html",
                mime="text/html",
            )

    ctx.check_cancelled()
    ctx.progress("netrecon.stage.export", 0.94, "netrecon.message.exporting")
    export_bytes = build_multilayer_export_zip(
        result, labels_df=labels_df, cluster_sizes_df=cluster_sizes_df
    )
    export_path = ctx.output_dir / "multi_layer_idopnetwork_export.zip"
    export_path.write_bytes(export_bytes)
    ctx.add_artifact(
        export_path,
        "multi_layer_idopnetwork_export.zip",
        "multi_layer_idopnetwork_export",
        kind="zip",
        mime="application/zip",
        is_primary=True,
    )
    summary_df = build_network_summary(result)
    summary_csv = write_text(
        ctx.output_dir / "tables" / "network_summary.csv",
        summary_df.to_csv(index=False),
    )
    ctx.add_artifact(
        summary_csv,
        "tables/network_summary.csv",
        "netrecon_table",
        kind="csv",
        mime="text/csv",
    )
    skipped_df = pd.DataFrame(skipped)
    skipped_path = write_text(
        ctx.output_dir / "tables" / "skipped_networks.csv",
        skipped_df.to_csv(index=False),
    )
    ctx.add_artifact(
        skipped_path,
        "tables/skipped_networks.csv",
        "netrecon_table",
        kind="csv",
        mime="text/csv",
    )
    summary = {
        "task": "netrecon.multi",
        "conditions": condition_names,
        "clusters": selected_clusters,
        "scope": params.scope,
        "algorithm": params.algorithm,
        "adjacency_aggregation": aggregation,
        "member_cap": int(params.member_cap),
        "n_inter_networks": len(inter_cluster),
        "n_intra_networks": sum(len(map_) for map_ in intra_cluster.values()),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "networks": summary_df.to_dict(orient="records"),
        "params": params.model_dump(),
    }
    summary_path = write_text(
        ctx.output_dir / "multilayer_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "multilayer_summary.json",
        "multilayer_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress(
        "netrecon.stage.export",
        1.0,
        "netrecon.message.multi_done",
        inter=len(inter_cluster),
        intra=sum(len(map_) for map_ in intra_cluster.values()),
        skipped=len(skipped),
    )


register(
    TaskSpec(
        key="netrecon.multi",
        module="netrecon",
        title_key="tasks.netrecon_multi.title",
        description_key="tasks.netrecon_multi.description",
        params_model=NetReconMultiParams,
        input_roles=("funclu_zip",),
        stages=MI_STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=14400,
        run=_run_netrecon_multi,
    )
)
