# idopNetwork Web

A web computing platform for the **idopNetwork** complex-systems analysis framework. It provides four computational modules as a guided pipeline:

1. **Curve Fitting** — power-law fitting of quasi-dynamic data, or SWT denoising of time-series data.
2. **FunClu** — functional clustering with EM Gaussian mixtures.
3. **NetRecon** — single-layer and multi-layer interaction network reconstruction.
4. **NetAnal** — GLMY persistent path homology analysis.

The platform is being built in phases. Phases 0–5 are complete: anonymous workspace, durable job queue, Curve Fitting, FunClu, NetRecon (single and multi), and NetAnal.

## Architecture

```
frontend/   React + TypeScript + Ant Design 5 (Vite)
backend/    FastAPI + SQLite; API process + worker process
            worker claims a run → spawns a child process → child executes the task
shared/     job message keys and translations (zh/en)
data/       runtime data (gitignored)
```

No registration or login is required. A signed anonymous workspace cookie groups a browser's own uploads and runs. Runs and artifacts have unguessable IDs and are readable by link; only the owning workspace can cancel or retry them.

## Local development

Requirements: Python 3.12+, Node.js 20+.

The API and worker must run with a Python interpreter that has the backend dependencies installed. If `python` on your PATH points to another installation, activate the right environment first or use its full path.

Install the backend dependencies (from the repository root):

```powershell
python -m pip install -r backend/requirements-dev.txt
```

Start the API, the worker, and the frontend in three terminals:

```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

```powershell
cd backend
python -m app.worker
```

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Status

- Phase 0: platform skeleton, anonymous workspace, job queue, demo task, frontend shell.
- Phase 1: Curve Fitting — quasi-dynamic power fitting and dynamic SWT denoising, with the export ZIP, publication-grade HTML figures (per-lead ECG figures in the upstream full-record/zoom style), and table downloads.
- Phase 2: FunClu — EM functional clustering with a FunClu-K export ZIP and cluster-profile HTML, plus a BIC scan over K × S1–S4 covariance structures (best of n_init starts, AIC/ΔBIC/selected table, comparison plot) that can be cancelled at K boundaries.
- Phase 3: NetRecon single-layer — ASGL, static idop and dynamic idopECG reconstruction with an interactive network HTML (offline vis-network), effect-decomposition HTML, adjacency heatmap HTML, and the `single_layer_idopnetwork_export.zip` contract.
- Phase 4: NetRecon multi-layer — inter-module and intra-module networks from a FunClu-K export, with a publication HTML overview matrix, per-network interactive HTML, skipped-network reporting, and `multi_layer_idopnetwork_export.zip`.
- Phase 5: NetAnal — regular GLMY path homology on a subnetwork of a single- or multi-layer NetRecon export, with all-edges-coloured or sign-separated figures (PNG/SVG/PDF), tables, and a `glmy_<mode>.zip` package.
- Later phase: platform hardening (derived charts, quotas and cleanup, docs, cross-browser checks).

### FunClu engine

FunClu runs on the **Fast-Funclu** estimator (`fast-funclu`, pinned at commit `583cebbeb8416678f06d979e03b7461701d1c74d`, provided and authorized by its author). It clusters the observed `quasi_dynamic` tables of a Curve Fitting export, never the sampled power curves in `curve_sample`, and is adapted for this platform: the component mean is a power law `exp(alpha + beta * z)` on the supplied scale, with additive SAD(1) residuals on that scale. A feature is clustered only when its observations and its fitted curve are finite in every condition. Initialization defaults to `log_c_beta` (falling back to KMeans for non-positive responses); KMeans/MiniBatchKMeans and random starts are selectable. Covariance structures S1–S4 share the SAD(1) `phi`/`gamma` parameters across conditions; the FunClu-K task defaults to S1, and BIC can compare any subset (default S1). In the FunClu-K export, `cluster_center_curve_sample.csv` holds the fitted cluster means on each condition's `curve_sample` grid, and `cluster_center_quasi_dynamic.csv` holds the mean of the members' observations. The cluster profile figure always draws those fitted means and can show the members either as observations or as fitted curves.

## License

MIT. See [LICENSE](LICENSE).
