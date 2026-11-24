# idopNetwork Web

A web computing platform for the **idopNetwork** complex-systems analysis framework. It provides four computational modules as a guided pipeline:

1. **Curve Fitting** — power-law fitting of quasi-dynamic data, or SWT denoising of time-series data.
2. **FunClu** — functional clustering with EM Gaussian mixtures.
3. **NetRecon** — single-layer and multi-layer interaction network reconstruction.
4. **NetAnal** — GLMY persistent path homology analysis.

The platform is being built in phases. Phase 0 (current) provides the anonymous workspace, upload storage, a durable job queue with a separate worker process, and the application shell.

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
- Later phases: Curve Fitting, FunClu, NetRecon, NetAnal, and platform hardening.

## License

MIT. See [LICENSE](LICENSE).
