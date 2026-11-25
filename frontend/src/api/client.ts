import type {
  Artifact,
  ArtifactTable,
  CreateRunPayload,
  Dataset,
  DerivedChartPayload,
  InspectResult,
  LogEntry,
  RunDetail,
  RunSummary,
  SessionInfo,
  TaskSpec,
} from "../types";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string) {
    super(code);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`/api${path}`, {
    credentials: "include",
    ...init,
    headers,
  });
  if (!response.ok) {
    let code = `http.${response.status}`;
    try {
      const payload = await response.json();
      const detail = payload?.detail;
      if (typeof detail === "string") {
        code = detail;
      } else if (detail?.code) {
        code = detail.code;
      }
    } catch {
      // keep the HTTP fallback code
    }
    throw new ApiError(response.status, code);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  session: () => request<SessionInfo>("/session"),
  tasks: () => request<TaskSpec[]>("/tasks"),

  createRun: (payload: CreateRunPayload) =>
    request<RunDetail>("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listRuns: (limit = 50) => request<RunSummary[]>(`/runs?limit=${limit}`),
  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),
  cancelRun: (runId: string) =>
    request<RunDetail>(`/runs/${runId}/cancel`, { method: "POST" }),
  retryRun: (runId: string) =>
    request<RunDetail>(`/runs/${runId}/retry`, { method: "POST" }),
  derivedChart: (runId: string, payload: DerivedChartPayload) =>
    request<Artifact>(`/runs/${runId}/derived-charts`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  logs: (runId: string, afterSeq = 0) =>
    request<LogEntry[]>(`/runs/${runId}/logs?after_seq=${afterSeq}`),

  datasets: () => request<Dataset[]>("/datasets"),
  inspectDataset: (datasetId: string) =>
    request<InspectResult>(`/datasets/${datasetId}/inspect`),
  uploadDataset: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body: form });
  },
  deleteDataset: (datasetId: string) =>
    request<void>(`/datasets/${datasetId}`, { method: "DELETE" }),

  artifactUrl: (artifactId: string, mode: "content" | "download" = "content") =>
    `/api/artifacts/${artifactId}/${mode}`,
  listArtifacts: (semanticType?: string) => {
    const query = new URLSearchParams({ limit: "100" });
    if (semanticType) {
      query.set("semantic_type", semanticType);
    }
    return request<Artifact[]>(`/artifacts?${query.toString()}`);
  },
  inspectArtifact: (artifactId: string) =>
    request<InspectResult>(`/artifacts/${artifactId}/inspect`),
  artifactTable: (artifactId: string, limit = 500) =>
    request<ArtifactTable>(`/artifacts/${artifactId}/table?limit=${limit}`),
  artifactJson: <T>(artifactId: string) =>
    request<T>(`/artifacts/${artifactId}/content`),
};
