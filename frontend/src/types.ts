export interface SessionInfo {
  created_at: string;
  retention_days: number;
  max_csv_bytes: number;
  max_zip_bytes: number;
}

export interface StageSpec {
  key: string;
  weight: number;
}

export interface TaskSpec {
  key: string;
  module: string;
  title_key: string;
  description_key: string;
  params_schema: Record<string, unknown>;
  input_roles: string[];
  stages: StageSpec[];
  maturity: "stable" | "beta";
  enabled: boolean;
}

export interface RunMessage {
  key: string | null;
  params: Record<string, unknown>;
}

export interface Artifact {
  id: string;
  run_id: string;
  logical_name: string;
  semantic_type: string;
  kind: string;
  mime: string;
  size_bytes: number;
  sha256: string;
  is_primary: boolean;
  created_at: string;
}

export interface RunSummary {
  id: string;
  module: string;
  task_key: string;
  status: RunStatus;
  progress: number;
  message: RunMessage;
  created_at: string;
  finished_at: string | null;
}

export interface RunInput {
  role: string;
  ordinal: number;
  dataset_id: string | null;
  artifact_id: string | null;
  filename: string;
}

export interface RunDetail extends RunSummary {
  stage_key: string | null;
  error_text: string | null;
  params: Record<string, unknown>;
  params_version: number;
  started_at: string | null;
  inputs: RunInput[];
  artifacts: Artifact[];
}

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface BicSummary {
  recommended_k: number;
  selected_model: string;
  k_values: number[];
  k_min: number;
  k_max: number;
  selected_at_boundary?: boolean;
  /** Missing in summaries written before the field existed. */
  boundary_side?: "upper" | "lower" | null;
  min_size_violated?: boolean;
  params?: { min_cluster_size?: number };
}

export interface LogEntry {
  seq: number;
  ts: string;
  level: string;
  message_key: string | null;
  message_params: Record<string, unknown>;
  raw_text: string | null;
}

export interface Dataset {
  id: string;
  original_filename: string;
  size_bytes: number;
  sha256: string;
  detected_kind: string;
  content_type: string;
  created_at: string;
}

export interface InspectResult {
  kind: string;
  conditions: string[];
  members: string[];
  features?: number | null;
  clusters?: string[];
}

export interface ArtifactTable {
  columns: string[];
  rows: Array<Array<string | number | boolean | null>>;
  total: number;
}

export interface CreateRunPayload {
  task_key: string;
  params: object;
  inputs: Array<{
    role: string;
    ordinal?: number;
    dataset_id?: string;
    artifact_id?: string;
  }>;
}

export interface DerivedChartPayload {
  kind: "network" | "effect" | "heatmap";
  member?: string;
  top_edges?: number;
}
