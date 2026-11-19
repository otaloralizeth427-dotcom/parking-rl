// Contrato entre la web y el servidor de la simulación (api/server.py).
// Unidades del ambiente: metros, radianes, segundos (ver docs/ficha_ambiente.md).

export type LevelId = "N1" | "N2" | "N3" | "N4" | "N5" | string;
export type Status = "idle" | "running" | "paused" | "success" | "collision" | "timeout";
export type CollisionKind = "pedestrian" | "obstacle" | "boundary";
export type Outcome = "success" | CollisionKind | "timeout";

export interface Slot {
  id: number; x: number; y: number; width: number; depth: number;
  row: number | string; opening: number; name?: string;
}

export interface Obstacle {
  kind: string; x: number; y: number; hx: number; hy: number; angle: number;
  color?: number; facing?: number;
}

export interface Layout {
  kind: "aisle" | "garage" | string;
  lot: { width: number; height: number };
  slots: Slot[];
  obstacles: Obstacle[];
}

export interface Car { x: number; y: number; heading: number; speed: number; steer: number; length: number; width: number }
export interface Sensor { angle: number; distance: number; origin?: [number, number]; range?: number }
export interface Pedestrian { x: number; y: number; dir: number; moving: boolean; heading?: number }

export interface Training {
  running: boolean; episode: number; total: number; success_rate: number; returns: number[];
  experiment_id?: string; run_id?: string; error?: string;
}

export interface ModelIdentity {
  git_commit?: string; model_sha256?: string; seed?: number; test_success_rate?: number | null;
  mlflow_run_id?: string; mlflow_experiment?: string;
}

/** Estado de un paso (respuesta a reset, step, state y cada cuadro de steps). */
export interface Frame {
  type: "state" | "reset";
  car: Car;
  target: { id: number; x: number; y: number; angle: number };
  sensors: Sensor[];
  sensor_range: number;
  distance: number;
  heading_error: number;
  in_slot: boolean;
  steps: number;
  pedestrians: Pedestrian[];
  episode: number;
  level: LevelId;
  environment_version: string;
  collision_kind: CollisionKind | null;
  status: Exclude<Status, "idle" | "paused"> | "idle";
  reward: number;
  total_reward: number;
  action: number | null;
  action_name: string | null;
  agent: string;
  agent_label?: string;
  mode: string;
  model: ModelIdentity | null;
  collisions_total: number;
  successes_total: number;
  training: Training;
}

export interface ResetFrame extends Frame { type: "reset"; layout: Layout }

export interface CatalogItem extends ModelIdentity {
  id: string; label: string; mode: string; available: boolean; status?: string;
  agent?: string; environment_version?: string; created_at?: string;
  test_mean_time_s?: number | null; diagnostic?: string | null;
  hypothesis?: string | null; change_vs_baseline?: string | null; evaluation_suite_id?: string | null;
  parent_run_id?: string | null; mean_return?: number | null; outcomes?: Partial<Record<Outcome, number>>;
}

export interface ServingEntry {
  run_id: string; action: "promote" | "rollback"; reason: string; environment_version: string;
  model_sha256: string; test_success_rate: number | null; at: string;
}

export interface Serving { current: ServingEntry; previous: string | null; history: ServingEntry[] }

export interface Config {
  type: "config";
  actions: string[];
  environment_version: string;
  levels: { id: LevelId; name: string }[];
  level: LevelId;
  sensor_noise: number;
  agent: string;
  agents: CatalogItem[];
  serving: Serving | null;
  can_train: boolean;
}

export interface Summary {
  episodes: number; success_rate: number; outcomes: Record<Outcome, number>;
  mean_return: number; mean_time_s: number; mean_final_distance_m: number;
  mean_final_heading_error_deg: number; mean_latency_ms: number;
}

export interface Evaluation {
  type: "evaluation"; level: LevelId; suite_id: string; seed_start: number; episodes: number;
  results: {
    agent: string; label: string; environment_version: string; model_sha256?: string;
    git_commit?: string; summary: Summary; outcomes_by_case: Outcome[];
  }[];
}

export interface Metrics {
  uptime_s: number; steps: number; git_commit: string | null; live_sessions: number;
  step_latency_ms: { mean: number; p95: number; window: number };
  levels: Record<LevelId, { episodes: number; success_rate: number; outcomes: Partial<Record<Outcome, number>> }>;
  served: Record<LevelId, string | null>;
}

/** Comandos del WebSocket y lo que responde cada uno. */
export interface Commands {
  config: { payload: Record<string, never>; response: Config };
  reset: { payload: { seed?: number; level?: LevelId; sensor_noise?: number }; response: ResetFrame };
  step: { payload: { action?: number }; response: Frame };
  steps: { payload: { n: number }; response: { type: "states"; frames: Frame[] } };
  state: { payload: Record<string, never>; response: Frame };
  agent: { payload: { agent: string }; response: Config };
  train: { payload: { episodes: number }; response: { type: "training"; training: Training } };
  evaluate: { payload: { agents: string[]; episodes: number }; response: Evaluation };
}

/** Selector de nivel: GET /api/levels */
export interface LevelInfo {
  id: LevelId; name: string; environment_version: string; served: string | null; success_rate: number | null;
}

export interface PolicyResult extends Summary { label: string }

/** Métricas de un nivel: GET /api/level-metrics?level=… (resultados versionados del pipeline) */
export interface LevelMetrics {
  level: LevelId; name: string; environment_version: string; description?: string;
  served: (ModelIdentity & {
    run_id: string; agent: string; experiment_id: string; hypothesis?: string; environment_version: string;
    evaluation_suite_id?: string; parent_run_id?: string | null; created_at?: string;
    budget?: { timesteps?: number; episodes?: number; wall_time_s?: number };
  }) | null;
  suite: { level_id: string; suite_id: string; policies: Record<string, PolicyResult> } | null;
  final_test: { suite_id: string; evaluated_at?: string; policies: Record<string, PolicyResult> } | null;
  has_curves: boolean;
}
