export type RunStatus = "running" | "completed" | "stopped" | "failed";
export type StageId = "prepare" | "baseline" | "research" | "refit" | "result";
export type ExperimentStatus = "running" | "ok" | "failed" | "rejected" | "timeout";
export type Level = "smoke" | "full";
export type Direction = "higher" | "lower";
export type SearchMode = "qwen" | "optuna" | "baseline";

export type Params = Record<string, string | number | boolean>;

export interface Experiment {
  id: string;
  parentId: string | null;
  level: Level;
  status: ExperimentStatus;
  /** Какая ошибка замечена */
  observed: string;
  /** Что предложено */
  proposed: string;
  /** Что фактически запущено */
  executed: string;
  /** Что измерено */
  measured: string;
  proposedParams: Params;
  executedParams: Params;
  featureChanges: string[];
  score: number | null;
  /** Разность score с родителем в единицах метрики; «лучше или хуже» зависит от direction запуска */
  delta: number | null;
  durationSec: number | null;
  error: string | null;
  code: string | null;
  backend: string;
  /** llm_request — запрос к Qwen, завершившийся ошибкой до запуска эксперимента */
  kind?: "experiment" | "llm_request";
}

export interface Budget {
  timeSec: number;
  tokens: number;
  threads: number;
}

export interface Usage {
  timeSec: number;
  tokens: number;
  tokensUncertain: boolean;
}

export interface RunSummary {
  id: string;
  title: string;
  taskKind: string;
  metric: string;
  direction: Direction;
  status: RunStatus;
  stage: StageId;
  baseline: number | null;
  best: number | null;
  elapsedSec: number;
  startedAt: string;
  demo: boolean;
}

export interface Run extends RunSummary {
  model: string;
  mode: SearchMode;
  budget: Budget;
  used: Usage;
  experiments: Experiment[];
  /** Существует ли CSV, который можно скачать прямо сейчас */
  csvAvailable: boolean;
  finalCsv: boolean;
  /** Пояснение статуса: причина остановки, отказ Qwen, timeout */
  note: string | null;
}

export type FileKey = "train" | "test" | "dictionary" | "sample";

export interface NewRunInput {
  title: string;
  description: string;
  files: Partial<Record<FileKey, { name: string; size: number; file?: File }>>;
  budget: Budget;
  seed: number;
  memoryGb: number;
  workerTimeoutSec: number;
  mode: SearchMode;
}

export interface QwenStatus {
  available: boolean;
  model: string;
  detail: string;
}
