import type { NewRunInput, QwenStatus, Run, RunSummary } from "./types";

export type DownloadKind = "current" | "final" | "report";

export interface PlaybackState {
  step: number;
  total: number;
  playing: boolean;
}

/**
 * Единая граница доступа к данным. Экраны знают только этот интерфейс:
 * демо-источник заменяется HTTP-клиентом без изменения компонентов.
 */
export interface DataSource {
  readonly kind: "demo" | "api";
  /** Период опроса активного запуска, мс; null — источник сам уведомляет */
  readonly pollMs: number | null;
  listRuns(): Promise<RunSummary[]>;
  getRun(id: string): Promise<Run | null>;
  createRun(input: NewRunInput): Promise<string>;
  abortRun(id: string): Promise<void>;
  qwenStatus(): Promise<QwenStatus>;
  download(run: Run, kind: DownloadKind): Promise<void>;
  subscribe(listener: () => void): () => void;
  /** Только демо: управление воспроизведением */
  playback?: {
    get(id: string): PlaybackState | null;
    start(id: string): void;
    pause(id: string): void;
    repeat(id: string): void;
  };
  /** Только демо: очистка и восстановление истории */
  demoHistory?: {
    isEmpty(): boolean;
    setEmpty(empty: boolean): void;
  };
}

export function saveBlob(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function saveBlobData(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
