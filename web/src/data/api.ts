import { saveBlobData } from "./source";
import type { DataSource, DownloadKind } from "./source";
import type { NewRunInput, QwenStatus, Run, RunSummary } from "./types";

const FILE_KEYS = ["train", "test", "dictionary", "sample"] as const;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "Проверьте заполненные поля";
    } catch {
      /* тело не JSON */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/** HTTP-клиент локального web_api; экраны используют его через DataSource */
export class ApiSource implements DataSource {
  readonly kind = "api" as const;
  readonly pollMs = 2000;
  private listeners = new Set<() => void>();

  private emit() {
    this.listeners.forEach((l) => l());
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  listRuns() {
    return call<RunSummary[]>("/runs");
  }

  async getRun(id: string): Promise<Run | null> {
    try {
      return await call<Run>(`/runs/${id}`);
    } catch (e) {
      if (e instanceof Error && e.message === "Исследование не найдено") return null;
      throw e;
    }
  }

  async createRun(input: NewRunInput): Promise<string> {
    const form = new FormData();
    form.set("title", input.title);
    form.set("description", input.description);
    form.set("time_sec", String(input.budget.timeSec));
    form.set("tokens", String(input.budget.tokens));
    form.set("threads", String(input.budget.threads));
    form.set("seed", String(input.seed));
    form.set("memory_gb", String(input.memoryGb));
    form.set("worker_timeout_sec", String(input.workerTimeoutSec));
    form.set("mode", input.mode);
    for (const key of FILE_KEYS) {
      const file = input.files[key]?.file;
      if (!file) throw new Error(`Файл ${key} не выбран`);
      form.set(key, file);
    }
    const { id } = await call<{ id: string }>("/runs", { method: "POST", body: form });
    await call(`/runs/${id}/start`, { method: "POST" });
    this.emit();
    return id;
  }

  async abortRun(id: string) {
    await call(`/runs/${id}/abort`, { method: "POST" });
    this.emit();
  }

  qwenStatus() {
    return call<QwenStatus>("/qwen");
  }

  async download(run: Run, kind: DownloadKind) {
    const response = await fetch(`/api/runs/${run.id}/download/${kind}`);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: string };
      throw new Error(body.detail ?? `Ошибка скачивания (${response.status})`);
    }
    saveBlobData(`${run.id}-${kind}.${kind === "report" ? "md" : "csv"}`, await response.blob());
  }
}
