import { PLAYBACK_START, SCENARIOS } from "./scenarios";
import { saveBlob } from "./source";
import type { DataSource, DownloadKind, PlaybackState } from "./source";
import type { Experiment, NewRunInput, Run, RunSummary, StageId } from "./types";

const STEP_MS = 1600;

interface Step {
  stage: StageId;
  /** индекс эксперимента, который стартует / завершается на этом шаге */
  start?: number;
  finish?: number;
}

function buildSteps(run: Run): Step[] {
  const steps: Step[] = [{ stage: "prepare" }, { stage: "baseline" }];
  run.experiments.forEach((_, i) => {
    steps.push({ stage: "research", start: i }, { stage: "research", finish: i });
  });
  if (run.status !== "failed") steps.push({ stage: "refit" }, { stage: "result" });
  return steps;
}

const better = (run: Run, a: number, b: number) => (run.direction === "higher" ? a > b : a < b);

/** Состояние запуска на шаге k: демо только раскрывает заранее заданные факты, ничего не вычисляя заново */
export function snapshot(full: Run, k: number, stopped: boolean): Run {
  const steps = buildSteps(full);
  const last = steps.length - 1;
  const at = Math.min(k, last);
  const done = at === last;
  const step = steps[at];
  const shown: Experiment[] = [];
  full.experiments.forEach((e, i) => {
    const startedAt = steps.findIndex((s) => s.start === i);
    const finishedAt = steps.findIndex((s) => s.finish === i);
    if (at < startedAt) return;
    if (at < finishedAt) {
      shown.push({
        ...e,
        status: "running",
        executed: e.proposed,
        measured: "Выполняется…",
        score: null,
        delta: null,
        durationSec: null,
        error: null,
      });
    } else shown.push(e);
  });

  const baselineKnown = at >= 2 || full.baseline === null;
  const baseline = baselineKnown ? full.baseline : null;
  let best = baseline;
  if (done) best = full.best;
  else
    for (const e of shown)
      if (e.status === "ok" && e.level === "full" && e.score !== null)
        if (best === null || better(full, e.score, best)) best = e.score;

  const progress = last === 0 ? 1 : at / last;
  const terminal = full.status === "running" ? "completed" : full.status;
  const status = stopped ? "stopped" : done ? terminal : "running";
  return {
    ...full,
    status,
    stage: stopped ? step.stage : done ? full.stage : step.stage,
    baseline,
    best,
    elapsedSec: Math.round(full.elapsedSec * progress),
    used: {
      ...full.used,
      timeSec: Math.round(full.used.timeSec * progress),
      tokens: Math.round(full.used.tokens * progress),
    },
    experiments: shown,
    csvAvailable: baselineKnown && at >= 2,
    finalCsv: !stopped && done && terminal === "completed" && (full.finalCsv || full.status === "running"),
    note: done || stopped ? (stopped ? "Запуск прерван пользователем. Существующий CSV сохранён." : full.note) : null,
  };
}

const toSummary = (r: Run): RunSummary => {
  const { id, title, taskKind, metric, direction, status, stage, baseline, best, elapsedSec, startedAt, demo } = r;
  return { id, title, taskKind, metric, direction, status, stage, baseline, best, elapsedSec, startedAt, demo };
};

function fakeCsv(run: Run, kind: DownloadKind): string {
  const rng = (n: number) => ((n * 9301 + 49297) % 233280) / 233280;
  const rows = Array.from({ length: 20 }, (_, i) => `${100000 + i},${rng(i + run.id.length).toFixed(6)}`);
  return ["id,prediction", ...rows].join("\n") + `\n# демонстрационный файл (${kind}), не результат обучения\n`;
}

function fakeReport(run: Run): string {
  const lines = [
    `# Отчёт: ${run.title}`,
    "",
    "> Демонстрационные данные: файл сформирован интерфейсом, обучение не запускалось.",
    "",
    `- Метрика: ${run.metric}`,
    `- Baseline: ${run.baseline ?? "—"}`,
    `- Лучший результат: ${run.best ?? "—"}`,
    "",
    "## Эксперименты",
    ...run.experiments.map((e) => `- ${e.id}: ${e.measured}`),
  ];
  return lines.join("\n") + "\n";
}

interface Play {
  step: number;
  playing: boolean;
  stopped: boolean;
}

export class DemoSource implements DataSource {
  readonly kind = "demo" as const;
  readonly pollMs = null;
  private listeners = new Set<() => void>();
  private fulls = new Map<string, Run>();
  private play = new Map<string, Play>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private empty = false;
  private seq = 0;

  constructor() {
    this.resetHistory();
  }

  private resetHistory() {
    this.fulls.clear();
    this.play.clear();
    for (const r of SCENARIOS) {
      this.fulls.set(r.id, r);
      const total = buildSteps(r).length;
      this.play.set(r.id, { step: PLAYBACK_START[r.id] ?? total - 1, playing: false, stopped: false });
    }
  }

  private emit() {
    this.listeners.forEach((l) => l());
  }

  private ensureTimer() {
    const anyPlaying = [...this.play.values()].some((p) => p.playing);
    if (anyPlaying && !this.timer) {
      this.timer = setInterval(() => this.tick(), STEP_MS);
    } else if (!anyPlaying && this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private tick() {
    for (const [id, p] of this.play) {
      if (!p.playing) continue;
      const total = buildSteps(this.fulls.get(id)!).length;
      p.step = Math.min(p.step + 1, total - 1);
      if (p.step >= total - 1) p.playing = false;
    }
    this.ensureTimer();
    this.emit();
  }

  private current(id: string): Run | null {
    const full = this.fulls.get(id);
    const p = this.play.get(id);
    if (!full || !p || this.empty) return null;
    return snapshot(full, p.step, p.stopped);
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async listRuns() {
    if (this.empty) return [];
    return [...this.fulls.keys()]
      .map((id) => this.current(id)!)
      .sort((a, b) => b.startedAt.localeCompare(a.startedAt))
      .map(toSummary);
  }

  async getRun(id: string) {
    return this.current(id);
  }

  async createRun(input: NewRunInput) {
    const id = `demo-new-${++this.seq}`;
    const template = SCENARIOS.find((s) => s.id === "demo-improved")!;
    const mode = input.mode;
    const full: Run = {
      ...template,
      id,
      title: input.title.trim() || "Новое исследование",
      startedAt: new Date().toISOString().slice(0, 19),
      budget: input.budget,
      mode,
      experiments: mode === "baseline" ? [] : template.experiments,
      best: mode === "baseline" ? template.baseline : template.best,
      note: mode === "baseline" ? "Режим baseline: поиск гипотез не запускался." : null,
    };
    this.fulls.set(id, full);
    this.play.set(id, { step: 0, playing: true, stopped: false });
    this.empty = false;
    this.ensureTimer();
    this.emit();
    return id;
  }

  async abortRun(id: string) {
    const p = this.play.get(id);
    if (!p) return;
    p.playing = false;
    p.stopped = true;
    this.ensureTimer();
    this.emit();
  }

  async qwenStatus() {
    return { available: true, model: "Qwen3-Coder", detail: "Демо: Qwen не вызывается" };
  }

  async download(run: Run, kind: DownloadKind) {
    if (kind === "report") saveBlob(`${run.id}-report.md`, fakeReport(run), "text/markdown");
    else saveBlob(`${run.id}-${kind}.csv`, fakeCsv(run, kind), "text/csv");
  }

  playback = {
    get: (id: string): PlaybackState | null => {
      const p = this.play.get(id);
      const f = this.fulls.get(id);
      return p && f ? { step: p.step, total: buildSteps(f).length, playing: p.playing } : null;
    },
    start: (id: string) => {
      const p = this.play.get(id);
      const f = this.fulls.get(id);
      if (!p || !f || p.stopped) return;
      if (p.step >= buildSteps(f).length - 1) p.step = 0;
      p.playing = true;
      this.ensureTimer();
      this.emit();
    },
    pause: (id: string) => {
      const p = this.play.get(id);
      if (!p) return;
      p.playing = false;
      this.ensureTimer();
      this.emit();
    },
    repeat: (id: string) => {
      const p = this.play.get(id);
      if (!p) return;
      p.step = 0;
      p.stopped = false;
      p.playing = true;
      this.ensureTimer();
      this.emit();
    },
  };

  demoHistory = {
    isEmpty: () => this.empty,
    setEmpty: (empty: boolean) => {
      if (!empty) this.resetHistory();
      this.empty = empty;
      this.ensureTimer();
      this.emit();
    },
  };
}
