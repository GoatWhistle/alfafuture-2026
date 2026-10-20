import { AlertTriangle, CheckCircle2, CircleDot, OctagonX, XCircle, Clock3, Ban } from "lucide-react";
import type { ReactNode } from "react";
import type { Direction, ExperimentStatus, RunStatus, StageId } from "./data/types";

export function fmtDuration(sec: number | null): string {
  if (sec === null) return "—";
  const s = Math.round(sec);
  if (s < 60) return `${s} с`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} мин ${String(s % 60).padStart(2, "0")} с`;
  return `${Math.floor(m / 60)} ч ${String(m % 60).padStart(2, "0")} мин`;
}

export function fmtScore(v: number | null, metric = ""): string {
  if (v === null) return "—";
  return /rmse|mae|mse/i.test(metric) ? v.toFixed(2) : v.toFixed(4);
}

export function fmtDelta(v: number | null, metric = ""): string {
  if (v === null) return "—";
  const digits = /rmse|mae|mse/i.test(metric) ? 2 : 4;
  const sign = v > 0 ? "+" : v < 0 ? "−" : "±";
  return `${sign}${Math.abs(v).toFixed(digits)}`;
}

export function isBetter(direction: Direction, delta: number): boolean {
  return direction === "higher" ? delta > 0 : delta < 0;
}

export function deltaTone(direction: Direction, delta: number | null): "good" | "bad" | "flat" {
  if (delta === null || delta === 0) return "flat";
  return isBetter(direction, delta) ? "good" : "bad";
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("ru-RU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / 1024 ** 2).toFixed(1)} МБ`;
}

export const fmtInt = (n: number) => n.toLocaleString("ru-RU");

export const STATUS_LABEL: Record<RunStatus, string> = {
  running: "Выполняется",
  completed: "Завершено",
  stopped: "Остановлено",
  failed: "Ошибка",
};

export const STATUS_ICON: Record<RunStatus, ReactNode> = {
  running: <CircleDot size={14} aria-hidden />,
  completed: <CheckCircle2 size={14} aria-hidden />,
  stopped: <OctagonX size={14} aria-hidden />,
  failed: <AlertTriangle size={14} aria-hidden />,
};

export const EXP_LABEL: Record<ExperimentStatus, string> = {
  running: "Выполняется",
  ok: "Измерено",
  failed: "Ошибка",
  rejected: "Отклонено",
  timeout: "Timeout",
};

export const EXP_ICON: Record<ExperimentStatus, ReactNode> = {
  running: <CircleDot size={14} aria-hidden />,
  ok: <CheckCircle2 size={14} aria-hidden />,
  failed: <XCircle size={14} aria-hidden />,
  rejected: <Ban size={14} aria-hidden />,
  timeout: <Clock3 size={14} aria-hidden />,
};

export const STAGES: { id: StageId; label: string }[] = [
  { id: "prepare", label: "Подготовка" },
  { id: "baseline", label: "Baseline" },
  { id: "research", label: "Исследования" },
  { id: "refit", label: "Финальное обучение" },
  { id: "result", label: "Результат" },
];
