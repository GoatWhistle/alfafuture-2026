import { Check } from "lucide-react";
import type { RunStatus, StageId } from "../data/types";
import { STAGES } from "../format";

/** Показывает текущий этап без процента готовности */
export function StageBar({ stage, status }: { stage: StageId; status: RunStatus }) {
  const cur = STAGES.findIndex((s) => s.id === stage);
  return (
    <ol className="stagebar" aria-label="Этапы исследования">
      {STAGES.map((s, i) => {
        const state =
          status === "completed" ? "done" : i < cur ? "done" : i === cur ? (status === "running" ? "active" : "halt") : "todo";
        return (
          <li key={s.id} className={`stage stage-${state}`} aria-current={state === "active" ? "step" : undefined}>
            <span className="stage-dot">{state === "done" ? <Check size={13} aria-hidden /> : i + 1}</span>
            <span>{s.label}</span>
            {state === "halt" && <em>остановлено здесь</em>}
          </li>
        );
      })}
    </ol>
  );
}
