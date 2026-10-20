import { ArrowRight, Plus, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { StatusBadge } from "../components/Badges";
import { useLoaded } from "../data/context";
import type { RunStatus } from "../data/types";
import { deltaTone, fmtDate, fmtDelta, fmtDuration, fmtScore } from "../format";

type Filter = "all" | RunStatus;
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "Все" },
  { id: "running", label: "Выполняются" },
  { id: "completed", label: "Завершены" },
  { id: "stopped", label: "Остановлены" },
  { id: "failed", label: "Ошибки" },
];

export function Runs() {
  const nav = useNavigate();
  const [filter, setFilter] = useState<Filter>("all");
  const { data, loading, error } = useLoaded((s) => s.listRuns(), [], true);
  const runs = data ?? [];
  const shown = filter === "all" ? runs : runs.filter((r) => r.status === filter);

  return (
    <div className="page">
      <header className="page-head">
        <h1>История исследований</h1>
        <Link to="/runs/new" className="btn btn-primary">
          <Plus size={18} aria-hidden /> Новое исследование
        </Link>
      </header>

      {error && <div className="alert alert-error">Не удалось загрузить список: {error}</div>}

      {!loading && runs.length === 0 && !error ? (
        <div className="empty">
          <div className="empty-art" aria-hidden>
            <i /> <i /> <i />
          </div>
          <h2>Исследований пока нет</h2>
          <p>Загрузите данные задачи, задайте бюджет — агент проверит гипотезы и подготовит предсказания.</p>
          <div className="row">
            <Link to="/runs/new" className="btn btn-primary">
              <Plus size={18} aria-hidden /> Создать исследование
            </Link>
            <button className="btn btn-secondary" onClick={() => nav("/runs/demo-improved")}>
              <Sparkles size={18} aria-hidden /> Посмотреть демо
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="filters" role="group" aria-label="Фильтр по состоянию">
            {FILTERS.map((f) => (
              <button key={f.id} className="chip" aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
                {f.label}
                <span className="chip-n">{f.id === "all" ? runs.length : runs.filter((r) => r.status === f.id).length}</span>
              </button>
            ))}
          </div>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Задача</th>
                  <th>Тип</th>
                  <th>Состояние</th>
                  <th className="num">Лучшая метрика</th>
                  <th className="num">К baseline</th>
                  <th className="num">Время</th>
                  <th>Дата</th>
                  <th aria-label="Открыть" />
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => {
                  const delta = r.best !== null && r.baseline !== null ? r.best - r.baseline : null;
                  return (
                    <tr key={r.id} className="row-click" onClick={() => nav(`/runs/${r.id}`)}>
                      <td>
                        <Link to={`/runs/${r.id}`} className="run-link" onClick={(e) => e.stopPropagation()}>
                          {r.title}
                        </Link>
                        {r.demo && <span className="tag-demo">демо</span>}
                      </td>
                      <td>{r.taskKind}</td>
                      <td>
                        <StatusBadge status={r.status} />
                      </td>
                      <td className="num">
                        <span className="muted">{r.metric}</span> {fmtScore(r.best, r.metric)}
                      </td>
                      <td className={`num tone-${deltaTone(r.direction, delta)}`}>
                        {delta === null || delta === 0 ? "без изменений" : fmtDelta(delta, r.metric)}
                      </td>
                      <td className="num">{fmtDuration(r.elapsedSec)}</td>
                      <td>{fmtDate(r.startedAt)}</td>
                      <td>
                        <ArrowRight size={16} aria-hidden />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {shown.length === 0 && <div className="table-empty">Нет запусков в этом состоянии.</div>}
          </div>
        </>
      )}
    </div>
  );
}
