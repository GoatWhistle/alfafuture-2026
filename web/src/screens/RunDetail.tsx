import { ArrowLeft, Download, FileText, Pause, Play, RotateCcw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ExpBadge, StatusBadge } from "../components/Badges";
import { ExperimentDrawer } from "../components/ExperimentDrawer";
import { Modal } from "../components/Modal";
import { QualityChart } from "../components/QualityChart";
import { StageBar } from "../components/StageBar";
import { useLoaded, useSource } from "../data/context";
import type { DownloadKind } from "../data/source";
import type { Run } from "../data/types";
import { deltaTone, fmtDelta, fmtDuration, fmtInt, fmtScore } from "../format";

export function RunDetail() {
  const { id = "" } = useParams();
  const source = useSource();
  const [polling, setPolling] = useState(true);
  const { data: run, loading, error, reload } = useLoaded((s) => s.getRun(id), [id], polling);
  const running = run?.status === "running";
  useEffect(() => {
    // опрос прекращается, когда запуск перестал выполняться
    if (run) setPolling(running);
  }, [run, running]);
  const [selected, setSelected] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  if (loading && !run) return <div className="page muted">Загрузка…</div>;
  if (error && !run) return <div className="page"><div className="alert alert-error">Ошибка загрузки: {error}</div></div>;
  if (!run)
    return (
      <div className="page">
        <Link to="/runs" className="back"><ArrowLeft size={16} aria-hidden /> К истории</Link>
        <h1>Исследование не найдено</h1>
      </div>
    );

  const exp = run.experiments.find((e) => e.id === selected) ?? null;
  const delta = run.best !== null && run.baseline !== null ? run.best - run.baseline : null;
  const fullDone = run.experiments.filter((e) => e.level === "full" && e.status !== "running").length;
  const play = source.playback?.get(run.id) ?? null;

  const download = async (kind: DownloadKind) => {
    setActionError(null);
    try {
      await source.download(run, kind);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  };

  const abort = async () => {
    setConfirm(false);
    try {
      await source.abortRun(run.id);
      reload();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="page">
      <Link to="/runs" className="back">
        <ArrowLeft size={16} aria-hidden /> К истории
      </Link>
      <header className="run-head">
        <div>
          <h1>{run.title}</h1>
          <div className="row meta">
            <StatusBadge status={run.status} />
            <span className="tag">{run.model}</span>
            <span className="muted">{run.taskKind} · {run.metric}</span>
            <span className="muted">Время: {fmtDuration(run.elapsedSec)}</span>
          </div>
        </div>
        <div className="row actions">
          <button className="btn btn-secondary" disabled={!run.csvAvailable} onClick={() => download("current")}>
            <Download size={16} aria-hidden /> Скачать текущий CSV
          </button>
          <button className="btn btn-secondary" disabled={!run.finalCsv} onClick={() => download("final")}>
            <Download size={16} aria-hidden /> Скачать итоговый CSV
          </button>
          <button className="btn btn-secondary" disabled={run.status === "running"} onClick={() => download("report")}>
            <FileText size={16} aria-hidden /> Скачать отчёт
          </button>
          {run.status === "running" && (
            <button className="btn btn-danger" onClick={() => setConfirm(true)}>
              <Square size={16} aria-hidden /> Прервать запуск
            </button>
          )}
        </div>
      </header>

      {actionError && <div className="alert alert-error" role="alert">{actionError}</div>}
      {run.note && (
        <div className={`alert ${run.status === "failed" ? "alert-error" : "alert-info"}`} role="status">
          {run.note}
          {run.csvAvailable && " Файл «текущий CSV» остаётся доступным."}
        </div>
      )}

      {play && source.playback && (
        <div className="playback" role="group" aria-label="Управление демонстрацией">
          <span className="muted small">Воспроизведение сценария · шаг {play.step + 1} из {play.total}</span>
          {play.playing ? (
            <button className="btn btn-ghost btn-sm" onClick={() => source.playback!.pause(run.id)}>
              <Pause size={14} aria-hidden /> Пауза
            </button>
          ) : (
            <button className="btn btn-ghost btn-sm" onClick={() => source.playback!.start(run.id)} disabled={run.status === "stopped"}>
              <Play size={14} aria-hidden /> Старт
            </button>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => source.playback!.repeat(run.id)}>
            <RotateCcw size={14} aria-hidden /> Повтор
          </button>
        </div>
      )}

      <StageBar stage={run.stage} status={run.status} />

      <section className="summary" aria-label="Сводка">
        <Stat label="Baseline" value={fmtScore(run.baseline, run.metric)} />
        <Stat label="Лучший full" value={fmtScore(run.best, run.metric)} accent />
        <Stat
          label="Δ к baseline"
          value={delta === null || delta === 0 ? "без изменений" : fmtDelta(delta, run.metric)}
          tone={deltaTone(run.direction, delta)}
        />
        <Stat label="Завершено full" value={String(fullDone)} />
        <Stat label="Осталось" value={fmtDuration(Math.max(0, run.budget.timeSec - run.used.timeSec))} sub={`токенов: ${fmtInt(Math.max(0, run.budget.tokens - run.used.tokens))}`} />
      </section>

      <div className="cols">
        <section className="card">
          <h2 className="h-sm">Качество</h2>
          <QualityChart run={run} onSelect={setSelected} />
        </section>
        <Resources run={run} />
      </div>

      <section className="card">
        <h2 className="h-sm">Гипотезы</h2>
        {run.experiments.length === 0 ? (
          <p className="muted">{run.mode === "baseline" ? "В этом режиме гипотезы не проверяются." : "Гипотезы появятся после baseline."}</p>
        ) : (
          <ol className="feed">
            {run.experiments.map((e) => (
              <li key={e.id}>
                <button className="feed-item" onClick={() => setSelected(e.id)}>
                  <div className="feed-top">
                    <strong>Эксперимент {e.id}</strong>
                    <ExpBadge status={e.status} />
                    <span className="tag">{e.level}</span>
                    <span className="feed-score">
                      {e.score !== null && <>{fmtScore(e.score, run.metric)} </>}
                      {e.delta !== null && <span className={`tone-${deltaTone(run.direction, e.delta)}`}>{fmtDelta(e.delta, run.metric)}</span>}
                    </span>
                  </div>
                  <dl className="feed-steps">
                    <div><dt>Замечено</dt><dd>{e.observed}</dd></div>
                    <div><dt>Предложено</dt><dd>{e.proposed}</dd></div>
                    <div><dt>Запущено</dt><dd>{e.executed}</dd></div>
                    <div><dt>Измерено</dt><dd>{e.measured}</dd></div>
                  </dl>
                  {e.error && <div className="feed-err">{e.error}</div>}
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>

      {exp && <ExperimentDrawer run={run} exp={exp} onClose={() => setSelected(null)} />}
      {confirm && (
        <Modal
          title="Прервать запуск"
          onClose={() => setConfirm(false)}
          footer={
            <>
              <button className="btn btn-ghost" onClick={() => setConfirm(false)}>Продолжить исследование</button>
              <button className="btn btn-danger" onClick={abort}>Прервать</button>
            </>
          }
        >
          <div className="dialog-body">
            <h2>Прервать запуск?</h2>
            <p>Уже сохранённый CSV останется доступным. Финальное переобучение после прерывания не выполняется.</p>
          </div>
        </Modal>
      )}
    </div>
  );
}

function Stat({ label, value, sub, tone, accent }: { label: string; value: string; sub?: string; tone?: string; accent?: boolean }) {
  return (
    <div className={`stat ${accent ? "stat-accent" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${tone ? `tone-${tone}` : ""}`}>{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

function Resources({ run }: { run: Run }) {
  const t = Math.min(100, (run.used.timeSec / run.budget.timeSec) * 100);
  const k = Math.min(100, (run.used.tokens / run.budget.tokens) * 100);
  return (
    <section className="card">
      <h2 className="h-sm">Ресурсы</h2>
      <Meter label="Время" text={`${fmtDuration(run.used.timeSec)} из ${fmtDuration(run.budget.timeSec)}`} pct={t} />
      <Meter
        label="Токены"
        text={`${run.used.tokensUncertain ? "≈ " : ""}${fmtInt(run.used.tokens)} из ${fmtInt(run.budget.tokens)}`}
        pct={k}
      />
      {run.used.tokensUncertain && (
        <p className="note">Usage части запросов неизвестен — неучтённый расход оценён консервативно.</p>
      )}
      <p className="muted small">Потоков: {run.budget.threads}</p>
    </section>
  );
}

function Meter({ label, text, pct }: { label: string; text: string; pct: number }) {
  return (
    <div className="meter">
      <div className="meter-top"><span>{label}</span><span>{text}</span></div>
      <div className="meter-bar" role="progressbar" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <i style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
