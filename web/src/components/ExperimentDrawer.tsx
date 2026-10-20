import { X } from "lucide-react";
import type { Experiment, Params, Run } from "../data/types";
import { deltaTone, fmtDelta, fmtDuration, fmtScore } from "../format";
import { ExpBadge } from "./Badges";
import { Modal } from "./Modal";

function ParamTable({ params, other }: { params: Params; other?: Params }) {
  const keys = Object.keys(params);
  if (!keys.length) return <p className="muted small">Параметры не заданы.</p>;
  return (
    <dl className="kv">
      {keys.map((k) => (
        <div key={k} className={other && String(other[k]) !== String(params[k]) ? "diff" : ""}>
          <dt>{k}</dt>
          <dd>{String(params[k])}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ExperimentDrawer({ run, exp, onClose }: { run: Run; exp: Experiment; onClose: () => void }) {
  const tone = deltaTone(run.direction, exp.delta);
  const ran = Object.keys(exp.executedParams).length > 0;
  return (
    <Modal title={`Эксперимент ${exp.id}`} onClose={onClose} side>
      <header className="drawer-head">
        <div>
          <h2>{exp.kind === "llm_request" ? "Запрос к Qwen" : `Эксперимент ${exp.id}`}</h2>
          <div className="row">
            <ExpBadge status={exp.status} />
            <span className="tag">{exp.kind === "llm_request" ? "запрос Qwen" : exp.level}</span>
            <span className="tag">{exp.backend}</span>
          </div>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Закрыть">
          <X size={18} aria-hidden />
        </button>
      </header>

      <div className="drawer-body">
        <dl className="kv">
          <div>
            <dt>Родитель</dt>
            <dd>{exp.parentId === "baseline" ? "baseline" : (exp.parentId ?? "—")}</dd>
          </div>
          <div>
            <dt>{run.metric}</dt>
            <dd>{fmtScore(exp.score, run.metric)}</dd>
          </div>
          <div>
            <dt>Δ к родителю</dt>
            <dd className={`tone-${tone}`}>{fmtDelta(exp.delta, run.metric)}</dd>
          </div>
          <div>
            <dt>Длительность</dt>
            <dd>{fmtDuration(exp.durationSec)}</dd>
          </div>
        </dl>

        {exp.level === "smoke" && exp.kind !== "llm_request" && (
          <p className="note">Smoke-проверка: метрика не сравнивается с полной validation.</p>
        )}

        <h3>Гипотеза</h3>
        <p>
          <b>Замечено:</b> {exp.observed}
        </p>
        <p>
          <b>Предложено:</b> {exp.proposed}
        </p>

        <h3>Параметры</h3>
        <div className="two">
          <div>
            <h4>Предложено</h4>
            <ParamTable params={exp.proposedParams} other={exp.executedParams} />
          </div>
          <div>
            <h4>Запущено</h4>
            {ran ? <ParamTable params={exp.executedParams} other={exp.proposedParams} /> : <p className="muted small">Не запускалось.</p>}
          </div>
        </div>

        <h3>Изменения признаков</h3>
        {exp.featureChanges.length ? (
          <ul className="plain">
            {exp.featureChanges.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        ) : (
          <p className="muted small">Без изменений признаков.</p>
        )}

        <h3>Результат</h3>
        <p>{exp.measured}</p>

        {exp.error && (
          <div className="alert alert-error">
            <b>Ошибка:</b> {exp.error}
          </div>
        )}

        {exp.code && (
          <details className="code">
            <summary>Код преобразования</summary>
            <pre>
              <code>{exp.code}</code>
            </pre>
          </details>
        )}
      </div>
    </Modal>
  );
}
