import { CheckCircle2, FileUp, Loader2, Play, RefreshCw, XCircle } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLoaded, useSource } from "../data/context";
import type { FileKey, NewRunInput, SearchMode } from "../data/types";
import { fmtDuration, fmtInt, fmtSize } from "../format";

const FILES: { key: FileKey; name: string; hint: string }[] = [
  { key: "train", name: "train.csv", hint: "Обучающая выборка с целевой колонкой" },
  { key: "test", name: "test.csv", hint: "Данные для предсказаний" },
  { key: "dictionary", name: "data_dictionary.csv", hint: "Описание признаков" },
  { key: "sample", name: "sample_submission.csv", hint: "Образец формата ответа" },
];

type FileState = { name: string; size: number; file?: File; state: "loading" | "ready" | "error"; error?: string };

const MODES: { id: SearchMode; label: string }[] = [
  { id: "qwen", label: "Qwen" },
  { id: "optuna", label: "Optuna" },
  { id: "baseline", label: "Только baseline" },
];

export function NewRun() {
  const source = useSource();
  const nav = useNavigate();
  const qwen = useLoaded((s) => s.qwenStatus(), []);
  const [step, setStep] = useState<1 | 2>(1);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<Partial<Record<FileKey, FileState>>>({});
  const [timeMin, setTimeMin] = useState(30);
  const [tokens, setTokens] = useState(32000);
  const [threads, setThreads] = useState(4);
  const [seed, setSeed] = useState(42);
  const [memoryGb, setMemoryGb] = useState(8);
  const [workerMin, setWorkerMin] = useState(10);
  const [mode, setMode] = useState<SearchMode>("qwen");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [triedNext, setTriedNext] = useState(false);

  const missing = FILES.filter((f) => files[f.key]?.state !== "ready");
  const titleOk = title.trim().length > 0;
  const descOk = description.trim().length > 0 || !!files.train;
  const step1Ok = titleOk && missing.length === 0;

  const pick = (key: FileKey, file: File | undefined) => {
    if (!file) return;
    const expected = FILES.find((f) => f.key === key)!.name;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setFiles((p) => ({ ...p, [key]: { name: file.name, size: file.size, state: "error", error: `Нужен файл ${expected}` } }));
      return;
    }
    setFiles((p) => ({ ...p, [key]: { name: file.name, size: file.size, file, state: "loading" } }));
    // В демо загрузка имитируется; с API здесь передаётся сам файл
    setTimeout(() => setFiles((p) => ({ ...p, [key]: { name: file.name, size: file.size, file, state: "ready" } })), 600);
  };

  const launch = async () => {
    setStarting(true);
    setStartError(null);
    const input: NewRunInput = {
      title,
      description,
      files: Object.fromEntries(
        Object.entries(files).map(([k, v]) => [k, { name: v!.name, size: v!.size, file: v!.file }]),
      ) as NewRunInput["files"],
      budget: { timeSec: timeMin * 60, tokens, threads },
      seed,
      memoryGb,
      workerTimeoutSec: workerMin * 60,
      mode,
    };
    try {
      const id = await source.createRun(input);
      nav(`/runs/${id}`);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
      setStarting(false);
    }
  };

  return (
    <div className="page page-narrow">
      <header className="page-head">
        <h1>Новое исследование</h1>
      </header>
      <ol className="steps" aria-label="Шаги">
        <li className={step === 1 ? "on" : "done"}>1 · Материалы</li>
        <li className={step === 2 ? "on" : ""}>2 · Ресурсы</li>
      </ol>

      {step === 1 && (
        <section className="card form">
          <label className="field">
            <span>Название исследования</span>
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Например, отток клиентов" />
            {triedNext && !titleOk && <small className="err">Укажите название</small>}
          </label>
          <label className="field">
            <span>Описание задачи</span>
            <textarea
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Текст задачи или содержимое task.md"
            />
            <TaskMdButton onText={setDescription} />
            {triedNext && !descOk && <small className="err">Опишите задачу текстом или загрузите task.md</small>}
          </label>

          <h2 className="h-sm">Файлы данных</h2>
          <ul className="files">
            {FILES.map((f) => (
              <FileRow key={f.key} def={f} state={files[f.key]} onPick={(file) => pick(f.key, file)} />
            ))}
          </ul>
          {triedNext && missing.length > 0 && (
            <div className="alert alert-error" role="alert">
              Не загружено: {missing.map((m) => m.name).join(", ")}
            </div>
          )}
          <div className="form-actions">
            <button
              className="btn btn-primary"
              onClick={() => {
                setTriedNext(true);
                if (step1Ok) setStep(2);
              }}
            >
              Далее: ресурсы
            </button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="card form">
          <div className="grid3">
            <NumField label="Время, мин" value={timeMin} min={1} max={240} onChange={setTimeMin} />
            <NumField label="Токены" value={tokens} min={1000} max={500000} step={1000} onChange={setTokens} />
            <NumField label="Потоки" value={threads} min={1} max={32} onChange={setThreads} />
          </div>
          <details className="adv">
            <summary>Дополнительные настройки</summary>
            <div className="grid3">
              <NumField label="Seed" value={seed} min={0} max={999999} onChange={setSeed} />
              <NumField label="Память, ГБ" value={memoryGb} min={1} max={256} onChange={setMemoryGb} />
              <NumField label="Worker timeout, мин" value={workerMin} min={1} max={120} onChange={setWorkerMin} />
            </div>
            <fieldset className="field">
              <legend>Режим поиска</legend>
              <div className="seg">
                {MODES.map((m) => (
                  <label key={m.id} className={mode === m.id ? "on" : ""}>
                    <input type="radio" name="mode" checked={mode === m.id} onChange={() => setMode(m.id)} />
                    {m.label}
                  </label>
                ))}
              </div>
            </fieldset>
          </details>

          <div className="summary-box">
            <h2 className="h-sm">Перед запуском</h2>
            <dl>
              <dt>Исследование</dt>
              <dd>{title}</dd>
              <dt>Файлы</dt>
              <dd>{FILES.map((f) => files[f.key]?.name).join(", ")}</dd>
              <dt>Бюджет</dt>
              <dd>
                {fmtDuration(timeMin * 60)} · {fmtInt(tokens)} токенов · {threads} потока
              </dd>
              <dt>Режим</dt>
              <dd>{MODES.find((m) => m.id === mode)!.label}</dd>
              <dt>Qwen</dt>
              <dd>
                {qwen.loading ? (
                  <span className="muted">Проверка…</span>
                ) : qwen.data?.available ? (
                  <span className="tone-good">
                    <CheckCircle2 size={14} aria-hidden /> Доступна · {qwen.data.model}
                  </span>
                ) : (
                  <span className="tone-bad">
                    <XCircle size={14} aria-hidden /> Недоступна: {qwen.data?.detail ?? qwen.error}
                  </span>
                )}
              </dd>
            </dl>
            <p className="muted small">Ключи вводить не нужно: сервер использует локальную конфигурацию.</p>
          </div>

          {startError && (
            <div className="alert alert-error" role="alert">
              Не удалось запустить: {startError}
            </div>
          )}
          <div className="form-actions between">
            <button className="btn btn-ghost" onClick={() => setStep(1)} disabled={starting}>
              Назад
            </button>
            <button
              className="btn btn-primary"
              onClick={launch}
              disabled={starting || (mode === "qwen" && qwen.data?.available === false)}
            >
              {starting ? <Loader2 size={18} className="spin" aria-hidden /> : <Play size={18} aria-hidden />} Запустить
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

function TaskMdButton({ onText }: { onText: (t: string) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={ref}
        type="file"
        accept=".md,.txt"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (f) onText(await f.text());
          e.target.value = "";
        }}
      />
      <button type="button" className="btn btn-ghost btn-sm self-start" onClick={() => ref.current?.click()}>
        <FileUp size={14} aria-hidden /> Загрузить task.md
      </button>
    </>
  );
}

function FileRow({
  def,
  state,
  onPick,
}: {
  def: (typeof FILES)[number];
  state: FileState | undefined;
  onPick: (f: File | undefined) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <li className={`file file-${state?.state ?? "empty"}`}>
      <div className="file-main">
        <strong>{def.name}</strong>
        <span className="muted small">{state ? `${state.name} · ${fmtSize(state.size)}` : def.hint}</span>
        {state?.state === "error" && <span className="err small">{state.error}</span>}
      </div>
      <span className="file-state">
        {state?.state === "loading" && (
          <>
            <Loader2 size={14} className="spin" aria-hidden /> Загрузка…
          </>
        )}
        {state?.state === "ready" && (
          <span className="tone-good">
            <CheckCircle2 size={14} aria-hidden /> Загружен
          </span>
        )}
        {state?.state === "error" && (
          <span className="tone-bad">
            <XCircle size={14} aria-hidden /> Ошибка
          </span>
        )}
      </span>
      <input
        ref={ref}
        type="file"
        accept=".csv"
        hidden
        onChange={(e) => {
          onPick(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <button className="btn btn-secondary btn-sm" onClick={() => ref.current?.click()} aria-label={`${state ? "Заменить" : "Выбрать"} ${def.name}`}>
        {state ? <RefreshCw size={14} aria-hidden /> : <FileUp size={14} aria-hidden />} {state ? "Заменить" : "Выбрать"}
      </button>
    </li>
  );
}

function NumField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, Math.round(n))));
        }}
      />
    </label>
  );
}
