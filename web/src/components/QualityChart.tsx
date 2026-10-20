import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Run } from "../data/types";
import { fmtScore } from "../format";

interface Point {
  x: string;
  score: number;
  id: string;
}

/** Только full-эксперименты: smoke не сравнивается с full и на графике не показывается */
export function QualityChart({ run, onSelect }: { run: Run; onSelect: (id: string) => void }) {
  const points: Point[] = run.experiments
    .filter((e) => e.level === "full" && e.status === "ok" && e.score !== null)
    .map((e) => ({ x: e.id, score: e.score as number, id: e.id }));
  const smoke = run.experiments.filter((e) => e.level === "smoke").length;
  const values = [...points.map((p) => p.score), ...(run.baseline !== null ? [run.baseline] : [])];
  const span = values.length ? Math.max(...values) - Math.min(...values) : 0;
  const pad = span === 0 ? Math.abs(values[0] ?? 1) * 0.01 + 0.0001 : span * 0.25;
  const domain: [number, number] = values.length ? [Math.min(...values) - pad, Math.max(...values) + pad] : [0, 1];
  const arrow = run.direction === "higher" ? "↑ выше — лучше" : "↓ ниже — лучше";

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <span>
          {run.metric} на полной validation · <b>{arrow}</b>
        </span>
        {smoke > 0 && <span className="muted">Smoke-проверок скрыто: {smoke}</span>}
      </div>
      {points.length === 0 ? (
        <div className="chart-empty">Полных результатов пока нет. Baseline: {fmtScore(run.baseline, run.metric)}</div>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={points} margin={{ top: 12, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="x" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--line)" />
            <YAxis
              domain={domain}
              tickFormatter={(v: number) => fmtScore(v, run.metric)}
              tick={{ fill: "var(--muted)", fontSize: 12 }}
              stroke="var(--line)"
              width={64}
            />
            <Tooltip
              formatter={(v: number) => [fmtScore(v, run.metric), run.metric]}
              labelFormatter={(l) => `Эксперимент ${l}`}
              contentStyle={{ borderRadius: 10, border: "1px solid var(--line)", fontFamily: "inherit" }}
            />
            {run.baseline !== null && (
              <ReferenceLine
                y={run.baseline}
                stroke="var(--blue)"
                strokeDasharray="5 4"
                label={{ value: "baseline", position: "insideTopLeft", fill: "var(--blue)", fontSize: 12 }}
              />
            )}
            <Line
              type="monotone"
              dataKey="score"
              stroke="var(--red)"
              strokeWidth={2.5}
              isAnimationActive={false}
              dot={(props: { cx?: number; cy?: number; payload?: Point; index?: number }) => {
                const { cx, cy, payload, index } = props;
                if (cx === undefined || cy === undefined || !payload) return <g key={index} />;
                const isBest = run.best !== null && payload.score === run.best;
                return (
                  <circle
                    key={payload.id}
                    cx={cx}
                    cy={cy}
                    r={isBest ? 8 : 6}
                    fill={isBest ? "var(--lime)" : "var(--surface)"}
                    stroke={isBest ? "var(--ink)" : "var(--red)"}
                    strokeWidth={2.5}
                    style={{ cursor: "pointer" }}
                    onClick={() => onSelect(payload.id)}
                  />
                );
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
      <ul className="sr-only">
        {points.map((p) => (
          <li key={p.id}>
            Эксперимент {p.id}: {fmtScore(p.score, run.metric)}
          </li>
        ))}
      </ul>
    </div>
  );
}
