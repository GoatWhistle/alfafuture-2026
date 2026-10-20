import type { Experiment, Params, Run } from "./types";

const CODE_FREQ = `class FreqEncoder:
    def fit(self, X):
        self.maps = {c: X[c].value_counts(normalize=True) for c in ["region", "channel"]}
        return self

    def transform(self, X):
        out = X.copy()
        for c, m in self.maps.items():
            out[c + "_freq"] = out[c].map(m).fillna(0.0)
        return out.drop(columns=list(self.maps))

def build_transformer():
    return FreqEncoder()`;

function exp(p: Partial<Experiment> & Pick<Experiment, "id">): Experiment {
  return {
    parentId: null,
    level: "full",
    status: "ok",
    observed: "",
    proposed: "",
    executed: "",
    measured: "",
    proposedParams: {},
    executedParams: {},
    featureChanges: [],
    score: null,
    delta: null,
    durationSec: null,
    error: null,
    code: null,
    backend: "catboost",
    ...p,
  };
}

const base = (id: string, title: string, over: Partial<Run>): Run => ({
  id,
  title,
  taskKind: "Бинарная классификация",
  metric: "ROC AUC",
  direction: "higher",
  status: "completed",
  stage: "result",
  baseline: 0.8412,
  best: null,
  elapsedSec: 0,
  startedAt: "2026-10-20T09:12:00",
  demo: true,
  model: "Qwen3-Coder",
  mode: "qwen",
  budget: { timeSec: 1800, tokens: 32000, threads: 4 },
  used: { timeSec: 0, tokens: 0, tokensUncertain: false },
  experiments: [],
  csvAvailable: true,
  finalCsv: true,
  note: null,
  ...over,
});

const catParams = (depth: number, lr: number): Params => ({ depth, learning_rate: lr, iterations: 600 });

/** Улучшение: смоук + две полные гипотезы, вторая даёт прирост */
const improved = base("demo-improved", "Отток клиентов, банковский сегмент", {
  best: 0.8667,
  elapsedSec: 1244,
  used: { timeSec: 1244, tokens: 21480, tokensUncertain: false },
  experiments: [
    exp({
      id: "e1",
      parentId: "baseline",
      level: "smoke",
      observed: "Ошибки сосредоточены в классе «ушёл» среди клиентов с редкими регионами.",
      proposed: "Заменить категорию region на частоту в train.",
      executed: "Frequency encoding для region, channel на 10% строк.",
      measured: "Smoke выполнен без ошибок за 38 с. Метрика smoke не сравнивается с full.",
      proposedParams: catParams(6, 0.05),
      executedParams: catParams(6, 0.05),
      featureChanges: ["region → region_freq", "channel → channel_freq"],
      score: 0.8398,
      durationSec: 38,
      code: CODE_FREQ,
    }),
    exp({
      id: "e2",
      parentId: "baseline",
      observed: "Ошибки сосредоточены в классе «ушёл» среди клиентов с редкими регионами.",
      proposed: "Заменить категорию region на частоту в train.",
      executed: "Frequency encoding для region, channel; параметры без изменений.",
      measured: "AUC 0.8521 против 0.8412 у baseline: +0.0109 на полной validation.",
      proposedParams: catParams(6, 0.05),
      executedParams: catParams(6, 0.05),
      featureChanges: ["region → region_freq", "channel → channel_freq"],
      score: 0.8521,
      delta: 0.0109,
      durationSec: 212,
      code: CODE_FREQ,
    }),
    exp({
      id: "e3",
      parentId: "e2",
      observed: "Остаточные ошибки — клиенты с большим разбросом сумм операций.",
      proposed: "signed_log1p для amount_*, depth 8, learning_rate 0.04, iterations 800.",
      executed: "signed_log1p для amount_sum, amount_max; depth 8, lr 0.04, iterations 600.",
      measured: "AUC 0.8667: +0.0146 к родителю, +0.0255 к baseline.",
      proposedParams: { depth: 8, learning_rate: 0.04, iterations: 800 },
      executedParams: { depth: 8, learning_rate: 0.04, iterations: 600 },
      featureChanges: ["amount_sum → signed_log1p", "amount_max → signed_log1p"],
      score: 0.8667,
      delta: 0.0146,
      durationSec: 341,
    }),
  ],
});

/** Нет улучшения: baseline остаётся лучшим */
const noImprovement = base("demo-noimp", "Прогноз просрочки платежа", {
  startedAt: "2026-10-19T17:40:00",
  baseline: 0.7934,
  best: 0.7934,
  elapsedSec: 1502,
  used: { timeSec: 1502, tokens: 27910, tokensUncertain: false },
  note: "Ни одна гипотеза не превзошла baseline. В итоговый CSV вошла baseline-модель.",
  experiments: [
    exp({
      id: "e1",
      parentId: "baseline",
      observed: "Модель ошибается на клиентах с короткой историей.",
      proposed: "Удалить признаки tenure_*, увеличить l2_leaf_reg до 6.",
      executed: "Удалены tenure_months, tenure_bucket; l2_leaf_reg 6.",
      measured: "AUC 0.7902: −0.0032 относительно baseline.",
      proposedParams: { l2_leaf_reg: 6, depth: 6 },
      executedParams: { l2_leaf_reg: 6, depth: 6 },
      featureChanges: ["drop tenure_months", "drop tenure_bucket"],
      score: 0.7902,
      delta: -0.0032,
      durationSec: 190,
    }),
    exp({
      id: "e2",
      parentId: "baseline",
      observed: "Разрыв между train и validation указывает на переобучение.",
      proposed: "depth 4, learning_rate 0.03.",
      executed: "depth 4, learning_rate 0.03, iterations 600.",
      measured: "AUC 0.7921: −0.0013 относительно baseline.",
      proposedParams: catParams(4, 0.03),
      executedParams: catParams(4, 0.03),
      score: 0.7921,
      delta: -0.0013,
      durationSec: 176,
    }),
    exp({
      id: "e3",
      parentId: "baseline",
      observed: "Редкие категории в merchant_type.",
      proposed: "Frequency encoding для merchant_type.",
      executed: "Frequency encoding для merchant_type.",
      measured: "AUC 0.7930: −0.0004 относительно baseline.",
      proposedParams: catParams(6, 0.05),
      executedParams: catParams(6, 0.05),
      featureChanges: ["merchant_type → merchant_type_freq"],
      score: 0.793,
      delta: -0.0004,
      durationSec: 204,
      code: CODE_FREQ,
    }),
  ],
});

/** Отказ Qwen */
const qwenFail = base("demo-qwenfail", "Скоринг заявок на кредит", {
  startedAt: "2026-10-19T12:05:00",
  baseline: 0.8105,
  best: 0.8105,
  elapsedSec: 410,
  status: "failed",
  stage: "research",
  finalCsv: false,
  used: { timeSec: 410, tokens: 4120, tokensUncertain: true },
  note: "Qwen недоступна: 3 попытки завершились ошибкой 503. Сохранён CSV baseline.",
  experiments: [
    exp({
      id: "e1",
      parentId: "baseline",
      status: "failed",
      level: "smoke",
      observed: "Ошибки по классу «отказ» у клиентов без кредитной истории.",
      proposed: "Предложение не получено.",
      executed: "Эксперимент не запускался.",
      measured: "Нет измерений. Usage последнего запроса неизвестен — учтён консервативно.",
      error: "LLM API: 503 Service Unavailable после 3 попыток (задержки 2, 4, 8 с).",
    }),
  ],
});

/** Отклонение по ресурсам и timeout полного обучения */
const timeout = base("demo-timeout", "Выручка точек продаж", {
  taskKind: "Регрессия",
  metric: "RMSE",
  direction: "lower",
  startedAt: "2026-10-18T21:30:00",
  baseline: 14.62,
  best: 14.31,
  elapsedSec: 1790,
  used: { timeSec: 1790, tokens: 29800, tokensUncertain: false },
  note: "Последний эксперимент остановлен по deadline. Лучший CSV сохранён.",
  experiments: [
    exp({
      id: "e1",
      parentId: "baseline",
      observed: "Наибольшие ошибки у точек с выручкой выше 90-го перцентиля.",
      proposed: "signed_log1p для revenue_prev, depth 6.",
      executed: "signed_log1p для revenue_prev; depth 6.",
      measured: "RMSE 14.31 против 14.62: −0.31 (лучше).",
      proposedParams: catParams(6, 0.05),
      executedParams: catParams(6, 0.05),
      featureChanges: ["revenue_prev → signed_log1p"],
      score: 14.31,
      delta: -0.31,
      durationSec: 260,
      backend: "lightgbm",
    }),
    exp({
      id: "e2",
      parentId: "e1",
      status: "rejected",
      level: "smoke",
      observed: "Остаточная нелинейность по сезонным признакам.",
      proposed: "iterations 4000, depth 10, tune=true.",
      executed: "Не запускалось: оценка 1 140 с превышает остаток 610 с.",
      measured: "Отклонено по ресурсам: прогноз 1 140 с > доступных 610 с. Параметры не уменьшались.",
      proposedParams: { iterations: 4000, depth: 10, tune: true },
      error: "Отклонено допуском по ресурсам.",
      backend: "lightgbm",
    }),
    exp({
      id: "e3",
      parentId: "e1",
      status: "timeout",
      observed: "Остаточная нелинейность по сезонным признакам.",
      proposed: "iterations 1500, depth 8.",
      executed: "iterations 1500, depth 8; остановлено по deadline 420 с.",
      measured: "Worker не завершился за 420 с. Результата нет.",
      proposedParams: { iterations: 1500, depth: 8 },
      executedParams: { iterations: 1500, depth: 8 },
      durationSec: 420,
      error: "Worker timeout 420 с.",
      backend: "lightgbm",
    }),
  ],
});

/** Завершён, поиск не запускался — сохранён baseline */
const baselineKept = base("demo-baseline", "Сегментация обращений в поддержку", {
  taskKind: "Мультикласс",
  metric: "Macro F1",
  startedAt: "2026-10-17T10:20:00",
  baseline: 0.612,
  best: 0.612,
  elapsedSec: 640,
  used: { timeSec: 640, tokens: 0, tokensUncertain: false },
  mode: "baseline",
  note: "Режим baseline: поиск гипотез не запускался, в CSV вошла baseline-модель.",
});

/** Выполняется: проигрывается пошагово */
const running = base("demo-running", "Кросс-продажи карты, сегмент 2", {
  status: "running",
  stage: "research",
  startedAt: "2026-10-20T10:02:00",
  baseline: 0.7788,
  best: 0.7882,
  elapsedSec: 1100,
  finalCsv: false,
  used: { timeSec: 1100, tokens: 18200, tokensUncertain: true },
  experiments: [
    exp({
      id: "e1",
      parentId: "baseline",
      observed: "Ошибки в сегменте клиентов с одним продуктом.",
      proposed: "Frequency encoding для product_code.",
      executed: "Frequency encoding для product_code.",
      measured: "AUC 0.7864: +0.0076 к baseline.",
      proposedParams: catParams(6, 0.05),
      executedParams: catParams(6, 0.05),
      featureChanges: ["product_code → product_code_freq"],
      score: 0.7864,
      delta: 0.0076,
      durationSec: 187,
      code: CODE_FREQ,
    }),
    exp({
      id: "e2",
      parentId: "e1",
      observed: "Систематическая недооценка у клиентов с высоким доходом.",
      proposed: "signed_log1p для income, depth 7.",
      executed: "signed_log1p для income; depth 7.",
      measured: "AUC 0.7851: −0.0013 к родителю.",
      proposedParams: catParams(7, 0.05),
      executedParams: catParams(7, 0.05),
      featureChanges: ["income → signed_log1p"],
      score: 0.7851,
      delta: -0.0013,
      durationSec: 201,
    }),
    exp({
      id: "e3",
      parentId: "e1",
      observed: "Сохраняется разброс в регионах с малым числом клиентов.",
      proposed: "Увеличить l2_leaf_reg до 5.",
      executed: "l2_leaf_reg 5.",
      measured: "AUC 0.7882: +0.0018 к родителю.",
      proposedParams: { l2_leaf_reg: 5 },
      executedParams: { l2_leaf_reg: 5 },
      score: 0.7882,
      delta: 0.0018,
      durationSec: 195,
    }),
  ],
});

export const SCENARIOS: Run[] = [running, improved, noImprovement, qwenFail, timeout, baselineKept];
/** Шаг, на котором стоит проигрывание при первом открытии */
export const PLAYBACK_START: Record<string, number> = { "demo-running": 4 };
