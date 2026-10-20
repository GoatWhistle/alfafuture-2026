import { describe, expect, it } from "vitest";
import { snapshot } from "./demo";
import { SCENARIOS } from "./scenarios";

const get = (id: string) => SCENARIOS.find((s) => s.id === id)!;

describe("snapshot", () => {
  it("не показывает baseline и эксперименты до их шагов", () => {
    const s = snapshot(get("demo-improved"), 1, false);
    expect(s.baseline).toBeNull();
    expect(s.experiments).toHaveLength(0);
    expect(s.status).toBe("running");
  });

  it("показывает выполняющийся эксперимент без score", () => {
    const s = snapshot(get("demo-improved"), 2, false);
    expect(s.experiments[0].status).toBe("running");
    expect(s.experiments[0].score).toBeNull();
    expect(s.csvAvailable).toBe(true);
  });

  it("лучший результат считает только по full", () => {
    const s = snapshot(get("demo-improved"), 5, false);
    expect(s.best).toBe(0.8521);
  });

  it("последний шаг возвращает итоговое состояние сценария", () => {
    const s = snapshot(get("demo-running"), 999, false);
    expect(s.status).toBe("completed");
    expect(s.finalCsv).toBe(true);
  });

  it("отказ Qwen остаётся ошибкой без итогового CSV, но с текущим", () => {
    const s = snapshot(get("demo-qwenfail"), 999, false);
    expect(s.status).toBe("failed");
    expect(s.finalCsv).toBe(false);
    expect(s.csvAvailable).toBe(true);
  });

  it("прерывание сохраняет текущий CSV и не даёт итогового", () => {
    const s = snapshot(get("demo-running"), 4, true);
    expect(s.status).toBe("stopped");
    expect(s.csvAvailable).toBe(true);
    expect(s.finalCsv).toBe(false);
  });
});
