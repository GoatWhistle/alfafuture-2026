"""Predeclared paired research-budget curves; holdout scores never reach the agent.

Both modes use the same profiling, fixed final training and CSV selection. This
isolates research, excludes AutoGluon, and reports actual runtime as well as caps.
Credentials are read only by the launcher and are never saved with run artifacts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from compare import dataset


def worker(source, task, budget, final_fit_seconds=180):
    sys.path.insert(0, str(Path(source).resolve()))
    import models
    import linear_search
    import solution
    from agent_utils import StageTimeout, shutdown, start_watchdog, time_limit
    from task_profile import build_profile

    task = Path(task)
    start_watchdog(task.parent)
    started = time.monotonic()
    try:
        prof = build_profile(task)
        target = prof["target_cols"][0]
        current = {target: models.constant_predictions(prof, target)}
        models.write_submission(prof, current)
        quick = models.quick_model(prof, target)
        current[target] = quick["test_pred"]
        models.write_submission(prof, current)
        prof["base_features"] = list(prof["features"])
        work = {**prof, "features": list(prof["features"])}
        begin = time.monotonic()
        research_status = "complete"
        try:
            with time_limit(budget):
                research = solution.run_autoeda(work, target, budget, use_llm=bool(os.environ.get("OPENAI_API_KEY")))
        except StageTimeout:
            work = {**prof, "features": list(prof["features"])}
            research = {"accepted": []}
            research_status = "deadline_fallback"
        research_seconds = time.monotonic() - begin
        # Both modes use identical final fitting/selection, on the same discovered CV.
        base = {**prof, "cv": work.get("cv", {"kind": "kfold"})}
        end = time.time() + final_fit_seconds
        baseline = models.quick_oof(base, target, deadline_ts=end)
        cands = [{"name": "base", **baseline}] if baseline is not None else []
        if research["accepted"]:
            candidate = models.quick_oof(work, target, deadline_ts=end)
            if candidate is not None:
                cands.append({"name": "recipe", **candidate})
        linear = linear_search.search(base, target, min(30, end - time.time()))
        solution.select_final(work, target, current, quick, cands, current[target],
                              extra_cands=[linear] if linear is not None else [])
        (task.parent / "result.json").write_text(json.dumps(dict(
            research_seconds=research_seconds, research_status=research_status, total_seconds=time.monotonic()-started,
            accepted=research["accepted"], cv=work.get("cv"), experiments=solution.EXPERIMENTS),
            ensure_ascii=False, default=str))
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        (task.parent / "failure.json").write_text(json.dumps(dict(error=type(exc).__name__)))
    finally:
        shutdown(0 if (task.parent / "result.json").exists() else 1, task.parent)


def prepare_case(args, name, seed, root):
    """Preserve competition target names, task meaning and dictionary in local splits."""
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    train, test, truth, metric, provenance = dataset(name, seed, args.public_root,
                                                    args.max_train_rows, args.max_test_rows)
    target = "target"
    task_text = {"rmse": "Регрессия. Метрика RMSE.", "log_loss": "Бинарная классификация. Метрика log loss.",
                 "f1_macro": "Многоклассовая классификация. Метрика Macro F1."}[metric]
    if name.startswith("public_"):
        kind = name.removeprefix("public_")
        sub = "multiclass" if kind == "classification" else "regression"
        target = "attack_cat" if kind == "classification" else "site_eui"
        original = args.public_root / kind
        train, truth = train.rename(columns={"target": target}), truth.rename(columns={"target": target})
        dictionary = original / sub / "participants" / "data_dictionary.csv"
        (root / "data_dictionary.csv").write_bytes(dictionary.read_bytes())
        task_text = (original / "task.md").read_text().replace(f"{sub}/participants/", "")
        if kind == "classification":
            task_text = re.sub(r"Test — stratified-выборка.*?UNSW-NB15\.",
                               "Локальный test — стратифицированный holdout из исходного train.",
                               task_text, flags=re.S)
        task_text = (f"Локальная проверка исходной задачи: train содержит {len(train)} строк, "
                     f"test — {len(test)} строк без ответов. Размеры ниже описывают исходный набор.\n\n" + task_text)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    placeholder = str(train[target].iloc[0]) if metric == "f1_macro" else 0.
    pd.DataFrame({"id": test.id, target: placeholder}).to_csv(root / "sample_submission.csv", index=False)
    (root / "task.md").write_text(task_text + " Используйте train.csv, test.csv и sample_submission.csv.")
    truth_dir = args.output / "holdout_labels"
    truth_dir.mkdir(exist_ok=True)
    truth.to_csv(truth_dir / f"{name}-{seed}.csv", index=False)
    return truth, metric, provenance, target, len(train), len(test)


def run_one(args, name, seed, mode, budget, credentials):
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn.metrics import f1_score, log_loss, mean_squared_error

    root = args.output / name / str(seed) / str(budget) / mode
    truth, metric, provenance, target, n_train, n_test = prepare_case(args, name, seed, root)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "AGENT_"))}
    env.update(credentials)
    env.update(AGENT_N_CPUS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               AGENT_RESEARCH_MODE=mode, AGENT_LLM_MAX_TOKENS=str(args.max_tokens), AGENT_LLM_REQUEST_SEC="90",
               AGENT_LLM_REASONING_TOKENS="1024",
               AGENT_TIME_LIMIT_SEC=str(budget + args.final_fit_seconds + 120),
               AGENT_SAFETY_SEC="10", AGENT_HARD_MARGIN_SEC="10")
    peak, started = 0, time.monotonic()
    with (root / "log.txt").open("w") as stream:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(args.source),
                                 str(root / "task.md"), str(budget), str(args.final_fit_seconds)], cwd=root, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
        monitor = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - started > budget + args.final_fit_seconds + 150:
                for child in monitor.children(recursive=True):
                    child.kill()
                proc.kill()
                proc.wait()
                break
            time.sleep(.2)
    record = dict(task=name, seed=seed, mode=mode, research_budget_seconds=budget, metric=metric,
                  provenance=provenance, seconds=time.monotonic()-started, peak_rss_mb=peak/2**20,
                  exit_code=proc.returncode, success=False, target=target, train_rows=n_train, holdout_rows=n_test)
    try:
        pred = pd.read_csv(root / "submission.csv")
        assert list(pred.columns) == ["id", target] and np.array_equal(pred.id, truth.id)
        score = float(np.sqrt(mean_squared_error(truth[target], pred[target]))) if metric == "rmse" else (
            float(log_loss(truth[target], pred[target], labels=[0, 1])) if metric == "log_loss" else
            float(f1_score(truth[target], pred[target], average="macro")))
        result = json.loads((root / "result.json").read_text())
        record.update(score=score, success=proc.returncode == 0 and bool(np.isfinite(score)),
                      research_seconds=result["research_seconds"], research_status=result["research_status"],
                      accepted=result["accepted"])
    except Exception as exc:
        record["error"] = type(exc).__name__
    logs = (root / "log.txt").read_text()
    record["llm_responses"] = logs.count(" usage: ")
    record["llm_valid_json"] = len(re.findall(r"LLM\[[^\]]+\] \(попытка", logs))
    record["llm_truncated"] = logs.count("finish_reason=length")
    record["llm_cost_rub"] = sum(float(x) for x in re.findall(r"'cost_rub': ([0-9.eE+-]+)", logs))
    paths = list((root / "research").glob("*.json")) if (root / "research").exists() else []
    if paths:
        report = json.loads(paths[0].read_text())
        record.update(history=report["history"], stop_reason=report["stop_reason"],
                      trials=sum(e["event"] == "trial" for e in report["events"]),
                      rounds=sum(e["event"] == "round" for e in report["events"]))
    return record


def main():
    if sys.argv[1:2] == ["--worker"]:
        return worker(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("solution"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--tasks", nargs="+", default=["diabetes", "nested_segments"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--budgets", nargs="+", type=int, default=[600, 1800, 5400])
    parser.add_argument("--modes", nargs="+", choices=["legacy", "iterative"], default=["legacy", "iterative"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--final-fit-seconds", type=int, default=180)
    parser.add_argument("--public-root", type=Path, default=Path("data"))
    parser.add_argument("--max-train-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=6000)
    args = parser.parse_args()
    args.source, args.output, args.public_root = args.source.resolve(), args.output.resolve(), args.public_root.resolve()
    credentials = json.loads(args.credentials_file.read_text()) if args.credentials_file else {}
    credentials = {k: v for k, v in credentials.items() if k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {k: v for k, v in vars(args).items() if k != "credentials_file"}
    protocol.update(llm=bool(credentials.get("OPENAI_API_KEY")), model=credentials.get("OPENAI_MODEL"),
                    cpus_per_worker=2, autogluon=False, llm_request_seconds=90,
                    llm_reasoning=dict(max_tokens=1024),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source.glob('*.py')},
                    note="Budgets are research caps. Early stopping on exhausted search is recorded, not counted as full wall time. "
                         "Each cap is an independent run; profile and final training are outside the research cap. "
                         "Holdout labels and scores never enter the task or LLM context. "
                         "Full public regression uses fixed year-5 holdout; different seeds shuffle rows, not independent cohorts.")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    records = []
    with ThreadPoolExecutor(args.jobs) as pool, (args.output / "results.jsonl").open("w") as stream:
        futures = [pool.submit(run_one, args, task, seed, mode, budget, credentials)
                   for budget in args.budgets for seed in args.seeds for task in args.tasks for mode in args.modes]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            line = json.dumps(record, ensure_ascii=False, default=str)
            stream.write(line + '\n')
            stream.flush()
            print(json.dumps({k: v for k, v in record.items() if k not in ('history', 'accepted')}, ensure_ascii=False), flush=True)
            (args.output / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
