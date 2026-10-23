"""Paired offline benchmark; holdout labels live outside each agent's task folder.

Example: python verification/compare.py --baseline /path/old/solution
    --candidate /path/new/solution --output .scratch/benchmark --seeds 17 29 43
This short-budget comparison excludes LLM and AutoGluon and is not a platform score.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(source, task):
    sys.path.insert(0, str(Path(source).resolve()))
    import solution
    from agent_utils import shutdown

    # Exercise AutoEDA at a short budget; identical adjustment in both revisions.
    solution.MIN_AUTOEDA_SEC = 1
    sys.argv = [str(Path(source) / "solution.py"), str(task)]
    try:
        solution.main()
        (Path(task).parent / "experiments.json").write_text(
            json.dumps(solution.EXPERIMENTS, ensure_ascii=False, default=str))
    finally:
        shutdown(0 if (Path(task).parent / "submission.csv").exists() else 1, Path(task).parent)


def dataset(name, seed, public_root=None, max_train_rows=15000, max_test_rows=6000):
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed)
    if name in ("public_classification", "public_regression"):
        kind = name.removeprefix("public_")
        sub = "multiclass" if kind == "classification" else "regression"
        df = pd.read_csv(public_root / kind / sub / "participants" / "train.csv")
        target = "attack_cat" if kind == "classification" else "site_eui"
        if kind == "classification":
            train, holdout = train_test_split(df, test_size=0.2, stratify=df[target], random_state=seed)
            if len(train) > max_train_rows:
                train, _ = train_test_split(train, train_size=max_train_rows, stratify=train[target], random_state=seed)
            if len(holdout) > max_test_rows:
                holdout, _ = train_test_split(holdout, train_size=max_test_rows, stratify=holdout[target], random_state=seed)
            metric = "f1_macro"
        else:
            holdout = df[df.Year_Factor == df.Year_Factor.max()]
            train = df[df.Year_Factor < df.Year_Factor.max()]
            train = train.sample(min(len(train), max_train_rows), random_state=seed)
            holdout = holdout.sample(min(len(holdout), max_test_rows), random_state=seed)
            metric = "rmse"
        train = train.rename(columns={target: "target"})
        truth = holdout[["id", target]].rename(columns={target: "target"})
        return train, holdout.drop(columns=target), truth, metric, "competition train; independent local holdout"
    if name in ("diabetes", "breast_cancer", "wine"):
        data = getattr(datasets, "load_" + name)()
        X = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])
        y = data.target
        metric = {"diabetes": "rmse", "breast_cancer": "log_loss", "wine": "f1_macro"}[name]
        if name == "wine":
            y = np.array(["class_a", "class_b", "class_c"])[y]
        provenance = "sklearn bundled real dataset"
    elif name == "category_codes":
        n, k = 3200, 64
        codes = rng.randint(k, size=n)
        category_prob = rng.uniform(0.05, 0.95, size=k)
        X = pd.DataFrame({"product_code": codes, "measurement": rng.randn(n)})
        y = (rng.rand(n) < category_prob[codes]).astype(int)
        metric, provenance = "log_loss", "synthetic arbitrary category codes"
    elif name == "nested_segments":
        n = 3200
        fine = rng.randint(4, size=n)
        x, z = rng.randn(2, n)
        X = pd.DataFrame({"fine": fine, "coarse": fine // 2, "x": x, "z": z})
        y = np.array([-5, -2, 2, 5])[fine] * x + rng.randn(n) * 0.8
        metric, provenance = "rmse", "synthetic nested regimes"
    else:
        raise ValueError(name)
    train, test = train_test_split(np.arange(len(X)), test_size=0.25, random_state=seed,
                                   stratify=None if metric == "rmse" else y)
    X.insert(0, "id", np.arange(len(X)))
    tr, te = X.iloc[train].copy(), X.iloc[test].copy()
    tr["target"] = y[train]
    truth = pd.DataFrame({"id": te.id, "target": y[test]})
    return tr, te, truth, metric, provenance


def run_one(args, name, seed, variant, source):
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn.metrics import f1_score, log_loss, mean_squared_error

    root = args.output / name / str(seed) / variant
    root.mkdir(parents=True, exist_ok=True)
    tr, te, truth, metric, provenance = dataset(name, seed, args.public_root, args.max_train_rows, args.max_test_rows)
    tr.to_csv(root / "train.csv", index=False)
    te.to_csv(root / "test.csv", index=False)
    sample = pd.DataFrame({"id": te.id, "target": "class_a" if metric == "f1_macro" else 0.0})
    sample.to_csv(root / "sample_submission.csv", index=False)
    text = {"rmse": "Регрессия. Метрика RMSE.", "log_loss": "Бинарная классификация. Метрика log loss.",
            "f1_macro": "Многоклассовая классификация. Метрика Macro F1."}[metric]
    if name == "category_codes":
        text += " product_code — код категории товара; measurement — числовое измерение."
    if name.startswith("public_"):
        kind = name.removeprefix("public_")
        sub = "multiclass" if kind == "classification" else "regression"
        target = "attack_cat" if kind == "classification" else "site_eui"
        original = args.public_root / kind
        text += "\n" + (original / "task.md").read_text().replace(target, "target")
        dictionary = pd.read_csv(original / sub / "participants" / "data_dictionary.csv")
        dictionary = dictionary.replace(target, "target")
        dictionary.to_csv(root / "data_dictionary.csv", index=False)
    (root / "task.md").write_text(text + " Используйте train.csv, test.csv и sample_submission.csv.")
    truth_dir = args.output / "holdout_labels"
    truth_dir.mkdir(exist_ok=True)
    truth.to_csv(truth_dir / f"{name}-{seed}-{variant}.csv", index=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_") and not k.startswith("AGENT_")}
    env.update(AGENT_N_CPUS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               AGENT_TIME_LIMIT_SEC=str(args.budget), AGENT_SAFETY_SEC="10", AGENT_HARD_MARGIN_SEC="10",
               AGENT_PREDICT_RESERVE_SEC="30", AGENT_KFOLD_RESERVE_SEC="30", AGENT_AUTOEDA_SEC=str(args.autoeda))
    record = {"task": name, "seed": seed, "variant": variant, "metric": metric, "provenance": provenance}
    t0, peak = time.monotonic(), 0
    with (root / "log.txt").open("w") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(source),
                                 str(root / "task.md")], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - t0 > args.budget + 20:
                for child in monitor.children(recursive=True):
                    child.kill()
                proc.kill()
                proc.wait()
                break
            time.sleep(0.1)
    record.update(seconds=round(time.monotonic() - t0, 3), peak_rss_mb=round(peak / 2**20, 2), exit_code=proc.returncode)
    try:
        pred = pd.read_csv(root / "submission.csv")
        assert list(pred.columns) == ["id", "target"] and np.array_equal(pred.id, truth.id)
        if metric == "rmse":
            value = float(np.sqrt(mean_squared_error(truth.target, pred.target)))
        elif metric == "log_loss":
            value = float(log_loss(truth.target, pred.target, labels=[0, 1]))
        else:
            value = float(f1_score(truth.target, pred.target, average="macro"))
        assert np.isfinite(value)
        record.update(score=value, success=proc.returncode == 0)
    except Exception as exc:
        record.update(score=None, success=False, error=str(exc))
    log = (root / "log.txt").read_text()
    record.update(autoeda_completed="AutoEDA: принято" in log,
                  kfold_completed="quick_oof[target]" in log, tracebacks=log.count("Traceback (most recent call last)"))
    return record


def summarize(output, records):
    import numpy as np

    result = []
    for name in sorted({r["task"] for r in records}):
        subset = [r for r in records if r["task"] == name]
        pairs = []
        for seed in sorted({r["seed"] for r in subset}):
            pair = {r["variant"]: r for r in subset if r["seed"] == seed and r["success"]}
            if len(pair) == 2:
                a, b = pair["baseline"], pair["candidate"]
                sign = 1 if a["metric"] == "f1_macro" else -1
                pairs.append({"seed": seed, "baseline": a["score"], "candidate": b["score"],
                              "improvement_pct": sign * (b["score"] - a["score"]) / max(abs(a["score"]), 1e-12) * 100})
        result.append({"task": name, "metric": subset[0]["metric"], "pairs": pairs,
                       "median_improvement_pct": float(np.median([p["improvement_pct"] for p in pairs])) if pairs else None})
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    if sys.argv[1:2] == ["--worker"]:
        return worker(sys.argv[2], sys.argv[3])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--tasks", nargs="+", default=["diabetes", "breast_cancer", "wine", "category_codes", "nested_segments"])
    p.add_argument("--budget", type=int, default=240)
    p.add_argument("--autoeda", type=int, default=60)
    p.add_argument("--public-root", type=Path)
    p.add_argument("--max-train-rows", type=int, default=15000)
    p.add_argument("--max-test-rows", type=int, default=6000)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    args.output, args.baseline, args.candidate = args.output.resolve(), args.baseline.resolve(), args.candidate.resolve()
    if args.public_root:
        args.public_root = args.public_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(vars(args), default=str, indent=2))
    records = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool, (args.output / "results.jsonl").open("w") as stream:
        futures = [pool.submit(run_one, args, name, seed, variant, source)
                   for name in args.tasks for seed in args.seeds
                   for variant, source in [("baseline", args.baseline), ("candidate", args.candidate)]]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            line = json.dumps(record)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)
            summarize(args.output, records)
    print(json.dumps(summarize(args.output, records), indent=2), flush=True)


if __name__ == "__main__":
    main()
