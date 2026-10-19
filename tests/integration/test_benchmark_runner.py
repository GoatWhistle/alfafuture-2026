"""Actual fake CLI processes exercise telemetry, cache integrity and equal ceilings."""

import importlib.util
import json
from pathlib import Path

import psutil
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "verification/run_benchmarks.py"
module_spec = importlib.util.spec_from_file_location("offline_benchmark_runner", SCRIPT)
benchmark = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(benchmark)


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    task = root / "task.md"
    task.write_text("RMSE\nOutput: `answer.csv`\n")
    (root / "train.csv").write_text("id,x,y\ntr0,0,0\ntr1,1,1\ntr2,2,2\ntr3,3,3\ntr4,4,4\n")
    (root / "test.csv").write_text("id,x\n001,9\nNA,8\n")
    (root / "data_dictionary.csv").write_text(
        "feature,role,dtype\nid,id,string\nx,feature,numeric\ny,target,numeric\n"
    )
    (root / "sample_submission.csv").write_text("id,y\n001,0\nNA,0\n")
    fake = tmp_path / "fake_solution.py"
    fake.write_text("""import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("task",type=Path)
p.add_argument("--baseline-only",action="store_true",required=True)
for name in ("seed","backend","threads","time-budget","worker-timeout","memory-limit-mb","run-dir"):
    p.add_argument("--"+name,required=True)
a=p.parse_args()
assert "OPENAI_API_KEY" not in os.environ and "QWEN_KEY" not in os.environ
root=a.task.parent
mode=os.environ.get("BENCHMARK_FAKE_MODE","")
with (Path(__file__).parent/"calls.jsonl").open("a") as f:
    f.write(json.dumps({"seed":a.seed,"backend":a.backend,"threads":a.threads,
                      "time_budget":a.time_budget,"memory_mb":a.memory_limit_mb})+"\\n")
run=Path(a.run_dir)
run.mkdir(parents=True)
childcode="import time; data=bytearray(16*1024*1024); time.sleep("
child=subprocess.Popen([sys.executable,"-c",childcode+("5" if mode=="hang" else "0.12")+")"])
(run/"child.json").write_text(json.dumps({"pid":child.pid}))
if mode=="hang": time.sleep(5)
child.wait()
protocol={"fingerprint":"protocol-"+a.seed+("-"+a.backend if mode=="mismatch" else ""),
          "level":"full","train_indices":[0,1,2],"validation_indices":[3,4],"control_indices":[]}
spec={"train_path":str(root/"train.csv"),"test_path":str(root/"test.csv"),"submission_path":str(root/"answer.csv"),"id_column":"id",
      "target":"y","submission_columns":["id","y"],"kind":"regression","metric":"rmse",
      "prediction_kind":"values","classes":[]}
(run/"prepared.json").write_text(json.dumps({"spec":spec,"protocol":protocol}))
score={"value":1.0,"metric":"rmse","protocol_id":protocol["fingerprint"],"level":"full"}
summary={"score":score,"submission_path":spec["submission_path"],"finalization":{
         "selected":"baseline","refit_status":"succeeded","control_scores":{},
         "selection_protocol":"search_no_control_partition"}}
if mode=="fallback":
    summary={"status":"fallback_preserved","submission_path":spec["submission_path"]}
else:
    (run/"baseline.json").write_text(json.dumps({"score":score}))
    d=run/"validation"
    d.mkdir()
    (d/"result.json").write_text(json.dumps({"ids":["tr3","tr4"],"predictions":[3,4],"classes":[]}))
(run/"summary.json").write_text(json.dumps(summary))
value="NaN" if mode=="invalid" else ("1" if a.backend=="catboost" else "2")
(root/"answer.csv").write_text("id,y\\n001,"+value+"\\nNA,"+value+"\\n")
""")
    return task, fake, tmp_path / "results"


def run(case, **kwargs):
    task, fake, output = case
    return benchmark.run_benchmarks(
        [task],
        output,
        solution=fake,
        seeds=(7,),
        backends=("catboost", "lightgbm"),
        threads=1,
        time_budget=5,
        memory_limit_mb=256,
        **kwargs,
    )


def test_sequential_real_process_telemetry_copies_and_cache(case, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-cli")
    monkeypatch.setenv("QWEN_KEY", "must-not-reach-cli")
    report = run(case)
    assert report["mode"] == "baseline_only" and report["success_fraction"] == 1
    assert report["full_agent_comparison"]["status"] == "blocked"
    assert len(report["runs"]) == 2
    for result in report["runs"]:
        assert result["returncode"] == 0 and result["wall_seconds"] > 0
        assert result["peak_rss_bytes"] > 20 * 1024 * 1024
        assert result["rss_scope"] == "cli_and_descendants"
        assert result["score_status"] == "measured"
        assert result["submission"]["rows"] == 2
    first, second = [Path(item["submission"]["path"]).read_text() for item in report["runs"]]
    assert first != second and "001" in first and "NA" in second
    calls = (case[1].parent / "calls.jsonl").read_text()
    assert run(case) == report
    assert (case[1].parent / "calls.jsonl").read_text() == calls
    configs = [json.loads(line) for line in calls.splitlines()]
    assert all(item["time_budget"] == "5" and item["memory_mb"] == "256" for item in configs)


def test_config_and_cached_copy_changes_fail_without_rerun(case):
    report = run(case)
    with pytest.raises(ValueError, match="configuration differs"):
        benchmark.run_benchmarks(
            [case[0]], case[2], solution=case[1], seeds=(42,), backends=("catboost",), threads=1
        )
    Path(report["runs"][0]["submission"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(case)
    assert len((case[1].parent / "calls.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("mode", ["fallback", "invalid"])
def test_success_fraction_uses_measured_artifacts_not_cli_zero_exit(case, monkeypatch, mode):
    monkeypatch.setenv("BENCHMARK_FAKE_MODE", mode)
    report = run(case)
    assert report["success_fraction"] == 0
    assert all(item["returncode"] == 0 for item in report["runs"])
    if mode == "fallback":
        assert report["valid_submission_fraction"] == 1
        assert all(item["fallback_status"] for item in report["runs"])
    else:
        assert report["valid_submission_fraction"] == 0
        assert all(item["score_status"] == "measured" for item in report["runs"])


def test_backend_comparison_requires_same_protocol(case, monkeypatch):
    monkeypatch.setenv("BENCHMARK_FAKE_MODE", "mismatch")
    with pytest.raises(ValueError, match="protocol fingerprint mismatch"):
        run(case)


def test_actual_time_ceiling_kills_cli_and_memory_allocating_child(case, monkeypatch):
    monkeypatch.setenv("BENCHMARK_FAKE_MODE", "hang")
    report = benchmark.run_benchmarks(
        [case[0]],
        case[2],
        solution=case[1],
        seeds=(7,),
        backends=("catboost",),
        threads=1,
        time_budget=0.3,
        memory_limit_mb=256,
    )
    result = report["runs"][0]
    assert result["termination"] == "time_limit" and result["returncode"] != 0
    assert result["wall_seconds"] < 3 and report["success_fraction"] == 0
    child = json.loads(next(case[2].glob("*/*/harness/child.json")).read_text())["pid"]
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_input_data_change_invalidates_cache_and_holdout_targets_are_not_inputs(case):
    run(case)
    (case[0].parent / "holdout_targets.csv").write_text("private external evaluation only")
    run(case)
    train = case[0].parent / "train.csv"
    train.write_text(train.read_text() + "tr5,5,5\n")
    with pytest.raises(ValueError, match="configuration differs"):
        run(case)


def test_dependency_versions_and_lockfile_hashes_bind_cached_runs(case, monkeypatch):
    run(case)
    configuration = json.loads((case[2] / "configuration.json").read_text())
    assert len(configuration["code"]["pyproject"]) == 64
    assert len(configuration["code"]["uv_lock"]) == 64
    assert set(configuration["runtime_versions"]) == {
        "catboost",
        "lightgbm",
        "autogluon.tabular",
        "scikit-learn",
        "numpy",
        "psutil",
    }
    altered = {**configuration["runtime_versions"], "numpy": "changed-test-version"}
    monkeypatch.setattr(benchmark, "_runtime_versions", lambda: altered)
    with pytest.raises(ValueError, match="configuration differs"):
        run(case)
    assert len((case[1].parent / "calls.jsonl").read_text().splitlines()) == 2
