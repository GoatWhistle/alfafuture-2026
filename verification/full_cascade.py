"""Paired public-case evaluation of the production pipeline, including AutoGluon.

Only observers wrap production functions. Holdout labels and scores never enter
the worker. Both modes receive the same overall budget and research allocation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from research_curve import prepare_case


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)


def worker(source, task):
    sys.path.insert(0, str(Path(source).resolve()))
    import numpy as np
    import pandas as pd
    import solution
    from agent_utils import shutdown

    task = Path(task)
    meta = dict(research=[], autogluon=[], fits=[], pipeline_complete=False)

    def persist():
        save_json(task.parent / 'telemetry.json', meta)

    original_research = solution.run_autoeda
    def research(*args, **kwargs):
        event = dict(budget=args[2], started=time.monotonic(), status='running')
        meta['research'].append(event)
        persist()
        try:
            result = original_research(*args, **kwargs)
            event.update(status='complete', accepted=result['accepted'])
            return result
        except BaseException as exc:
            event.update(status=type(exc).__name__)
            raise
        finally:
            event['seconds'] = time.monotonic() - event.pop('started')
            persist()

    original_fit = solution.fit_autogluon
    def fit(*args, **kwargs):
        event = dict(budget=args[2], status='running')
        meta['fits'].append(event)
        start = time.monotonic()
        persist()
        try:
            predictor = original_fit(*args, **kwargs)
            event.update(status='complete', retained_models=predictor.model_names())
            try:
                event['leaderboard'] = json.loads(predictor.leaderboard(silent=True).to_json(orient='records'))
            except Exception as exc:
                event['leaderboard_error'] = type(exc).__name__
            return predictor
        except BaseException as exc:
            event.update(status=type(exc).__name__)
            raise
        finally:
            event['seconds'] = time.monotonic() - start
            persist()

    original_ag = solution.run_autogluon
    def autogluon(*args, **kwargs):
        start = time.monotonic()
        event = dict(budget=args[2], status='running', evaluated=False)
        meta['autogluon'].append(event)
        persist()
        try:
            result = original_ag(*args, **kwargs)
            event.update(status='complete', evaluated=bool(result is not None and
                result.get('score') is not None and np.isfinite(result['score'])),
                score=None if result is None else result.get('score'))
            if result is not None and result.get('test_pred') is not None:
                # An observation of AG alone, before the final blend/postprocessing.
                frame = pd.read_csv(task.parent / 'sample_submission.csv')
                frame[args[1]] = result['test_pred']
                frame.to_csv(task.parent / 'autogluon_submission.csv', index=False)
            return result
        except BaseException as exc:
            event.update(status=type(exc).__name__)
            raise
        finally:
            event['seconds'] = time.monotonic() - start
            persist()

    solution.run_autoeda, solution.fit_autogluon, solution.run_autogluon = research, fit, autogluon
    sys.argv = [str(Path(source) / 'solution.py'), str(task)]
    try:
        solution.main()
        meta.update(pipeline_complete=True, experiments=solution.EXPERIMENTS)
        persist()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        meta['error'] = type(exc).__name__
        persist()
    finally:
        shutdown(0 if meta['pipeline_complete'] else 1, task.parent)


def score_csv(path, truth, target, metric):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, mean_squared_error
    pred = pd.read_csv(path)
    if list(pred.columns) != ['id', target] or not np.array_equal(pred.id, truth.id):
        raise ValueError('submission contract mismatch')
    score = (float(np.sqrt(mean_squared_error(truth[target], pred[target]))) if metric == 'rmse'
             else float(f1_score(truth[target], pred[target], average='macro')))
    if not np.isfinite(score):
        raise ValueError('non-finite metric')
    return score


def run_one(args, name, seed, mode, credentials):
    import psutil
    root = args.output / name / str(seed) / mode
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite run artifacts: {root}')
    truth, metric, provenance, target, n_train, n_test = prepare_case(args, name, seed, root)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_', 'RAY_'))}
    ray_tmp = tempfile.mkdtemp(prefix='alfa-ray-', dir='/tmp')
    env.update(credentials)
    env.update(AGENT_N_CPUS='2', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               AGENT_RESEARCH_MODE=mode, AGENT_RESEARCH_SHARE='0.3',
               AGENT_TIME_LIMIT_SEC=str(args.total_seconds), AGENT_AUTOEDA_SEC='600',
               AGENT_SAFETY_SEC='60', AGENT_HARD_MARGIN_SEC='15',
               AGENT_PREDICT_RESERVE_SEC='120', AGENT_KFOLD_RESERVE_SEC='600',
               AGENT_LLM_MAX_TOKENS='8192', AGENT_LLM_REQUEST_SEC='90', AGENT_LLM_REASONING_TOKENS='1024',
               RAY_TMPDIR=ray_tmp, MPLCONFIGDIR=str(root / 'mpl'))
    peak, started, children = 0, time.monotonic(), {}
    with (root/'log.txt').open('w') as stream:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                                 str(args.source), str(root/'task.md')], cwd=root, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
        monitor = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                descendants = monitor.children(recursive=True)
                children.update({p.pid:p for p in descendants})
                rss = monitor.memory_info().rss
                for child in descendants:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - started > args.total_seconds + 30:
                proc.kill()
                proc.wait()
                break
            time.sleep(.5)
    for child in children.values():
        try:
            if child.is_running():
                child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(list(children.values()), timeout=3)
    shutil.rmtree(ray_tmp, ignore_errors=True)
    record = dict(task=name, seed=seed, mode=mode, metric=metric, target=target, provenance=provenance,
                  total_budget_seconds=args.total_seconds, seconds=time.monotonic()-started,
                  peak_tree_rss_mib=peak/2**20, exit_code=proc.returncode,
                  train_rows=n_train, holdout_rows=n_test, success=False)
    telemetry = json.loads((root/'telemetry.json').read_text()) if (root/'telemetry.json').exists() else {}
    record['telemetry'] = telemetry
    record['autogluon_evaluated'] = any(e.get('evaluated') for e in telemetry.get('autogluon', []))
    try:
        record['score'] = score_csv(root/'submission.csv', truth, target, metric)
        record['submission_valid'] = True
        record['success'] = proc.returncode == 0 and telemetry.get('pipeline_complete', False)
    except Exception as exc:
        record['error'] = type(exc).__name__
    if (root/'autogluon_submission.csv').exists():
        try:
            record['autogluon_standalone_score'] = score_csv(root/'autogluon_submission.csv', truth, target, metric)
        except Exception as exc:
            record['autogluon_submission_error'] = type(exc).__name__
    log = (root/'log.txt').read_text()
    record.update(llm_responses=log.count(' usage: '),
                  llm_cost_rub=sum(float(x) for x in re.findall(r"'cost_rub': ([0-9.eE+-]+)", log)),
                  llm_timeouts=len(re.findall(r'LLM\[[^\]]+\].*(?:Request timed out|нет ответа за)', log)),
                  final_source=(re.findall(r'Финальный источник: ([^\n]+)', log) or [None])[-1])
    record['input_sha256'] = {n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                              for n in ['train.csv','test.csv','sample_submission.csv','task.md','data_dictionary.csv']}
    save_json(root/'result.json', record)
    return record


def main():
    if sys.argv[1:2] == ['--worker']:
        return worker(sys.argv[2], sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('solution'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, default=Path('data'))
    parser.add_argument('--tasks', nargs='+', default=['public_regression','public_classification'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[17,29])
    parser.add_argument('--modes', nargs='+', choices=['legacy','iterative'], default=['legacy','iterative'])
    parser.add_argument('--total-seconds', type=int, default=3000)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    args.source, args.output, args.public_root = args.source.resolve(), args.output.resolve(), args.public_root.resolve()
    args.max_train_rows = args.max_test_rows = 100000
    if args.output.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output}')
    args.output.mkdir(parents=True)
    credentials = {k:v for k,v in json.loads(args.credentials_file.read_text()).items()
                   if k in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL']}
    if not credentials.get('OPENAI_API_KEY'):
        raise ValueError('LLM credentials are required for this comparison')
    protocol = {k:v for k,v in vars(args).items() if k != 'credentials_file'}
    protocol.update(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source.glob('*.py')},
        harness_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__),Path(__file__).with_name('research_curve.py'),Path(__file__).with_name('compare.py')]},
        python=platform.python_version(), system=platform.system(), machine=platform.machine(),
        packages={n:version(n) for n in ['autogluon.tabular','numpy','pandas','scikit-learn','lightgbm',
                                      'catboost','xgboost','torch','fastai','ray','pyarrow','scipy','psutil','openai']},
        cpus_per_worker=2, research_cap_seconds=600, research_share=.3,
        model=credentials.get('OPENAI_MODEL'), llm_max_tokens=8192, llm_request_cap_seconds=90,
        llm_reasoning_tokens=1024, safety_seconds=60, predict_reserve_seconds=120, kfold_reserve_seconds=600,
        note='Production main; observers do not change decisions. Both modes use research share .3 to reserve AG time. '
             'Unused search time is available to AG. Tree RSS sums parent/children and may count shared pages twice. '
             'Standalone AG CSV is observational, before final blend/postprocessing; not a separate equal-budget control. '
             'Local held-out train labels, not official hidden test. Regression seeds reorder the same year cohorts.')
    save_json(args.output/'protocol.json', protocol)
    records = []
    with ThreadPoolExecutor(args.jobs) as pool, (args.output/'results.jsonl').open('w') as stream:
        futures = [pool.submit(run_one,args,task,seed,mode,credentials)
                   for seed in args.seeds for task in args.tasks for mode in args.modes]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
            stream.flush()
            save_json(args.output/'results.json', records)
            print(json.dumps({k:v for k,v in record.items() if k not in ('telemetry','input_sha256')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
