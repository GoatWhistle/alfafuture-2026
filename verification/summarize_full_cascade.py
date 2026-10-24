"""Validate and summarize a finished full_cascade.py experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np
import pandas as pd


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def stage_seconds(record, stage):
    events = record['telemetry'].get(stage, [])
    if any('seconds' not in event for event in events):
        return None
    return sum(event['seconds'] for event in events)


def complete_mean(values):
    values = list(values)
    return None if any(value is None for value in values) else mean(values)


def recover_lightgbm(root, folder, record, train):
    """Post-hoc diagnosis only: invert a regression blend when clipping lost no information."""
    if record['metric'] != 'rmse' or not (record.get('final_source') or '').startswith('blend '):
        return None
    blends = [e for e in record['telemetry'].get('experiments', []) if e['name']=='blend']
    if len(blends)!=1:
        return None
    match = re.fullmatch(r'смесь autogluon и quick_kfold, alpha=([0-9.]+)', blends[0]['hypothesis'])
    if not match:
        return None
    alpha = float(match[1])
    if not 0 < alpha < 1 or not np.isclose(alpha * 10, round(alpha * 10), rtol=0, atol=1e-12):
        return None
    final = pd.read_csv(folder/'submission.csv')
    ag = pd.read_csv(folder/'autogluon_submission.csv')
    target = record['target']
    values = final[target].to_numpy(float)
    lo, hi = train[target].min(), train[target].max()
    boundary = np.isclose(values, lo, rtol=0, atol=1e-12) | np.isclose(values, hi, rtol=0, atol=1e-12)
    if boundary.any():
        return dict(available=False,reason='clipping boundary rows prevent exact reconstruction')
    truth = pd.read_csv(root/'holdout_labels'/f"{record['task']}-{record['seed']}.csv")
    assert np.array_equal(final.id,ag.id) and np.array_equal(final.id,truth.id)
    component = (values-alpha*ag[target].to_numpy(float))/(1-alpha)
    return dict(available=True,alpha_autogluon=alpha,
                rmse=float(np.sqrt(np.mean((truth[target].to_numpy(float)-component)**2))),
                method='Invert final = alpha*AG + (1-alpha)*LightGBM; no test predictions at clipping boundaries. '
                       'Alpha is exact on the production 0.1 grid. Regression has no subsequent label/weight postprocessing. '
                       'This is the component within the same completed run, not a separately trained control.')


def summarize(root):
    protocol = json.loads((root/'protocol.json').read_text())
    runs = [json.loads(line) for line in (root/'results.jsonl').read_text().splitlines()]
    expected = len(protocol['tasks']) * len(protocol['seeds']) * len(protocol['modes'])
    if len(runs) != expected:
        raise ValueError(f'Experiment incomplete: {len(runs)}/{expected} runs')
    identities = {(r['task'],r['seed'],r['mode']) for r in runs}
    assert len(identities) == expected, 'duplicate runs'
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((Path(protocol['source'])/name).read_bytes()).hexdigest() == digest, name
    for name,digest in protocol['harness_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == digest, name
    for r in runs:
        folder = root/r['task']/str(r['seed'])/r['mode']
        for name,digest in r['input_sha256'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
        train, test = pd.read_csv(folder/'train.csv'), pd.read_csv(folder/'test.csv')
        assert not set(train.id) & set(test.id), 'train/holdout overlap'
        assert r['target'] not in test, 'holdout labels exposed'
        r['full_cascade_valid'] = bool(r['success'] and r['autogluon_evaluated'] and
                                        'autogluon_standalone_score' in r)
        csv = folder/'submission.csv'
        r['submission_sha256'] = hashlib.sha256(csv.read_bytes()).hexdigest() if csv.exists() else None
        log = (folder/'log.txt').read_text(errors='replace')
        r['ray_worker_error_messages'] = log.count('The worker died unexpectedly')
        r['watchdog_triggered'] = 'Жёсткий дедлайн: останавливаю работу' in log
        for stage in ['research', 'autogluon']:
            r[f'{stage}_seconds_recorded'] = stage_seconds(r, stage)
        r['selection_validation'] = [dict(candidate=name, metric=r['metric'],
                                          score=float(value), rows=int(count))
            for name,value,count in re.findall(
                rf'\] ([\w]+): {re.escape(r["metric"])}=([-+0-9.eE]+) на (\d+) общих строках', log)]
        journals = list((folder/'research').glob('*.json')) if (folder/'research').exists() else []
        if journals:
            r['research_journal'] = json.loads(journals[0].read_text())
        recovered = recover_lightgbm(root,folder,r,train)
        if recovered is not None:
            r['reconstructed_lightgbm_component'] = recovered
    pairs, summary = [], []
    for task in protocol['tasks']:
        rows = [r for r in runs if r['task']==task]
        for seed in protocol['seeds']:
            modes = {r['mode']:r for r in rows if r['seed']==seed}
            a,b = modes['legacy'],modes['iterative']
            assert a['input_sha256']==b['input_sha256'], (task,seed,'unequal inputs')
            eligible = a['full_cascade_valid'] and b['full_cascade_valid']
            delivered = a.get('submission_valid', False) and b.get('submission_valid', False)
            direction = -1 if a['metric']=='rmse' else 1
            gain = direction*(b['score']-a['score'])/abs(a['score'])*100 if eligible else None
            pairs.append(dict(task=task,seed=seed,metric=a['metric'],eligible=eligible,
                              legacy=a.get('score'),iterative=b.get('score'),improvement_pct=gain,
                              iterative_minus_legacy=b['score']-a['score'] if eligible else None,
                              submission_pair_valid=bool(delivered),
                              delivered_improvement_pct=direction*(b['score']-a['score'])/abs(a['score'])*100
                                  if delivered else None))
        paired_seeds = {p['seed'] for p in pairs if p['task']==task and p['eligible']}
        delivered_seeds = {p['seed'] for p in pairs if p['task']==task and p['submission_pair_valid']}
        stats = {}
        for mode in protocol['modes']:
            subset = [r for r in rows if r['mode']==mode]
            stats[mode] = dict(attempts=len(subset),
                pipeline_success=sum(r['success'] for r in subset),
                full_cascade_valid=sum(r['full_cascade_valid'] for r in subset),
                valid_submissions=sum(bool(r.get('submission_valid')) for r in subset),
                watchdog_stops=sum(r['watchdog_triggered'] for r in subset),
                score=mean(r['score'] for r in subset if r['seed'] in paired_seeds),
                delivered_score=mean(r['score'] for r in subset if r['seed'] in delivered_seeds),
                seconds=mean(r['seconds'] for r in subset),
                peak_tree_rss_mib=mean(r['peak_tree_rss_mib'] for r in subset),
                research_seconds=complete_mean(r['research_seconds_recorded'] for r in subset),
                autogluon_seconds=complete_mean(r['autogluon_seconds_recorded'] for r in subset),
                research_timing_observations=sum(r['research_seconds_recorded'] is not None for r in subset),
                autogluon_timing_observations=sum(r['autogluon_seconds_recorded'] is not None for r in subset),
                llm_cost_rub=mean(r['llm_cost_rub'] for r in subset))
        summary.append(dict(task=task,metric=rows[0]['metric'],paired_seeds=sorted(paired_seeds),
                            delivered_paired_seeds=sorted(delivered_seeds),mean=stats))
    for key in ['source','output','public_root']:
        protocol.pop(key,None)
    return dict(protocol=protocol,runs=runs,pairs=pairs,summary=summary,
                successful_pipelines=sum(r['success'] for r in runs),
                full_cascade_valid=sum(r['full_cascade_valid'] for r in runs),
                valid_submissions=sum(bool(r.get('submission_valid')) for r in runs),
                watchdog_stops=sum(r['watchdog_triggered'] for r in runs),
                returned_usage_cost_rub=sum(r['llm_cost_rub'] for r in runs),
                limitations=[
                    'Local holdout from public train, not official hidden test.',
                    'Regression repeats use the same time cohorts; classification uses two different random splits.',
                    f"Both modes have a {protocol['total_seconds']}-second overall cap and the same .3 research share / 600-second research cap.",
                    'Actual AG time depends on earlier stages; unused time is not padded.',
                    'Standalone AG scores precede final blending/postprocessing; not a separate equal-budget control.',
                    'Previous no-AG research harness had a different final selection pipeline and budget.',
                    'CPU macOS environment; memory is aggregate process-tree RSS, possibly double-counting shared pages.',
                    'Two configured CPUs per worker are model/thread settings, not a hard OS quota.',
                    'Both modes use tabular[all], a wider optional backend set than the repository requirements.',
                    'Selection validation scores are parsed from logs and rounded to five decimals.',
                    'Delivered score includes valid fallback CSVs from incomplete pipelines; full-cascade score excludes them.',
                    'Stage timing is null if interrupted before telemetry persisted elapsed seconds; it is never imputed as zero.',
                    'Reported API cost comes from returned usage and may omit charged timed-out requests.'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['summary','pairs','successful_pipelines','full_cascade_valid','returned_usage_cost_rub']},ensure_ascii=False,indent=2))
