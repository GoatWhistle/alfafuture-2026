"""Validate and summarize a finished full_cascade.py experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import pandas as pd


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


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
        journals = list((folder/'research').glob('*.json')) if (folder/'research').exists() else []
        if journals:
            r['research_journal'] = json.loads(journals[0].read_text())
    pairs, summary = [], []
    for task in protocol['tasks']:
        rows = [r for r in runs if r['task']==task]
        for seed in protocol['seeds']:
            modes = {r['mode']:r for r in rows if r['seed']==seed}
            a,b = modes['legacy'],modes['iterative']
            assert a['input_sha256']==b['input_sha256'], (task,seed,'unequal inputs')
            eligible = a['full_cascade_valid'] and b['full_cascade_valid']
            direction = -1 if a['metric']=='rmse' else 1
            gain = direction*(b['score']-a['score'])/abs(a['score'])*100 if eligible else None
            pairs.append(dict(task=task,seed=seed,metric=a['metric'],eligible=eligible,
                              legacy=a.get('score'),iterative=b.get('score'),improvement_pct=gain))
        paired_seeds = {p['seed'] for p in pairs if p['task']==task and p['eligible']}
        stats = {}
        for mode in protocol['modes']:
            subset = [r for r in rows if r['mode']==mode]
            stats[mode] = dict(attempts=len(subset),
                pipeline_success=sum(r['success'] for r in subset),
                full_cascade_valid=sum(r['full_cascade_valid'] for r in subset),
                score=mean(r['score'] for r in subset if r['seed'] in paired_seeds),
                seconds=mean(r['seconds'] for r in subset),
                peak_tree_rss_mib=mean(r['peak_tree_rss_mib'] for r in subset),
                research_seconds=mean(sum(e.get('seconds',0) for e in r['telemetry'].get('research',[])) for r in subset),
                autogluon_seconds=mean(sum(e.get('seconds',0) for e in r['telemetry'].get('autogluon',[])) for r in subset),
                llm_cost_rub=mean(r['llm_cost_rub'] for r in subset))
        summary.append(dict(task=task,metric=rows[0]['metric'],paired_seeds=sorted(paired_seeds),mean=stats))
    for key in ['source','output','public_root']:
        protocol.pop(key,None)
    return dict(protocol=protocol,runs=runs,pairs=pairs,summary=summary,
                successful_pipelines=sum(r['success'] for r in runs),
                full_cascade_valid=sum(r['full_cascade_valid'] for r in runs),
                returned_usage_cost_rub=sum(r['llm_cost_rub'] for r in runs),
                limitations=[
                    'Local holdout from public train, not official hidden test.',
                    'Regression repeats use the same time cohorts; classification uses two different random splits.',
                    f"Both modes have a {protocol['total_seconds']}-second overall cap and the same .3 research share / 600-second research cap.",
                    'Actual AG time depends on earlier stages; unused time is not padded.',
                    'Standalone AG scores precede final blending/postprocessing; not a separate equal-budget control.',
                    'Previous no-AG research harness had a different final selection pipeline and budget.',
                    'CPU macOS environment; memory is aggregate process-tree RSS, possibly double-counting shared pages.',
                    'Reported API cost comes from returned usage and may omit charged timed-out requests.'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['summary','pairs','successful_pipelines','full_cascade_valid','returned_usage_cost_rub']},ensure_ascii=False,indent=2))
