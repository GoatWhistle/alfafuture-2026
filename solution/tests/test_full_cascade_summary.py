"""Only reconstruct a component when the saved blend preserves its information."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'verification'))
from summarize_full_cascade import recover_lightgbm, summarize


@pytest.fixture
def blend(tmp_path):
    folder = tmp_path / 'run'
    folder.mkdir()
    (tmp_path / 'holdout_labels').mkdir()
    pd.DataFrame({'id': [9, 2], 'target': [3., 7.]}).to_csv(
        tmp_path / 'holdout_labels' / 'case-17.csv', index=False)
    pd.DataFrame({'id': [9, 2], 'target': [4., 8.]}).to_csv(
        folder / 'autogluon_submission.csv', index=False)
    # AG=[4,8], LightGBM=[2,6], alpha=.9 => blend=[3.8,7.8].
    pd.DataFrame({'id': [9, 2], 'target': [3.8, 7.8]}).to_csv(
        folder / 'submission.csv', index=False)
    record = dict(metric='rmse', final_source='blend (rmse=1.0)',
                  target='target', task='case', seed=17,
                  telemetry={'experiments': [dict(name='blend', hypothesis=
                      'смесь autogluon и quick_kfold, alpha=0.90')]})
    return tmp_path, folder, record, pd.DataFrame({'target': [0., 10.]})


def test_reconstruct_known_component(blend):
    result = recover_lightgbm(*blend)
    assert result['available']
    assert result['rmse'] == pytest.approx(1.)


def test_clipping_boundary_cannot_be_inverted(blend):
    root, folder, record, train = blend
    pd.DataFrame({'id': [9, 2], 'target': [0., 7.8]}).to_csv(
        folder / 'submission.csv', index=False)
    assert not recover_lightgbm(root, folder, record, train)['available']


def test_misaligned_saved_component_rejected(blend):
    root, folder, record, train = blend
    path = folder / 'autogluon_submission.csv'
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    with pytest.raises(AssertionError):
        recover_lightgbm(root, folder, record, train)


@pytest.mark.parametrize('alpha', ['0.00', '1.00', '0.93'])
def test_noninvertible_or_unknown_weight_rejected(blend, alpha):
    root, folder, record, train = blend
    record['telemetry']['experiments'][0]['hypothesis'] = (
        f'смесь autogluon и quick_kfold, alpha={alpha}')
    assert recover_lightgbm(root, folder, record, train) is None


@pytest.mark.parametrize('fallback_returns_normally', [True, False])
def test_fallback_score_is_reported_but_not_counted_as_completed_autogluon(tmp_path, fallback_returns_normally):
    protocol = dict(tasks=['case'], seeds=[17], modes=['legacy', 'iterative'],
                    total_seconds=3000, source_sha256={}, harness_sha256={})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    records = []
    for mode, complete, value in [('legacy', True, .8), ('iterative', False, .7)]:
        folder = tmp_path/'case'/'17'/mode
        folder.mkdir(parents=True)
        pd.DataFrame({'id':[1], 'target':['a']}).to_csv(folder/'train.csv', index=False)
        pd.DataFrame({'id':[2]}).to_csv(folder/'test.csv', index=False)
        pd.DataFrame({'id':[2], 'target':['a']}).to_csv(folder/'submission.csv', index=False)
        (folder/'log.txt').write_text('' if complete else 'Жёсткий дедлайн: останавливаю работу')
        record = dict(task='case', seed=17, mode=mode, metric='f1_macro', target='target',
                      success=complete, autogluon_evaluated=complete, submission_valid=True,
                      score=value, seconds=10, peak_tree_rss_mib=100, llm_cost_rub=0,
                      telemetry={}, input_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ['train.csv', 'test.csv']})
        if complete:
            record['autogluon_standalone_score'] = value
        else:
            record['success'] = fallback_returns_normally
        records.append(record)
    (tmp_path/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    result = summarize(tmp_path)
    assert result['valid_submissions'] == 2
    assert result['full_cascade_valid'] == 1
    assert result['successful_pipelines'] == 1 + int(fallback_returns_normally)
    assert result['watchdog_stops'] == 1
    pair = result['pairs'][0]
    assert not pair['eligible'] and pair['improvement_pct'] is None
    assert pair['submission_pair_valid']
    assert pair['delivered_improvement_pct'] == pytest.approx(-12.5)
    stats = result['summary'][0]
    assert stats['paired_seeds'] == []
    assert stats['delivered_paired_seeds'] == [17]
    for mode, value in [('legacy', .8), ('iterative', .7)]:
        assert stats['mean'][mode]['score'] is None
        assert stats['mean'][mode]['delivered_score'] == value
