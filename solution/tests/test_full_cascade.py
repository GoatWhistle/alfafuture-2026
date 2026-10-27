"""External evaluation rejects misaligned CSVs and uses the task's real metric."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'verification'))
from full_cascade import score_csv


def test_rmse_and_row_alignment(tmp_path):
    truth = pd.DataFrame({'id':[9,2], 'site_eui':[3.,7.]})
    pred = pd.DataFrame({'id':[9,2], 'site_eui':[0.,3.]})
    path = tmp_path/'submission.csv'
    pred.to_csv(path,index=False)
    assert score_csv(path,truth,'site_eui','rmse') == pytest.approx((25/2)**.5)
    pred.iloc[::-1].to_csv(path,index=False)
    with pytest.raises(ValueError,match='contract'):
        score_csv(path,truth,'site_eui','rmse')


def test_macro_f1_not_accuracy(tmp_path):
    truth = pd.DataFrame({'id':[1,2,3,4], 'attack_cat':['normal','normal','normal','rare']})
    path = tmp_path/'submission.csv'
    truth.assign(attack_cat='normal').to_csv(path,index=False)
    assert score_csv(path,truth,'attack_cat','f1_macro') == pytest.approx(3/7)


def test_nonfinite_regression_rejected(tmp_path):
    truth = pd.DataFrame({'id':[1,2], 'site_eui':[1.,2.]})
    path = tmp_path/'submission.csv'
    truth.assign(site_eui=[float('inf'),1.]).to_csv(path,index=False)
    with pytest.raises(ValueError):
        score_csv(path,truth,'site_eui','rmse')
