"""Dataset descriptions never learn from validation targets or feature values."""

from dataclasses import replace

import pandas as pd
import pytest
from test_prepare_task import task_fixture

from ml_harness.adapters.data.profile import column_details, dataset_profile
from ml_harness.adapters.data.task import CsvTaskData
from ml_harness.domain.models import ContractError


def test_profile_uses_train_partition_and_read_only_details(tmp_path):
    task = CsvTaskData().prepare(str(task_fixture(tmp_path)), 42)
    task = replace(
        task, protocol=replace(task.protocol, train_indices=(1, 2), validation_indices=(3, 4))
    )
    profile = dataset_profile(task)
    assert profile["columns"]["x"]["quantiles"]["1.0"] == 2
    assert profile["columns"]["x"]["rows"] == 2
    assert profile["target"]["quantiles"]["1.0"] == pytest.approx(2 / 3)
    assert profile["columns"]["x"]["test_missing"] == 1
    assert column_details(profile, ["x"])["x"]["dictionary"]["dtype"] == "object"
    data = pd.read_csv(task.spec.train_path)
    data.loc[3:, ["x", "response"]] = 999999
    data.to_csv(task.spec.train_path, index=False)
    assert dataset_profile(task) == profile
    with pytest.raises(ContractError, match="Unknown"):
        column_details(profile, ["response"])
    with pytest.raises(ContractError, match="eight"):
        column_details(profile, ["x"] * 9)
