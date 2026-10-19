"""Explicit conversion of frozen search protocols to final control evaluation."""

import hashlib
import json
from dataclasses import replace

from ml_harness.domain.models import ContractError, PreparedTask, ValidationProtocol


def control_task(task: PreparedTask) -> PreparedTask:
    """Train on development rows and predict the untouched control partition once."""
    original = task.protocol
    if not original.control_indices or original.level != "full":
        raise ContractError("Full protocol with a frozen control partition required")
    indices = original.train_indices + original.validation_indices + original.control_indices
    if any(index >= task.train_rows for index in indices):
        raise ContractError("Control protocol contains indices outside training data")
    payload = (original.fingerprint, "control", original.control_indices)
    fingerprint = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    protocol = ValidationProtocol(
        fingerprint,
        original.data_fingerprint,
        original.kind,
        original.seed,
        tuple(sorted(original.train_indices + original.validation_indices)),
        original.control_indices,
        level="full",
    )
    return replace(task, protocol=protocol)
