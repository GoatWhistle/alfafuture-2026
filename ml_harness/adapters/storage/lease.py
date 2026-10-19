"""Hold one nonblocking process lease for the lifetime of a CLI run."""

import fcntl
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from ml_harness.domain.models import ContractError


class RunLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "RunLease":
        if self._stream is not None:
            raise ContractError("Run lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise ContractError("Run directory is in use by another process") from None
        self._stream = stream
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream, fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None
