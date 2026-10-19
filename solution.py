"""Competition entrypoint; wall-clock includes application import time."""

import time


def _entrypoint() -> int:
    started = time.monotonic()
    from ml_harness.entrypoints.cli import main

    return main(started_at=started)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
