"""Общие утилиты агента: логирование, бюджет времени и жёсткий дедлайн."""

import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

START = time.time()

# Полный лимит платформы — 3 часа. Резерв на предикт/запись — отдельно.
TOTAL_BUDGET_SEC = float(os.environ.get("AGENT_TIME_LIMIT_SEC", 3 * 3600))
# Жёсткий запас: к этому моменту submission.csv уже должен лежать финальный.
SAFETY_SEC = float(os.environ.get("AGENT_SAFETY_SEC", 600))
# Сторожевой поток завершает процесс за столько секунд до лимита, что бы ни висело.
HARD_MARGIN_SEC = float(os.environ.get("AGENT_HARD_MARGIN_SEC", 120))


def available_cpus() -> int:
    """CPU, доступные процессу: AGENT_N_CPUS, иначе минимум из affinity и квоты cgroup.

    В контейнере os.cpu_count() возвращает ядра хоста, а лимит платформы
    (8 CPU) задан квотой cgroup — без учёта квоты потоков слишком много.
    """
    if os.environ.get("AGENT_N_CPUS"):
        return max(1, int(os.environ["AGENT_N_CPUS"]))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 8
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),
    ):
        try:
            if parse is not None:
                quota, period = parse(Path(path).read_text())
            else:
                quota = Path(path).read_text().strip()
                period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(int(quota) / int(period))))
            break
        except (OSError, ValueError):
            continue
    return n


def log(msg: str) -> None:
    print(f"[agent +{time.time() - START:8.1f}s] {msg}", flush=True)


def elapsed() -> float:
    return time.time() - START


def time_left() -> float:
    """Сколько секунд осталось до дедлайна с учётом запаса."""
    return TOTAL_BUDGET_SEC - SAFETY_SEC - elapsed()


def deadline() -> float:
    """Абсолютное время (time.time()), к которому планируемая работа должна закончиться."""
    return START + TOTAL_BUDGET_SEC - SAFETY_SEC


def hard_deadline() -> float:
    return START + TOTAL_BUDGET_SEC - HARD_MARGIN_SEC


def kill_children() -> None:
    """Убивает все дочерние процессы (воркеры AutoGluon/joblib и т.п.)."""
    try:
        import psutil

        procs = psutil.Process().children(recursive=True)
        for p in procs:
            try:
                p.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(procs, timeout=5)
    except Exception as exc:  # noqa: BLE001
        log(f"kill_children: {exc}")


def shutdown(code: int = 0, workdir: Path | None = None) -> None:
    """Немедленное завершение: дочерние процессы и фоновые потоки не удерживают процесс.

    submission.csv пишется атомарно, поэтому на диске остаётся последний валидный файл.
    """
    kill_children()
    try:
        import models

        models.cleanup_autogluon()
    except Exception:  # noqa: BLE001
        pass
    if workdir is not None:
        for tmp in Path(workdir).glob(".submission.*.tmp"):
            tmp.unlink(missing_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def start_watchdog(workdir: Path | None = None) -> threading.Thread:
    """Сторожевой поток: на hard_deadline() завершает процесс с кодом 0."""

    def run():
        while True:
            left = hard_deadline() - time.time()
            if left <= 0:
                break
            time.sleep(min(left, 5.0))
        log("Жёсткий дедлайн: останавливаю работу, остаётся последний валидный submission.csv")
        shutdown(0, workdir)

    t = threading.Thread(target=run, name="watchdog", daemon=True)
    t.start()
    return t


def run_with_timeout(fn, timeout: float, *args, **kwargs):
    """fn(*args) в daemon-потоке: (True, результат) или (False, None) по таймауту.

    Исключение fn пробрасывается. Брошенный по таймауту поток не мешает
    завершению процесса; fn не должна менять общие объекты.
    """
    try:
        return True, Background(fn, *args, **kwargs).result(timeout)
    except TimeoutError:
        return False, None


class Background:
    """Задача в daemon-потоке: не удерживает процесс при завершении (в отличие от ThreadPoolExecutor)."""

    def __init__(self, fn, *args, **kwargs):
        self._box: dict = {}

        def run():
            try:
                self._box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                self._box["error"] = exc

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def result(self, timeout: float):
        self._thread.join(max(0.0, timeout))
        if self._thread.is_alive():
            raise TimeoutError
        if "error" in self._box:
            raise self._box["error"]
        return self._box.get("result")


class StageTimeout(BaseException):
    """Этап превысил лимит. BaseException: широкие `except Exception` этапа его не глотают."""


@contextmanager
def time_limit(seconds: float):
    """Прерывает блок в главном потоке через SIGALRM по истечении seconds.

    Тяжёлые этапы остаются в главном потоке (LightGBM в рабочем потоке в
    разы медленнее из-за пулов OpenMP). Сигнал обрабатывается между
    шагами Python — в т.ч. на callback каждой итерации LightGBM; долгий
    вызов C без возврата в Python прервёт только сторожевой поток.
    """
    if threading.current_thread() is not threading.main_thread() or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise StageTimeout

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
