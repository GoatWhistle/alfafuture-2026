"""Запуск агента с зависшим этапом: python hang_runner.py task.md <stage> <pidfile>.

stage=fit — k-fold LightGBM висит и держит дочерний процесс;
stage=autogluon — AutoGluon висит (как зависший predict).
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solution  # noqa: E402
from agent_utils import shutdown  # noqa: E402

task, stage, pidfile = sys.argv[1], sys.argv[2], sys.argv[3]


def hang(*args, **kwargs):
    child = subprocess.Popen(["sleep", "1000"])
    Path(pidfile).write_text(str(child.pid))
    while True:
        time.sleep(1)


if stage == "fit":
    solution.quick_oof = hang
elif stage == "autogluon":
    solution.run_autogluon = hang
sys.argv = ["solution.py", task]
solution.main()
shutdown(0, Path(task).parent)
