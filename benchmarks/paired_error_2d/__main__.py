"""``python -m benchmarks.paired_error_2d``: every (map, cloud) problem on the shared runner.

Each problem logs one JSON line per observation to ``runs/toy-refactor/<problem name>.log``.
"""
from benchmarks.toy_runner import main
from .task import CLOUDS, TASKS, PairedError2D

if __name__ == "__main__":
    raise SystemExit(max([main(PairedError2D(task, cloud)) for task in TASKS for cloud in CLOUDS]))
