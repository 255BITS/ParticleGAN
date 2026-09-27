"""Catch changed test conditions in the comparison; the harness never owns an optimizer."""
import ast
from pathlib import Path

from benchmarks.locked_shared.recorded_recipes import GAN_V1, GAN_V2

from benchmarks.transfer_suite.compare_defaults import effective_spec, plan
from benchmarks.transfer_suite.formulations import axes

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks/transfer_suite"
# Vector/image hosts (vector_tasks, image_tasks, image_solvability) belong to
# their own migrations; everything else here is harness or D-research code.
HOST_OWNED = {"vector_tasks.py", "image_tasks.py", "image_solvability.py", "stress_tasks.py"}


def test_both_defaults_keep_all_nineteen_targets_architectures_and_resources():
    jobs = plan()
    assert len(jobs) == 19
    assert sum(j['spec']['runner'] == 'legacy' for j in jobs) == 9
    for job in jobs:
        old, new = [effective_spec(job['spec'], recipe) for recipe in (GAN_V1, GAN_V2)]
        if old['runner'] == 'legacy':
            assert old == new
        else:
            a, b = axes(old, old['runner']), axes(new, new['runner'])
            for key in ('target', 'architecture', 'resources'):
                assert a[key] == b[key]


def _violations(path):
    """Optimizer construction, LR writes and schedule formulas in one source file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and node.attr in {"Adam", "FixedControl", "control_host_schedules"}:
            found.append(node.attr)
        if isinstance(node, ast.Name) and node.id in {"learning_rate_scale", "policy_multipliers",
                                                      "schedule_optimizer"}:
            found.append(node.id)
        if isinstance(node, ast.alias) and node.name in {"learning_rate_scale", "policy_multipliers",
                                                         "schedule_optimizer", "FixedControl"}:
            found.append(node.name)
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "lr"):
                    found.append("group['lr'] =")
    return found


def test_harness_builds_no_optimizer_and_writes_no_learning_rate():
    paths = [*sorted(HARNESS.glob("*.py")), HARNESS.parent / "toy_suite.py"]
    offenders = {path.name: _violations(path) for path in paths if path.name not in HOST_OWNED}
    assert {name: found for name, found in offenders.items() if found} == {}
