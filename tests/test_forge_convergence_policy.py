"""Restored accuracy/endurance failures must block qualification, not become held results."""
from pathlib import Path

import pytest

from experiments.forge.views import load_view, qualify
from test_forge_views import receipt, simple_task, simple_view

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('discriminator_stability', 'gaussian1d_stability'),
    ('discriminator_stability', 'grid100'),
    ('quality_coverage', 'rotated100'),
    ('quality_coverage', 'staggered100'),
    ('clockfree_continuous', 'grid100_14k'),
    ('clockfree_continuous', 'rotated100_14k'),
    ('clockfree_continuous', 'staggered100_14k'),
    ('adaptation', 'target_shift_recovery'),
    ('clockfree_continuous', 'clockfree_audit'),
]


@pytest.mark.parametrize('view_id,task_id', CASES)
def test_restored_gate_failure_blocks_an_otherwise_passing_candidate(view_id, task_id):
    assignment = next(a for a in load_view(ROOT, view_id)['assignments'] if a['task'] == task_id)
    tier = assignment['qualification_tier']
    assignments = [('smoke', 1, 'required')]
    if tier == 3:
        assignments.append(('quality', 2, 'required'))
    assignments.append((task_id, tier, assignment['importance']))
    view = simple_view(*assignments)
    tasks = {name: simple_task(name) for name, _, _ in assignments}
    passing = [receipt(tasks[name]) for name in tasks]
    assert qualify(view, tasks, passing)['status'] == 'PASS'
    failing = [receipt(tasks[name], bad_steps=range(1, 25), final=2.) if name == task_id
               else receipt(tasks[name]) for name in tasks]
    result = qualify(view, tasks, failing)
    assert result['status'] == 'FAIL'
    assert result['qualified_tier'] == tier - 1
