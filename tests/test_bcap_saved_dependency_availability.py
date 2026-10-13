"""Own-checkpoint reporting cannot borrow another arm's passing producer."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "reports/forge/bcap-three-phase/publication_adapter.py"
spec = importlib.util.spec_from_file_location("saved_dependency_availability", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture():
    tasks = dict(producer={}, hold=dict(dependencies=[dict(task="producer", kind="checkpoint")]), other={})
    cells = [dict(scope="research_diagnostic", role=role, task_id=task, gate_status=status)
             for role, task, status in [("baseline", "producer", "PASS"), ("baseline", "hold", "UNMEASURED"),
                                       ("candidate", "producer", "INCOMPLETE"), ("candidate", "hold", "UNMEASURED"),
                                       ("candidate", "other", "UNMEASURED")]]
    return dict(cells=cells, final=[dict(certified="immutable")], accounting=[dict(paid_seconds=900)],
                scopes=[dict(scope="research_diagnostic", submissions={
                    role: dict(request=dict(tasks=tasks)) for role in ("baseline", "candidate")})])


def test_failed_own_producer_cannot_use_baseline_pass():
    collection = fixture()
    original = deepcopy(collection)
    adapter.annotate(collection)
    assert collection["cells"][3]["gate_status"] == "BLOCKED"
    assert "INCOMPLETE" in collection["cells"][3]["reason"][0]
    assert collection["cells"][:3] == original["cells"][:3]
    assert collection["cells"][4] == original["cells"][4]
    assert collection["final"] == original["final"]
    assert collection["accounting"] == original["accounting"]


def test_measured_hold_verdict_and_foreign_scope_are_preserved():
    collection = fixture()
    collection["cells"][3]["gate_status"] = "FAIL"
    collection["cells"][2]["scope"] = "foreign"
    original = deepcopy(collection)
    adapter.annotate(collection)
    assert collection == original


def test_missing_own_producer_is_known_blocker_and_annotation_is_idempotent():
    collection = fixture()
    collection["cells"] = [cell for cell in collection["cells"]
                           if not (cell["role"] == "candidate" and cell["task_id"] == "producer")]
    adapter.annotate(collection)
    hold = next(cell for cell in collection["cells"] if cell["role"] == "candidate" and cell["task_id"] == "hold")
    assert hold["gate_status"] == "BLOCKED" and "UNMEASURED" in hold["reason"][0]
    once = deepcopy(collection)
    adapter.annotate(collection)
    assert collection == once


def retry_fixture():
    audit_path = path.with_name('audit_track.py')
    audit_spec = importlib.util.spec_from_file_location('saved_retry_authorization', audit_path)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    from experiments.forge.contracts import stable_hash
    previous = dict(attempt_id='previous', candidate_revision='frozen', raw=dict(attempt_status='timeout'),
                    task_results=[dict(gate_status='INCOMPLETE')])
    approval = dict(predecessor_attempt_id='previous', predecessor_result_hash=stable_hash(previous),
                    reason='User-authorized repaired execution environment')
    current = dict(attempt_id='current', candidate_revision='frozen', retry_of=dict(
        attempt_id='previous', result_hash=stable_hash(previous), reason=approval['reason']))
    state = dict(jobs=dict(job=dict(attempts=[dict(attempt_id='previous'), dict(attempt_id='current')])))
    collection = dict(attempts={item['attempt_id']: dict(result=item) for item in (previous, current)})
    return audit, state, collection, dict(retries=[approval])


def test_authorized_infrastructure_retry_preserves_predecessor_binding():
    audit, state, collection, approval = retry_fixture()
    assert audit.authorized_retries(state, collection, approval) == [dict(
        predecessor='previous', successor='current', reason=approval['retries'][0]['reason'])]
    with pytest.raises(ValueError, match='explicitly authorized'):
        audit.authorized_retries(state, collection, {})


def test_scientific_failure_or_changed_formulation_cannot_be_execution_retry():
    audit, state, collection, approval = retry_fixture()
    collection['attempts']['previous']['result']['raw']['attempt_status'] = 'completed'
    collection['attempts']['previous']['result']['task_results'][0]['gate_status'] = 'FAIL'
    with pytest.raises(ValueError, match='incomplete execution predecessor'):
        audit.authorized_retries(state, collection, approval)
    audit, state, collection, approval = retry_fixture()
    collection['attempts']['current']['result']['candidate_revision'] = 'changed'
    with pytest.raises(ValueError, match='incomplete execution predecessor'):
        audit.authorized_retries(state, collection, approval)
