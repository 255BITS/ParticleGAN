"""Reject request-level seed-sweep bypasses before any queue mutation."""
from copy import deepcopy

import pytest

from experiments.forge.api import CapabilityError, FormulationContext
from experiments.forge.promotion import validate_screening_submission
from experiments.forge.queue import Queue
from test_forge_queue import request, campaign


def test_fixed_screening_request_is_idempotent_without_running(tmp_path):
    req = request(tmp_path)
    validate_screening_submission(req)
    q = Queue(tmp_path / "queue")
    a = q.submit(req, campaign())
    b = q.submit(deepcopy(req), campaign())
    assert a == b and len(q.inspect()["submissions"]) == 1


@pytest.mark.parametrize("bypass", ["protocol", "rng", "binding", "job_seed", "job_protocol", "job_rng",
                                    "candidate_seed", "top_seed", "task_seed", "seed_only",
                                    "missing_protocol", "missing_rng", "missing_science", "marker"])
def test_every_screening_seed_override_is_rejected_before_queue_creation(tmp_path, bypass):
    req = request(tmp_path)
    if bypass == "protocol":
        req["protocol"]["seed"] = 1
    elif bypass == "rng":
        req["rng"] = FormulationContext(seed=1).streams.manifest()
    elif bypass == "binding":
        key = next(iter(req["rng"]["bindings"]))
        req["rng"]["bindings"][key]["seed"] += 1
    elif bypass == "job_seed":
        req["jobs"][0]["science"]["seed"] = 1
    elif bypass == "job_protocol":
        req["jobs"][0]["science"]["protocol"] = {"id": "screening", "seed": 1}
    elif bypass == "job_rng":
        req["jobs"][0]["science"]["rng"] = FormulationContext(seed=1).streams.manifest()
    elif bypass == "candidate_seed":
        req["candidate"]["seed"] = 1
    elif bypass == "top_seed":
        req["seed"] = 1
    elif bypass == "task_seed":
        req["tasks"]["t1"]["execution"] = {"seed": 1}
    elif bypass == "seed_only":
        req["candidate"]["changed_factors"] = ["seed"]
    elif bypass == "missing_protocol":
        req.pop("protocol")
    elif bypass == "missing_rng":
        req.pop("rng")
    elif bypass == "missing_science":
        req["jobs"][0].pop("science")
    else:
        req["protocol"]["promotion"] = {"registration_id": "invented"}
    q = Queue(tmp_path / "rejected-queue")
    with pytest.raises((CapabilityError, ValueError)):
        q.submit(req, campaign())
    assert not (q.root / "queue/state.json").exists()


def test_bare_promotion_stamp_cannot_authorize_seed_variation(tmp_path):
    req = request(tmp_path)
    req["protocol"]["seed"] = 1729
    req["promotion"] = {"registration_id": "invented", "seed": 1729, "subject": "a"}
    q = Queue(tmp_path / "rejected-queue", report_root=tmp_path / "reports/forge")
    with pytest.raises((CapabilityError, ValueError), match="registration"):
        q.submit(req, campaign())
    assert not (q.root / "queue/state.json").exists()
