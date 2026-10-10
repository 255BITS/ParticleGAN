"""Archived CUDA checks of batch-only binding, data coupling and retention.

These checks execute at the original source identity, with verified original
archives. They grant no qualification or numerical parity claim to live code.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

from benchmarks.toy_audit import tier1_batch_size as study

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="batch study requires CUDA")


@pytest.fixture(scope="module")
def archived_batch_source(tmp_path_factory):
    if os.environ.get("PARTICLEGAN_ARCHIVED_BATCH_CHILD"):
        return None
    checkout = Path(__file__).resolve().parents[1]
    provenance = json.loads((checkout / "reports/forge/tier1-batch-size/provenance.json").read_text())
    revision = provenance["executed_commit"]
    override = os.environ.get("PARTICLEGAN_TIER1_ARCHIVE_DIR")
    archives = {}
    for name in ("tier1-prior-smoke", "tier1-prior-duration"):
        receipt = json.loads((checkout / f"reports/forge/{name}/artifact-provenance.json").read_text())
        record = receipt["archive"]
        path = Path(override) / Path(record["path"]).name if override else Path(record["local_path"])
        if not path.is_file():
            pytest.skip(f"unavailable original archive {path}; archived batch source {revision}")
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], f"original archive changed: {path}"
        archives[f"{name}-v1"] = data
    paths = ["particlegan", "benchmarks", "experiments", "configs", "lib",
             "reports/transfer_suite/unadjusted/leading_profile.json",
             "reports/forge/tier1-batch-size/protocol.json"]
    snapshot = subprocess.run(["git", "archive", revision, *paths], cwd=checkout, capture_output=True)
    if snapshot.returncode:
        pytest.skip(f"unavailable original batch source {revision}: {snapshot.stderr.decode().strip()}")
    root = tmp_path_factory.mktemp("archived-batch-source")
    with tarfile.open(fileobj=io.BytesIO(snapshot.stdout)) as archive:
        archive.extractall(root, filter="data")
    source = root / provenance["training_source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["training_source"]["sha256"]
    protocol_path = root / "reports/forge/tier1-batch-size/protocol.json"
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == provenance["protocol_sha256"]
    protocol = json.loads(protocol_path.read_text())
    for name, data in archives.items():
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for relative, expected in protocol["inputs"].items():
                if not relative.startswith(f"runs/api/{name}/"):
                    continue
                content = archive.extractfile(relative.removeprefix("runs/api/")).read()
                assert hashlib.sha256(content).hexdigest() == expected, f"archived input changed: {relative}"
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    for relative, expected in {**protocol["inputs"], **protocol["scientific_implementation"]}.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    test_path = root / "tests/test_tier1_batch_size.py"
    test_path.parent.mkdir()
    test_path.write_bytes(Path(__file__).read_bytes())
    return root


def run_archived_check(root, request):
    if root is None:
        return False
    environment = dict(os.environ, PYTHONPATH=str(root), PARTICLEGAN_ARCHIVED_BATCH_CHILD="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = root.parent / f"{request.node.name}.log"
    with log.open("w") as output:
        child = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                f"tests/test_tier1_batch_size.py::{request.node.name}"],
                               cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=60)
    assert child.returncode == 0, log.read_text()
    return True


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_batch_change_matches_archived_initial_models_and_streams(task_id, archived_batch_source, request):
    if run_archived_check(archived_batch_source, request):
        return
    protocol = study.declaration()
    for batch in (128, 512):
        context, trainer, _ = study.build(task_id, batch, "cuda:0")
        assert study.verify_initial(context, protocol["tasks"][task_id])["matched"]
        assert trainer.recipe.batch_size == batch
        assert trainer.recipe.total_steps == protocol["tasks"][task_id]["original_schedule_horizon"]
        assert trainer.recipe.lr_floor == trainer.recipe.network_lr_floor == 1.
        assert trainer.prior.z.device.type == "cuda"


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_larger_batch_preserves_flat_real_example_stream(task_id, archived_batch_source, request):
    if run_archived_check(archived_batch_source, request):
        return
    context, _, task = study.build(task_id, 512, "cuda:0")
    target, _ = study.scorer(task_id)
    a = context.streams.generator("data", component="target", purpose="training", device="cpu")
    b = torch.Generator(device="cpu"); b.set_state(a.get_state())
    large, _ = study.grouped_real(target, task["execution"]["host_definition"], 512, a, 0)
    small = torch.cat([study.grouped_real(target, task["execution"]["host_definition"], 128, b, i)[0]
                       for i in range(4)])
    assert torch.equal(large.to("cuda:0"), small.to("cuda:0"))
    assert torch.equal(a.get_state(), b.get_state())


def test_a_failed_hold_check_cannot_be_replaced_by_a_good_endpoint(archived_batch_source, request):
    if run_archived_check(archived_batch_source, request):
        return
    protocol = study.declaration()
    context, _, task = study.build("gaussian1d_acquisition", 512, "cuda:0")
    target, score = study.scorer("gaussian1d_acquisition")
    stream = context.streams.generator("data", component="target", purpose="oracle", device="cpu")
    metrics = score(target(task["execution"]["host_definition"], 4096, stream, 0).to("cuda:0"),
                    task["execution"]["host_definition"], 0)
    rows = [dict(step=s, full_pass=True, metrics=metrics) for s in study.checkpoints("gaussian1d_acquisition", protocol)]
    assert study.summarize(rows, "gaussian1d_acquisition", protocol)["combined_verdict"] == "PASS"
    bad = deepcopy(rows)
    bad[30]["full_pass"] = False
    summary = study.summarize(bad, "gaussian1d_acquisition", protocol)
    assert summary["acquisition_verdict"] == "PASS"
    assert summary["final_terminal_suffix"] >= 5
    assert summary["combined_verdict"] == summary["hold_verdict"] == "FAIL"
    with pytest.raises(ValueError, match="missing"):
        study.summarize(rows[:-1], "gaussian1d_acquisition", protocol)


def test_cpu_fallback_rejected_before_output_creation(tmp_path):
    path = tmp_path / "not-created"
    with pytest.raises(ValueError, match="requires CUDA"):
        study.execute(path, device="cpu")
    assert not path.exists()
