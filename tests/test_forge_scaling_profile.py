"""Profiling must preserve the actual public trainer trajectory."""
import json

import pytest

from particlegan import get_recipe
from experiments.forge import scaling_profile
from experiments.forge.scaling_benchmark import run_benchmark


def test_component_wrappers_preserve_complete_public_update(tmp_path):
    spec = dict(image_size=32, batch_size=2, base_channels=2, max_channels=8,
                latent_dim=4, n_particles=8, warmup_updates=1, measured_updates=2,
                max_seconds=30.)
    recipe = get_recipe("bispf", batch_size=2, z_dim=4, num_particles=8,
                        total_steps=3, prior_kind="mog", standardize=False)
    plain = run_benchmark(recipe, spec, tmp_path / "plain", "cpu")
    output = tmp_path / "components"
    output.mkdir()
    spans = scaling_profile.ComponentSpans("cpu", output)
    with spans.attach():
        components = run_benchmark(recipe, spec, output, "cpu")
    assert plain["status"] == components["status"] == "PASS"
    for key in ("initial_model_sha256", "initial_parameter_sha256", "initialization_sha256",
                "batch_sequence_sha256", "final_parameter_sha256"):
        assert plain["identity"][key] == components["identity"][key]
    rows = [json.loads(line) for line in (output / "components.jsonl").read_text().splitlines()]
    names = {row["name"] for row in rows}
    assert {"complete_update", "critic.optimizer", "generator.optimizer", "critic.penalty",
            "critic.sinkhorn", "prior.regularizer"} <= names
    assert all(0 <= row["exclusive_ms"] <= row["inclusive_ms"] for row in rows)
    assert (output / "torch-trace.json").is_file()
    assert (output / "torch-table.txt").is_file()


def test_immutable_source_declaration_blocks_mutation_before_execution(tmp_path):
    root = tmp_path / "source"
    for directory in ("particlegan", "experiments/forge", "benchmarks/toy_audit",
                      "configs/forge/ideas", "configs/forge/tasks"):
        (root / directory).mkdir(parents=True)
    source = root / "particlegan/implementation.py"
    source.write_text("first\n")
    (root / "benchmarks/toy_audit/reproducibility.py").write_text("fixture\n")
    (root / "configs/forge/ideas/candidate.json").write_text("{}")
    (root / "configs/forge/tasks/task.json").write_text("{}")
    output = tmp_path / "artifacts"
    first = scaling_profile.declare(root, output, candidates=["candidate"], tasks=["task"], source_commit="frozen")
    assert first["budget"]["paid_retries"] == 0
    assert first == scaling_profile.declare(root, output, candidates=["candidate"], tasks=["task"], source_commit="frozen")
    source.write_text("second\n")
    with pytest.raises(ValueError, match="immutable"):
        scaling_profile.declare(root, output, candidates=["candidate"], tasks=["task"], source_commit="frozen")
    with pytest.raises(ValueError, match="frozen profiling source changed"):
        scaling_profile.run(root, output, candidate_id="candidate", task_id="task", device="cpu")
