"""External demo observation must preserve original updates and RNG streams."""
from pathlib import Path
import random
import runpy
from types import SimpleNamespace
from unittest.mock import patch
import sys
import json

import numpy as np
import pytest
import torch

from benchmarks.toy_audit import source_demos as demos
from benchmarks.toy_audit import source_demo_report as reports


ROOT = Path(__file__).resolve().parents[1]
CHARS = "abcdefghijklmnopqrstuvwxyz_ "
WORDS = ("apple", "grape", "lemon", "melon", "berry")


def test_progress_read_cannot_abort_the_training_supervisor(tmp_path):
    path = tmp_path / "progress.json"
    assert demos.read_progress(path) == {}
    path.write_text('{"completed_update_pairs":')
    assert demos.read_progress(path) == {"progress_read": "pending atomic receipt"}
    demos.write(path, {"completed_update_pairs": 38})
    assert demos.read_progress(path) == {"completed_update_pairs": 38}
    assert json.loads(path.read_text()) == {"completed_update_pairs": 38}


def test_passing_partial_frames_cannot_certify_a_missing_terminal_receipt(tmp_path):
    demos.write(tmp_path / "source-receipt.json", {"wall_cap_seconds": 120})
    demos.write(tmp_path / "raw/progress.json", {"completed_update_pairs": 1000})
    observations = [{"step": step, "seconds": step / 10,
                     "live": {"passed": True}, "ema": {"passed": True}}
                    for step in (900, 920, 940, 960, 980, 1000)]
    (tmp_path / "raw/observations.jsonl").write_text(
        "".join(json.dumps(shot) + "\n" for shot in observations))
    summary = reports.attempt_summary(tmp_path, "quickstart")
    assert summary["completed_update_pairs"] == 1000
    assert summary["live_terminal"]["passing_suffix"] == 6
    assert summary["measurement_complete"] is False
    assert reports.gate_status(summary, "live") == "INCOMPLETE"
    assert reports.gate_status(summary, "ema") == "INCOMPLETE"


def software_scoring():
    def probabilities(logits):
        values = np.exp(logits - logits.max(1, keepdims=True))
        return values / values.sum(1, keepdims=True)
    return SimpleNamespace(CHARS=CHARS, WORDS=WORDS, word_probabilities=probabilities,
                           five_word_metrics=lambda *_: {"passed": False},
                           gaussian_metrics=lambda *_: {"passed": False})


def checkpoint(modules, optimizers):
    return dict(modules=[m.state_dict() for m in modules],
                optimizers=[o.state_dict() for o in optimizers],
                torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                modes=[[c.training for c in m.modules()] for m in modules])


def test_word_sampling_uses_noisy_prior_law_and_preserves_caller_state():
    from particlegan import MoGParticlePrior
    torch.manual_seed(8)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(168, 2))
    generator = torch.nn.Sequential(torch.nn.Linear(2, 168), torch.nn.Unflatten(1, (28, 6)))
    prior = MoGParticlePrior(num_particles=5, z_dim=2, sigma=0.2)
    encoder.train(); generator.eval(); prior.train()
    canonical = torch.zeros(5, 28, 6)
    before = demos.state_hash(checkpoint((encoder, generator, prior), ()))
    generated, reconstructed, _ = demos.word_draw(encoder, generator, prior, canonical, count=64)
    after = demos.state_hash(checkpoint((encoder, generator, prior), ()))
    assert before == after
    with torch.no_grad():
        latent, _ = prior.sample(64, generator=torch.Generator().manual_seed(demos.EVALUATION_SEED))
        assert np.array_equal(generated, generator(latent).numpy())
        assert np.array_equal(reconstructed, generator(encoder(canonical)).numpy())
        assert len(generated) != len(prior.z)


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font")
def test_five_source_observer_preserves_original_model_optimizer_and_rng(tmp_path):
    from matplotlib import pyplot as plt
    torch.set_num_threads(1)
    module = runpy.run_path(str(ROOT / "examples/five_modes.py"))
    receipts = []
    for observed in (False, True):
        np.random.seed(5)
        refs = {}
        observer = demos.Observer("five_modes", tmp_path / "observed", software_scoring(), count=32) if observed else None
        with demos.five_observer(observer, module, baseline_refs=refs):
            prior, encoder, generator, critic = module["train"](
                total_steps=1, batch_size=8, viz_interval=1, frame_interval=1,
                log_interval=1, out_dir=str(tmp_path / ("plots-observed" if observed else "plots-baseline")))
        receipts.append(demos.state_hash(dict(
            state=checkpoint((prior, encoder, generator, critic), refs["optimizers"]),
            training_input_sha256=refs["training_inputs"])))
        assert len(refs["training_inputs"]) == 4
        plt.close("all")
        if observed:
            assert observer.completed == 2  # The original inclusive loop.
            assert [r["step"] for r in observer.rows] == [1, 2]
    assert receipts[0] == receipts[1]


def test_quick_observer_preserves_original_checkpoint_and_all_training_streams(tmp_path):
    from particlegan import GANTrainer, get_recipe, init
    checkpoints = []
    for observed in (False, True):
        torch.manual_seed(0)
        recipe = get_recipe(total_steps=2, num_particles=32, batch_size=16)
        generator = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.LeakyReLU(.2), torch.nn.Linear(8, 2))
        critic = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Softplus(), torch.nn.Linear(8, 1))
        init.deterministic_orthogonal_(generator, seed=0)
        init.deterministic_orthogonal_(critic, seed=1)
        prior = init.deterministic_orthogonal_(recipe.make_prior())
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0)
        data_rng = torch.Generator().manual_seed(0)
        observer = demos.Observer("quickstart", tmp_path / "observed", software_scoring(), count=32)
        def real():
            return .2 * torch.randn(16, 2, generator=data_rng) + 1
        if observed:
            with demos.quick_observer(observer, GANTrainer):
                for _ in range(2):
                    trainer.step(real(), generator_real=real)
        else:
            for _ in range(2):
                trainer.step(real(), generator_real=real)
        checkpoints.append(demos.state_hash(dict(trainer=trainer.state_dict(), data_rng=data_rng.get_state())))
    assert checkpoints[0] == checkpoints[1]
    assert observer.completed == 2
    assert [r["step"] for r in observer.rows] == [0, 1]


def test_cpu128_profile_preserves_the_same_unobserved_original_example(tmp_path):
    from particlegan import GANTrainer
    hashes = []
    for observed in (False, True):
        output = tmp_path / ("observed.pt" if observed else "baseline.pt")
        source = ROOT / "examples/quickstart_gan.py"
        argv = [str(source), "--stop-after", "2", "--output", str(output)]
        observer = demos.Observer("quickstart", tmp_path / "observer", software_scoring(), count=32)
        with demos.recipe_profile("cpu128"), patch.object(sys, "argv", argv):
            if observed:
                with demos.quick_observer(observer, GANTrainer):
                    runpy.run_path(str(source), run_name="__main__")
            else:
                runpy.run_path(str(source), run_name="__main__")
        checkpoint = torch.load(output, weights_only=True)
        assert checkpoint["trainer"]["recipe"]["batch_size"] == 128
        assert checkpoint["trainer"]["recipe"]["total_steps"] == 1000
        hashes.append(demos.state_hash(checkpoint))
    assert hashes[0] == hashes[1]
