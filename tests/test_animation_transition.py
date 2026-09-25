"""Sprite animation world model: simulator, dataset, models, evaluation, trainer and leaderboard."""
import json
import math

import numpy as np
import pytest
import torch

from lib import sprite_animation as sa
from lib.animation_transition import (RECORD_DIM, AnimationCritics, AnimationEncoder, AnimationGenerator,
                                      StateScaler, join, rollout)
from lib.animation_evaluation import HORIZONS, evaluate_split, persistence_predictor


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sprite_data")
    sa.make_dataset(out, scale=.01)
    return out


def starts(n=64, seed=0):
    return sa.sample_starts(n, (.2, .6), (0., 1.), (0., 2 * math.pi), np.random.default_rng(seed))


def test_step_deterministic_numpy_torch_and_bounded():
    s = starts()
    a, b = sa.simulate(s, 200), sa.simulate(s.copy(), 200)
    np.testing.assert_array_equal(a, b)
    t = sa.simulate(torch.as_tensor(s), 200).numpy()
    np.testing.assert_allclose(a, t, rtol=0, atol=1e-12)
    assert a[..., :2].min() >= sa.RADIUS and a[..., :2].max() <= 1 - sa.RADIUS
    np.testing.assert_allclose(a[..., 4] ** 2 + a[..., 5] ** 2, 1, atol=1e-9)
    energy = sa.GRAVITY * a[..., 1] + .5 * (a[..., 2] ** 2 + a[..., 3] ** 2)
    assert (energy[:, -1] <= energy[:, 0] + 1e-2).all()


def test_render_shape_range_and_notch_moves():
    s = torch.tensor([[.5, .5, 0, 0, 0., 1.], [.5, .5, 0, 0, 1., 0.]])
    frames = sa.render(s)
    assert frames.shape == (2, 1, sa.IMAGE_SIZE, sa.IMAGE_SIZE)
    assert frames.min() >= -1 and frames.max() <= 1 and frames.max() > .9
    assert (frames[0] - frames[1]).abs().sum() > 1


def test_make_dataset_splits(data_dir):
    meta = json.loads((data_dir / "metadata.json").read_text())
    for name in sa.SPLITS:
        states = sa.load_episodes(data_dir, name)
        assert states.shape[1:] == (sa.EPISODE_LENGTH + 1, sa.STATE_DIM)
    for name in ("train", "validation", "test"):
        assert sa.load_episodes(data_dir, name)[..., 1].max() < sa.OOD_BAND
    assert meta["ood_ceiling_bounce_episodes"] > 0
    assert meta["splits"]["ood_test"]["max_y"] > 1 - sa.RADIUS - .02


def test_model_shapes():
    g, e = AnimationGenerator(4, 16, 8), AnimationEncoder(4, 16)
    from particlegan import get_recipe
    prior = get_recipe(prior_kind="mog", sigma_rel=.025, z_dim=4, num_particles=8, total_steps=10,
                       batch_size=8, lr=1e-3).make_prior(device="cpu")
    assert g(torch.randn(5, 4)).shape == (5, RECORD_DIM)
    for joint in (True, False):
        d = AnimationCritics(16, 8, 4, joint=joint)
        assert d.roles() == (("joint",) if joint else ()) + ("state", "next_state", "image")
        record = torch.randn(3, RECORD_DIM)
        for role in d.roles():
            assert d.critic_for(role)(d.inputs(role, record)).reshape(-1).shape == (3,)
    states, frames = rollout(e, g, prior, torch.randn(3, sa.STATE_DIM), 4)
    assert states.shape == (3, 5, sa.STATE_DIM) and frames.shape == (3, 4, sa.IMAGE_SIZE ** 2)


def test_evaluate_persistence_and_oracle(data_dir):
    episodes = sa.load_episodes(data_dir, "test").double()
    train = sa.load_episodes(data_dir, "train").double().reshape(-1, sa.STATE_DIM)
    scaler = StateScaler.fit(train).double()
    result = evaluate_split(persistence_predictor(scaler), scaler, episodes)
    assert all(math.isfinite(v) for v in result.values())
    assert result["frame_self_mse"] == 0 and result["dream_score"] > 0

    def oracle(state):
        physical = scaler.inverse(state)
        return join(state, scaler(sa.step(physical)), sa.render(physical).flatten(1).double())

    exact = evaluate_split(oracle, scaler, episodes)
    assert exact["dream_score"] < 1e-8 and exact["one_step_mse"] < 1e-8
    assert exact["first_fail_median"] == max(HORIZONS) + 1


def test_smoke_train_and_leaderboard(data_dir, tmp_path):
    from experiments.train_animation_transition import DEFAULTS, train
    from experiments.animation_leaderboard import build
    results = tmp_path / "results"
    tiny = dict(width=16, encoder_width=16, channels=8, d_width=16, marginal_width=8, d_channels=4, z_dim=4,
                num_particles=8, batch_size=8, steps=3, checkpoints=[3], log_interval=1, device="cpu",
                data_dir=str(data_dir), live_log=str(tmp_path / "live.log"))
    arms = dict(gan=dict(arm="gan"), direct=dict(arm="direct"), persistence=dict(arm="persistence"))
    for name, switches in arms.items():
        summary = train({**DEFAULTS, **tiny, **switches, "name": name, "out_dir": str(results / name)})
        assert math.isfinite(summary["evaluation"]["test"]["dream_score"])
    assert (results / "gan" / "dream.gif").exists() and not (results / "persistence" / "dream.gif").exists()
    out = tmp_path / "board"
    table = build(results, out)
    readme = (out / "README.md").read_text()
    assert [row["name"] for row in table] == sorted(arms, key=lambda n: json.loads(
        (results / n / "summary.json").read_text())["evaluation"]["test"]["dream_score"])
    for rank, row in enumerate(table, 1):
        assert f"| {rank} | {row['name']} |" in readme
    assert "## Prior and encoder health" in readme and (out / "gan_dream.gif").exists()
    readme_text = readme.replace("_Filled in after the runs complete._", "Keep me.\n\n### Detail\nmore")
    (out / "README.md").write_text(readme_text)
    build(results, out)
    assert "Keep me.\n\n### Detail\nmore\n\n## OOD animation" in (out / "README.md").read_text()
