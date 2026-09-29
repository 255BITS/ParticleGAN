import importlib.util
import json
from pathlib import Path

import pytest
import torch

from experiments.config import read_config
from experiments.run_grid import load_config, trainer_defaults
from lib.toy_models import SimpleMLPDiscriminator
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[1]


def example():
    spec = importlib.util.spec_from_file_location('pacgan_example', ROOT / 'examples/100gaussians.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packing_matches_explicit_concatenation_and_backpropagates():
    packed = SimpleMLPDiscriminator(pack_size=8)
    reference = SimpleMLPDiscriminator(in_dim=16)
    reference.load_state_dict(packed.state_dict())
    # Non-contiguous input must also preserve order and every point's gradient.
    points = torch.randn(2, 32).t().requires_grad_()
    expected = points.detach().clone().requires_grad_()
    actual_logits = packed(points)
    expected_logits = reference(expected.reshape(4, 16))
    torch.testing.assert_close(actual_logits, expected_logits)
    actual_logits.sum().backward()
    expected_logits.sum().backward()
    torch.testing.assert_close(points.grad, expected.grad)
    assert (points.grad.abs().sum(1) > 0).all()
    assert actual_logits.shape == (4,)


@pytest.mark.parametrize('batch', [torch.empty(0, 2), torch.randn(15, 2), torch.randn(16, 3)])
def test_packing_rejects_incomplete_or_invalid_batches(batch):
    with pytest.raises(ValueError, match='divisible'):
        SimpleMLPDiscriminator(pack_size=8)(batch)


def test_no_regularizer_bypasses_recipe_stabilizers(tmp_path, monkeypatch):
    grid = example()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError('no_regularizer called a stabilization factory or noise schedule')

    for name in ('make_optimizers', 'make_generator_optimizer', 'make_critic_penalty', 'make_prior_regularizer'):
        monkeypatch.setattr(Recipe, name, forbidden)
    monkeypatch.setattr(grid, 'input_noise_std', forbidden)
    monkeypatch.setattr(grid, 'output_noise_std', forbidden)
    result = grid.train(epochs=1, steps_per_epoch=3, batch_size=16, num_particles=32,
                        pack_size=8, no_regularizer=True, reg_coeff=0., lambda_ep=0.,
                        device_str='cpu', save_plots=False, return_details=True, out_dir=str(tmp_path))
    assert result['D'](torch.randn(16, 2)).shape == (2,)
    for name in ('G', 'D', 'prior'):
        parameters = list(result[name].parameters())
        assert all(torch.isfinite(p).all() for p in parameters)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 3]
    assert all(row['prior_regularization'] == 0 for row in rows)
    assert not list(tmp_path.glob('*.png'))


@pytest.mark.parametrize('options, message', [
    ({'pack_size': 0}, 'positive integer'),
    ({'pack_size': True}, 'positive integer'),
    ({'pack_size': 8, 'batch_size': 17}, 'divisible'),
    ({'pack_size': 8}, 'only for'),
    ({'no_regularizer': True}, 'requires'),
    ({'no_regularizer': True, 'reg_coeff': 0., 'use_training_api': True}, 'requires'),
])
def test_invalid_training_settings_fail_before_setup(options, message):
    with pytest.raises(ValueError, match=message):
        example().train(**options)


def test_experiment_config_resolves_in_grid_runner():
    defaults = trainer_defaults(str(ROOT / 'experiments/train_100gaussians.py'))
    cfg = load_config(ROOT / 'configs/100gaussians/pacgan8_no_reg.toml', defaults)
    assert cfg['no_regularizer'] and cfg['pack_size'] == 8
    assert cfg['reg_coeff'] == cfg['lambda_ep'] == 0
    assert cfg['batch_size'] // cfg['pack_size'] == 256
    assert cfg['epochs'] * cfg['steps_per_epoch'] == 7000
    assert cfg['seed'] == read_config(ROOT / 'configs/100gaussians/default.toml')['seed']
