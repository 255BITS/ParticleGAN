"""Contract checks for the isolated full-cloud prior probe."""

import torch

from benchmarks.toy100.train import _set_output_sigma, make_trainer, resolve_config
from reports.toy100.accuracy_full_prior_probe import install_full_cloud_prior_regularizer


def _trainer(num_particles):
    config, recipe = resolve_config(dict(
        problem="grid100", steps=2, seed=31, device="cpu", threads=1,
        num_particles=num_particles, batch_size=16, g_hidden=16, d_hidden=16,
        n_hidden=1, output_noise_std=.029, output_noise_warmup=.5,
        input_noise_std=.5, input_noise_anneal_end=.5,
        eval_samples=1024, snapshot_samples=32, early_eval_steps=[0, 2],
    ))
    trainer = make_trainer(config, recipe)
    _set_output_sigma(trainer, config, 0)
    return trainer, config


def test_small_prior_training_is_bitwise_identical_and_rng_free():
    torch.set_num_threads(1)
    control, _ = _trainer(64)
    probe, _ = _trainer(64)
    install_full_cloud_prior_regularizer(probe)
    real = torch.randn(16, 2, generator=torch.Generator().manual_seed(77))
    rng = torch.get_rng_state().clone()
    reference = control.step(real)
    reference_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    observed = probe.step(real)
    assert torch.equal(reference_rng, torch.get_rng_state())
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        torch.testing.assert_close(observed[key], reference[key], rtol=0, atol=0)
    for key in ("G", "D", "prior", "ema_G", "ema_prior"):
        control_state, probe_state = getattr(control, key).state_dict(), getattr(probe, key).state_dict()
        assert control_state.keys() == probe_state.keys()
        for name in control_state:
            torch.testing.assert_close(probe_state[name], control_state[name], rtol=0, atol=0)


def test_large_prior_regularizes_all_rows_without_rng():
    trainer, _ = _trainer(2048)
    sample = trainer.prior.z[:16]
    reference = trainer.prior_regularizer(sample)
    reference_grad = torch.autograd.grad(reference, trainer.prior.z)[0]
    assert torch.count_nonzero(reference_grad[16:]) == 0
    install_full_cloud_prior_regularizer(trainer)
    rng = torch.get_rng_state().clone()
    value = trainer.prior_regularizer(sample)
    assert torch.equal(rng, torch.get_rng_state())
    gradient = torch.autograd.grad(value, trainer.prior.z)[0]
    assert torch.count_nonzero(gradient[16:]) > 0
    assert torch.isfinite(gradient).all()
