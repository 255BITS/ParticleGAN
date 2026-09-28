"""Check that sampling calibration leaves a matched training update unchanged."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'candidate/package'))
from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import ParticlePrior

torch.set_num_threads(2)


def make():
    torch.manual_seed(91701)
    options = json.loads((ROOT / 'candidate/overrides.json').read_text())
    options.update(num_particles=64, z_dim=2, batch_size=32, initialization=None)
    recipe = get_recipe(**options)
    prior = ParticlePrior(64, 2, init_std=.1)
    generator = torch.nn.Linear(2, 2)
    with torch.no_grad():
        generator.weight.copy_(torch.eye(2))
        generator.bias.zero_()
    critic = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Softplus(),
                                 torch.nn.Linear(16, 1))
    return GANTrainer(recipe, generator, critic, prior=prior, seed=91702,
                      serial_backward=True)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    real = .12 * torch.randn(16384, 2, generator=torch.Generator().manual_seed(91703))
    calibrated = make()
    uncalibrated = make()
    for trainer in (calibrated, uncalibrated):
        trainer.row_em.observe_real(real)
        for group, role, tester in zip(trainer.opt_g.param_groups, trainer.roles[0],
                                       trainer.lr_settle.testers[0]):
            if role == 'generator' and not group.get('sigma_group'):
                tester.s = 1 / 64
    uncalibrated.row_em.armed = False
    log_sigma_before = calibrated.log_output_sigma.detach().clone()
    bd_before = calibrated.birth_death
    stream_before = {name: getattr(calibrated, name).get_state().clone()
                     for name in calibrated._STREAMS}
    global_before = torch.get_rng_state().clone()
    event = calibrated.row_em.maybe_apply(calibrated)
    assert event and calibrated.row_em.active
    assert torch.equal(calibrated.log_output_sigma, log_sigma_before)
    assert calibrated.birth_death is bd_before
    assert torch.equal(torch.get_rng_state(), global_before)
    assert all(torch.equal(state, getattr(calibrated, name).get_state())
               for name, state in stream_before.items())
    assert same(calibrated.birth_death.state_dict(), uncalibrated.birth_death.state_dict())
    assert calibrated.prior.row_weights_active
    assert calibrated.output_sigma() == event['sigma_after']
    assert uncalibrated.output_sigma() != calibrated.output_sigma()
    batch = real[:32]
    before_update = torch.get_rng_state().clone()
    result_a = calibrated.step(batch)
    torch.set_rng_state(before_update)
    result_b = uncalibrated.step(batch)
    assert same(result_a, result_b), 'calibration changed adversarial update'
    a, b = calibrated.state_dict(), uncalibrated.state_dict()
    for key in ('G', 'D'):
        assert same(a['models'][key], b['models'][key]), key
    assert torch.equal(a['models']['prior']['z'], b['models']['prior']['z'])
    for key in ('optimizers', 'controller', 'lr_settle', 'birth_death', 'streams'):
        assert same(a[key], b[key]), key
    assert torch.equal(a['output_noise']['log_sigma'], b['output_noise']['log_sigma'])
    assert calibrated.last_output_sigma == uncalibrated.last_output_sigma
    replay = make()
    replay.load_state_dict(a)
    assert same(a, replay.state_dict())
    stream = torch.Generator().manual_seed(91902)
    draw = calibrated.sample(256, generator=stream)
    assert torch.equal(draw, replay.sample(256, generator=torch.Generator().manual_seed(91902)))
    assert calibrated.birth_death is bd_before
    report = {'status': 'PASS', 'fit': event, 'training_sigma': calibrated.last_output_sigma,
              'checks': ['private fit RNG', 'unchanged GAN and birth-death update',
                         'sampling width separate from training width', 'checkpoint replay']}
    (ROOT / 'smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
