"""One exact historical diagnostic setup, with state and RNG witnesses."""
from pathlib import Path
import hashlib

RECEIPTS = []
SETUP_SHA = '3b3c84174d91f0e724ce8109ffc646a9de50d3e27332de39e9d8d3ae9f72c7e7'


def apply(torch, host, trainer, declaration, stream):
    assert declaration['candidate'] == 'API-RP1-CUDA-EAGER-new-init-diagnostic'
    assert declaration['initial_optimizer_state'] == 'declared_eager'
    assert declaration['optimizer_step_devices'] == {'G': 'parameter', 'D': 'parameter'}
    assert trainer.completed_steps == 0
    assert not trainer.opt_g.state and not trainer.opt_d.state
    path = Path(__file__).with_name('historical_eager_setup.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SETUP_SHA
    from historical_eager_setup import apply_historical_eager_state

    def material():
        state = trainer.state_dict()
        del state['optimizers']
        return host.digest([state, stream.get_state()])

    before = material()
    default_device = str(torch.get_default_device())
    apply_historical_eager_state(trainer)
    after = material()
    assert before == after, 'historical setup changed non-optimizer state or RNG'
    assert str(torch.get_default_device()) == default_device
    counts = {}
    for role, optimizer in (('G', trainer.opt_g), ('D', trainer.opt_d)):
        counts[role] = 0
        for group in optimizer.param_groups:
            for parameter in group['params']:
                state = optimizer.state[parameter]
                assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                assert state['step'].shape == () and state['step'].dtype == torch.float32
                assert state['step'].device == parameter.device and float(state['step']) == 0
                for name in ('exp_avg', 'exp_avg_sq'):
                    assert state[name].shape == parameter.shape
                    assert state[name].device == parameter.device and state[name].dtype == parameter.dtype
                    assert bool(torch.all(state[name] == 0))
                counts[role] += 1
    RECEIPTS.append(dict(scope='HISTORICAL_WORKER_EAGER_DIAGNOSTIC_NOT_PUBLIC_API',
        setup_sha256=SETUP_SHA, non_optimizer_and_rng_before=before,
        non_optimizer_and_rng_after=after, counts=counts, completed_steps=0,
        default_device=default_device, only_declared_zero_optimizer_state_created=True))
