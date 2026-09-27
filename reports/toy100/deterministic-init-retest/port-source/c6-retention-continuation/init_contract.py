"""CPU-only initialization conformance; no backward or optimizer step."""
from copy import deepcopy
import hashlib
import json


def tensor_receipt(torch, tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def material_receipt(torch, value):
    if isinstance(value, torch.Tensor):
        return tensor_receipt(torch, value)
    if isinstance(value, dict):
        return {str(k): material_receipt(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [material_receipt(torch, v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unregistered checkpoint value: {type(value)}')


def initial_material(torch, trainer):
    """Every checkpointed learner value except independent sampling RNG state."""
    state = trainer.state_dict()
    for key in ('streams', 'cpu_rng', 'cuda_rng'):
        state.pop(key, None)
    # Include all named buffers/parameters independently of checkpoint coverage.
    modules = {}
    for role in ('G', 'D', 'prior', 'ema_G', 'ema_D', 'ema_prior'):
        module = getattr(trainer, role)
        modules[role] = dict(parameters=dict(module.named_parameters()), buffers=dict(module.named_buffers()))
    return material_receipt(torch, dict(state=state, all_parameters_and_buffers=modules))


def assert_geometry_after_initialization(torch, trainer):
    """Detect the concrete stale random-prior bandwidth failure in DV16 ports."""
    controller = getattr(trainer, 'controller', None)
    if controller is None or not hasattr(controller, 'latent_bandwidth'):
        return {'geometry_check': 'not applicable: no declared bandwidth controller'}
    bandwidth = controller.latent_bandwidth
    if bandwidth is None:
        raise AssertionError('candidate did not initialize its prior geometry')
    z = trainer.prior.z.detach()
    expected = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
    if not torch.equal(bandwidth, expected):
        raise AssertionError('initial bandwidth was not derived after deterministic z overwrite')
    return dict(geometry_check='PASS', latent_bandwidth=tensor_receipt(torch, bandwidth))


def check(torch, package, host, declaration):
    from mode_hold_contract import construct
    if str(torch.get_default_device()) != 'cpu':
        raise RuntimeError('CPU preflight refuses a non-CPU default')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    stream = torch.Generator(device='cpu').manual_seed(0)
    from particlegan import initialization
    rng_checks = []
    originals = {name: getattr(initialization, name) for name in ('_initialize', '_initialize_prior')}
    def wrap(name):
        def witnessed(*args, **kwargs):
            before_cpu, before_stream = torch.get_rng_state().clone(), stream.get_state().clone()
            result = originals[name](*args, **kwargs)
            after_cpu, after_stream = torch.get_rng_state(), stream.get_state()
            if not torch.equal(before_cpu, after_cpu) or not torch.equal(before_stream, after_stream):
                raise AssertionError(f'{name} consumed construction/sampling RNG')
            rng_checks.append(dict(operation=name, cpu_before=tensor_receipt(torch, before_cpu),
                                   cpu_after=tensor_receipt(torch, after_cpu),
                                   sampling_before=tensor_receipt(torch, before_stream),
                                   sampling_after=tensor_receipt(torch, after_stream)))
            return result
        return witnessed
    construction_start = dict(cpu=tensor_receipt(torch, torch.get_rng_state()),
                              sampling=tensor_receipt(torch, stream.get_state()))
    try:
        for name in originals:
            setattr(initialization, name, wrap(name))
        first, _ = construct(torch, package, host, declaration, 'cpu', stream)
    finally:
        for name, function in originals.items():
            setattr(initialization, name, function)
    construction_end = dict(cpu=tensor_receipt(torch, torch.get_rng_state()),
                            sampling=tensor_receipt(torch, stream.get_state()))
    if {x['operation'] for x in rng_checks} != set(originals):
        raise AssertionError('candidate bypassed the witnessed public initializer paths')
    if construction_start == construction_end:
        raise AssertionError('ordinary constructor draws were unexpectedly omitted')
    first_material = initial_material(torch, first)
    geometry = assert_geometry_after_initialization(torch, first)
    # No second seed and no reset: random construction values must differ while
    # every final initialized parameter/derived buffer remains identical.
    torch.rand(97)
    torch.rand(113, generator=stream)
    second, _ = construct(torch, package, host, declaration, 'cpu', stream)
    second_material = initial_material(torch, second)
    if first_material != second_material:
        raise AssertionError('fresh model/derived state depends on preceding RNG consumption')
    assert_geometry_after_initialization(torch, second)
    from particlegan import qr_bz_pq_init
    if initialization._external_init is not None:
        raise AssertionError('global research initializer hook is active')
    expected = qr_bz_pq_init.qmc_draw(0, (12, 4), ('normal', 0., .5), rows_as_points=True).to(first.prior.z)
    if not torch.equal(first.prior.z, expected):
        raise AssertionError('recipe factory did not produce the c720 R2 prior at std .5')
    # Explicit network initializer is RNG-neutral and optimizer factories do
    # not overwrite parameters which the public initializer already marked.
    probe = torch.nn.Linear(4, 3)
    before = torch.get_rng_state().clone()
    package.initialize_(probe)
    if not torch.equal(before, torch.get_rng_state()):
        raise AssertionError('explicit public initializer consumed RNG')
    if first.completed_steps != 0 or second.completed_steps != 0:
        raise AssertionError('initialization preflight advanced the learner')
    return dict(status='PASS', scope='CPU construction only; no forward/backward/optimizer step or quality score',
                seed=0, no_seed_sweep=True, repeated_without_rng_reset=True,
                constructor_rng_start=construction_start, constructor_rng_end=construction_end,
                initializer_rng_neutrality=rng_checks,
                all_initial_material=first_material, **geometry)
