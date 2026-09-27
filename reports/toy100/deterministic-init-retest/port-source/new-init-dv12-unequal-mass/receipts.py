"""Source-only copied receipt helpers; no Torch import or model execution."""
import hashlib
import json

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def material(torch, value):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), device=str(value.device), dtype=str(value.dtype),
                    sha256=sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()))
    if isinstance(value, dict):
        return {str(k): material(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [material(torch, v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unregistered checkpoint object: {type(value)}')

def optimizer_proof(torch, trainer, declaration, include_values=True):
    rows = {}
    for role, optimizer in (('G', trainer.opt_g), ('D', trainer.opt_d)):
        entries = []
        for group in optimizer.param_groups:
            assert group.get('capturable', False) is False
            assert group['foreach'] is False and group['fused'] is False
            for parameter in group['params']:
                state = optimizer.state.get(parameter)
                if not state:
                    assert trainer.completed_steps == 0 and declaration['initial_optimizer_state'] == 'native_lazy'
                    entries.append({'state_missing': True, 'parameter': material(torch, parameter)})
                    continue
                clock_device = (str(parameter.device) if declaration['optimizer_step_devices'][role] == 'parameter'
                                else 'cpu')
                assert str(state['step'].device) == clock_device
                assert float(state['step']) == trainer.completed_steps
                assert state['exp_avg'].device == state['exp_avg_sq'].device == parameter.device
                entry = dict(shape=list(parameter.shape), step=float(state['step']),
                             step_device=str(state['step'].device), parameter_device=str(parameter.device))
                if include_values:
                    entry['state'] = material(torch, state)
                entries.append(entry)
        rows[role] = entries
    return rows

def geometry(torch, trainer):
    controller = getattr(trainer, 'controller', None)
    if controller is None or not hasattr(controller, 'latent_bandwidth'):
        return None
    z = trainer.prior.z.detach()
    expected = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
    assert torch.equal(controller.latent_bandwidth, expected)
    return material(torch, expected)
