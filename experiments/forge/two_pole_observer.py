"""Task-declared, RNG-neutral observations of the existing two-pole loop."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch


def diagnostic_contract(task):
    execution = task['execution']
    contract = execution.get('horizon_diagnostic')
    horizon = execution.get('original_schedule_horizon', execution['steps'])
    if type(horizon) is not int or horizon < 1:
        raise ValueError('original_schedule_horizon must be a positive integer')
    if contract is None:
        if 'original_schedule_horizon' in execution and horizon != execution['steps']:
            raise ValueError('a separate behavioral schedule horizon requires horizon_diagnostic')
        return None
    if (task.get('adapter') != 'transfer_behavior' or execution.get('host') != 'two_pole'
            or task['id'] == 'two_pole'):
        raise ValueError('horizon_diagnostic requires a separately declared two_pole host task')
    if (not isinstance(contract, dict) or set(contract) != {'schema_version', 'kind', 'checkpoints', 'prefix_horizon'}
            or type(contract['schema_version']) is not int or contract['schema_version'] != 1
            or contract['kind'] != 'two_pole_force_v1'):
        raise ValueError('unsupported horizon_diagnostic contract')
    steps, prefix = execution['steps'], contract['prefix_horizon']
    checkpoints = contract['checkpoints']
    if (type(steps) is not int or type(prefix) is not int or not 0 < prefix < steps
            or 'original_schedule_horizon' not in execution or horizon not in (prefix, steps)
            or not isinstance(checkpoints, list) or not checkpoints
            or any(type(step) is not int or not 0 < step <= steps for step in checkpoints)
            or checkpoints != sorted(set(checkpoints)) or prefix not in checkpoints or steps not in checkpoints):
        raise ValueError('horizon_diagnostic must bind its execution, prefix, schedule and ordered checkpoints')
    return deepcopy(contract)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return deepcopy(value)


def _norm(tensor):
    return float(tensor.detach().norm())


class TwoPoleObserver:
    """Optimizer hooks inspect already-computed gradients and actual updates."""
    def __init__(self, task, output):
        self.contract = diagnostic_contract(task)
        if self.contract is None:
            raise ValueError('a declared horizon diagnostic is required')
        self.task, self.output = task, Path(output)
        self.records, self.milestones, self.checkpoint_states, self.handles = [], [], {}, []
        self.prefix_steps = {math.ceil(i * self.contract['prefix_horizon'] / 24) for i in range(1, 25)}
        self.observation_steps = self.prefix_steps | set(self.contract['checkpoints'])
        self.step = 0
        self.stream = (self.output / 'force-trace.jsonl').open('w')

    def bind(self, components, particles, generator_optimizer, critic_optimizer):
        from benchmarks.legacy.locked_shared import LOCKED_SHARED
        self.components, self.particles = components, particles
        self.generator_optimizer, self.critic_optimizer = generator_optimizer, critic_optimizer
        self.critic = components.models['discriminator']
        self.critic_parameters = list(self.critic.parameters())
        self.particle_l2 = LOCKED_SHARED.particle_l2
        self.initial = self._state()
        self.handles = [generator_optimizer.register_step_pre_hook(self._before),
                        generator_optimizer.register_step_post_hook(self._after),
                        critic_optimizer.register_step_pre_hook(self._critic_before),
                        critic_optimizer.register_step_post_hook(self._critic_after)]

    def penalty_gradient(self, penalty):
        gradients = (torch.autograd.grad(penalty, self.critic_parameters, retain_graph=True, allow_unused=True)
                     if penalty.requires_grad else [None] * len(self.critic_parameters))
        self.penalty_parameter_gradient = torch.cat([
            (torch.zeros_like(parameter) if gradient is None else gradient.detach()).flatten()
            for parameter, gradient in zip(self.critic_parameters, gradients)])

    def _critic_before(self, optimizer, args, kwargs):
        self.critic_positions_before = torch.cat([parameter.detach().flatten() for parameter in self.critic_parameters]).clone()
        self.critic_gradient_before_guard = torch.cat([parameter.grad.detach().flatten() for parameter in self.critic_parameters]).clone()

    def _critic_after(self, optimizer, args, kwargs):
        self.critic_gradient_after_guard = torch.cat([parameter.grad.detach().flatten() for parameter in self.critic_parameters]).clone()
        self.critic_displacement = (torch.cat([parameter.detach().flatten() for parameter in self.critic_parameters])
                                    - self.critic_positions_before)

    def _state(self):
        return _snapshot({'positions': self.particles, 'critic': self.critic.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'streams': self.components.context.streams.state_dict(), 'torch_rng_state': torch.get_rng_state()})

    def _before(self, optimizer, args, kwargs):
        self.before = self.particles.detach().clone()
        self.gradient = self.particles.grad.detach().clone()
        self.scheduled_lr = float(optimizer.param_groups[0]['lr'])
        self.group_betas = tuple(optimizer.param_groups[0]['betas'])

    def _critic_diagnostics(self):
        from benchmarks.locked_shared.two_pole import real_batch
        # HostCritic is deterministic and has no mutable forward buffers. Use
        # the base critic, without input/output-noise wrappers or sample draws.
        with torch.enable_grad():
            real = real_batch(self.particles.shape[0]).to(self.particles)
            values = torch.cat((real, self.particles.detach())).requires_grad_(True)
            scores = self.critic(values)
            gradients = torch.autograd.grad(scores.sum(), values)[0]
        count = real.shape[0]
        return {'real_scores': scores[:count].detach().clone(), 'particle_scores': scores[count:].detach().clone(),
                'real_input_gradients': gradients[:count].detach().clone(),
                'particle_input_gradients': gradients[count:].detach().clone(),
                'clean_gradient_median': float(gradients.abs().median()),
                'real_score_mean': float(scores[:count].detach().mean()),
                'particle_score_mean': float(scores[count:].detach().mean())}

    def _after(self, optimizer, args, kwargs):
        self.step += 1
        positions = self.particles.detach().clone()
        displacement = positions - self.before
        l2_gradient = 2. * self.particle_l2 * self.before / self.before.numel()
        adversarial_gradient = self.gradient - l2_gradient
        response = getattr(optimizer, 'direct_response', None)
        gain = response.last_gain if response is not None else 1.
        betas = response.betas if response is not None else self.group_betas
        penalty = dict(self.components.penalty.bound.last_stats)
        scalar = {'step': self.step, 'mean_abs': float(positions.abs().mean()),
            'minimum_position': float(positions.min()), 'maximum_position': float(positions.max()),
            'particle_std': float(positions.std(unbiased=False)),
            'positive_particles': int((positions > 0).sum()), 'negative_particles': int((positions < 0).sum()),
            'zero_particles': int((positions == 0).sum()),
            'nearest_pole_distance': float((positions.flatten().abs() - 1.).abs().mean()),
            'scheduled_generator_lr': self.scheduled_lr, 'direct_gain': float(gain),
            'actual_generator_lr': self.scheduled_lr * gain,
            'actual_generator_betas': list(betas),
            'critic_lr': float(self.critic_optimizer.param_groups[0]['lr']),
            'input_noise_std': float(self.components.noise.input_sigma),
            'output_noise_std': float(self.components.noise.output_sigma),
            'total_gradient_norm': _norm(self.gradient), 'adversarial_gradient_norm': _norm(adversarial_gradient),
            'l2_gradient_norm': _norm(l2_gradient), 'displacement_norm': _norm(displacement),
            'mean_signed_adversarial_force': float(-adversarial_gradient.mean()),
            'mean_signed_l2_force': float(-l2_gradient.mean()), 'penalty': penalty,
            'critic_total_gradient_norm': _norm(self.critic_gradient_before_guard),
            'critic_penalty_gradient_norm': _norm(self.penalty_parameter_gradient),
            'critic_payoff_gradient_norm': _norm(self.critic_gradient_before_guard - self.penalty_parameter_gradient),
            'critic_applied_gradient_norm': _norm(self.critic_gradient_after_guard),
            'critic_displacement_norm': _norm(self.critic_displacement)}
        record = {**scalar, 'positions_before': self.before, 'positions': positions,
                  'total_gradient': self.gradient, 'adversarial_gradient': adversarial_gradient,
                  'particle_l2_gradient': l2_gradient, 'displacement': displacement,
                  'critic_total_parameter_gradient': self.critic_gradient_before_guard,
                  'critic_penalty_parameter_gradient': self.penalty_parameter_gradient,
                  'critic_payoff_parameter_gradient': self.critic_gradient_before_guard - self.penalty_parameter_gradient,
                  'critic_applied_parameter_gradient': self.critic_gradient_after_guard,
                  'critic_parameter_displacement': self.critic_displacement}
        if self.step in self.contract['checkpoints']:
            diagnostics = self._critic_diagnostics()
            record['critic_diagnostics'] = diagnostics
            self.milestones.append({**scalar, **{key: value for key, value in diagnostics.items()
                                                 if not isinstance(value, torch.Tensor)}})
            self.checkpoint_states[self.step] = self._state()
        self.records.append(_snapshot(record))
        self.stream.write(json.dumps(scalar, allow_nan=False, sort_keys=True) + '\n')
        self.stream.flush()

    def finish(self):
        self.close()
        torch.save(self.records, self.output / 'force-trace.pt')
        torch.save({'initial': self.initial, 'checkpoints': self.checkpoint_states}, self.output / 'diagnostic-checkpoints.pt')
        artifacts = []
        for name, count, kind in (('force-trace.pt', len(self.records), 'direct_coordinate_updates_v1'),
                                  ('force-trace.jsonl', len(self.records), 'direct_coordinate_scalar_updates_v1'),
                                  ('diagnostic-checkpoints.pt', len(self.checkpoint_states), 'public_component_states_v1')):
            path = self.output / name
            artifacts.append({'path': name, 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'record_count': count, 'kind': kind})
        return {'schema_version': 1, 'kind': self.contract['kind'], 'contract': self.contract,
            'execution_updates': self.task['execution']['steps'],
            'schedule_horizon': self.components.recipe.total_steps,
            'optimizer_updates_added': 0, 'sampling_draws_added': 0,
            'known_particle_l2_coefficient': self.particle_l2,
            'force_definition': 'negative loss gradient; adversarial gradient = observed total minus exact existing particle-L2 gradient',
            'critic_diagnostics_sampling': 'clean deterministic stored-host critic on fixed reals and current particles',
            'milestones': self.milestones, 'artifacts': artifacts}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stream.close()
