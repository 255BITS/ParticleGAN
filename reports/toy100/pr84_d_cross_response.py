"""One explicit actual-D response to PR84's accepted G/prior movement.

After unchanged PR84 produces D* and G+, materialize
  D+ = D* - P_D [F_D(D*,G+) - F_D(D*,G0)],
where P_D is the already advanced Adam metric and both D fields replay the
same frozen host D block and RNG. There is no response gain, solve, clipping,
new objective or clock. G0/G+ include the prior. Extra D queries do not call
Adam; moments and normal optimizer callbacks remain unchanged. This scratch
method has no neural stability guarantee or shared-gate eligibility.
"""

import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch

from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.extra_adam_scratch import HOSTS, sha


METHOD = 'pr84_accepted_g_discriminator_cross_response'


def response_point(materialized, metric, base_field, moved_field):
    """Return the unbounded, fixed-metric secant response; never mutate inputs."""
    if not (len(materialized) == len(metric) == len(base_field) == len(moved_field)):
        raise ValueError('D response tensor collections differ')
    result = []
    for point, p, base, moved in zip(materialized, metric, base_field, moved_field):
        if not (point.shape == p.shape == base.shape == moved.shape):
            raise ValueError('D response tensor shapes differ')
        if (not all(torch.isfinite(value).all() for value in (point, p, base, moved))
                or not (p > 0).all()):
            raise FloatingPointError('invalid D response point, metric or field')
        corrected = (point.double() - p.double() *
                     (moved.double() - base.double())).to(point.dtype)
        if not torch.isfinite(corrected).all():
            raise FloatingPointError('nonfinite D cross response; no clipping fallback')
        result.append(corrected)
    return result


class DCrossResponseRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, correction=True):
        super().__init__(start_step=start_step)
        self.correction = bool(correction)
        self.cross_records = []
        self.additional_d_fields = 0
        self.d_query_rng_verified = 0

    def phases(self, step, opt_d, opt_g, local):
        active = self.enabled and step >= self.start_step and self.correction
        if not active:
            yield from super().phases(step, opt_d, opt_g, local)
            return
        streams = [value for value in local.values() if isinstance(value, torch.Generator)]
        policy = local.get('noise_policy')
        if policy is not None:
            streams.extend(value for name in ('input_stream', 'output_stream')
                           if isinstance((value := getattr(policy, name, None)), torch.Generator))
        self._cross_streams = list({id(value): value for value in streams}.values())
        before_rng = self._rng(self._cross_streams)
        buffers = [(buffer, buffer.detach().clone())
                   for name in ('generator', 'critic', 'prior')
                   if isinstance((module := local.get(name)), torch.nn.Module)
                   for buffer in module.buffers()]
        self._base_d_query_field = self._base_d_query_rng = None
        self._cross_query_field = None
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
        after_rng = self._rng(self._cross_streams)
        accepted_g = [p.detach().clone() for p in self._params(opt_g)]
        accepted_d = [p.detach().clone() for p in self._params(opt_d)]
        accepted_buffers = [(buffer, buffer.detach().clone()) for buffer, _ in buffers]
        if any(not torch.equal(a, b) for a, b in zip(accepted_d, self.d_star)):
            raise RuntimeError('ordinary PR84 D did not finish at D*')
        reuse = all(torch.equal(a, b) for a, b in zip(self.d1, self.d_star))
        # The phase-1 D field is at D1,G0. It can be reused only when D1=D*.
        base_field = self._base_d_query_field if reuse else None
        fields = []
        completed = False
        try:
            query_names = ([] if reuse else ['base']) + ['moved']
            for index, query in enumerate(query_names):
                self._set_rng(self._cross_streams, before_rng)
                with torch.no_grad():
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                    for parameter, saved in zip(self._params(opt_d), accepted_d):
                        parameter.copy_(saved)
                    for parameter, saved in zip(self._params(opt_g),
                                                self.g_base if query == 'base' else accepted_g):
                        parameter.copy_(saved)
                self.phase = 3 + index
                self._smooth_on = False
                self._cross_query_field = None
                yield self.phase
                if self._cross_query_field is None:
                    raise RuntimeError('extra D block did not produce its field')
                if not all(torch.equal(a, b) for a, b in
                           zip(self._base_d_query_rng, self._rng(self._cross_streams))):
                    raise RuntimeError('extra D field consumed a different D-block RNG pattern')
                self.d_query_rng_verified += 1
                fields.append(query)
                if query == 'base':
                    base_field = self._cross_query_field
                else:
                    moved_field = self._cross_query_field
            corrected = response_point(accepted_d, self.metric_d, base_field, moved_field)
            with torch.no_grad():
                for parameter, saved in zip(self._params(opt_g), accepted_g):
                    parameter.copy_(saved)
                for parameter, value in zip(self._params(opt_d), corrected):
                    parameter.copy_(value)
            completed = True
            sq = lambda values: sum(float(value.double().square().sum()) for value in values)
            correction_norm = sq([a - b for a, b in zip(corrected, accepted_d)]) ** .5
            ordinary_norm = sq([a - b for a, b in zip(accepted_d, self.d0)]) ** .5
            row = dict(outer_step=self.outer_steps, base_field_reused=reuse,
                       extra_d_field_queries=len(fields), queries=fields,
                       correction_parameter_norm=correction_norm,
                       ordinary_d_step_parameter_norm=ordinary_norm,
                       correction_to_ordinary_d_norm=(correction_norm / ordinary_norm
                                                      if ordinary_norm else None),
                       d_field_difference_norm=sq([a - b for a, b in
                                                   zip(moved_field, base_field)]) ** .5,
                       correction_clipped=False)
            self.cross_records.append(row)
            self.records[-1]['d_cross_response'] = row
        finally:
            self._set_rng(self._cross_streams, after_rng)
            with torch.no_grad():
                for buffer, saved in accepted_buffers:
                    buffer.copy_(saved)
                for parameter, saved in zip(self._params(opt_g), accepted_g):
                    parameter.copy_(saved)
                if not completed:
                    for parameter, saved in zip(self._params(opt_d), accepted_d):
                        parameter.copy_(saved)
            self.phase = None
            self._smooth_on = False

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if (not self.passthrough and self.correction and self.optimizers is not None
                and optimizer is self.optimizers[0]):
            if self.phase == 0:
                self._base_d_query_rng = self._rng(self._cross_streams)
            elif self.phase == 1:
                self._base_d_query_field = [p.grad.detach().clone() for p in self._params(optimizer)]
        return super().step(optimizer, ordinary_step, closure)

    @torch.no_grad()
    def capture_d_query(self, optimizer):
        if self.phase not in (3, 4) or optimizer is not self.optimizers[0]:
            raise RuntimeError('D response query outside its declared phase')
        self._cross_query_field = [p.grad.detach().clone() for p in self._params(optimizer)]
        if not all(torch.isfinite(value).all() for value in self._cross_query_field):
            raise FloatingPointError('nonfinite extra D field')
        self.additional_d_fields += 1

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
                     correction=self.correction,
                     response_rule='D+=Dstar-PD*(FD(Dstar,Gaccepted)-FD(Dstar,Gbase))',
                     response_gain=1., moment_updates_per_outer_step=1,
                     ordinary_optimizer_callbacks_per_player_per_outer_step=3,
                     gradient_evaluations_per_outer_step=None,
                     d_gradient_evaluations_per_outer_step=(
                         3 + self.additional_d_fields / self.outer_steps if self.outer_steps else None),
                     g_gradient_evaluations_per_outer_step=3,
                     d_gradient_evaluations_total=3*self.outer_steps+self.additional_d_fields,
                     g_gradient_evaluations_total=3*self.outer_steps,
                     additional_d_field_queries=self.additional_d_fields,
                     additional_g_field_queries=0, d_query_rng_verified=self.d_query_rng_verified,
                     original_curvature_scope='original G and D proposals only; extra D response is unbounded',
                     cross_records=self.cross_records,
                     adapter_sha256=sha(Path(__file__).read_bytes()))
        return value


def d_only_query_source(source):
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == '_extra_phase']
    if len(loops) != 1:
        raise RuntimeError('expected one frozen gradient-block loop')
    loop = loops[0]
    positions = [i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.unparse(node.value.func) == 'opt_d.step']
    if len(positions) != 1:
        raise RuntimeError('expected one ordinary D optimizer callback')
    index = positions[0]
    original = deepcopy(tree)
    check = ast.parse('_extra_phase < 3', mode='eval').body
    loop.body[index:] = [
        ast.If(test=deepcopy(check), body=[loop.body[index]], orelse=[ast.parse(
            '_extra_state.capture_d_query(opt_d)').body[0]]),
        ast.If(test=deepcopy(check), body=loop.body[index + 1:], orelse=[]),
    ]
    ast.fix_missing_locations(tree)
    inverse = deepcopy(tree)
    changed = next(node for node in ast.walk(inverse) if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == '_extra_phase')
    changed.body[index:] = changed.body[index].body + changed.body[index + 1].body
    if ast.dump(original, include_attributes=False) != ast.dump(inverse, include_attributes=False):
        raise RuntimeError('D-only query transform changed the original gradient block')
    return tree, ast.unparse(tree) + '\n'


@contextmanager
def pr84_d_cross_response(*, task='mode_hold', start_step=0, correction=True):
    from benchmarks.locked_shared import mode_hold, trajectory
    module = {'mode_hold': mode_hold, 'trajectory': trajectory}[task]
    def factory(*, start_step=0):
        return DCrossResponseRecorder(start_step=start_step, correction=correction)
    with patch.object(frozen, 'SmoothedBothBoundRecorder', factory):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            tree, modified = d_only_query_source(source)
            recorder.host_source.update(original_pr84_generated_sha256=sha(source.encode()),
                                        generated_function_sha256=sha(modified.encode()))
            namespace = {}
            exec(compile(tree, f'<pr84-D-cross-{task}>', 'exec'), module.__dict__, namespace)
            with patch.object(module, HOSTS[task], namespace[HOSTS[task]]):
                yield recorder, modified
