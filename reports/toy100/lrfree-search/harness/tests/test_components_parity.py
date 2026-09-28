"""custom22 engine and host-binding checks (reports/custom22-design.md §3, "Implementation").

  /tmp/pr38-default-env/bin/python -m unittest -v harness/tests/test_components_parity.py
  (LRFREE_TEST_CUDA=0 skips the CUDA parity; LRFREE_TEST_STEPS overrides the scalar parity length)

(a) L1 scalar parity: the engine's re-expression vs the candidate's own GANTrainer.step, bitwise after every
    update (c1 mode_hold resources, c2 sparse table), dv12-ams-rc3 and st-10, CPU and CUDA; plus a mutation
    check that the comparator does catch a one-ulp LR change.
(b) per-host CPU smoke through screen.py (30 updates, both candidates), run twice: metrics/rates identical (L4).
(c) host copies with the engine disabled (HostPolicy = the host's own learner) reproduce the ORIGINAL PR #155
    hosts: every optimizer step's (lr, params, grads) sha256, the observation curve and the final metrics.
(c') the same with output noise ON (one duck-typed noise policy on both sides): identical digests, curve,
    final metrics and training noise-call sequence -> the bindings' noise sites/order are the host's.
L3a two-table A2 composition, L3b role-union penalty value, source/spec pins.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HARNESS = Path(__file__).resolve().parents[1]
BASE = HARNESS.parent
PY = '/tmp/pr38-default-env/bin/python'
PR155 = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search')
CANDIDATES = {'dv12-ams-rc3': BASE / 'candidates/dv12-ams/package', 'st-10': BASE / 'candidates/dv12-st/package'}
TASKS = ('two_pole', 'trajectory', 'residual_student', 'unipolar', 'ae_gan_hold', 'cover_leftover',
         'unused_token_hold', 'mid_scale_identity')
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def overrides(cand):
    return json.loads((BASE / 'runs' / cand / 'candidate.json').read_text())['overrides']


def options(cand):
    return json.loads((BASE / 'runs' / cand / 'candidate.json').read_text())['candidate_options']


def run(args, env=None, timeout=3600):
    return subprocess.run([PY, *map(str, args)], capture_output=True, text=True, env=env or ENV, timeout=timeout)


class ScalarParityTest(unittest.TestCase):
    """(a) L1: engine scalar_step == package GANTrainer.step, bitwise, every update."""

    def check(self, cand, device, steps):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'parity.json'
            env = dict(ENV, CUDA_VISIBLE_DEVICES='0' if device.startswith('cuda') else '')
            proc = run([HARNESS / 'components_parity.py', '--package-root', CANDIDATES[cand],
                        '--overrides', json.dumps(overrides(cand)), '--steps', steps, '--device', device,
                        '--out', out], env=env)
            self.assertTrue(out.exists(), proc.stderr[-2000:])
            record = json.loads(out.read_text())
        for row in record['results']:
            self.assertEqual(row['status'], 'PASS', f'{cand} {device} {row}')
            self.assertEqual(row['steps_compared'], steps)
        self.assertEqual(record['status'], 'PASS')
        return record

    def test_cpu(self):
        steps = int(os.environ.get('LRFREE_TEST_STEPS', 60))
        for cand in CANDIDATES:
            with self.subTest(cand=cand):
                self.check(cand, 'cpu', steps)

    @unittest.skipIf(os.environ.get('LRFREE_TEST_CUDA') == '0', 'CUDA parity disabled')
    def test_cuda(self):
        steps = int(os.environ.get('LRFREE_TEST_STEPS', 50))
        for cand in CANDIDATES:
            with self.subTest(cand=cand):
                self.check(cand, 'cuda:0', steps)

    def test_comparator_catches_one_ulp(self):
        """Mutation: nudge the critic LR by one ulp inside the engine -> the parity check must FAIL."""
        code = (
            'import sys, json, math, importlib; sys.path[:0] = [%r, %r]\n'
            'import torch, components, components_parity as cp\n'
            'package = importlib.import_module("particlegan")\n'
            'enter = components.Update._enter\n'
            'def mutated(self):\n'
            '    enter(self)\n'
            '    if self.eng.completed_steps == 3:\n'
            '        g = self.eng.opt_d.param_groups[0]; g["lr"] = math.nextafter(g["lr"], 1.0)\n'
            '        self.lrs = self.eng.current_lrs()\n'
            'components.Update._enter = mutated\n'
            'row = cp.run_config(package, components, json.loads(%r), "c1", 8, "cpu", torch)\n'
            'print(json.dumps(dict(status=row["status"], mismatch=row["first_mismatch"])))\n'
        ) % (str(CANDIDATES['dv12-ams-rc3']), str(HARNESS), json.dumps(overrides('dv12-ams-rc3')))
        proc = run(['-c', code], env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out['status'], 'FAIL')
        self.assertTrue(out['mismatch'].startswith('step4'), out)


class ComponentTest(unittest.TestCase):
    """L3a two-table A2 composition (exact) and L3b role-union KA2 value (documented tolerance)."""

    def test_two_table_a2(self):
        code = r'''
import sys, importlib, copy, json
sys.path[:0] = [%r, %r]
import torch
package = importlib.import_module("particlegan")
from particlegan.k3p import LatentRowDamping
recipe = package.get_recipe(**{**json.loads(%r), "num_particles": 12, "z_dim": 4, "batch_size": 32})
torch.manual_seed(0)
tables = [package.ParticlePrior(64, 4, init_std=.05, generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
twin = copy.deepcopy(tables)
betas = recipe.prior_betas or recipe.betas
lr = recipe.lr * recipe.prior_lr_mult
opt = recipe.make_generator_optimizer([{"params": [tables[0].z], "lr": lr, "betas": betas}],
                                      latent_table=tables[0].z, foreach=False, fused=False)
opt.add_param_group({"params": [tables[1].z], "lr": lr, "betas": betas})
extra = LatentRowDamping(tables[1].z, torch.zeros_like(tables[1].z), recipe.latent_damping_max_rate)
sep = [recipe.make_generator_optimizer([{"params": [t.z], "lr": lr, "betas": betas}], latent_table=t.z,
                                       foreach=False, fused=False) for t in twin]
gen = torch.Generator().manual_seed(5)
active = 0
for step in range(200):
    idx = [t.sample_indices(16, generator=gen) for t in tables]
    target = torch.randn(16, 4, generator=gen)
    for pair, opts in ((tables, None), (twin, sep)):
        loss = sum(((t.z[i] - target) ** 2).sum() * (k + 1) for k, (t, i) in enumerate(zip(pair, idx)))
        if opts is None:
            opt.zero_grad(); loss.backward()
            with extra.around(opt):
                opt.step()
        else:
            for o in opts: o.zero_grad()
            loss.backward()
            for o in opts: o.step()
    assert all(torch.equal(a.z, b.z) for a, b in zip(tables, twin)), step
    active += int(extra.started and extra.observed / extra.total < extra.max_rate)   # rho-damping path applied
print(json.dumps(dict(ok=True, started=extra.started, rate=extra.observed / extra.total, active=active)))
''' % (str(CANDIDATES['dv12-ams-rc3']), str(HARNESS), json.dumps(overrides('dv12-ams-rc3')))
        proc = run(['-c', code], env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(out['ok'] and out['started'] and out['rate'] < .5 and out['active'] > 150, out)

    def test_role_union_penalty(self):
        code = r'''
import sys, importlib, copy, json, math
sys.path[:0] = [%r, %r]
import torch
from torch import nn
package = importlib.import_module("particlegan")
import components
class ScaleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.LeakyReLU(.2), nn.Linear(32, 1))
    def score(self, z, scale):
        return self.net(torch.cat([z, z.new_full((z.shape[0], 1), float(scale))], -1)).squeeze(-1)
torch.manual_seed(0)
recipe = package.get_recipe(**json.loads(%r))
critic = ScaleCritic()
opt = recipe.make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic), foreach=False, fused=False)
pen = recipe.make_critic_penalty(opt)
scales, rows = (-1., 0., .5, 1.), 8
view = components.RoleView(critic, scales, rows)
out = {}
for phase, calls in (("A", 5), ("blend", 900)):
    opt.record.calls = calls
    for step in range(30):   # completed critic steps (surprise samples; blend: anchor started, EMA lagging)
        x = torch.randn(rows * len(scales), 4)
        opt.zero_grad(); (pen(view, x, x + .1).mean() + view(x).mean()).backward(); opt.step()
    opt.record.calls = calls
    state = copy.deepcopy(opt.record.state_dict())
    real, fake = torch.randn(rows * len(scales), 4), torch.randn(rows * len(scales), 4)
    pen.collect_stats = True
    union = pen(view, real, fake)
    stats = dict(pen.last_stats)
    pen.collect_stats = False
    parts = []
    for i, s in enumerate(scales):
        opt.record.load_state_dict(copy.deepcopy(state))
        role = lambda z, s=s: critic.score(z, s)
        blk = slice(i * rows, (i + 1) * rows)
        parts.append(pen.regularizer.penalty(role, real[blk], fake[blk], opt.record.observed_steps + 1, False,
            ema_critic=lambda x, s=s: opt.anchor.forward(lambda m, inp: m.score(inp, s), x))[0])
    total = sum(p / len(scales) for p in parts)
    out[phase] = [float(union), float(total), abs(float(union) - float(total)) / max(abs(float(total)), 1e-12),
                  stats.get("phase"), stats.get("prox")]
print(json.dumps(out))
''' % (str(CANDIDATES['dv12-ams-rc3']), str(HARNESS), json.dumps(overrides('dv12-ams-rc3')))
        proc = run(['-c', code], env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out['A'][3], 'a')
        self.assertEqual(out['blend'][3], 'blend')
        self.assertGreater(out['blend'][4], 0.)   # the EMA-anchor term is exercised through the RoleView
        for phase, (union, total, rel, _, _) in out.items():
            self.assertLess(rel, 1e-5, f'{phase}: union {union} vs sum_r cap_r/R {total}')


class HostSmokeTest(unittest.TestCase):
    """(b) + L4: every host through screen.py for both candidates, 30 updates, twice; identical outputs."""

    def screen(self, cand, task, out):
        env = dict(ENV, LRFREE_CUSTOM_TEST_STEPS='30')
        proc = run([HARNESS / 'screen.py', '--package-root', CANDIDATES[cand], '--overrides',
                    json.dumps(overrides(cand)), '--candidate-options', json.dumps(options(cand)), '--task', task,
                    '--output', out, '--cand', cand, '--device', 'cuda:0'], env=env)
        result = json.loads((out / 'result.json').read_text())
        self.assertIn(result['status'], ('PASS', 'FAIL'), result.get('traceback', proc.stderr[-2000:]))
        return result

    def test_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cand in CANDIDATES:
                for task in TASKS:
                    with self.subTest(cand=cand, task=task):
                        runs = [Path(tmp) / f'{cand}-{task}-{k}' for k in 'ab']
                        results = [self.screen(cand, task, r) for r in runs]
                        self.assertEqual(results[0]['completed_steps'], 30)
                        self.assertEqual(results[0]['custom22']['penalty_calls'], 30)
                        self.assertTrue(results[0]['custom22']['construction_global_rng_untouched'])
                        rates = [(r / 'rates.jsonl').read_text() for r in runs]
                        self.assertEqual(rates[0], rates[1])
                        self.assertEqual(len(rates[0].splitlines()), 30)
                        rows = [[{k: v for k, v in json.loads(line).items() if k != 'seconds'}
                                 for line in (r / 'metrics.jsonl').read_text().splitlines()] for r in runs]
                        self.assertEqual(rows[0], rows[1])
                        self.assertEqual(len(rows[0]), 24)
                        if task in ('two_pole', 'unipolar', 'cover_leftover', 'unused_token_hold',
                                    'mid_scale_identity'):
                            self.assertTrue(results[0]['noisy_equals_clean'])


class HostCopyTest(unittest.TestCase):
    """(c) engine disabled: the re-expressed loops on the verbatim copies == the original PR #155 hosts."""

    STEPS = 12

    def test_copies_reproduce_originals(self):
        script = HARNESS / 'tests' / 'custom22_host_policy.py'
        with tempfile.TemporaryDirectory() as tmp:
            for task in TASKS:
                with self.subTest(task=task):
                    outs = {}
                    for which in ('copy', 'original'):
                        path = Path(tmp) / f'{task}-{which}.json'
                        proc = run([script, '--which', which, '--task', task, '--steps', self.STEPS, '--out', path],
                                   env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
                        self.assertTrue(path.exists(), proc.stderr[-2000:])
                        outs[which] = json.loads(path.read_text())
                    a, b = outs['copy'], outs['original']
                    self.assertEqual(a['particlegan'], b['particlegan'])
                    self.assertEqual(len(a['records']), 2 * self.STEPS)
                    self.assertEqual([(r[0], r[2]) for r in a['records']], [(r[0], r[2]) for r in b['records']])
                    self.assertEqual(len(a['curve']), len(b['curve']))
                    for x, y in zip(a['curve'], b['curve']):
                        self.assertEqual(x, {k: v for k, v in y.items() if k in x})
                        self.assertEqual(set(y) - set(x), set())
                    for key, value in b['final'].items():
                        if key in a['final'] or isinstance(value, (int, float)) and not isinstance(value, bool):
                            if key in a['final']:
                                self.assertEqual(a['final'][key], value, key)


class HostNoiseSiteTest(unittest.TestCase):
    """(c') as (c) with output noise ON (same duck-typed policy both sides): the bindings apply output noise at
    the host's frozen noise_policy sites, in the host's order (harness/tests/custom22_host_noise.py)."""

    STEPS = 12

    def test_noise_sites_match_originals(self):
        script = HARNESS / 'tests' / 'custom22_host_noise.py'

        def training(calls):
            return [c for c in calls if c[0] == 'set' or c[1] == 'train']

        def subsequence(a, b):
            it = iter(b)
            return all(x in it for x in a)
        with tempfile.TemporaryDirectory() as tmp:
            for task in TASKS:
                with self.subTest(task=task):
                    outs = {}
                    for which in ('copy', 'original'):
                        path = Path(tmp) / f'{task}-{which}.json'
                        proc = run([script, '--which', which, '--task', task, '--steps', self.STEPS, '--out', path],
                                   env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
                        self.assertTrue(path.exists(), proc.stderr[-2000:])
                        outs[which] = json.loads(path.read_text())
                    a, b = outs['copy'], outs['original']
                    self.assertEqual(len(a['records']), 2 * self.STEPS)
                    self.assertEqual([(r[0], r[2]) for r in a['records']], [(r[0], r[2]) for r in b['records']])
                    self.assertTrue(any(c[0] == 'out' for c in a['noise_calls']))
                    self.assertEqual(training(a['noise_calls']), training(b['noise_calls']))
                    # originals add logging-only evaluations (residual_student _emit, ae_gan_hold _log)
                    self.assertTrue(subsequence(a['noise_calls'], b['noise_calls']))
                    self.assertEqual(len(a['curve']), len(b['curve']))
                    for x, y in zip(a['curve'], b['curve']):
                        self.assertEqual(x, {k: v for k, v in y.items() if k in x})
                        self.assertEqual(set(y) - set(x), set())
                    for key, value in b['final'].items():
                        if key in a['final']:
                            self.assertEqual(a['final'][key], value, key)


class SourceTest(unittest.TestCase):
    """Copies byte-identical below their headers, verdict functions verbatim, specs from the frozen plan."""

    def test_sources(self):
        code = (f'import sys; sys.path[:0] = [{str(HARNESS)!r}, {str(PR155)!r}]\n'
                'import custom22, json; print(json.dumps(custom22.verify_sources()["source_commit"]))')
        proc = run(['-c', code], env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])

    def test_verdict_functions_verbatim(self):
        import ast
        import hashlib
        import importlib.util
        spec = importlib.util.spec_from_file_location('fv_text', HARNESS / 'hosts/custom/frozen_verdict.py')
        text = (HARNESS / 'hosts/custom/frozen_verdict.py').read_text()
        tree = ast.parse(text)
        local = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest()
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
        for rel, names in (('benchmarks/locked_shared/baseline.py', ['score_metrics']),
                           ('benchmarks/transfer_suite/protocol.py', ['requirements', 'test_verdict'])):
            source = (PR155 / rel).read_text()
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    want = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                    self.assertEqual(local[node.name], want, f'{rel}::{node.name}')

    def test_specs(self):
        specs = json.loads((HARNESS / 'tasks/custom22_specs.json').read_text())
        plan = json.loads((PR155 / 'benchmarks/transfer_suite/plans/default_comparison.json').read_text())
        frozen = {j['spec']['name']: j['spec'] for j in plan}
        self.assertEqual(list(specs['tasks']), list(TASKS))
        for task, entry in specs['tasks'].items():
            self.assertEqual(entry['spec'], frozen[task])


if __name__ == '__main__':
    unittest.main(verbosity=2)
