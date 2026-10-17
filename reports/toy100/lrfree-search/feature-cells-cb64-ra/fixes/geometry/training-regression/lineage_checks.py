"""Shared focused fixtures for CPU and coordinator-owned CUDA lineage checks."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parent
FIXES = ROOT.parent.parent
BASE = FIXES / 'pkg-CB64-RA2'
SEED = 90229  # Existing geometry integration fixture; never varied.
DEVICE = torch.device('cpu')
INPUTS = None
REFERENCE = None
Recipe = ParticlePrior = GANTrainer = LatentLineage = BoundedLatentGeometry = None
DETAILS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(package):
    return {str(p.relative_to(package)): sha(p)
            for p in sorted((Path(package) / 'particlegan').rglob('*.py'))}


def digest(value):
    """Typed fingerprint of semantic state; observational durations are excluded."""
    h = hashlib.sha256()
    def token(x):
        b = x if isinstance(x, bytes) else str(x).encode()
        h.update(str(len(b)).encode() + b':' + b)
    def add(x):
        if isinstance(x, torch.Tensor):
            token('tensor'); token(tuple(x.shape)); token(x.dtype); token(x.device)
            token(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            keys = sorted((k for k in x if k != 'eval_seconds'), key=lambda k: (type(k).__name__, repr(k)))
            token('dict'); token(len(keys))
            for key in keys:
                add(key); add(x[key])
        elif isinstance(x, (list, tuple)):
            token(type(x).__name__); token(len(x))
            for v in x:
                add(v)
        elif isinstance(x, float):
            token('float64'); token(struct.pack('!d', x))
        else:
            token(type(x).__name__); token(repr(x))
    add(value)
    return h.hexdigest()


def stream(state=None):
    result = torch.Generator(device=DEVICE)
    return result.manual_seed(SEED) if state is None else result.set_state(state.cpu())


def ids(values):
    return torch.tensor(values, device=DEVICE, dtype=torch.long)


def make_trainer(rank=8):
    # Reuse the original focused integration fixture, with deterministic
    # two-cluster table/real rows and an explicit identity generator.
    torch.manual_seed(SEED)
    options = dict(INPUTS['config'])
    options.update(num_particles=1024, z_dim=2, initialization=None,
                   birth_death_metric_rank=rank,
                   output_noise_mode='fixed', output_noise_std=0., serve_average=0.)
    G = torch.nn.Linear(2, 2).to(DEVICE)
    with torch.no_grad():
        G.weight.copy_(torch.eye(2, device=DEVICE)); G.bias.zero_()
    D = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(DEVICE)
    with torch.no_grad():
        D[0].weight[:, 0] = torch.where(D[0].weight[:, 0] >= 0, 100., -100.)
        D[0].weight[:, 1] = 0.; D[0].bias.zero_()
    prior = ParticlePrior(1024, 2, init_std=0., generator=stream(), device=DEVICE)
    with torch.no_grad():
        prior.z[:, 0] = 3.; prior.z[:102, 0] = -3.
    return GANTrainer(Recipe(**options), G, D, prior=prior, seed=SEED, serial_backward=True)


def real_batch():
    real = torch.zeros(1024, 2, device=DEVICE)
    real[:, 0] = -3.; real[:102, 0] = 3.
    return real[torch.randperm(1024, generator=stream(), device=DEVICE)]


def reference_module():
    name = 'particlegan._lineage_ra2_reference'
    spec = importlib.util.spec_from_file_location(name, BASE / 'particlegan' / 'feature_cells.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(package, device):
    global DEVICE, INPUTS, REFERENCE, Recipe, ParticlePrior, GANTrainer, LatentLineage, BoundedLatentGeometry
    import sys
    DEVICE = torch.device(device)
    sys.path.insert(0, str(package))
    from particlegan import Recipe as recipe, ParticlePrior as prior, GANTrainer as trainer
    from particlegan.feature_cells import LatentLineage as lineage, BoundedLatentGeometry as geometry
    Recipe, ParticlePrior, GANTrainer, LatentLineage, BoundedLatentGeometry = recipe, prior, trainer, lineage, geometry
    INPUTS = torch.load(FIXES / 'geometry' / 'gpu-inputs.pt', map_location='cpu', weights_only=False)
    REFERENCE = reference_module()
    DETAILS.clear()


class LineageTests(unittest.TestCase):
    def check_graph(self, graph):
        graph.validate(graph.neighbors)
        self.assertLessEqual(int((graph.neighbors >= 0).sum(1).max()), graph.degree)

    def test_01_empty_graph_ra2_bits_rng_and_rank_edges(self):
        case = next(c for c in INPUTS['cases'] if c['name'] == 'mnist_E22_0')
        points, noise, width = (case[key].to(DEVICE) for key in ('prior', 'noise', 'bandwidth'))
        rows = torch.arange(len(noise), device=DEVICE)
        prior = SimpleNamespace(z=points)
        results = []
        for rank in (1, 8, 64, 128):
            graph = LatentLineage(len(points), min(rank, 64, len(points)-1), DEVICE)
            new = BoundedLatentGeometry(rank=rank, chunk=37, lineage=graph)
            old = REFERENCE.BoundedLatentGeometry(rank=rank, chunk=37)
            query = points[rows]
            a = old._local_geometry(query, prior)
            b = new._local_geometry(query, prior, rows=rows)
            self.assertTrue(torch.equal(a[0], b[0]), f'radius rank={rank}')
            self.assertTrue(torch.equal(a[1], b[1]), f'width rank={rank}')
            self.assertTrue(torch.equal(old.displacement(query, prior, width, noise),
                                       new.displacement(query, prior, width, noise, rows=rows)))
            self.assertLessEqual(new.work['max_candidates'], 64+graph.degree)
            results.append(dict(rank=rank, degree=graph.degree, radius_width_displacement_bit_identical=True))
        trainer = make_trainer()
        new = trainer.birth_death
        old = REFERENCE.FeatureCellBirthDeath(trainer, SEED)
        rows = torch.arange(63, device=DEVICE)
        query = trainer.prior.z[rows]
        a, b = stream(), stream()
        self.assertTrue(torch.equal(old.perturb_latent(query, a), new.perturb_latent(query, b, rows=rows)))
        self.assertTrue(torch.equal(a.get_state(), b.get_state()))
        for rank, n, expected in ((128, 1024, 64), (128, 17, 16), (8, 1024, 8)):
            graph = LatentLineage(n, min(rank, 64, n-1), DEVICE)
            self.assertEqual(graph.degree, expected)
        self.assertEqual(make_trainer(rank=128).birth_death.settings['lineage_degree'], 64)
        self.assertEqual(new.settings['latent_candidate_bound'], 72)
        DETAILS['empty_graph'] = dict(ranks=results, perturbation_rng_state_bit_identical=True,
                                     default_candidate_bound=72, high_rank_degree_cap=64)

    def test_02_retained_saved_close_pair_is_considered(self):
        path = Path('/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training/toy/E22/checkpoint-1000.pt')
        saved = torch.load(path, map_location='cpu', weights_only=False)['trainer']
        points = saved['models']['prior']['z'].to(DEVICE)
        prior = SimpleNamespace(z=points)
        old = REFERENCE.BoundedLatentGeometry()
        bounded = old.radius(points, prior)
        exact, parents = [], []
        # A test oracle only: full-vector exact pairs are never used by the
        # production graph or its overwrite path.
        for block in points.split(16):
            distance = (block[:, None]-points[None]).square().sum(-1)
            distance.masked_fill_(distance == 0, float('inf'))
            value, parent = distance.min(1)
            exact.append(.5*value.sqrt()); parents.append(parent)
        exact, parents = torch.cat(exact), torch.cat(parents)
        child = (bounded-exact).argmax().reshape(1)
        parent = parents[child]
        self.assertGreater(float(bounded[child]-exact[child]), .01)
        graph = LatentLineage(len(points), 8, DEVICE)
        graph.register_copies(child, parent)
        geometry = BoundedLatentGeometry(lineage=graph)
        actual = geometry.radius(points[child], prior, rows=child)
        self.assertTrue(torch.equal(actual, exact[child]))
        self.assertLess(float(actual), float(bounded[child]))
        self.check_graph(graph)
        DETAILS['saved_pair'] = dict(checkpoint=str(path), checkpoint_sha256=sha(path),
            child=int(child), parent=int(parent), bounded_radius=float(bounded[child]),
            exact_radius=float(exact[child]), lineage_radius=float(actual),
            max_candidates=geometry.work['max_candidates'])

    def test_03_overwrites_evictions_and_simultaneous_incarnations(self):
        graph = LatentLineage(20, 3, DEVICE)
        graph.register_copies(ids([1, 2, 3]), ids([0, 0, 0]))
        self.assertEqual(graph.neighbors[0].tolist(), [3, 2, 1])
        graph.register_copies(ids([4]), ids([0]))
        self.assertEqual(graph.neighbors[0].tolist(), [4, 3, 2])
        self.assertTrue(bool((graph.neighbors[1] == -1).all()))
        graph.register_copies(ids([3]), ids([5]))
        self.assertFalse(bool((graph.neighbors[0] == 3).any()))
        graph.register_copies(ids([0, 5]), ids([5, 6]))
        self.assertTrue(bool((graph.neighbors[0] == -1).all()))
        self.assertEqual(graph.neighbors[5, 0].item(), 6)
        self.assertTrue(bool((graph.neighbors[2] == -1).all()))
        self.assertTrue(bool((graph.neighbors[3] == -1).all()))
        self.assertTrue(bool((graph.neighbors[4] == -1).all()))
        self.check_graph(graph)
        # Both full parents were linked. One parent's eviction must not be
        # reintroduced by the other parent's preserved old-row snapshot.
        graph = LatentLineage(20, 2, DEVICE)
        for child, parent in ((1, 0), (2, 0), (3, 1)):
            graph.register_copies(ids([child]), ids([parent]))
        graph.register_copies(ids([4, 5]), ids([0, 1]))
        self.assertEqual(graph.neighbors[0].tolist(), [4, 2])
        self.assertEqual(graph.neighbors[1].tolist(), [5, 3])
        self.check_graph(graph)
        before = graph.neighbors.clone()
        graph.register_copies(ids([]), ids([]))
        self.assertTrue(torch.equal(before, graph.neighbors))

    def test_04_repeated_manual_parents_and_degree_cap(self):
        graph = LatentLineage(20, 3, DEVICE)
        graph.register_copies(ids(list(range(1, 9))), ids([0]*8))
        self.assertEqual(graph.neighbors[0].tolist(), [8, 7, 6])
        self.assertTrue(bool((graph.neighbors[1:6] == -1).all()))
        self.assertTrue(bool((graph.neighbors[6:9, 0] == 0).all()))
        self.check_graph(graph)
        with self.assertRaises(ValueError):
            LatentLineage(1024, 65, DEVICE)
        with self.assertRaises(ValueError):
            LatentLineage(17, 17, DEVICE)
        with self.assertRaises(ValueError):
            graph.register_copies(ids([1, 1]), ids([2, 3]))

    def test_05_batched_reference_graph_sequence(self):
        n, degree = 37, 8
        graph = LatentLineage(n, degree, DEVICE)
        reference = [[] for _ in range(n)]
        def remove(a, b):
            if b in reference[a]: reference[a].remove(b)
            if a in reference[b]: reference[b].remove(a)
        for turn in range(23):
            children = [(turn*3+j*7) % n for j in range(5)]
            parents = [(turn*11+j//2*13) % n for j in range(5)]
            for child in children:
                for parent in list(reference[child]): remove(child, parent)
            incoming = {}
            for child, parent in zip(children, parents):
                if parent not in children:
                    incoming.setdefault(parent, []).insert(0, child)
            evictions = [(p, q) for p, new in incoming.items()
                         for q in reference[p][max(0, degree-len(new)):]]
            for a, b in evictions: remove(a, b)
            for parent, new in incoming.items():
                for child in reversed(new[:degree]):
                    reference[parent].insert(0, child); reference[child].insert(0, parent)
            graph.register_copies(ids(children), ids(parents))
            self.check_graph(graph)
            self.assertEqual([[v for v in row if v >= 0] for row in graph.neighbors.tolist()], reference,
                             f'copy batch {turn}')
        DETAILS['reference_graph'] = dict(batches=23, population=n, degree=degree,
                                         exact_ordered_adjacency_match=True)

    def test_06_no_population_scan_during_copy_and_query_bounds(self):
        receipts = []
        for population in (1024, 100000):
            graph = LatentLineage(population, 8, DEVICE)
            graph.register_copies(ids(list(range(1, 9))), ids([0]*8))
            graph.register_copies(ids([9, 10]), ids([0, 1]))
            graph.register_copies(ids([9, 3]), ids([2, 4]))
            self.check_graph(graph)
            self.assertLessEqual(graph.work['maximum_reciprocal_slots'], 8*8*8)
            receipts.append(dict(population=population, **graph.work))
        self.assertEqual(receipts[0]['reciprocal_slots'], receipts[1]['reciprocal_slots'])
        case = next(c for c in INPUTS['cases'] if c['name'] == 'native_cpu_saved_density')
        points = case['prior'].to(DEVICE)
        graph = LatentLineage(len(points), 8, DEVICE)
        graph.register_copies(ids([0, 1]), ids([17, 18]))
        rows = torch.arange(73, device=DEVICE)
        geometry = BoundedLatentGeometry(chunk=19, lineage=graph)
        geometry.radius(points[rows], SimpleNamespace(z=points), rows=rows)
        self.assertEqual(geometry.work['distance_pairs'], len(rows)*72)
        self.assertLessEqual(geometry.work['max_query_rows'], 19)
        self.assertEqual(geometry.work['max_candidates'], 72)
        DETAILS['bounded_work'] = dict(copy_updates=receipts, geometry=dict(geometry.work))

    def test_07_training_capture_copy_share_rows_and_rng_law(self):
        trainer = make_trainer(); bd = trainer.birth_death
        # These are actual row moves through the inherited table/EMA/Adam path.
        bd._move(trainer, ids([30, 31]), ids([300, 301]))
        rows = ids([30, 300, 31, 301, 500])
        latent = trainer.prior.z[rows]
        noise = torch.randn(latent.shape, generator=stream(), device=DEVICE)
        delta = bd.latent_geometry.displacement(latent, trainer.prior,
                   trainer.controller.latent_bandwidth, noise, rows=rows)
        self.assertTrue(torch.equal(trainer._generate(trainer.G, latent, 0., stream(), rows=rows),
                                   trainer.G(latent+delta)))
        bd.stream.set_state(stream().get_state()); bd.sample_shape = (2,)
        captured = bd._capture_generated(trainer, latent, jitter=True, rows=rows)
        expected = bd._features(trainer, trainer.G(latent+delta), chunk=256)
        self.assertTrue(torch.equal(captured, expected))
        bd.stream.set_state(stream().get_state())
        child = ids([700, 701, 702, 703, 704])
        saved_stream = trainer.noise_generator.get_state().clone()
        before = latent.detach().clone()
        bd._move(trainer, child, rows)
        self.assertTrue(torch.equal(trainer.prior.z[child], before+delta))
        self.assertTrue(torch.equal(saved_stream, trainer.noise_generator.get_state()))
        self.check_graph(bd.lineage)
        self.assertIsNone(bd._copy_parent_rows)
        # Detached radius/width must leave the ordinary latent gradient intact.
        trainer.opt_g.zero_grad()
        trainer._generate(trainer.G, trainer.prior.z[child], 0., stream(), rows=child).sum().backward()
        self.assertTrue(torch.equal(trainer.prior.z.grad[child], torch.ones(5, 2, device=DEVICE)))

    def test_08_copy_preserves_ema_optimizer_history_and_private_stream(self):
        trainer = make_trainer(); bd = trainer.birth_death
        child, parent = ids([0, 1]), ids([300, 301])
        with torch.no_grad(): trainer.ema_prior.z.copy_(trainer.prior.z+10.)
        state = trainer.opt_g.state[trainer.prior.z]
        rows = torch.arange(2048, dtype=torch.float32, device=DEVICE).reshape(1024, 2)
        for name, offset in (('exp_avg', 1.), ('exp_avg_sq', 2.), ('max_exp_avg_sq', 3.)):
            state[name] = rows+offset
        state['step'] = torch.tensor(7., device=DEVICE)
        trainer.opt_g.latent_history.copy_(rows+4.)
        ema = trainer.ema_prior.z.detach().clone()
        moment, history = deepcopy(state), trainer.opt_g.latent_history.clone()
        prior_before = trainer.prior.z.detach().clone()
        draw = stream(bd.stream.get_state())
        delta = bd.latent_geometry.displacement(prior_before[parent], trainer.prior,
                trainer.controller.latent_bandwidth,
                torch.randn((2, 2), generator=draw, device=DEVICE), rows=parent)
        bd._move(trainer, child, parent)
        self.assertTrue(torch.equal(trainer.ema_prior.z[child], ema[parent]+delta))
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            self.assertTrue(torch.equal(state[name][child], moment[name][parent]))
        self.assertTrue(torch.equal(state['step'], moment['step']))
        self.assertTrue(torch.equal(trainer.opt_g.latent_history[child], history[parent]))
        self.assertTrue(torch.equal(bd.stream.get_state(), draw.get_state()))
        self.assertTrue(bool((bd.lineage.neighbors[child, 0] == parent).all()))

    def test_09_ema_current_distances_and_sampling_streams(self):
        trainer = make_trainer(); bd = trainer.birth_death
        bd._move(trainer, ids([0, 1]), ids([300, 301]))
        with torch.no_grad(): trainer.ema_prior.z.mul_(.125)
        rows = torch.arange(40, device=DEVICE)
        live_radius = bd.latent_geometry.radius(trainer.prior.z[rows], trainer.prior, rows=rows)
        ema_radius = bd.latent_geometry.radius(trainer.ema_prior.z[rows], trainer.ema_prior, rows=rows)
        self.assertTrue(torch.allclose(ema_radius, live_radius*.125))
        self.assertFalse(torch.equal(live_radius, ema_radius))
        graph = bd.lineage.neighbors.clone()
        states = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS}
        private = bd.stream.get_state().clone(); global_rng = torch.get_rng_state().clone()
        for ema in (False, True):
            model, prior = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
            expected_stream = stream()
            latent, rows = prior.sample(47, generator=expected_stream)
            expected = trainer._generate(model, latent, 0., expected_stream, rows=rows)
            actual = trainer.sample(47, ema=ema, generator=stream())
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(graph, bd.lineage.neighbors))
        self.assertTrue(torch.equal(private, bd.stream.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        for name, before in states.items():
            self.assertTrue(torch.equal(before, getattr(trainer, name).get_state()))

    def test_10_training_row_ids_and_semantic_checkpoint_replay(self):
        trainer = make_trainer(); real = real_batch(); bd = trainer.birth_death
        bd._move(trainer, ids([0, 1, 2]), ids([300, 301, 302]))
        calls = []
        original = bd.perturb_latent
        def observed(latent, *args, rows=None, **kw):
            self.assertIsNotNone(rows)
            self.assertTrue(torch.equal(latent.detach(), trainer.prior.z.detach()[rows]))
            calls.append(rows.detach().clone())
            return original(latent, *args, rows=rows, **kw)
        bd.perturb_latent = observed
        trainer.step(real)
        bd.perturb_latent = original
        self.assertEqual(len(calls), 2)
        self.check_graph(bd.lineage)
        saved = trainer.state_dict()
        self.assertEqual(saved['birth_death']['backend_schema'], bd.BACKEND_SCHEMA)
        self.assertGreaterEqual(bd.BACKEND_SCHEMA, 4)
        self.assertEqual(saved['schema'], 4)  # Trainer schema was already 4 in RA2.
        self.assertNotIn('latent_geometry', saved['birth_death'])
        self.assertGreater(int((saved['birth_death']['lineage_neighbors'] >= 0).sum()), 0)
        # Exercise actual serialization, not an alias of an in-memory graph.
        buffer = io.BytesIO(); torch.save(saved, buffer); buffer.seek(0)
        saved = torch.load(buffer, weights_only=False)
        restoration_digest = digest(saved)
        first_losses, endpoints = [], []
        for _ in range(2):
            result = trainer.step(real)
            first_losses.append({k: result[k].detach().clone() for k in
                                 ('loss_d', 'loss_g', 'loss_gan', 'prior_regularization', 'penalty')})
            endpoints.append(digest(trainer.state_dict()))
        restored = make_trainer(); restored.load_state_dict(saved)
        self.assertEqual(digest(restored.state_dict()), restoration_digest)
        self.assertFalse(restored.birth_death.latent_geometry._entries)
        self.assertIsNone(restored.birth_death.snapshot)
        for loss, endpoint in zip(first_losses, endpoints):
            result = restored.step(real)
            self.assertEqual(digest(loss), digest({k: result[k].detach().clone() for k in loss}))
            self.assertEqual(endpoint, digest(restored.state_dict()))
        DETAILS['checkpoint'] = dict(backend_schema=bd.BACKEND_SCHEMA, trainer_schema=4, restored_bit_identical=True,
            updates_replayed=2, per_update_loss_and_semantic_state_bit_identical=True,
            graph_serialized=True, derived_caches_cleared=True,
            excluded_observational_fields=['birth_death.last.eval_seconds'],
            semantic_endpoint_sha256=endpoints[-1], ordinary_moves=bd.last.get('ordinary_moves'),
            total_lineage_edges=int((bd.lineage.neighbors >= 0).sum())//2)

    def test_11_invalid_graph_and_old_kernel_checkpoint_rejected(self):
        trainer = make_trainer(); bd = trainer.birth_death
        bd._move(trainer, ids([0]), ids([300]))
        good = deepcopy(bd.state_dict())
        mutations = {}
        bad = deepcopy(good); bad['backend_schema'] = 3; mutations['old_backend_schema'] = bad
        bad = deepcopy(good); bad['settings']['latent_kernel'] = 'bounded_local_dv12'; mutations['old_kernel'] = bad
        bad = deepcopy(good); bad.pop('lineage_neighbors'); mutations['missing_graph'] = bad
        for name in ('out_of_range', 'negative', 'self_edge', 'duplicate', 'asymmetric', 'wrong_dtype', 'wrong_shape'):
            bad = deepcopy(good); graph = bad['lineage_neighbors']
            if name == 'out_of_range': graph[0, 0] = bd.N
            elif name == 'negative': graph[0, 0] = -2
            elif name == 'self_edge': graph[0, 0] = 0
            elif name == 'duplicate': graph[0, 1] = graph[0, 0]
            elif name == 'asymmetric': graph[300].fill_(-1)
            elif name == 'wrong_dtype': bad['lineage_neighbors'] = graph.float()
            elif name == 'wrong_shape': bad['lineage_neighbors'] = graph[:, :1]
            mutations[name] = bad
        before = digest(bd.state_dict())
        for name, bad in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError): bd.load_state_dict(bad)
            self.assertEqual(digest(bd.state_dict()), before)
        # Graph state owns its tensor; mutating a returned checkpoint is safe.
        returned = bd.state_dict()
        returned['lineage_neighbors'].fill_(-1)
        self.assertEqual(digest(bd.state_dict()), before)
        self.assertNotEqual(digest(bd.state_dict()), digest(returned))
        if DEVICE.type == 'cuda':
            cpu_stored = deepcopy(bd.state_dict())
            cpu_stored['lineage_neighbors'] = cpu_stored['lineage_neighbors'].cpu()
            bd.load_state_dict(cpu_stored)
            self.assertEqual(bd.lineage.neighbors.device, DEVICE)
        DETAILS['checkpoint_rejection'] = dict(rejected=list(mutations), failure_atomic=True,
                                               independent_graph_tensor=True)


def run(package, output, device):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists(): raise RuntimeError(f'receipt already exists: {output}')
    before = sources(package)
    setup(package, device)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LineageTests))
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), seconds=time.perf_counter()-start,
        seed=SEED, new_seed_experiments=0, device=str(DEVICE), cpu_threads=torch.get_num_threads(),
        cuda_initialized=torch.cuda.is_initialized(),
        scope='focused lineage geometry/copy/EMA/RNG/semantic checkpoint contracts; no quality acceptance',
        package_root=str(package), source_sha256=before,
        source_unchanged=before == sources(package),
        reference_sha256=sha(BASE / 'particlegan' / 'feature_cells.py'),
        fixed_input_sha256=sha(FIXES / 'geometry' / 'gpu-inputs.pt'),
        test_script_sha256=sha(Path(__file__)), details=DETAILS,
        failure_details=[dict(test=str(test), traceback=tb) for test, tb in result.failures+result.errors])
    if not receipt['source_unchanged']: receipt['status'] = 'FAIL'
    if DEVICE.type == 'cpu' and receipt['cuda_initialized']: receipt['status'] = 'FAIL'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(event='lineage_checks_complete', status=receipt['status'],
                         tests=result.testsRun, device=str(DEVICE), output=str(output))), flush=True)
    return 0 if receipt['status'] == 'PASS' else 1
