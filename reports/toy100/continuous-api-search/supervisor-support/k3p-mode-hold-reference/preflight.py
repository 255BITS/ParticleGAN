"""Standard-library-only sealed source and fixture preflight; never imports Torch."""
from pathlib import Path
import ast
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import sys
import zipfile
sys.dont_write_bytecode = True
from execution_contract import EXECUTION, host_cuda, serial_step, validate_checkpoint
from raw_fixture import inspect_checkpoint, sha


def node_named(tree, name):
    matches = [n for n in tree.body if getattr(n, "name", None) == name or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, name
    return matches[0]


def check_context_contracts():
    class Generator:
        def __new__(cls, device="cpu"):
            value = object.__new__(cls)
            value.device = device
            return value

        def __init__(self, device="cpu"):
            pass

    class FakeTorch:
        def __init__(self):
            self.current_device = "cpu"
            self.multithreading = True
            self.autograd = self

        def get_default_device(self):
            return self.current_device

        @contextmanager
        def device(self, name):
            before = self.current_device
            self.current_device = name
            try:
                yield
            finally:
                self.current_device = before

        def is_multithreading_enabled(self):
            return self.multithreading

        @contextmanager
        def set_multithreading_enabled(self, value):
            before = self.multithreading
            self.multithreading = value
            try:
                yield
            finally:
                self.multithreading = before

    FakeTorch.Generator = Generator
    fake = FakeTorch()
    for raise_error in (False, True):
        try:
            with host_cuda(fake):
                assert fake.get_default_device() == "cuda:0"
                assert fake.Generator().device == "cuda:0"
                assert fake.Generator(device="cpu").device == "cpu"
                if raise_error:
                    raise LookupError("deliberate restoration check")
        except LookupError:
            assert raise_error
        assert fake.get_default_device() == "cpu" and fake.Generator is Generator
    for previous in (True, False):
        for raise_error in (False, True):
            fake.multithreading = previous
            try:
                with serial_step(fake):
                    assert not fake.multithreading
                    with serial_step(fake):
                        assert not fake.multithreading
                    if raise_error:
                        raise LookupError("deliberate restoration check")
            except LookupError:
                assert raise_error
            assert fake.multithreading is previous
    envelope = dict(schema=1, execution=EXECUTION, identity={"pinned": "source"}, trainer={"schema": 3}, data_rng=None)
    validate_checkpoint(envelope, envelope["identity"])
    for key, value in (("execution", {"serial_backward": False}), ("identity", {}),
                       ("schema", 2), ("trainer", {"schema": 4})):
        bad = deepcopy(envelope)
        bad[key] = value
        try:
            validate_checkpoint(bad, envelope["identity"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid checkpoint contract accepted")


def verify(bundle=None):
    bundle = Path(__file__).resolve().parent if bundle is None else Path(bundle)
    manifest = json.loads((bundle / 'bundle-sha256.json').read_text())
    assert manifest['schema'] == 1
    actual_files = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()
                    and p.name not in ('bundle-sha256.json', 'preflight-result.json')}
    assert actual_files == set(manifest['files']), 'unexpected or missing bundle files'
    for name, want in manifest['files'].items():
        path = bundle / name
        assert path.resolve().is_relative_to(bundle.resolve())
        assert sha(path.read_bytes()) == want, name
        if path.suffix == '.py':
            compile(ast.parse(path.read_text()), str(path), 'exec')
    d = json.loads((bundle / 'declaration.json').read_text())
    assert d['schema'] == 1 and d['status'] == 'PREPARED_NOT_RUN'
    assert d['requires_independent_source_review_before_run'] is True
    assert d['release']['commit'] == '0ff9a7afe5dcb828239369446cfe71971bce687b'
    assert d['release']['tag'] == 'v0.8.0' and d['release']['patches'] == []
    receipt = json.loads((bundle / 'release-receipt.json').read_text())
    assert receipt['sha256'] == d['release']['source_sha256']
    assert sha((bundle / 'public-k3p-v0.8.0.zip').read_bytes()) == d['release']['archive_sha256']
    with zipfile.ZipFile(bundle / 'public-k3p-v0.8.0.zip') as z:
        assert len(z.namelist()) == 12 and set(z.namelist()) == set(receipt['sha256'])
        for name, want in receipt['sha256'].items():
            assert sha(z.read(name)) == sha((bundle / 'source' / name).read_bytes()) == want
        recipe = node_named(ast.parse(z.read('particlegan/recipes.py')), 'Recipe')
        defaults = {n.target.id: ast.literal_eval(n.value) for n in recipe.body if isinstance(n, ast.AnnAssign)}
        assert json.loads(json.dumps({**defaults, **d['recipe_overrides']})) == d['recipe']
    assert set((bundle / 'source/particlegan').glob('*.py')) == {bundle / 'source' / n for n in receipt['sha256']}
    assert d['recipe_overrides'] == dict(total_steps=1200, num_particles=12, z_dim=4, batch_size=128)
    assert d['recipe']['input_noise_anneal_end'] * 1200 == 120
    assert d['recipe']['output_noise_warmup'] * 1200 == 240
    assert min(1200, d['recipe']['network_lr_horizon_cap']) * d['recipe']['lr_anneal_start'] == 720
    assert sha((bundle / 'frozen-mode-hold-host.zip').read_bytes()) == d['source_preparation']['host_source_zip_sha256']
    with zipfile.ZipFile(bundle / 'frozen-mode-hold-host.zip') as z:
        for name, want in d['source_preparation']['original_source_sha256'].items():
            assert sha(z.read(name)) == sha((bundle / 'original-host' / name).read_bytes()) == want
        plan = json.loads(z.read('benchmarks/transfer_suite/plans/default_comparison.json'))
        assert next(x['spec'] for x in plan if x['spec']['name'] == 'mode_hold') == d['spec']
        for function in ('ring_means', 'sample_ring', 'diversity'):
            current = node_named(ast.parse(z.read('benchmarks/locked_shared/mode_hold.py')), function)
            canonical = node_named(ast.parse(z.read('reports/reversible-precision/canonical-mode_hold.py.txt')), function)
            assert ast.dump(current) == ast.dump(canonical)
        assert z.read('benchmarks/locked_shared/mlp.py') == z.read('reports/reversible-precision/canonical-mlp.py.txt')
    frozen = ast.parse((bundle / 'source/frozen_host.py').read_text())
    extracts = json.loads((bundle / 'extraction.json').read_text())
    for selected in extracts:
        original = (bundle / 'original-host' / selected['original']).read_text()
        span = ''.join(original.splitlines(keepends=True)[selected['first_line'] - 1:selected['last_line']])
        assert sha(span.encode()) == selected['source_span_sha256']
        assert ast.dump(node_named(ast.parse(original), selected['name'])) == ast.dump(node_named(frozen, selected['name']))
    fixture = json.loads((bundle / 'fixture-receipt.json').read_text())
    assert sha((bundle / 'fixture-receipt.json').read_bytes()) == d['fixture_receipt_sha256']
    for who, sources in fixture['source_receipts'].items():
        for name, source in sources.items():
            path = bundle / 'fixtures' / who / name
            assert sha(path.read_bytes()) == source['sha256']
            if name.endswith('.pt'):
                assert inspect_checkpoint(path) == fixture['raw_receipts'][who][name]
    rp5 = gzip.decompress((bundle / 'fixtures/rp5/batch-receipts.jsonl.gz').read_bytes())
    rp7 = gzip.decompress((bundle / 'fixtures/rp7/batch-receipts.jsonl.gz').read_bytes())
    assert rp5 == rp7 and sha(rp5) == fixture['batch_uncompressed_sha256']
    assert [json.loads(line)['step'] for line in rp5.decode().splitlines()] == list(range(1, 1201))
    for key in ('models', 'cpu_rng', 'cuda_rng', 'data_rng'):
        assert fixture['raw_receipts']['rp5']['initial-state.pt'][key] == fixture['raw_receipts']['rp7']['initial-state.pt'][key]
    for who in ('rp5', 'rp7'):
        assert fixture['raw_receipts'][who]['initial-state.pt']['completed_steps'] == 0
        final = fixture['raw_receipts'][who]['final-state.pt']
        assert final['completed_steps'] == 1200
        assert final['data_rng'] == final['streams']['latent_generator'] == fixture['expected_final_data_rng']
    assert d['evaluation']['observations'] == list(range(50, 1201, 50))
    assert d['evaluation']['samples'] == 4096 and d['evaluation']['latent_seed'] == 9
    assert d['evaluation']['final_passing_suffix'] == 5
    worker = ast.parse((bundle / 'worker.py').read_text())
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    recipes = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'get_recipe']
    assert len(recipes) == 1 and {k.arg: ast.literal_eval(k.value) for k in recipes[0].keywords} == d['recipe_overrides']
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'set_default_device') for n in calls)
    assert not any(isinstance(n, ast.Attribute) and ast.unparse(n) == 'torch.optim' for n in ast.walk(worker))
    steps = [n for n in calls if isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'trainer.step']
    serial = [n for n in ast.walk(worker) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Name)
        and i.context_expr.func.id == 'serial_step' for i in n.items)]
    assert len(steps) == len(serial) == 1 and steps[0] in list(ast.walk(serial[0]))
    trainers = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'GANTrainer']
    assert len(trainers) == 1
    assert not any(k.arg == 'serial_backward' for n in trainers for k in n.keywords), 'release has no serial constructor keyword'
    imports = [n for n in ast.walk(worker) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith(('benchmarks', 'reports')) for n in imports)
    check_context_contracts()
    assert 'torch' not in sys.modules, 'source preflight imported Torch'
    return dict(status='PASS', scope='stdlib-only source/fixture inspection; no Torch, model, GPU or training',
        package_files=12, exact_host_definitions=len(extracts), original_checkpoint_receipts=4, matching_batch_receipts=1200,
        checks=['complete seal', 'exact unpatched release bytes and finite recipe', 'original frozen plan/model/scorer source',
                'exact extracted host definitions', 'RP5/RP7 raw initial model/RNG and final caller cursor',
                'all1200 RP5/RP7 batch receipts equal', 'complete public step under restoring serial context',
                'host-only CUDA restoration and nested serial exceptions', 'checkpoint source/execution mode validation'],
        runtime_checks='NOT_RUN: import origins, generated CUDA initialization/draws, native Adam placement and all gates require authorized worker',
        independent_review='REQUIRED_BEFORE_EXECUTION',
        bundle_manifest_sha256=sha((bundle / 'bundle-sha256.json').read_bytes()))


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
