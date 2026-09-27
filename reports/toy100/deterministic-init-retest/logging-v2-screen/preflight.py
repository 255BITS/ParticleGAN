"""Source/fixture verification and optional CPU initialization-only checks."""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import importlib
import json
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(package_root, declaration_path):
    package_root = Path(package_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration = json.loads(declaration_path.read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    required = {'candidate', 'algorithm_source', 'initializer_commit', 'package_sha256',
                'recipe_overrides', 'serial_backward_argument', 'evaluation_generate',
                'initial_optimizer_state', 'optimizer_step_devices', 'port_changes'}
    if not required <= declaration.keys():
        raise ValueError(f'missing candidate declaration fields: {sorted(required - declaration.keys())}')
    if declaration['initializer_commit'] != protocol['initializer_commit']:
        raise ValueError('initializer pin differs')
    if declaration['evaluation_generate'] not in ('plain', 'indexed'):
        raise ValueError('unknown native generation binding')
    if declaration['initial_optimizer_state'] not in ('native_lazy', 'declared_eager'):
        raise ValueError('undeclared optimizer initialization')
    if (set(declaration['optimizer_step_devices']) != {'G', 'D'} or
            any(v not in ('cpu', 'parameter') for v in declaration['optimizer_step_devices'].values())):
        raise ValueError('declare G/D Adam scalar devices as cpu or parameter; no counter repair')
    if declaration['initial_optimizer_state'] == 'native_lazy' and any(
            v != 'cpu' for v in declaration['optimizer_step_devices'].values()):
        raise ValueError('ordinary noncapturable lazy Adam uses native CPU counters')
    if type(declaration['serial_backward_argument']) is not bool:
        raise ValueError('execution mode must be explicit')
    for name, value in protocol['source_sha256'].items():
        if sha((ROOT / name).read_bytes()) != value:
            raise ValueError(f'frozen host source drift: {name}')
    actual_names = {str(p.relative_to(package_root)) for p in (package_root / 'particlegan').rglob('*.py')}
    if actual_names != set(declaration['package_sha256']):
        raise ValueError('candidate manifest must seal the complete Python package')
    for name, value in declaration['package_sha256'].items():
        if sha((package_root / name).read_bytes()) != value:
            raise ValueError(f'candidate package drift: {name}')
    # The numerical initializer is pinned independently from candidate logic.
    initializer = {
        'particlegan/initialization.py': 'b769638eecc119b22138cd52fa91c56bb0ad3515aab3a567cfef8708dc81a366',
        'particlegan/qr_bz_pq_init.py': '3c7f4677d43f19e769453835b1619a17031cef5a5d3f72e9ecb731f310518ce3',
    }
    for name, value in initializer.items():
        if declaration['package_sha256'].get(name) != value:
            raise ValueError(f'candidate changed the declared initializer: {name}')
    source = ROOT / 'mode-hold-source'
    extracted = ast.parse((source / 'frozen_host.py').read_text())
    with zipfile.ZipFile(source / 'frozen-mode-hold-host.zip') as archive:
        for item in json.loads((source / 'extraction.json').read_text()):
            original = archive.read(item['original']).decode()
            span = ''.join(original.splitlines(keepends=True)[item['first_line'] - 1:item['last_line']])
            if sha(span.encode()) != item['source_span_sha256']:
                raise ValueError(f'host source span drift: {item["name"]}')
            def named(tree):
                return next(n for n in tree.body if getattr(n, 'name', None) == item['name'] or
                    isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == item['name'] for t in n.targets))
            if ast.dump(named(ast.parse(original))) != ast.dump(named(extracted)):
                raise ValueError(f'extracted host differs: {item["name"]}')
    fixture = json.loads((source / 'fixture-receipt.json').read_text())
    raw = gzip.decompress((source / 'batch-receipts.jsonl.gz').read_bytes())
    if sha(raw) != fixture['batch_uncompressed_sha256']:
        raise ValueError('old sampling fixture changed')
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if [r['step'] for r in rows] != list(range(1, 1201)):
        raise ValueError('incomplete frozen batch receipt sequence')
    # Parse all harness sources without importing Torch or compiling bytecode.
    for name in ('mode_hold_harness.py', 'mode_hold_contract.py', 'init_contract.py', 'preflight.py'):
        ast.parse((ROOT / name).read_text())
    return declaration, protocol, fixture, rows, dict(
        status='PASS', scope='stdlib-only package/source/fixture audit',
        candidate=declaration['candidate'], package_files=len(actual_names),
        declaration_sha256=sha(declaration_path.read_bytes()),
        protocol_sha256=sha((ROOT / 'protocol.json').read_bytes()),
        exact_host_definitions=len(json.loads((source / 'extraction.json').read_text())),
        frozen_batch_receipts=1200,
        harness_sha256={n: sha((ROOT / n).read_bytes()) for n in
                       ('mode_hold_harness.py', 'mode_hold_contract.py', 'init_contract.py', 'preflight.py')},
        sampling_runtime_check='NOT_RUN: exact CUDA draw parity must pass before quality execution')


def import_package(package_root, declaration):
    if any(n == 'particlegan' or n.startswith('particlegan.') for n in sys.modules):
        raise RuntimeError('fresh process required: package already imported')
    sys.path.insert(0, str(Path(package_root).resolve()))
    package = importlib.import_module('particlegan')
    verify_imports(package_root, declaration)
    return package


def verify_imports(package_root, declaration):
    root = Path(package_root).resolve()
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'particlegan' or name.startswith('particlegan.'):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(root))
            value = sha(path.read_bytes())
            if declaration['package_sha256'].get(relative) != value:
                raise RuntimeError(f'package import escaped seal: {name}')
            imports[name] = dict(path=str(path), sha256=value)
    return imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--cpu-init', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    declaration, _, _, _, result = verify(args.package_root, args.declaration)
    if args.cpu_init:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight refuses an initialized CUDA context')
        package = import_package(args.package_root, declaration)
        from mode_hold_contract import load_host
        from init_contract import check
        result['cpu_initialization'] = check(torch, package, load_host(), declaration)
        result['imports'] = verify_imports(args.package_root, declaration)
        result['torch'] = str(torch.__version__)
        result['cuda_initialized'] = torch.cuda.is_initialized()
        if result['cuda_initialized']:
            raise AssertionError('CPU initialization touched CUDA')
    else:
        assert 'torch' not in sys.modules
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cpu_initialization', 'imports')}, indent=2))


if __name__ == '__main__':
    main()
