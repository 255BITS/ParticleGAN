"""Stdlib-only API helper/input binding after genuine fixtures and composition."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-production'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collect(value, base, protected):
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and isinstance(item, str) and len(item) == 64:
                path = Path(name)
                if path.is_absolute() or (base / path).is_file():
                    path = path if path.is_absolute() else base / path
                    assert sha(path) == item, str(path)
                    assert str(path) not in protected or protected[str(path)] == item
                    protected[str(path)] = item
            collect(item, base, protected)
    elif isinstance(value, list):
        for item in value:
            collect(item, base, protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--hook-receipt', type=Path, required=True)
    parser.add_argument('--hook-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protected = {}
    source_paths = [OWNER / 'READY.json', OWNER / 'FROZEN.json', OWNER / 'SOURCE-FROZEN.json',
        ROOT / 'quality/ra10/COMPOSITION.json', args.hook_receipt, args.hook_freeze,
        HERE / 'SKELETON-FROZEN.json', HERE / 'SOURCE-REVIEW-FINAL-FROZEN.json',
        HERE / 'FIXTURE-V2-SOURCE-REVIEW-FROZEN.json']
    records = {}
    for path in source_paths:
        value = json.loads(path.read_text())
        collect(value, path.parent, protected)
        protected[str(path.resolve())] = sha(path)
        records[str(path.resolve())] = value
    ready = records[str((OWNER / 'READY.json').resolve())]
    composition = records[str((ROOT / 'quality/ra10/COMPOSITION.json').resolve())]
    hooks = records[str(args.hook_receipt.resolve())]
    assert ready['status'] == 'FROZEN_CPU_QUALIFIED' and hooks['status'] == 'PASS'
    assert ready['backend_schema'] == 9 and ready['trainer_schema'] == 5
    assert ready['package_source_sha256'] == composition['source_sha256']
    assert ready['package_sha256'] == composition['package_sha256']
    package = ROOT / 'pkg-CB64-RA10'
    actual = {str(p.relative_to(package / 'particlegan')): sha(p)
              for p in sorted((package / 'particlegan').rglob('*.py'))}
    assert actual == composition['source_sha256'] and len(actual) == 30
    proposal = Path(ready['package_root'])
    for name, digest in actual.items():
        for target in (proposal / 'particlegan' / name, package / 'particlegan' / name):
            assert sha(target) == digest
            protected[str(target)] = digest
    config = ROOT / 'configs/overrides-CB64-RA10.json'
    baseline_config = ROOT / 'configs/overrides-CB64-RA9.json'
    assert config.read_bytes() == baseline_config.read_bytes()
    assert sha(config) == composition['config_sha256'] == ready['config_sha256']
    for path in (config, baseline_config, Path(ready['config_path'])):
        assert path.read_bytes() == config.read_bytes()
        protected[str(path)] = sha(path)
    fixtures = {}
    for case in ('grid', 'toy'):
        folder = args.fixture_root / case
        fixtures[case] = {key: str(folder / name) for key, name in
            (('before','before.pt'),('after','after.pt'),('provenance','provenance.json'),
             ('trace','trace.json'),('hook','hook-state.pt'))}
        for path in folder.iterdir():
            if path.is_file():
                protected[str(path.resolve())] = sha(path)
        provenance = json.loads((folder / 'provenance.json').read_text())
        collect(provenance, folder, protected)
        assert provenance['case'] == case and provenance['device'] == 'cpu'
        assert provenance['completed_steps_before'] == 0 and provenance['completed_steps_after'] == 1
        assert provenance['no_optimizer_or_GAN_gradient_step'] is True
    helpers = [HERE / 'check_api.py', HERE / 'check_api_skeleton.py', Path(__file__),
        HERE / 'FINAL-PROTOCOL.md', ROOT / 'quality/ra8/integration-contract/check_api.py',
        Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py')]
    for path in helpers:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        protected[str(path.resolve())] = sha(path)
    result = dict(status='PRE_EXECUTION_API_SOURCE_INPUT_FROZEN', created_UTC=datetime.now(timezone.utc).isoformat(),
        backend_schema=9, trainer_schema=5, package_root=str(package),
        package_sha256=composition['package_sha256'], config_path=str(config), config_sha256=sha(config),
        owner_ready=str(OWNER / 'READY.json'), root_composition=str(ROOT / 'quality/ra10/COMPOSITION.json'),
        lineage_hook_receipt=str(args.hook_receipt.resolve()), fixtures=fixtures,
        protected_sha256=dict(sorted(protected.items())),
        scope='Raw-byte/stdlib binding only; no Torch/PT interpretation, forward, sample, test or optimizer step')
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], guarded_files=len(protected), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
