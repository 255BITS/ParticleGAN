"""Port a sealed API candidate by three-way source merge; never execute it.

Unknown conflicts are retained and rejected. Reviewed resolutions are usable
only for their exact candidate/old-base/new-base file hashes. No working tree,
old archive, initializer hook, optimizer state, or quality score is modified.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
OLD = 'fa511ce010120b502f494d717d01b14b8551eed8'
NEW = '25751c0864dd8259b00c5804f600cd41cce6e4cf'
INITIALIZER = 'c720645ecae6b648e9fc6034e9d6b48ccff06ed3'


def sha(data):
    return None if data is None else hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def git_package(repo, commit):
    resolved = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=repo, text=True).strip()
    if resolved != commit:
        raise ValueError('full immutable commit IDs required')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'particlegan'], cwd=repo, text=True).splitlines()
    return {name: subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo)
            for name in names if name.endswith('.py')}


def candidate_package(path, declaration):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive paths')
        package = {}
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('unsafe archive path')
            if name.startswith('particlegan/') and name.endswith('.py'):
                package[name] = archive.read(name)
    declared = {n: h for n, h in declaration['source_sha256'].items()
                if n.startswith('particlegan/') and n.endswith('.py')}
    if {n: sha(b) for n, b in package.items()} != declared:
        raise ValueError('complete candidate package does not match its declaration')
    return package


def methods(source):
    result = {}
    def visit(body, prefix=''):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = node
    visit(ast.parse(source).body)
    return result


def body_without_doc(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def preserved_statements(old, new):
    sequence = iter(ast.dump(n) for n in body_without_doc(new))
    return all(any(value == ast.dump(n) for value in sequence) for n in body_without_doc(old))


def audit_initialization_port(candidate, merged, new):
    """Source invariants for this reviewed port, not a quality qualification."""
    changed = sorted(n for n, b in candidate.items() if merged.get(n) != b)
    assert set(changed) <= {'particlegan/__init__.py', 'particlegan/recipes.py', 'particlegan/training.py'}
    for path, allowed in [('particlegan/training.py', {'GANTrainer.load_state_dict'}),
                          ('particlegan/recipes.py', {'Recipe.__post_init__', 'Recipe.make_prior', 'Recipe.make_optimizers'})]:
        old_methods, new_methods = methods(candidate[path]), methods(merged[path])
        assert old_methods.keys() == new_methods.keys()
        for name in old_methods.keys() - allowed:
            assert ast.dump(old_methods[name]) == ast.dump(new_methods[name]), (path, name)
    before, after = methods(candidate['particlegan/recipes.py']), methods(merged['particlegan/recipes.py'])
    for name in ('Recipe.__post_init__', 'Recipe.make_optimizers'):
        assert preserved_statements(before[name], after[name]), name
    class RemoveInitialization(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'initialize':
                assert len(node.args) == 1 and not node.keywords
                return node.args[0]
            return node
    restored = RemoveInitialization().visit(copy.deepcopy(after['Recipe.make_prior']))
    restored.body = [n for n in body_without_doc(restored)
                     if not (isinstance(n, ast.FunctionDef) and n.name == 'initialize')
                     and not (isinstance(n, ast.ImportFrom) and n.module is None
                              and [a.name for a in n.names] == ['initialization'])]
    prior = copy.deepcopy(before['Recipe.make_prior'])
    prior.body = body_without_doc(prior)
    assert ast.dump(restored) == ast.dump(prior), 'prior algorithm differs beyond initializer wrapper'
    factory = after['Recipe.make_prior']
    inner = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize')
    expected = next(n for n in methods(new['particlegan/recipes.py'])['Recipe.make_prior'].body
                    if isinstance(n, ast.FunctionDef) and n.name == 'initialize')
    assert ast.dump(inner) == ast.dump(expected)
    # ParticlePrior is byte-identical: width/shear allocate z-independent zeros.
    # GANTrainer.__init__ is AST-identical: observe_prior runs after make_prior.
    return dict(status='SOURCE_INVARIANTS_PASS_CPU_PREFLIGHT_REQUIRED',
                candidate_files_changed_only_for_init=changed,
                training_update_generation_sampling_controller_and_checkpoint_schema_unchanged=True,
                geometry_order='exact public R2 wrapper runs before make_prior returns; candidate-specific constructor/derived-state review remains required',
                initializer_rng_runtime_check='required CPU-only preflight; not executed by port tool')


def run(args):
    if args.old_base != OLD or args.new_base != NEW:
        raise ValueError('this reviewed tool pins old fa511 and merged API 25751')
    row = None
    if args.inventory:
        inventory = json.loads(args.inventory.read_text())
        row = next(r for r in inventory['candidate_rows'] if r['candidate'] == args.candidate)
        if row['lane'] == 'reference':
            raise ValueError('released reference is a separate public control, not a candidate port')
        args.source_zip = Path(row['source_authority']['path'])
        args.declaration = Path(row['configuration_authority']['path'])
        assert sha(args.source_zip.read_bytes()) == row['source_authority']['sha256']
        assert sha(args.declaration.read_bytes()) == row['configuration_authority']['sha256']
    if args.source_zip is None or args.declaration is None:
        raise ValueError('source ZIP and declaration, or inventory, are required')
    declaration = json.loads(args.declaration.read_text())
    if row is None and declaration.get('candidate') != args.candidate:
        raise ValueError('explicit candidate must match source declaration')
    base, new = git_package(args.git_repo, args.old_base), git_package(args.git_repo, args.new_base)
    candidate = candidate_package(args.source_zip, {'source_sha256': row['package_files']} if row else declaration)
    if not set(base) <= set(candidate):
        raise ValueError('candidate snapshot omits old package files; deletion requires explicit review')
    resolutions = json.loads(args.resolutions.read_text()) if args.resolutions else {'resolutions': {}}
    if resolutions.get('resolutions'):
        if (resolutions['candidate'], resolutions['old_base'], resolutions['new_base']) != (args.candidate, OLD, NEW):
            raise ValueError('resolution identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'port-tool.py').write_bytes(Path(__file__).read_bytes())
    receipts, merged, conflicts = {}, {}, []
    merge_dir = args.output / 'merge-inputs'
    merge_dir.mkdir()
    for name in sorted(set(base) | set(candidate) | set(new)):
        old, current, upstream = base.get(name), candidate.get(name), new.get(name)
        rec = dict(old_base_sha256=sha(old), candidate_sha256=sha(current), new_base_sha256=sha(upstream))
        if current == old:
            value, rec['method'] = upstream, 'new_base_candidate_unchanged'
        elif upstream == old or current == upstream:
            value, rec['method'] = current, 'candidate_preserved'
        elif None in (old, current, upstream):
            conflicts.append(name)
            rec['method'] = 'rejected_add_delete_conflict'
            value = None
        else:
            folder = merge_dir / name
            folder.mkdir(parents=True)
            for label, data in [('candidate', current), ('old-base', old), ('new-base', upstream)]:
                (folder / label).write_bytes(data)
            result = subprocess.run(['git', 'merge-file', '--stdout', '--diff3',
                '-L', 'candidate', '-L', 'old-base', '-L', 'new-base',
                str(folder / 'candidate'), str(folder / 'old-base'), str(folder / 'new-base')], capture_output=True)
            (folder / 'three-way-result').write_bytes(result.stdout)
            if result.returncode < 0 or result.returncode > 127:
                raise RuntimeError(result.stderr.decode())
            rec['merge_exit_code'] = result.returncode
            value, rec['method'] = result.stdout, 'git_merge_file_clean'
            if result.returncode:
                resolution = resolutions['resolutions'].get(name)
                if resolution and all(resolution[k] == rec[k] for k in ('candidate_sha256', 'old_base_sha256', 'new_base_sha256')):
                    path = args.resolutions.parent / resolution['resolution_file']
                    value = path.read_bytes()
                    if sha(value) != resolution['resolution_sha256']:
                        raise ValueError('reviewed resolution content changed')
                    rec.update(method='exact_hash_reviewed_resolution', resolution=resolution)
                else:
                    conflicts.append(name)
                    rec['method'] = 'rejected_unreviewed_conflict'
        if value is not None:
            merged[name] = value
            rec['output_sha256'] = sha(value)
        receipts[name] = rec
    manifest = dict(schema=1, candidate=args.candidate, old_base=OLD, new_api_base=NEW,
                    initializer_commit=INITIALIZER, source_zip_sha256=sha(args.source_zip.read_bytes()),
                    source_declaration_sha256=sha(args.declaration.read_bytes()),
                    port_tool_sha256=sha(Path(__file__).read_bytes()),
                    resolution_manifest_sha256=sha(args.resolutions.read_bytes()) if args.resolutions else None,
                    files=receipts, unresolved_conflicts=conflicts, no_quality_inherited=True)
    if row:
        dump(args.output / 'inventory-row.json', row)
        manifest['inventory_sha256'] = sha(args.inventory.read_bytes())
        manifest['inventory_row_sha256'] = sha((args.output / 'inventory-row.json').read_bytes())
    if conflicts:
        manifest['status'] = 'REJECTED_UNRESOLVED_CONFLICTS_NO_PACKAGE_EMITTED'
        dump(args.output / 'port-manifest.json', manifest)
        return manifest, 2
    for name, data in merged.items():
        if any(marker in data for marker in (b'<<<<<<< ', b'>>>>>>> ', b'||||||| ')):
            raise ValueError('conflict marker remains: ' + name)
        ast.parse(data, filename=name)
    initializer_files = sorted(set(new) - set(base))
    assert len(initializer_files) == 10
    assert all(merged[n] == new[n] for n in initializer_files), 'initializer changed'
    manifest['exact_new_initializer_files'] = {n: sha(merged[n]) for n in initializer_files}
    manifest['source_audit'] = audit_initialization_port(candidate, merged, new)
    manifest['status'] = 'SOURCE_PORTED_NOT_QUALITY_QUALIFIED'
    package = args.output / 'package'
    for name, data in merged.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with zipfile.ZipFile(args.output / 'package.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(merged.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest['package_sha256'] = {n: sha(b) for n, b in sorted(merged.items())}
    manifest['package_zip_sha256'] = sha((args.output / 'package.zip').read_bytes())
    diff = ''.join(''.join(difflib.unified_diff(candidate.get(n, b'').decode().splitlines(True), b.decode().splitlines(True),
                                              fromfile='candidate/' + n, tofile='ported/' + n)) for n, b in sorted(merged.items()))
    (args.output / 'initialization-port.patch').write_text(diff)
    (args.output / 'source-declaration.json').write_bytes(args.declaration.read_bytes())
    dump(args.output / 'port-manifest.json', manifest)
    if row or args.candidate == 'API-DV16':
        recipe = {**(row['declared_recipe'] if row else declaration['recipe']),
                  'num_particles': 12, 'z_dim': 4, 'batch_size': 128, 'initialization': 'batch_feature_zero'}
        generation = methods(merged['particlegan/training.py'])['GANTrainer._generate']
        indexed = any(a.arg == 'indices' for a in generation.args.args + generation.args.kwonlyargs)
        eager = recipe.get('adam_eager_state', False)
        if args.candidate == 'API-RP1-CUDA-EAGER':
            # This historical diagnostic mutated Adam outside the package.
            # Preserve its separate identity without silently scoring lazy RP1.
            manifest['integration_blocker'] = 'RP1 eager state was worker-owned and is not reproduced by this unchanged package; separate ownership review required.'
            dump(args.output / 'port-manifest.json', manifest)
            return manifest, 0
        harness = dict(candidate=args.candidate + '-new-init', algorithm_source=dict(kind='three_way_source_port',
            candidate=args.candidate, source_zip=str(args.source_zip.resolve()),
            source_zip_sha256=manifest['source_zip_sha256'], old_base=OLD, new_api_base=NEW,
            port_manifest_sha256=sha((args.output / 'port-manifest.json').read_bytes())),
            initializer_commit=INITIALIZER, package_sha256=manifest['package_sha256'],
            recipe_overrides={k: v for k, v in recipe.items() if k != 'name'}, resolved_recipe=recipe,
            serial_backward_argument=row['source_accepts_serial_backward_argument'] if row else True,
            evaluation_generate='indexed' if indexed else 'plain',
            initial_optimizer_state='declared_eager' if eager else 'native_lazy',
            optimizer_step_devices={role: 'parameter' if eager else 'cpu' for role in ('G', 'D')},
            port_changes=['Library-owned deterministic initialization only; exact-hash reviewed merge resolutions.',
                          'Compatible saved initialization metadata restored; continuous policy and serial/schema guards retained.'],
            status='SOURCE_ONLY_REQUIRES_CPU_PREFLIGHT_AND_NEW_QUALITY')
        dump(args.output / 'candidate-declaration.json', harness)
    return manifest, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-repo', type=Path, required=True)
    parser.add_argument('--source-zip', type=Path)
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--old-base', default=OLD)
    parser.add_argument('--new-base', default=NEW)
    parser.add_argument('--resolutions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, code = run(args)
    assert 'torch' not in sys.modules
    print(json.dumps({k: manifest[k] for k in ('status', 'candidate', 'unresolved_conflicts')}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
