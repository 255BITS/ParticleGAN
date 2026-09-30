"""Read-only structural audit; no lane preparation or numerical execution."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE = ROOT.parent / 'feature-cells-cuda-retest-20260929'
HERE = Path(__file__).resolve().parent
HELPER = ROOT / 'quality/prepare_lane.py'
TEMPLATE = ROOT / 'prepare_validation4.py'
VARIANT = 'CB64-LANE-AUDIT'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def run_statements(statements, namespace):
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<read-only construction>', 'exec'), namespace)


def main():
    inputs = [HELPER, TEMPLATE, ROOT / 'quality/PROTOCOL.md',
        ROOT / 'validation-ra4/launch.py', ROOT / 'pkg-CB64-RA4/particlegan/training.py',
        BASE / 'launch.py', BASE / 'screens/collect.py', BASE / 'screens/lane.py',
        Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')]
    before = {str(p): sha(p) for p in inputs}
    # run_path loads only stdlib definitions; main(), writes and subprocesses are not called.
    namespace = runpy.run_path(str(HELPER))
    main_node = function(ast.parse(HELPER.read_text()), 'main')
    start = next(i for i, n in enumerate(main_node.body) if assignment(n, 'source'))
    end = next(i for i, n in enumerate(main_node.body) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'compile')
    output = ROOT / ('validation-' + VARIANT.lower())
    script = ROOT / ('prepare-' + VARIANT.lower() + '-quality.py')
    ready = ROOT / 'quality/lane-audit/READY.json'
    namespace.update(args=SimpleNamespace(variant=VARIANT), ready=ready, output=output, script=script)
    run_statements(main_node.body[start:end + 1], namespace)
    generated = namespace['source']
    generated_tree = ast.parse(generated)
    generated_main = function(generated_tree, 'main')
    paths = {n.targets[0].id: ast.unparse(n.value) for n in generated_tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ('OUT', 'PACKAGE', 'CONFIG')}
    assert paths == dict(OUT=f'ROOT / {output.name!r}', PACKAGE=f"ROOT / 'pkg-{VARIANT}'",
        CONFIG=f"ROOT / 'configs/overrides-{VARIANT}.json'")
    assert 'integration/iteration-4/READY.json' not in generated
    assert str(ready.relative_to(ROOT)) in generated
    assert 'first invocation stops after learned toy' in generated
    assert '(--through 1)' in generated and '(--through 2)' in generated
    for name in ('adapted_source', 'launcher'):
        suffixes = [n.value.right.value for n in ast.walk(generated_main)
            if assignment(n, name) and isinstance(n.value, ast.BinOp)
            and isinstance(n.value.right, ast.Constant)]
        assert suffixes[-1] == '\n'

    # Execute only launcher string/AST construction, stopping before write_text().
    start = next(i for i, n in enumerate(generated_main.body) if assignment(n, 'launcher'))
    end = next(i for i, n in enumerate(generated_main.body) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == 'write_text' and isinstance(n.value.func.value, ast.BinOp)
        and isinstance(n.value.func.value.right, ast.Constant)
        and n.value.func.value.right.value == 'launch.py')
    local = dict(namespace, BASE=BASE, ROOT=output, OUT=output)
    run_statements(generated_main.body[start:end], local)
    launcher = local['launcher']
    compile(launcher, '<new launcher>', 'exec')
    launcher_tree = ast.parse(launcher)
    jobs_node = function(launcher_tree, 'jobs')
    tasks_node = next(n for n in launcher_tree.body if assignment(n, 'TASKS'))
    context = dict(ROOT=output, PYTHON='/tmp/pr38-default-env/bin/python')
    run_statements([tasks_node, jobs_node], context)
    jobs = list(context['jobs']())
    old_launcher_tree = ast.parse((ROOT / 'validation-ra4/launch.py').read_text().replace('CB64-RA4', VARIANT))
    old_jobs_node = function(old_launcher_tree, 'jobs')
    old_context = dict(ROOT=output, PYTHON=context['PYTHON'], TASKS=context['TASKS'])
    run_statements([old_jobs_node], old_context)
    old_jobs = list(old_context['jobs']())
    assert len(jobs) == len(old_jobs) == 19
    assert len([j for j in jobs if j['name'].startswith('screen-')]) == 16
    assert {j['name']: j for j in jobs} == {j['name']: j for j in old_jobs}
    launcher_tree.body[launcher_tree.body.index(jobs_node)] = old_jobs_node
    assert dump(launcher_tree) == dump(old_launcher_tree)

    # Run the actual inserted collector transform, replacing only its writes with capture.
    start = next(i for i, n in enumerate(generated_main.body) if assignment(n, 'tree'))
    end = next(i for i, n in enumerate(generated_main.body) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == 'write_text' and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == 'collector')
    collector_source = (BASE / 'screens/collect.py').read_text().replace('CB64-RA', VARIANT)
    class MemoryCollector:
        def read_text(self):
            return collector_source
    collector_context = dict(local, collector=MemoryCollector())
    run_statements(generated_main.body[start:end], collector_context)
    adapted = collector_context['adapted_source']
    compile(adapted, '<adapted collector>', 'exec')
    adapted_tree = ast.parse(adapted)
    adapted_collect = function(adapted_tree, 'collect')
    expected = next(n for n in ast.walk(adapted_collect) if assignment(n, 'expected_options'))
    value = next(k.value for k in expected.value.keywords if k.arg == 'evaluation_generate')
    assert value.value == 'indexed'
    value.value = 'plain'
    assert dump(adapted_tree) == dump(ast.parse(collector_source))

    # Exact original resolver and native fifth-argument dispatch are retained.
    screen_tree = ast.parse(inputs[-1].read_text())
    resolver = function(screen_tree, 'resolve_options')
    signature_check = "'indices' in inspect.signature(package.GANTrainer._generate).parameters"
    assert signature_check in ast.unparse(resolver)
    training_tree = ast.parse((ROOT / 'pkg-CB64-RA4/particlegan/training.py').read_text())
    trainer = next(n for n in training_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GANTrainer')
    generate = function(trainer, '_generate')
    assert len(generate.args.args) >= 6 and generate.args.args[5].arg == 'indices'
    native = function(screen_tree, 'run_native')
    assert 'arguments.append(indices)' in ast.unparse(native)
    assert 'trainer._generate(*arguments)' in ast.unparse(native)
    assert str(HELPER) in generated and str(TEMPLATE) in generated
    assert 'QUALITY-TARGETS.md' in generated and 'DECLARED-API-EXPECTATION.json' in generated
    after = {str(p): sha(p) for p in inputs}
    assert after == before
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU stdlib read-only construction; no actual lane preparation or numerical execution',
        source_sha256=before, generated_preparer_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        checks=dict(compile_helper_and_generated_sources=True, correct_newline_literals=True,
            candidate_output_config_ready_paths=True, collector_one_literal_AST_undo=True,
            full_launcher_AST_only_jobs_changes=True, all_19_job_specs_preserved=True,
            original_16_screens_preserved=True, original_resolver_and_positional_API_compatible=True,
            helper_template_and_declaration_freeze_coverage=True, all_inputs_unchanged=True),
        job_order=[j['name'] for j in jobs], defects=[],
        limitations=['Actual future candidate/READY/config existence and hashes are checked during preparation and its independent composition audit.',
            'The launcher requires the root supervisor to enforce toy/grid quality phase decisions through --through; it does not stop automatically on a quality FAIL.'])
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], jobs=len(jobs), screens=16, defects=receipt['defects'])))


if __name__ == '__main__':
    main()
