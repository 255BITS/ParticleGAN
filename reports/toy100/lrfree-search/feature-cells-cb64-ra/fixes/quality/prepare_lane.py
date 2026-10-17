"""Prepare a new frozen lane; toy and grid use the unchanged CUDA scorers."""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'prepare_validation4.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--ready', type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'CB64-[A-Z0-9_-]+', args.variant)
    ready = args.ready.resolve()
    assert ready.is_relative_to(ROOT) and ready.exists()
    package = ROOT / ('pkg-' + args.variant)
    config = ROOT / 'configs' / ('overrides-' + args.variant + '.json')
    assert package.exists() and config.exists()
    slug = args.variant.lower()
    output = ROOT / ('validation-' + slug)
    script = ROOT / ('prepare-' + slug + '-quality.py')
    assert not output.exists() and not script.exists()
    source = TEMPLATE.read_text().replace('CB64-RA4', args.variant)
    source = source.replace("OUT = ROOT / 'validation-ra4'", f'OUT = ROOT / {output.name!r}')
    source = source.replace('integration/iteration-4/READY.json', str(ready.relative_to(ROOT)))
    insertion = '''
    # Declare the existing indexed API before the new lane is frozen.
    api = ast.parse((PACKAGE / 'particlegan/training.py').read_text())
    trainer = next(n for n in api.body if isinstance(n, ast.ClassDef) and n.name == 'GANTrainer')
    generate = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == '_generate')
    assert len(generate.args.args) >= 6 and generate.args.args[5].arg == 'indices'
    collector = screens / 'collect.py'
    tree = ast.parse(collector.read_text())
    original_ast = ast.dump(tree, include_attributes=False)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'expected_options' for t in n.targets)]
    assert len(assignments) == 1
    value = next(k.value for k in assignments[0].value.keywords if k.arg == 'evaluation_generate')
    assert isinstance(value, ast.Constant) and value.value == 'plain'
    value.value = 'indexed'
    adapted_ast = ast.dump(tree, include_attributes=False)
    adapted_source = ast.unparse(tree) + '\\n'
    value.value = 'plain'
    assert ast.dump(tree, include_attributes=False) == original_ast
    collector.write_text(adapted_source)
    write(screens / 'DECLARED-API-EXPECTATION.json', dict(
        expected_generation_api='indexed', basis='fifth positional parameter named indices',
        original_ast_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),
        adapted_ast_sha256=hashlib.sha256(adapted_ast.encode()).hexdigest(),
        exact_change='collect.expected_options.evaluation_generate: plain -> indexed',
        all_other_collector_checks_exact=True, quality_gates_unchanged=True,
        declared_before_numerical_execution=True))
    (OUT / 'QUALITY-TARGETS.md').write_bytes((ROOT / 'quality/PROTOCOL.md').read_bytes())
'''
    marker = "    # Preserve all original fixture/scorer input hashes, replacing candidate and"
    assert source.count(marker) == 1
    source = source.replace(marker, insertion + '\n' + marker)
    new_jobs = f'''def jobs():
    def learned(problem):
        return dict(name=f'learned-{{problem}}-{args.variant}', threads=2,
            command=[PYTHON, '-u', str(ROOT / 'learned/run_training.py'),
                     '--problem', problem, '--variant', {args.variant!r}],
            result=ROOT / 'learned/training' / problem / {args.variant!r} / 'result.json')
    yield learned('toy')
    yield dict(name='screen-grid100', threads=1,
        command=[PYTHON, '-u', str(ROOT / 'screens/run_screen.py'), '--task', 'grid100'],
        result=ROOT / 'screens/runs/grid100/result.json')
    yield learned('mnist')
    yield dict(name='replay-{args.variant}', threads=2,
        command=[PYTHON, '-u', str(ROOT / 'learned/replay.py'), {args.variant!r}],
        result=ROOT / 'learned/replay-{args.variant}.json')
    for task in TASKS:
        if task == 'grid100':
            continue
        yield dict(name=f'screen-{{task}}', threads=1,
            command=[PYTHON, '-u', str(ROOT / 'screens/run_screen.py'), '--task', task],
            result=ROOT / 'screens/runs' / task / 'result.json')
'''
    launcher_insertion = f'''
    # Only the execution order changes: toy, grid, then original regressions.
    launcher_tree = ast.parse(launcher)
    old_jobs = next(n for n in launcher_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'jobs')
    new_jobs = ast.parse({new_jobs!r}).body[0]
    launcher_tree.body[launcher_tree.body.index(old_jobs)] = new_jobs
    launcher = ast.unparse(launcher_tree) + '\\n'
'''
    marker = "    (OUT / 'launch.py').write_text(launcher)"
    assert source.count(marker) == 1
    source = source.replace(marker, launcher_insertion + '\n' + marker)
    source = source.replace(
        'The first invocation may stop after\\nlearned toy, MNIST and replay (--through 3);',
        'The first invocation stops after learned toy (--through 1);\\n'
        'a passing toy proceeds to full grid (--through 2);')
    marker = '    external[str(Path(__file__))] = sha(Path(__file__))'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n' +
        f'    external[{str(Path(__file__).resolve())!r}] = sha({str(Path(__file__).resolve())!r})\n' +
        f'    external[{str(TEMPLATE)!r}] = sha({str(TEMPLATE)!r})\n')
    source = source.replace(marker, marker + '\n' +
        f'    external[{str(ROOT / "quality/nested_slot.py")!r}] = sha({str(ROOT / "quality/nested_slot.py")!r})\n')
    compile(source, str(script), 'exec')
    script.write_text(source)
    subprocess.run([sys.executable, '-B', str(script)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
