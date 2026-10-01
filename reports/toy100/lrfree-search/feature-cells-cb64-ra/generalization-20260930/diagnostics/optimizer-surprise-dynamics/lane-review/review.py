"""Source and raw hashes only. Never execute any model or scoring code."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2]
ORIGINAL = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
ROTATE = Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py')
OLD_SCREENS = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/screens')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(n):
    return ast.dump(n, include_attributes=False)


class UndoConstruction(ast.NodeTransformer):
    def __init__(self):
        self.counts = {'get_recipe': 0, 'make_prior': 0, 'GANTrainer': 0}

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ('frozen_recipe', 'frozen_prior', 'frozen_trainer'):
            kind = {'frozen_recipe': 'get_recipe', 'frozen_prior': 'make_prior', 'frozen_trainer': 'GANTrainer'}[node.func.id]
            self.counts[kind] += 1
            owner = node.args.pop(0)
            assert isinstance(owner, ast.Name) and owner.id == ('recipe' if kind == 'make_prior' else 'package')
            node.func = ast.Attribute(value=owner, attr=kind, ctx=ast.Load())
        return node


def main():
    prep = json.loads((HERE / 'INPUTS-FROZEN.json').read_text())
    for path, digest in prep['sha256'].items():
        assert sha(Path(path)) == digest, path
    oldlane, lane = STUDY / 'validation-ra13', STUDY / 'validation-ra13-r2'
    frozen = json.loads((lane / 'SOURCE-FREEZE.json').read_text())
    assert len(frozen['hashes']) == 103
    for path, digest in frozen['hashes'].items():
        assert sha(Path(path)) == digest, path
    for name in ('collect.py', 'lane.py'):
        path = OLD_SCREENS / name
        assert frozen['hashes'][str(path)] == sha(path)
    # The numerical adapter, collector and wrappers are unchanged in r2.
    for name in ('screen_current.py', 'current_api_fixtures.py', 'adapt_screen.py', 'collect.py',
                 'run_screen.py', 'run_moving.py', 'run_all.py', 'run_full_suite.py'):
        assert (oldlane / name).read_bytes() == (lane / name).read_bytes(), name
    original, current = ast.parse(ORIGINAL.read_text()), ast.parse((lane / 'screen_current.py').read_text())
    inverse = UndoConstruction()
    current = inverse.visit(current)
    assert inverse.counts == {'get_recipe': 5, 'make_prior': 3, 'GANTrainer': 5}
    current.body = [n for n in current.body if not (isinstance(n, ast.ImportFrom) and n.module == 'current_api_fixtures')]
    orig_harness = next(n for n in original.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'HARNESS' for t in n.targets))
    new_harness = next(n for n in current.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'HARNESS' for t in n.targets))
    new_harness.value = deepcopy(orig_harness.value)
    oldring = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'run_ring')
    newring = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'run_ring')
    oldmeasure = next(n for n in oldring.body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
    newmeasure = next(n for n in newring.body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
    ring_modes = [n for n in ast.walk(newmeasure) if isinstance(n, ast.keyword) and n.arg == 'output_noise']
    assert sorted(n.value.value for n in ring_modes) == [False, True]
    newring.body[newring.body.index(newmeasure)] = deepcopy(oldmeasure)
    assert dump(original) == dump(current), 'undeclared static screen AST delta'
    # Extract source-adapter functions only; no imported runtime or main call.
    tree = ast.parse((lane / 'run_moving.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('replace_once', 'adapted_source')]
    ns = {'ORIGINAL': ROTATE, 'PACKAGE': STUDY / 'pkg-RA13-settled', 'CONFIG': STUDY / 'configs/RA13-settled.json'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<source-adapter-only>', 'exec'), ns)
    moving = ns['adapted_source']()
    assert 'trainer._generate(model, latent, 0., latent_stream, indices=indices)' in moving
    assert "assert len(gate_rows) == 3 and len(ok) == 2" in moving
    # Undo only declared path/API/resource/instrumentation additions.
    for new, old in (
        (f'REPO = {str(STUDY / "pkg-RA13-settled")!r}', "REPO = '/ml2/hypergan/ParticleGAN-pr155-merge'"),
        (f'options = json.load(open({str(STUDY / "configs/RA13-settled.json")!r}))', "options = json.load(open(f'{REPO}/configs/100gaussians/e22-noout.json'))"),
        ('latent, indices = table.sample(n, generator=latent_stream)', 'latent, _ = table.sample(n, generator=latent_stream)'),
        ('trainer._generate(model, latent, 0., latent_stream, indices=indices)', 'trainer._generate(model, latent, 0., latent_stream)'),
        ('torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2, 0); torch.set_num_threads(1); torch.set_num_interop_threads(1)',
         'torch.cuda.set_device(0); torch.set_num_threads(1); torch.set_num_interop_threads(1)'),
        ("        torch.save(trainer.state_dict(), str(owned_output / f'checkpoint-{step:06d}.pt'))\n", ''),
        ("        print('MECHANISM ' + json.dumps(dict(step=step, surprise=None if trainer.surprise is None else trainer.surprise.diagnostics(), backend_selection=trainer.policy._feature_selection.state_dict(), reopen_guard=trainer.policy.reopen_guard.state_dict())), flush=True)\n", ''),
        ("    assert len(gate_rows) == 3 and len(ok) == 2, 'rotation gate requires both turns'\n", ''),
    ):
        assert moving.count(new) == 1, new
        moving = moving.replace(new, old)
    assert moving == ROTATE.read_text(), 'undeclared moving runner delta'
    h = hashlib.sha256()
    for path in sorted((STUDY / 'pkg-RA13-settled/particlegan').rglob('*.py')):
        h.update(str(path.relative_to(STUDY / 'pkg-RA13-settled/particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    assert h.hexdigest() == frozen['package_sha256']
    package = ast.parse((STUDY / 'pkg-RA13-settled/particlegan/training.py').read_text())
    cls = next(n for n in package.body if isinstance(n, ast.ClassDef) and n.name == 'GANTrainer')
    generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate')
    assert [a.arg for a in generate.args.args][-1] == 'indices'
    assert 'rows' in [a.arg for a in generate.args.kwonlyargs]
    collector = (lane / 'collect.py').read_text()
    assert 'native = validator.validate_native(output,task,result,reasons)' in collector
    assert "evaluation_generate='indexed'" in collector and 'eval_output_noise=True' in collector
    for path, digest in prep['sha256'].items():
        assert sha(Path(path)) == digest, path
    receipt = dict(status='PASS', scope='source/metadata only; no numerical model or scorer execution',
        lane=str(lane), source_freeze_sha256=sha(lane / 'SOURCE-FREEZE.json'), source_guards=103,
        superseded_lane_guard_gap='r1 omitted external collect.py/lane.py; r2 pins both unchanged before numerical execution',
        static_screen_full_ast_inverse_exact=True, moving_original_full_byte_inverse_exact=True,
        five_terminal_20k_and_100k_holdout='unchanged screen and directly reused original validate_native',
        primary_sampling='live noisy; ring explicit output_noise=True; native original indexed draw; moving actual row IDs',
        moving_nonvacuity='steps1500/turn_every500/degrees30; exactly3 period rows and2 tested turns',
        package_digest_contract='same ordered relative-particlegan path/NUL/raw-byte algorithm as original screen',
        remaining_blockers=[], no_torch_package_imports_pt_models_forwards_draws_updates_scoring_cuda=True)
    with (HERE / 'receipt.json').open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps(dict(status='PASS', receipt_sha256=sha(HERE / 'receipt.json'))))


if __name__ == '__main__':
    main()
