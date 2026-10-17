"""Adapt only construction and sampling API calls in the original screen.

Data, models, stream transactions, scorers, thresholds and budgets stay in
the original source. The adapted source is sealed before numerical execution.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')


class ConstructionAdapter(ast.NodeTransformer):
    def __init__(self):
        self.calls = {'get_recipe': 0, 'make_prior': 0, 'GANTrainer': 0}

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        attr, owner = node.func.attr, node.func.value
        if isinstance(owner, ast.Name) and owner.id == 'package' and attr in ('get_recipe', 'GANTrainer'):
            node.func = ast.Name(id='frozen_recipe' if attr == 'get_recipe' else 'frozen_trainer', ctx=ast.Load())
            node.args.insert(0, ast.Name(id='package', ctx=ast.Load()))
            self.calls[attr] += 1
        elif isinstance(owner, ast.Name) and owner.id == 'recipe' and attr == 'make_prior':
            node.func = ast.Name(id='frozen_prior', ctx=ast.Load())
            node.args.insert(0, ast.Name(id='recipe', ctx=ast.Load()))
            self.calls[attr] += 1
        return node


RING_MEASURE = '''
def measure(value, ema=False, both=False):
    with torch.random.fork_rng(devices=[0]):
        isolated = torch.Generator(device=device).manual_seed(9)
        noisy = host.diversity(value.sample(4096, ema=ema, generator=isolated, output_noise=True), means)
        sigma = ctx.sigma(value)
        if not sigma:
            pair = (noisy, dict(noisy))
        else:
            isolated = torch.Generator(device=device).manual_seed(9)
            clean = host.diversity(value.sample(4096, ema=ema, generator=isolated, output_noise=False), means)
            pair = (clean, noisy)
    if both:
        return pair
    return pair[1 if ctx.options['eval_output_noise'] else 0]
'''


def adapt():
    source = ORIGINAL.read_text()
    tree = ast.parse(source)
    adapter = ConstructionAdapter()
    tree = adapter.visit(tree)
    assert adapter.calls == {'get_recipe': 5, 'make_prior': 3, 'GANTrainer': 5}, adapter.calls
    ring = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_ring')
    measure = next(n for n in ring.body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
    assert any(isinstance(n, ast.Constant) and n.value == 'output_noise_std' for n in ast.walk(measure))
    ring.body[ring.body.index(measure)] = ast.parse(RING_MEASURE).body[0]
    # This file lives outside the frozen harness; preserve its original paths.
    harness = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'HARNESS' for t in n.targets))
    harness.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(ORIGINAL.parent))], keywords=[])
    at = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign))
    tree.body.insert(at, ast.ImportFrom(module='current_api_fixtures',
        names=[ast.alias(name=n) for n in ('frozen_recipe', 'frozen_prior', 'frozen_trainer')], level=0))
    ast.fix_missing_locations(tree)
    adapted = ast.unparse(tree) + '\n'
    compile(adapted, str(ROOT / 'screen_current.py'), 'exec')
    return adapted, adapter.calls


def main():
    adapted, calls = adapt()
    output = ROOT / 'screen_current.py'
    assert not output.exists(), 'retain every prepared adapter revision'
    output.write_text(adapted)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = dict(status='PREPARED_NOT_TRAINED', original_screen=str(ORIGINAL),
        original_screen_sha256=sha(ORIGINAL), adapted_screen_sha256=sha(output),
        adapter_sha256=sha(__file__), fixture_helper_sha256=sha(ROOT / 'current_api_fixtures.py'),
        changed_construction_calls=calls, ring_sample_api='explicit output_noise=True/False',
        scorer_changed=False, schedule_changed=False, thresholds_changed=False,
        hosts_changed=False, streams_changed=False)
    (ROOT / 'SCREEN-ADAPTER.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
