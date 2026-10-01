"""Prepare a prospective RA11 + current-PR155 R1 mechanism diagnostic.

This changes no frozen package and promotes no default. The recovery detector
and two recovery operations come directly from the current PR155 source.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
DEST = ROOT / 'pkg-RA11-R1'


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def extract(path, name, owner=None):
    text = path.read_text()
    tree = ast.parse(text)
    nodes = tree.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    start = min([node.lineno] + [n.lineno for n in getattr(node, 'decorator_list', [])])
    return '\n'.join(text.splitlines()[start - 1:node.end_lineno]) + '\n'


def main():
    assert not DEST.exists(), 'Use a new destination for a new preparation.'
    shutil.copytree(OLD / 'pkg-CB64-RA11', DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    pkg = DEST / 'particlegan'
    p = pkg / 'continuous.py'
    p.write_text(p.read_text().rstrip() + '\n\n\n' + extract(BASE / 'particlegan/continuous.py', 'OptimizerSurprise'))

    p = pkg / 'recipes.py'
    text = p.read_text()
    text = replace_once(text, '    reopen_signal: str = "data"\n', '    reopen_signal: str = "data"\n    reopen_anchor: str = "hold"\n')
    text = replace_once(text,
        '        if self.reopen_signal not in ("data", "none"):\n            raise ValueError("reopen_signal must be data or none")\n',
        '        if self.reopen_signal not in ("data", "none", "optimizer"):\n            raise ValueError("reopen_signal must be data, none or optimizer")\n'
        '        if self.reopen_anchor not in ("hold", "release"):\n            raise ValueError("reopen_anchor must be hold or release")\n'
        '        if self.reopen_anchor == "release" and self.reopen_signal != "optimizer":\n            raise ValueError("reopen_anchor release requires reopen_signal optimizer")\n')
    text = replace_once(text, '        result = asdict(self)\n',
        '        result = asdict(self)\n        if self.reopen_anchor == "hold":\n            result.pop("reopen_anchor")\n')
    p.write_text(text)

    p = pkg / 'training.py'
    text = p.read_text()
    text = replace_once(text, '        # particle_birth_death: Fisher-Rao moves',
        '        from .continuous import OptimizerSurprise\n'
        '        self.surprise = (OptimizerSurprise() if self.lr_settle is not None and recipe.reopen_signal == "optimizer" else None)\n'
        '        # particle_birth_death: Fisher-Rao moves')
    text = replace_once(text,
        '                reopen = self.controller.data_score > 3.\n            else:\n',
        '                reopen = self.controller.data_score > 3.\n'
        '            elif recipe.reopen_signal == "optimizer":\n'
        '                self.controller.observe_blind()\n'
        '                reopen = self.surprise.decide(self.completed_steps)\n'
        '                if reopen:\n                    self._reopen_moments()\n'
        '                if recipe.reopen_anchor == "release":\n                    self._anchor_release(reopen)\n'
        '            else:\n')
    text = replace_once(text,
        '        for group, rate, tester in zip(optimizer.param_groups, self.initial_lrs[index], self.lr_settle.testers[index]):',
        '        for j, (group, rate, tester) in enumerate(zip(optimizer.param_groups, self.initial_lrs[index], self.lr_settle.testers[index])):')
    text = replace_once(text,
        '                tester.observe(group["params"], group["lr"] / rate, step=self.completed_steps + 1)\n',
        '                tester.observe(group["params"], group["lr"] / rate, step=self.completed_steps + 1)\n'
        '                if self.surprise is not None:\n'
        '                    self.surprise.observe(f"{index}.{j}", (self.opt_g, self.opt_d)[index], group)\n')
    methods = '\n' + extract(BASE / 'particlegan/policy.py', '_anchor_release', 'UpdatePolicy')
    methods += '\n' + extract(BASE / 'particlegan/policy.py', '_reopen_moments', 'UpdatePolicy').replace('enumerate(self.optimizers)', 'enumerate((self.opt_g, self.opt_d))')
    text = replace_once(text, '    _STREAMS = ', methods + '\n    _STREAMS = ')
    text = replace_once(text,
        '            "schema": 5, "recipe": self.recipe.to_dict(),',
        '            **({"surprise": self.surprise.state_dict()} if self.surprise is not None else {}),\n'
        '            "schema": 5, "recipe": self.recipe.to_dict(),')
    text = replace_once(text,
        '        for name, values in state["models"].items():\n',
        '        if self.surprise is not None:\n            deepcopy(self.surprise).load_state_dict(state["surprise"])\n'
        '        for name, values in state["models"].items():\n')
    text = replace_once(text,
        '        self.initial_lrs, self.completed_steps = deepcopy(rates), steps\n',
        '        if self.surprise is not None:\n            self.surprise.load_state_dict(state["surprise"])\n'
        '        self.initial_lrs, self.completed_steps = deepcopy(rates), steps\n')
    ast.parse(text)
    p.write_text(text)

    config = json.loads((OLD / 'configs/overrides-CB64-RA11.json').read_text())
    config.update(lr=.00425, prior_lr_mult=2., d_lr_mult=1., reopen_signal='optimizer', reopen_anchor='release')
    (ROOT / 'configs').mkdir(exist_ok=True)
    (ROOT / 'configs/RA11-R1-historical-rates.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(DEST.rglob('*.py'))}
    changed = [p.name for p in sorted(pkg.glob('*.py'))
               if p.read_bytes() != (OLD / 'pkg-CB64-RA11/particlegan' / p.name).read_bytes()]
    receipt = dict(status='PREPARED_NOT_TRAINED', pr155_base='f459cb6d6aaaabeb1af076ec53ad7a963618de90',
                   package='pkg-RA11-R1', changed_modules=changed, hashes=files,
                   old_package_unchanged=True, default_promoted=False)
    (ROOT / 'R1-PREPARATION.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'hashes'}, sort_keys=True))


if __name__ == '__main__':
    main()
