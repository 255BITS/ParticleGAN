"""Read-only activation-branch audit of the saved1325 finite-bank secants.

This diagnoses whether two-sided finite differences cross LeakyReLU kinks.
The sixteen frozen banks, captured Adam metric and training state come from
``pr84_finite_bank_vr_diagnostic``; no optimizer or host update is run here.
"""

import argparse
from contextlib import contextmanager
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import torch


SCALES = (1e-4, 5e-5, 1e-7, 5e-8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def activation_capture(field):
    signs = []

    def record(_module, args):
        signs.append((args[0].detach() < 0).cpu().flatten())

    hooks = [module.register_forward_pre_hook(record)
             for model in (field.generator, field.critic)
             for module in model.modules()
             if isinstance(module, torch.nn.LeakyReLU)]
    try:
        yield signs
    finally:
        for hook in hooks:
            hook.remove()


def field_signs(field, point):
    with activation_capture(field) as signs:
        field.joint(point)
    return signs


def run(field, direction):
    rows = []
    for scale in SCALES:
        plus = field_signs(field, field.base + scale*direction)
        minus = field_signs(field, field.base - scale*direction)
        if len(plus) != len(minus):
            raise RuntimeError('different LeakyReLU call counts')
        if any(a.shape != b.shape for a, b in zip(plus, minus)):
            raise RuntimeError('different LeakyReLU input shapes')
        flips = sum(int(torch.count_nonzero(a ^ b)) for a, b in zip(plus, minus))
        elements = sum(a.numel() for a in plus)
        rows.append(dict(scale=scale, sign_flips=flips,
                         activation_inputs=elements, activation_calls=len(plus)))
    field.assign(field.base)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    initial = json.loads((args.initial/'result.json').read_text())
    source = initial['declaration']['source']
    for name, expected in source.items():
        if sha(root/name) != expected:
            raise RuntimeError(f'initial source changed: {name}')
    sys.path.insert(0, str(root))
    from reports.toy100 import pr84_finite_bank_vr_diagnostic as probe
    if sha(Path(probe.__file__)) != source['reports/toy100/pr84_finite_bank_vr_diagnostic.py']:
        raise RuntimeError('imported unexpected diagnostic source')
    if args.states.suffix == '.gz':
        bundle = torch.load(BytesIO(gzip.decompress(args.states.read_bytes())),
                            weights_only=True, map_location='cpu')
        if bundle['parent_states_sha256'] != probe.fit.STATE_SHA:
            raise RuntimeError('wrong original state archive')
        phases = bundle['states'][probe.STEP]
    else:
        if sha(args.states) != probe.fit.STATE_SHA:
            raise RuntimeError('wrong original state capture')
        phases = torch.load(args.states, weights_only=True, map_location='cpu')[probe.STEP]
    tensors = torch.load(args.initial/'tensors.pt', weights_only=True, map_location='cpu')
    recipe, _, _ = probe.declared_recipe(json.loads((root/probe.SOURCE_FILES[-1]).read_text()))
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        field = probe.FixedBankField(phases, tensors['d_rows'], tensors['g_rows'], recipe)
        direction = tensors['captured_delta'].clone()
        direction[:field.d_size] = 0
        rows = run(field, direction)
    if not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('global torch RNG changed')
    receipt = dict(scope='saved1325, 16 frozen native banks, G-only captured direction',
                   source_sha256=sha(Path(__file__)),
                   initial_result_sha256=sha(args.initial/'result.json'),
                   frozen_bank_tensors_sha256=sha(args.initial/'tensors.pt'),
                   states_sha256=sha(args.states), initial_source_sha256=source,
                   rows=rows, training_updates=0, global_rng_unchanged=True)
    (args.output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
