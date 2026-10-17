"""Export tiny public-API software states from a frozen source; no task scoring.

Checkpoints and tensors go to the external review archive, never into Git.
Two fixed CPU updates test compatibility; this supplies no scientific gate.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from torch import nn
    from particlegan import Recipe, get_recipe
    from experiments.forge.api import FormulationContext

    leader = json.loads((args.source / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())['recipe_overrides']
    cases = [('bcap_zero', 'bcap', {}), ('bcap_vicreg', 'bcap', {'prior_reg': .1}),
             ('bcap_frozen', 'bcap', {'prior_reg': .1, 'prior_update': 'frozen'}),
             ('bcap_none', 'bcap', {'prior_regularizer': 'none'}),
             ('ka2_zero', 'gan', {}), ('ka2_vicreg', 'gan', {'prior_reg': .1})]
    if args.role != 'develop':
        delta = ({'prior_proximal_strength': 1} if args.role == 'proximal' else
                 {'prior_regularizer': args.role, 'prior_reg': .001})
        cases = [(args.role + '_positive', 'bcap', delta)]
    packets = {}
    for name, preset, delta in cases:
        torch.manual_seed(0)
        settings = {**(leader if preset == 'bcap' else {}),
                    'num_particles': 12, 'z_dim': 2, 'batch_size': 4, 'total_steps': 4, **delta}
        ctx = FormulationContext(recipe_preset=preset, recipe_overrides=settings,
            prior=dict(kind='mog', sigma=.025, standardize=False), prior_contract='recipe_owned_v1', seed=0)
        g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component='generator')
        d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component='discriminator')
        trainer = ctx.build_trainer(g, d)
        initial = deepcopy(ctx.state_dict())
        real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
        first = deepcopy(trainer.step(real))
        checkpoint = deepcopy(ctx.state_dict())
        second = deepcopy(trainer.step(real))
        packets[name] = dict(preset=preset, settings=settings, initial=initial,
                            first_metrics=first, checkpoint=checkpoint,
                            second_metrics=second, final=deepcopy(ctx.state_dict()),
                            global_rng=torch.get_rng_state().clone())
    presets = {name: get_recipe(name).to_dict() for name in
               ('gan', 'mog', 'ddgan', 'ddgan_mog', 'ae_gan', 'vae_gan', 'ae_ddgan', 'bcap')}
    packets['_presets'] = presets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packets, args.output)
    print(json.dumps(dict(role=args.role, cases=len(cases), steps_per_case=2, task_scoring=False,
                         output=str(args.output))))


if __name__ == '__main__':
    main()
