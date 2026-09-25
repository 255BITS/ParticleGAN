"""Previous-command alignment, GAN gradient scopes, and saved inference parity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


from experiments.train_gym_previous_gan import DEFAULTS, train
from lib.gym_control import build_expert_records
from lib.gym_state_control import training_recipe
from lib.gym_transition import GymTransitionScaler
from lib.gym_previous_gan import (build_models, decode, fake_paths, real_record,
    adversarial_loss, control_action_details, hashes, load_checkpoint)


class PreviousGanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {**DEFAULTS, 'steps': 2, 'batch_size': 4, 'checkpoints': [1, 2],
                    'z_dim': 4, 'num_particles': 8, 'width': 8, 'encoder_width': 8,
                    'd_width': 8, 'marginal_width': 8, 'device': 'cpu'}

    def fixture(self, root):
        rng = np.random.default_rng(17)
        episodes = []
        for eid in range(3):
            states = rng.normal(size=(6, 8)).astype(np.float32)
            states[:, 6:] = rng.integers(0, 2, size=(6, 2))
            actions = rng.uniform(-1, 1, size=(6, 2)).astype(np.float32)
            episodes.append(dict(episode_id=eid, split='train' if eid < 2 else 'test', behavior='heuristic',
                states=states.tolist(), actions=actions.tolist(), next_states=states.tolist(), terrain=[-.5]*11))
        path = root / 'episodes.json'
        path.write_text(json.dumps(episodes))
        records = build_expert_records(path)
        return path, records

    def bundle_batch(self, records):
        triples = np.concatenate([records[k] for k in ('states', 'actions', 'next_states')], 1)
        bundle = build_models(self.cfg, GymTransitionScaler.fit(triples))
        batch = {k: torch.from_numpy(v[:4]) for k,v in records.items() if k not in ('episode_ids', 'steps')}
        return bundle, batch

    def test_previous_alignment_and_input_contract(self):
        with tempfile.TemporaryDirectory() as td:
            _, records = self.fixture(Path(td))
            self.assertEqual(len(records['states']), 12)
            for offset in (0, 6):
                np.testing.assert_array_equal(records['previous_actions'][offset], [-1., 0.])
                np.testing.assert_array_equal(records['previous_actions'][offset+1:offset+6], records['actions'][offset:offset+5])
            bundle, batch = self.bundle_batch(records)
            previous = batch['previous_actions'].clone().requires_grad_(True)
            output, _ = decode(bundle, batch['states'], previous, batch['terrain'])
            gradient = torch.autograd.grad(output[:, 8:10].sum(), previous)[0]
            self.assertGreater(float(gradient.abs().sum()), 0.)
            # Current action and successor targets cannot enter the encoder API.
            replay, _ = decode(bundle, batch['states'], previous, batch['terrain'])
            batch['actions'].fill_(999.)
            batch['next_states'].fill_(999.)
            changed, _ = decode(bundle, batch['states'], previous, batch['terrain'])
            torch.testing.assert_close(replay, changed, atol=0, rtol=0)

    def test_gan_reaches_control_encoder_and_all_generators(self):
        def has_grad(module):
            return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
        with tempfile.TemporaryDirectory() as td:
            _, records = self.fixture(Path(td))
            bundle, batch = self.bundle_batch(records)
            recipe = training_recipe(self.cfg)
            for path in ('prior', 'encoded'):
                for key in ('G', 'E', 'prior', 'D'):
                    bundle[key].zero_grad(set_to_none=True)
                bundle['D'].requires_grad_(False)
                fakes, _ = fake_paths(bundle, batch, torch.Generator().manual_seed(8), torch.Generator().manual_seed(9), True)
                loss, _ = adversarial_loss(bundle['D'], real_record(bundle, batch), {path: fakes[path]}, batch['terrain'], recipe.make_loss())
                loss.backward()
                self.assertTrue(all(has_grad(g) for g in bundle['G'].branches))
                self.assertTrue(has_grad(bundle['prior']))
                self.assertEqual(has_grad(bundle['E']), path == 'encoded')
                self.assertFalse(has_grad(bundle['D']))
            for key in ('G', 'E', 'prior', 'D'):
                bundle[key].zero_grad(set_to_none=True)
            bundle['D'].requires_grad_(True)
            fakes, _ = fake_paths(bundle, batch, torch.Generator().manual_seed(8), torch.Generator().manual_seed(9), True)
            loss, _ = adversarial_loss(bundle['D'], real_record(bundle, batch), fakes, batch['terrain'], recipe.make_loss(),
                reg=recipe.make_critic_regularizer(bundle['D']), rngs={r: torch.Generator().manual_seed(7) for r in bundle['D'].roles()})
            loss.backward()
            self.assertTrue(all(has_grad(d) for d in bundle['D'].critics.values()))
            self.assertTrue(all(not has_grad(bundle[k]) for k in ('G', 'E', 'prior')))

    def test_training_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, records = self.fixture(root)
            cfg = {**self.cfg, 'episodes': str(path), 'out_dir': str(root / 'run'), 'live_log': str(root / 'live.log')}
            summary = train(cfg)
            self.assertEqual(summary['gan_steps'], 2)
            self.assertEqual(summary['simulator_calls'], 0)
            self.assertEqual(summary['real_draws'], 16)
            saved = load_checkpoint(root / 'run/final.pt')
            for key,value in hashes(saved).items():
                self.assertNotEqual(value, summary['provenance']['initial_parameters'][key])
            replay = load_checkpoint(root / 'run/checkpoint_2.pt')
            inputs = [records[k][0] for k in ('states', 'previous_actions', 'terrain')]
            action, metadata = control_action_details(saved, *inputs)
            action2, metadata2 = control_action_details(replay, *inputs)
            np.testing.assert_array_equal(action, action2)
            self.assertEqual(metadata, metadata2)
            self.assertTrue(np.isfinite(action).all() and (np.abs(action) <= 1).all())


if __name__ == '__main__':
    unittest.main()
