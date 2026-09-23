"""Paired noise, exact cap, removal of reconstruction gradients, and configuration."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.train_gym_slider_gan import DEFAULTS, train
from lib.gym_previous_gan import build_models as build_l2, decode, real_record, control_action_details
from lib.gym_slider_gan import build_models, hashes, paired_loss, error_loss, load_checkpoint
from lib.gym_state_control import training_recipe
from lib.gym_transition import GymTransitionScaler
from lib.vendor.concept_slider_core.reference import noise_std
import tests.test_gym_previous_gan as fixtures


class SliderGanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {**DEFAULTS, 'steps': 4, 'batch_size': 4, 'checkpoints': [4], 'device': 'cpu',
            'z_dim': 4, 'num_particles': 8, 'width': 8, 'encoder_width': 8, 'd_width': 8,
            'marginal_width': 8, 'error_tokens': 2, 'error_width': 8, 'error_heads': 2}

    def test_normalization_noise_and_exact_cap(self):
        with tempfile.TemporaryDirectory() as td:
            _, records = fixtures.PreviousGanTests().fixture(Path(td))
            triples = torch.from_numpy(np.concatenate([records[k] for k in ('states', 'actions', 'next_states')], 1))
            scaler = GymTransitionScaler.fit(triples)
            targets = scaler(triples)
            bundle = build_models(self.cfg, scaler, targets)
            critic = bundle['R']
            edits = targets - targets.mean(0)
            std = edits.std(0).clamp_min(1e-4)
            expected = std * (edits/std).square().mean(1).sqrt().median().clamp_min(1e-4)
            torch.testing.assert_close(critic.target_std, expected)
            self.assertAlmostEqual(float((edits/expected).square().mean(1).sqrt().median()), 1., places=5)
            self.assertAlmostEqual(critic.sigma(1), float(critic.edit_rms)/.28)
            self.assertEqual(noise_std(2500, start=4., decay_steps=2500, hold=1.), 1.)
            # Identical real/fake coordinates give exactly the paired logistic floor.
            zero_targets = targets.clone()
            zero_targets[:, [6,7,16,17]] = .5
            zero_logits = zero_targets.clone()
            zero_logits[:, [6,7,16,17]] = 0.
            loss, _ = error_loss(critic, zero_logits, zero_targets, 1, torch.Generator().manual_seed(9))
            self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
            cap = training_recipe(self.cfg).make_gradient_penalty(lazy_k=4, coeff=1., kappa=1.)
            x = torch.randn(4,18)
            f = lambda a: 2*a[:,0]
            off,_ = cap.penalty(f,x,x,step=3)
            on,_ = cap.penalty(f,x,x,step=4)
            self.assertEqual(float(off),0.)
            self.assertAlmostEqual(float(on),4.,places=5)
            # Actual attention critic supports second derivatives for the exact cap.
            tight = training_recipe(self.cfg).make_gradient_penalty(lazy_k=4,kappa=0.)
            penalty,_=tight.penalty(critic,x,x,step=4)
            penalty.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters()))

    def test_all_scope_has_only_error_gan_gradients_and_unchanged_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            _, records = fixtures.PreviousGanTests().fixture(Path(td))
            triples = np.concatenate([records[k] for k in ('states','actions','next_states')],1)
            scaler=GymTransitionScaler.fit(triples)
            bundle=build_models(self.cfg,scaler,scaler(torch.from_numpy(triples)))
            baseline=build_l2(self.cfg,scaler)
            for k,v in hashes(bundle).items():
                if k!='R':self.assertEqual(v,hashes({**baseline,'R':bundle['R']})[k])
            batch={k:torch.from_numpy(v[:4]) for k,v in records.items() if k not in ('episode_ids','steps')}
            decoded,_=decode(bundle,batch['states'],batch['previous_actions'],batch['terrain'])
            real=real_record(bundle,batch)
            bundle['R'].requires_grad_(False)
            total,terms=paired_loss(bundle,decoded,real,1,torch.Generator().manual_seed(10))
            reference,_=error_loss(bundle['R'],decoded,real,1,torch.Generator().manual_seed(10))
            torch.testing.assert_close(total,reference,atol=0,rtol=0)
            torch.testing.assert_close(torch.autograd.grad(total,decoded,retain_graph=True)[0],
                                       torch.autograd.grad(reference,decoded,retain_graph=True)[0],atol=0,rtol=0)
            self.assertFalse(any(terms[k].requires_grad for k in ('action_loss','state_loss','next_loss')))
            total.backward()
            for module in [bundle['E'],bundle['prior'],*bundle['G'].branches]:
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))
            self.assertTrue(all(p.grad is None for p in bundle['R'].parameters()))

    def test_both_scopes_train_and_restore(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path,records=fixtures.PreviousGanTests().fixture(root)
            for scope in ('all','action'):
                out=root/scope
                cfg={**self.cfg,'slider_scope':scope,'episodes':str(path),'out_dir':str(out),'live_log':str(root/'live.log')}
                summary=train(cfg)
                bundle=load_checkpoint(out/'final.pt')
                self.assertEqual(len(bundle['R'].target_std),18 if scope=='all' else 2)
                self.assertNotEqual(hashes(bundle)['R'],summary['provenance']['initial_parameters']['R'])
                saved_sigma=bundle['R'].sigma(4)
                restored=load_checkpoint(out/'checkpoint_4.pt')
                self.assertEqual(saved_sigma,restored['R'].sigma(4))
                inputs=[records[k][0] for k in ('states','previous_actions','terrain')]
                a,m=control_action_details(bundle,*inputs);b,n=control_action_details(restored,*inputs)
                np.testing.assert_array_equal(a,b);self.assertEqual(m,n)
