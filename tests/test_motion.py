import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.train_motion import DEFAULTS, validate
from lib.denoising_toy import DiffusionSchedule, DrawSource
from lib.grad_regularizers import GradRegularizer
from lib.motion import MotionData, identify, normalize_window, joints, motion_metrics, baselines
from lib.trajectory import TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryCritic, generate


class MotionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_subject_split_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            for subject in (1, 2, 3):
                for action in range(1, 13):
                    rng = np.random.default_rng(subject*100+action)
                    x = rng.normal(size=(30, 24, 3)).astype(np.float32)
                    np.save(Path(directory)/f'P{subject:02}G01R01F0000T0030A{action:02}01.npy', x)
            cfg = {**DEFAULTS, 'data_dir': directory, 'expected_clips': 36,
                   'train_subjects': [1], 'validation_subjects': [2], 'test_subjects': [3]}
            data = MotionData(cfg)
            c, context, future = data.evaluation('test', 2)[:3]
            self.assertEqual(future.shape, (12, 72, 16))
            self.assertEqual(set(c.tolist()), set(range(12)))
            past = joints(context.reshape(12, 72, 8))
            torch.testing.assert_close(past[:, -1, 0], torch.zeros(12, 3))
            for path in Path(directory).glob('P03*.npy'):
                np.save(path, np.load(path)*100)
            changed = MotionData(cfg)
            self.assertEqual(data.scale, changed.scale)
            self.assertNotEqual(data.manifest['fingerprint'], changed.manifest['fingerprint'])
            self.assertEqual({data.rows[int(i)]['subject'] for i in data.group_indices}, {1})
            with self.assertRaises(ValueError):
                MotionData({**cfg, 'test_subjects': [1,3]})

    def test_future_cannot_change_observed_context(self):
        x = np.random.default_rng(4).normal(size=(24, 24, 3))
        a = normalize_window(x, 8, .4)
        x[8:] += 1000
        b = normalize_window(x, 8, .4)
        np.testing.assert_array_equal(a[:8], b[:8])
        np.testing.assert_allclose(a[8,0]-a[7,0], (x[8,0]-1000-x[7,0])/.4)
        self.assertEqual(identify('P10G01R02F0345T0464A0201.npy')['recording'], 'P10G01R02')

    def test_metrics_distinguish_motion_diversity_and_distortion(self):
        pose = torch.randn(4, 24, 3)
        full = pose[:,None]+torch.arange(24)[None,:,None,None]*.02
        past = full[:,:8].flatten(2).transpose(1,2).flatten(1)
        real = full[:,8:].flatten(2).transpose(1,2)
        c = torch.arange(4)%2
        refs = baselines(real, past, 8)
        torch.testing.assert_close(refs['constant_velocity'], real, atol=2e-6, rtol=1e-5)
        m = motion_metrics(real[:,None].repeat(1,2,1,1), real, past, 8, c)
        self.assertEqual(m['single_ade'], 0)
        self.assertEqual(m['diversity'], 0)
        self.assertLess(m['bone_relative_error'], 1e-6)
        self.assertLess(m['acceleration'], 1e-6)
        samples = torch.stack([real, real+.5], 1)
        varied = motion_metrics(samples, real, past, 8, c)
        self.assertEqual(varied['best_of_k_ade'], 0)
        self.assertGreater(varied['diversity'], .8)
        self.assertGreater(varied['boundary_acceleration'], .4)
        broken = motion_metrics((real*3)[:,None], real, past, 8, c)
        self.assertGreater(broken['bone_relative_error'], 1.9)

    def test_motion_shape_joint_ucd_bcap_and_particle_gradients(self):
        cfg = {**DEFAULTS, 'width': 8, 'd_width': 32, 'd_temporal_width': 8}
        rng = torch.Generator().manual_seed(7)
        for arch in ('hybrid', 'mlp'):
            cfg['d_architecture'] = arch
            g, d = TrajectoryGenerator(cfg), TrajectoryDiscriminator(cfg)
            c, t = torch.arange(12), torch.arange(12)%4+1
            context, x = torch.randn(12,576), torch.randn(12,72,16)
            schedule = DiffusionSchedule(cfg['alpha_bar'])
            real, xt = schedule.forward_pair(x, t, rng)
            z = torch.randn(12,32,requires_grad=True)
            clean = g(z,c,context,xt,t)
            fake = schedule.reverse(clean, xt, t, torch.randn_like(clean))
            torch.testing.assert_close(fake[t==1], clean[t==1], atol=0, rtol=0)
            d(fake,c,context,xt,t)[0].mean().backward()
            self.assertGreater(float(z.grad.norm()), 0)
            logits = d(real,c,context,xt,t)[1]
            self.assertEqual(logits.shape,(12,48))
            torch.testing.assert_close(logits, d(real,11-c,context,xt,5-t)[1])
            self.assertEqual(d.ucd_labels(c,t).tolist(), ((t-1)*12+c).tolist())
            d.zero_grad()
            reg=GradRegularizer('b_cap',1.,kappa=0.,lazy_k=4)
            reg.penalty(TrajectoryCritic(d,c,context,xt,t),real,fake.detach(),4,rng,collect_stats=False)[0].backward()
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in d.parameters()))
            if arch=='hybrid':
                self.assertGreater(float(d.stages[0][0].weight.grad.norm()),0)
                self.assertGreater(float(d.global_net[0].weight.grad.norm()),0)
            prior=DrawSource('learned',16,32,1,'cpu')
            noise=DrawSource('gaussian',16,72*16,2,'cpu')
            output=generate(g,prior,noise,schedule,c,context,[rng,rng,rng])
            self.assertEqual(output.shape,x.shape)
            self.assertTrue(bool(torch.isfinite(output).all()))

    def test_config_validation(self):
        validate(DEFAULTS)
        for bad in ({'context_dim':18},{'channels':2},{'eval_samples':1},{'past_length':0},
                    {'test_subjects':[9,10]},{'length':17}):
            with self.assertRaises(ValueError):
                validate({**DEFAULTS,**bad})


if __name__ == '__main__':
    unittest.main()
