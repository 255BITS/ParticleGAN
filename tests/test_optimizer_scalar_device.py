"""CUDA counters follow optimizer flags; closures keep the caller's context."""
import unittest

import torch

from particlegan.k3p import K3PGeneratorAdam
from particlegan.ka2 import KA2CriticAdam


def _run(kind, options, ambient):
    device = torch.device('cuda', torch.cuda.current_device())
    with torch.device(ambient):
        parameter = torch.nn.Parameter(torch.tensor([.25, -.75], device=device))
        kwargs = dict(lr=.007, betas=(.2, .9), eps=1e-8,
                      weight_decay=.12, amsgrad=True, **options)
        if kind == 'reference':
            optimizer = torch.optim.Adam([parameter], **kwargs)
        elif kind == 'generator':
            optimizer = K3PGeneratorAdam([parameter], **kwargs)
        else:
            critic = torch.nn.Module()
            critic.register_parameter('weight', parameter)
            optimizer = KA2CriticAdam([parameter], critic=critic, guard_ratio=0, **kwargs)
        closure_devices = []

        def closure():
            closure_devices.append(torch.empty(0).device)
            return torch.zeros((), device=device)

        for gradient in ([.2, -.1], [-.05, .3], [.125, -.25]):
            parameter.grad = torch.tensor(gradient, device=device)
            optimizer.step(closure)
        return parameter.detach().clone(), optimizer.state[parameter], closure_devices


@unittest.skipUnless(torch.cuda.is_available(), 'requires actual CUDA')
class OptimizerScalarDeviceTests(unittest.TestCase):
    def test_counter_placement_numerics_and_closure_context(self):
        device = torch.device('cuda', torch.cuda.current_device())
        configurations = ({'foreach': False},
                          {'capturable': True, 'foreach': False},
                          {'fused': True})
        for kind in ('generator', 'critic'):
            for options in configurations:
                with self.subTest(kind=kind, options=options):
                    reference, reference_state, _ = _run('reference', options, 'cpu')
                    left, left_state, left_closures = _run(kind, options, 'cpu')
                    right, right_state, right_closures = _run(kind, options, device)
                    expected = device if options.get('capturable') or options.get('fused') else torch.device('cpu')
                    self.assertEqual(left_state['step'].device, expected)
                    self.assertEqual(right_state['step'].device, expected)
                    self.assertEqual(left_closures, [torch.device('cpu')] * 3)
                    self.assertEqual(right_closures, [device] * 3)
                    torch.testing.assert_close(left, reference, rtol=0, atol=0)
                    torch.testing.assert_close(right, reference, rtol=0, atol=0)
                    for key in reference_state:
                        self.assertEqual(left_state[key].device, reference_state[key].device)
                        self.assertEqual(right_state[key].device, reference_state[key].device)
                        torch.testing.assert_close(left_state[key], reference_state[key], rtol=0, atol=0)
                        torch.testing.assert_close(right_state[key], reference_state[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
