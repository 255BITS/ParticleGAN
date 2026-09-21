import unittest

import torch

from lib.gym_transition import (
    DirectPredictor, GymTransitionCritics, GymTransitionEncoder,
    GymTransitionGenerator, GymTransitionScaler, composed_transition,
    contact_record, contact_state, encoded_transition, real_reconstruction,
    state_reconstruction, synthetic_reconstruction,
)
from particlegan import get_recipe


class GymTransitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.real = torch.randn(32, 18)
        self.real[:, [6, 7, 16, 17]] = torch.randint(2, (32, 4)).float()
        self.real[:, 8:10].tanh_()
        self.terrain = torch.randn(32, 11)
        self.scaler = GymTransitionScaler.fit(self.real)
        self.prior = get_recipe("mog", z_dim=32).make_prior(num_particles=32)
        self.g = GymTransitionGenerator(self.scaler, width=16)
        self.e = GymTransitionEncoder(width=16)

    def test_shared_training_scaler_roundtrip_and_serialization(self):
        x = self.scaler(self.real)
        torch.testing.assert_close(self.scaler.inverse(x), self.real)
        torch.testing.assert_close(x[:, [6, 7, 16, 17]], self.real[:, [6, 7, 16, 17]])
        expected = torch.cat([self.real[:, :6], self.real[:, 10:16]])
        torch.testing.assert_close(self.scaler.state_mean, expected.mean(0))
        torch.testing.assert_close(self.scaler.state_scale, expected.std(0, unbiased=False))
        twin = self.real.clone(); twin[:, 10:] = twin[:, :8]
        normalized = self.scaler(twin)
        torch.testing.assert_close(normalized[:, :8], normalized[:, 10:])
        restored = GymTransitionScaler(**self.scaler.state_dict())
        torch.testing.assert_close(restored(self.real), x)
        self.assertEqual(float(GymTransitionScaler.fit(torch.zeros(1, 18)).state_scale.min()),
                         float(torch.tensor(1e-3)))
        with self.assertRaises(ValueError):
            GymTransitionScaler.fit(torch.empty(0, 18))

    def test_generators_are_independent_and_actions_physically_bounded(self):
        ids = [{id(p) for p in branch.parameters()} for branch in self.g.branches]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        z = torch.randn(32, 32)
        fake = self.g(z, self.terrain)
        self.assertEqual(fake.shape, (32, 18))
        action = self.scaler.inverse_action(fake[:, 8:10])
        self.assertTrue((action.abs() <= 1).all())
        changed = self.terrain.clone(); changed[:, 0] += 2
        changed_fake = self.g(z, changed)
        self.assertTrue(all(not torch.equal(fake[:, s], changed_fake[:, s])
                            for s in (slice(0, 8), slice(8, 10), slice(10, 18))))

    def test_contacts_are_binary_with_sigmoid_st_gradient_and_seeded_draw(self):
        logits = torch.zeros(4096, 8, requires_grad=True)
        rng = torch.Generator().manual_seed(98)
        sampled = contact_state(logits, rng=rng, straight_through=True)
        self.assertTrue(((sampled[:, 6:] == 0) | (sampled[:, 6:] == 1)).all())
        self.assertLess(abs(float(sampled[:, 6:].detach().mean())-.5), .025)
        sampled[:, 6:].sum().backward()
        torch.testing.assert_close(logits.grad[:, 6:], torch.full((4096, 2), .25))
        torch.testing.assert_close(logits.grad[:, :6], torch.zeros(4096, 6))
        repeated = contact_state(logits.detach(), rng=torch.Generator().manual_seed(98))
        torch.testing.assert_close(sampled, repeated)
        probabilities = contact_state(logits, mode="probability")
        torch.testing.assert_close(probabilities[:, 6:], torch.full((4096, 2), .5))
        threshold = contact_state(logits, mode="threshold")
        self.assertTrue((threshold[:, 6:] == 1).all())

    def test_encoder_observation_only_and_live_composition_gradient(self):
        state_action = self.scaler(self.real)[:, :10].requires_grad_()
        decoded, encoding = encoded_transition(self.e, self.g, self.prior, state_action, self.terrain)
        decoded.square().mean().backward()
        for module in (self.e, self.g, self.prior):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0)
        self.assertGreater(float(state_action.grad.norm()), 0)
        centers = self.prior.means()[encoding.indices[:, 0]]
        self.assertLessEqual(float((encoding.codes[:, 0]-centers).detach().abs().max()),
                             3*float(self.prior.sigma)+1e-6)
        self.assertEqual(float(encoding.kl.sum()), 0)
        with self.assertRaises(ValueError):
            self.e(self.scaler(self.real), self.terrain, self.prior)
        fake = contact_record(self.g(self.prior.sample(32)[0], self.terrain), straight_through=True)
        composed, _, _ = composed_transition(self.e, self.g, self.prior, fake, self.terrain,
                                              straight_through=True)
        torch.testing.assert_close(composed[:, :10], fake[:, :10])
        grads = torch.autograd.grad(composed[:, 10:16].square().mean(),
                                    [branch[0].weight for branch in self.g.branches])
        self.assertTrue(all(torch.isfinite(grad).all() and float(grad.norm()) > 0 for grad in grads))

    def test_real_role_weighting_and_detached_synthetic_targets(self):
        decoded = torch.randn(32, 18, requires_grad=True)
        real = self.scaler(self.real)
        loss, terms = real_reconstruction(decoded, real, continuous_weight=2., contact_weight=.25)
        torch.testing.assert_close(loss, (terms["state"]+terms["action"]+terms["next_state"])/3)
        torch.testing.assert_close(terms["state"], 2*terms["state_continuous"]+.25*terms["state_contact"])
        target = real.detach().requires_grad_()
        synthetic, st = synthetic_reconstruction(decoded, target)
        torch.testing.assert_close(synthetic, (st["state"]+st["action"])/2)
        target_grad, decoded_grad = torch.autograd.grad(synthetic, (target, decoded), allow_unused=True)
        self.assertIsNone(target_grad)
        torch.testing.assert_close(decoded_grad[:, 10:], torch.zeros_like(decoded_grad[:, 10:]))
        self.assertGreater(float(decoded_grad[:, :10].norm()), 0)
        with self.assertRaises(ValueError):
            state_reconstruction(decoded[:, :8], real[:, :8], contact_weight=-1)

    def test_shared_state_critic_roles_and_gradient(self):
        critics = GymTransitionCritics(width=16, marginal_width=8)
        self.assertEqual(len(critics.critics), 3)
        self.assertIs(critics.critic_for("state"), critics.critic_for("next_state"))
        twin = self.real.clone(); twin[:, 10:] = twin[:, :8]
        real = self.scaler(twin).requires_grad_()
        state, current = critics.inputs("state", real, self.terrain)
        successor, following = critics.inputs("next_state", real, self.terrain)
        torch.testing.assert_close(state, successor)
        torch.testing.assert_close(current[:, :-1], following[:, :-1])
        self.assertTrue((current[:, -1] == 0).all())
        self.assertTrue((following[:, -1] == 1).all())
        scores = []
        for role in critics.roles():
            score, logits = critics.critic_for(role)(*critics.inputs(role, real, self.terrain))
            self.assertEqual(logits.shape, (32, 1))
            scores.append(score.mean())
        (scores[0]+sum(scores[1:])/3).backward()
        self.assertGreater(float(real.grad.norm()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in critics.parameters()))

    def test_direct_parameter_match_and_prediction_shape(self):
        target = sum(p.numel() for module in (self.e, self.g.branches[2], self.prior) for p in module.parameters())
        direct = DirectPredictor(target_parameters=target)
        count = sum(p.numel() for p in direct.parameters())
        self.assertLess(abs(count-target)/target, .03)
        self.assertEqual(direct(self.scaler(self.real)[:, :10], self.terrain).shape, (32, 8))


if __name__ == "__main__":
    unittest.main()
